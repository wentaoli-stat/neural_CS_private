#!/usr/bin/env python3
"""Fair Linear versus Positive(s, anchor) Stage-1 training.

Both arms start from the same untrained annual readout, receive identical
complete-dataset minibatches, and take the same number of optimizer updates.
Positive uses microbatch accumulation only to fit all 3,081 pair scores in
memory. Fixed-final EMA is primary; validation-best is secondary.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from maxstable_rainfall79.core import (
    DistanceBinMLPFullDatasetScore,
    DistanceBinPairMLPFullDatasetScore,
    FormalConfig,
    PairDesign,
    PositiveScoreAnchorPairGate,
    SELECTED_PROTOCOL_VERSION,
    TrainConfig,
    feature_statistics,
    jsonable,
    save_selected_stage1_checkpoint,
    set_seed,
    transform_inputs,
)


METHODS = ("linear", "positive_anchor")


def clone_state(model: torch.nn.Module, device: str = "cpu") -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device).clone()
        for key, value in model.state_dict().items()
    }


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def update_ema(
    ema: dict[str, torch.Tensor], model: torch.nn.Module, decay: float
) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if torch.is_floating_point(value):
                ema[key].mul_(float(decay)).add_(value, alpha=1.0 - float(decay))
            else:
                ema[key].copy_(value)


def load_prepared(prepared: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    compact = np.load(prepared / metadata["arrays"], allow_pickle=False)
    arrays: dict[str, np.ndarray] = {key: np.asarray(compact[key]) for key in compact.files}
    if metadata.get("array_format") != "separate_npy_memmap":
        raise ValueError("confirmatory runner requires separate pair/group memmaps")
    for key in (
        "train_feature",
        "val_feature",
        "train_grouped_feature",
        "val_grouped_feature",
    ):
        path = Path(metadata[key])
        if not path.is_absolute():
            path = prepared / path
        arrays[key] = np.load(path, mmap_mode="r")
    return metadata, arrays


def make_statistics(
    train_grouped: np.ndarray,
    train_anchor_w: np.ndarray,
    pair_mean: np.ndarray,
    pair_sd: np.ndarray,
    group_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    stats = feature_statistics(train_grouped, train_anchor_w)
    pair_mean = np.asarray(pair_mean, dtype=np.float64)
    pair_sd = np.asarray(pair_sd, dtype=np.float64)
    if pair_mean.shape != stats["feature_mean"].shape or pair_sd.shape != pair_mean.shape:
        raise ValueError("prepared pair statistics do not match grouped feature shape")
    stats["pair_feature_mean"] = pair_mean
    stats["pair_feature_sd"] = pair_sd
    stats["pair_block_mean"] = (stats["feature_mean"] - pair_mean) / pair_sd
    stats["pair_block_sd"] = np.where(
        stats["feature_sd"] / pair_sd < 1e-8,
        1.0,
        stats["feature_sd"] / pair_sd,
    )
    counts = np.bincount(
        np.asarray(group_ids, dtype=np.int64), minlength=pair_mean.shape[2]
    ).astype(np.float64)
    weights = counts / counts.sum()
    group_mean = pair_mean.reshape(pair_mean.shape[2], pair_mean.shape[3])
    group_sd = pair_sd.reshape(pair_sd.shape[2], pair_sd.shape[3])
    global_mean = np.sum(weights[:, None] * group_mean, axis=0)
    global_second = np.sum(
        weights[:, None] * (np.square(group_sd) + np.square(group_mean)), axis=0
    )
    global_sd = np.sqrt(np.maximum(global_second - np.square(global_mean), 1e-12))
    stats["gate_score_mean"] = global_mean
    stats["gate_score_sd"] = global_sd
    return {key: np.asarray(value) for key, value in stats.items()}


def pair_standardizers(
    stats: dict[str, np.ndarray], group_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(group_ids, dtype=np.int64)
    mean = np.asarray(stats["pair_feature_mean"], dtype=np.float32)[:, :, ids, :]
    sd = np.asarray(stats["pair_feature_sd"], dtype=np.float32)[:, :, ids, :]
    return mean, sd


def pair_batch(
    raw: np.ndarray,
    index: np.ndarray | slice,
    mean: np.ndarray,
    sd: np.ndarray,
    device: str,
) -> torch.Tensor:
    value = np.asarray(raw[index], dtype=np.float32)
    value = (value - mean) / sd
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def build_models(
    config: FormalConfig,
    design: PairDesign,
    stats: dict[str, np.ndarray],
    training: TrainConfig,
    seed: int,
    device: str,
) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    set_seed(seed)
    common_core = DistanceBinMLPFullDatasetScore(
        config.theta_dim,
        training.hidden,
        training.depth,
        design.group_geometry,
        design.population_counts,
    )
    common_state = clone_state(common_core)
    linear = DistanceBinPairMLPFullDatasetScore(
        config.theta_dim,
        training.hidden,
        training.depth,
        design.group_geometry,
        design.population_counts,
        design.group_ids,
        stats["pair_block_mean"],
        stats["pair_block_sd"],
    )
    linear.core.load_state_dict(common_state)

    # The positive gate is exactly identity at initialization.
    gate_seed = int(seed) + 1_000_003
    set_seed(gate_seed)
    positive_anchor = PositiveScoreAnchorPairGate(
        linear,
        training.gate_hidden,
        stats["pair_feature_mean"],
        stats["pair_feature_sd"],
        stats["feature_sd"],
        stats["gate_score_mean"],
        stats["gate_score_sd"],
    )
    models = {
        "linear": linear.to(device),
        "positive_anchor": positive_anchor.to(device),
    }
    core_hashes = {
        method: state_sha256(
            clone_state(model.core if method == "linear" else model.core.core)
        )
        for method, model in models.items()
    }
    if len(set(core_hashes.values())) != 1:
        raise RuntimeError("the two arms did not receive the same outer-core init")
    return models, {
        "common_outer_core_initial_sha256": next(iter(core_hashes.values())),
        "gate_hidden_initialization_seed": gate_seed,
    }


@torch.no_grad()
def validation_mse(
    method: str,
    model: torch.nn.Module,
    grouped_score: np.ndarray,
    anchor: np.ndarray,
    target: np.ndarray,
    raw_pair: np.ndarray,
    pair_mean: np.ndarray,
    pair_sd: np.ndarray,
    *,
    grouped_batch_size: int,
    pair_microbatch_size: int,
    device: str,
) -> float:
    model.eval()
    squared = 0.0
    count = 0
    batch_size = grouped_batch_size if method == "linear" else pair_microbatch_size
    for start in range(0, len(target), int(batch_size)):
        stop = min(start + int(batch_size), len(target))
        anchor_batch = torch.as_tensor(anchor[start:stop], dtype=torch.float32, device=device)
        target_batch = torch.as_tensor(target[start:stop], dtype=torch.float32, device=device)
        if method == "linear":
            score = torch.as_tensor(
                grouped_score[start:stop], dtype=torch.float32, device=device
            )
            prediction = model.core.forward_from_tokens(score, anchor_batch)
        else:
            score = pair_batch(raw_pair, slice(start, stop), pair_mean, pair_sd, device)
            prediction = model(score, anchor_batch)
        squared += float((prediction - target_batch).square().sum().cpu())
        count += int(target_batch.numel())
    return squared / count


def train_one_step(
    method: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    formal_index: np.ndarray,
    grouped_score: np.ndarray,
    anchor: np.ndarray,
    target: np.ndarray,
    raw_pair: np.ndarray,
    pair_mean: np.ndarray,
    pair_sd: np.ndarray,
    *,
    microbatch_size: int,
    grad_clip: float,
    device: str,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    formal_size = int(len(formal_index))
    theta_dim = int(target.shape[1])
    loss_total = 0.0
    if method == "linear":
        score = torch.as_tensor(
            grouped_score[formal_index], dtype=torch.float32, device=device
        )
        anchor_batch = torch.as_tensor(
            anchor[formal_index], dtype=torch.float32, device=device
        )
        target_batch = torch.as_tensor(
            target[formal_index], dtype=torch.float32, device=device
        )
        prediction = model.core.forward_from_tokens(score, anchor_batch)
        loss = (prediction - target_batch).square().sum() / (formal_size * theta_dim)
        loss.backward()
        loss_total = float(loss.detach().cpu())
    else:
        for start in range(0, formal_size, int(microbatch_size)):
            selected = formal_index[start : start + int(microbatch_size)]
            score = pair_batch(raw_pair, selected, pair_mean, pair_sd, device)
            anchor_batch = torch.as_tensor(
                anchor[selected], dtype=torch.float32, device=device
            )
            target_batch = torch.as_tensor(
                target[selected], dtype=torch.float32, device=device
            )
            prediction = model(score, anchor_batch)
            loss = (prediction - target_batch).square().sum() / (
                formal_size * theta_dim
            )
            loss.backward()
            loss_total += float(loss.detach().cpu())
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip)).detach().cpu()
    )
    optimizer.step()
    return loss_total, grad_norm


@torch.no_grad()
def identity_audit(
    models: dict[str, torch.nn.Module],
    raw_pair: np.ndarray,
    grouped_score: np.ndarray,
    anchor: np.ndarray,
    pair_mean: np.ndarray,
    pair_sd: np.ndarray,
    device: str,
) -> dict[str, float]:
    n = min(4, len(anchor))
    pair = pair_batch(raw_pair, slice(0, n), pair_mean, pair_sd, device)
    anchor_batch = torch.as_tensor(anchor[:n], dtype=torch.float32, device=device)
    linear_pair = models["linear"](pair, anchor_batch)
    output = {}
    for method, model in models.items():
        if method == "linear":
            continue
        output[f"{method}_vs_pair_linear_max_abs"] = float(
            (model(pair, anchor_batch) - linear_pair).abs().max().cpu()
        )
    grouped = torch.as_tensor(grouped_score[:n], dtype=torch.float32, device=device)
    linear_grouped = models["linear"].core.forward_from_tokens(grouped, anchor_batch)
    output["cached_group_vs_pair_replay_max_abs"] = float(
        (linear_grouped - linear_pair).abs().max().cpu()
    )
    return output


@torch.no_grad()
def gate_diagnostics(
    method: str,
    model: torch.nn.Module,
    raw_pair: np.ndarray,
    anchor: np.ndarray,
    pair_mean: np.ndarray,
    pair_sd: np.ndarray,
    *,
    limit: int,
    microbatch_size: int,
    device: str,
) -> dict[str, float]:
    n = len(raw_pair) if int(limit) <= 0 else min(int(limit), len(raw_pair))
    total = 0.0
    total_square = 0.0
    count = 0
    minimum = float("inf")
    maximum = float("-inf")
    sign_flips = 0
    for start in range(0, n, int(microbatch_size)):
        stop = min(start + int(microbatch_size), n)
        score = pair_batch(raw_pair, slice(start, stop), pair_mean, pair_sd, device)
        anchor_batch = torch.as_tensor(anchor[start:stop], dtype=torch.float32, device=device)
        raw = model.raw_score(score)
        if method != "positive_anchor":
            raise ValueError(f"unsupported nonlinear method: {method}")
        value = model.gate_values(raw, anchor_batch)
        transformed = raw * value
        value64 = value.to(torch.float64)
        total += float(value64.sum().cpu())
        total_square += float(value64.square().sum().cpu())
        count += int(value.numel())
        minimum = min(minimum, float(value.min().cpu()))
        maximum = max(maximum, float(value.max().cpu()))
        sign_flips += int(((raw * transformed) < 0.0).sum().cpu())
    mean = total / count
    return {
        "n_validation_datasets": n,
        "value_mean": mean,
        "value_sd": float(np.sqrt(max(total_square / count - mean * mean, 0.0))),
        "value_min": minimum,
        "value_max": maximum,
        "transformed_coordinate_sign_flip_fraction": sign_flips / count,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--iters", type=int, default=10_000)
    parser.add_argument("--formal-batch-size", type=int, default=128)
    parser.add_argument("--pair-microbatch-size", type=int, default=32)
    parser.add_argument("--validation-every", type=int, default=1_000)
    parser.add_argument("--grouped-validation-batch-size", type=int, default=512)
    parser.add_argument("--pair-validation-microbatch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--gate-diagnostic-limit",
        type=int,
        default=0,
        help="0 audits the complete validation support",
    )
    args = parser.parse_args(argv)

    prepared = args.prepared_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    metadata, arrays = load_prepared(prepared)
    if metadata.get("individual_pair_storage_dtype") != "float32":
        raise ValueError("confirmatory rerun requires a float32 individual-pair cache")
    config = FormalConfig(**metadata["config"])
    design = PairDesign(
        pairs=np.asarray(arrays["design_pairs"]),
        group_ids=np.asarray(arrays["design_group_ids"]),
        group_geometry=np.asarray(arrays["design_group_geometry"]),
        population_counts=np.asarray(arrays["design_population_counts"]),
    )
    train_raw = arrays["train_feature"]
    val_raw = arrays["val_feature"]
    train_grouped = arrays["train_grouped_feature"]
    val_grouped = arrays["val_grouped_feature"]
    train_anchor_w = np.asarray(arrays["train_anchor_w"])
    val_anchor_w = np.asarray(arrays["val_anchor_w"])
    train_target = np.asarray(arrays["train_target"], dtype=np.float32)
    val_target = np.asarray(arrays["val_target"], dtype=np.float32)
    stats = make_statistics(
        train_grouped,
        train_anchor_w,
        arrays["pair_feature_mean"],
        arrays["pair_feature_sd"],
        design.group_ids,
    )
    train_grouped_score, train_anchor = transform_inputs(
        train_grouped, train_anchor_w, stats
    )
    val_grouped_score, val_anchor = transform_inputs(val_grouped, val_anchor_w, stats)
    pair_mean, pair_sd = pair_standardizers(stats, design.group_ids)
    training = TrainConfig(
        sigma_q=float(metadata["sigma_q"]),
        hidden=int(args.hidden),
        depth=int(args.depth),
        gate_hidden=int(args.gate_hidden),
        iters=int(args.iters),
        batch_size=int(args.formal_batch_size),
        lr=float(args.lr),
        gate_lr=float(args.lr),
        joint_lr=float(args.lr),
        gate_only_steps=0,
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        patience=0,
        print_every=int(args.validation_every),
        ema_decay=float(args.ema_decay),
    )
    all_models, initialization = build_models(
        config, design, stats, training, int(args.seed), device
    )
    selected_methods = METHODS
    models = {method: all_models[method] for method in selected_methods}
    audit_models = {"linear": all_models["linear"], **models}
    identity = identity_audit(
        audit_models, val_raw, val_grouped_score, val_anchor, pair_mean, pair_sd, device
    )
    if any(
        identity[f"{method}_vs_pair_linear_max_abs"] > 2e-6
        for method in selected_methods
        if method != "linear"
    ):
        raise RuntimeError(f"raw nonlinear arms are not nested: {identity}")

    optimizers = {
        method: torch.optim.AdamW(
            model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
        for method, model in models.items()
    }
    ema = {
        method: {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        for method, model in models.items()
    }
    best_value = {method: float("inf") for method in selected_methods}
    best_state: dict[str, dict[str, torch.Tensor]] = {}
    best_selection: dict[str, dict[str, Any]] = {}
    traces: dict[str, list[dict[str, Any]]] = {method: [] for method in selected_methods}
    clip_count = {method: 0 for method in selected_methods}
    step_count = {method: 0 for method in selected_methods}
    batch_digest = hashlib.sha256()
    rng = np.random.default_rng(int(args.seed))

    for step in range(1, int(args.iters) + 1):
        formal_index = rng.integers(
            0, len(train_target), size=int(args.formal_batch_size), dtype=np.int64
        )
        batch_digest.update(formal_index.tobytes())
        train_metrics: dict[str, tuple[float, float]] = {}
        for method in selected_methods:
            loss, grad_norm = train_one_step(
                method,
                models[method],
                optimizers[method],
                formal_index,
                train_grouped_score,
                train_anchor,
                train_target,
                train_raw,
                pair_mean,
                pair_sd,
                microbatch_size=int(args.pair_microbatch_size),
                grad_clip=float(args.grad_clip),
                device=device,
            )
            update_ema(ema[method], models[method], float(args.ema_decay))
            train_metrics[method] = (loss, grad_norm)
            step_count[method] += 1
            clip_count[method] += int(grad_norm > float(args.grad_clip))

        if step == 1 or step % int(args.validation_every) == 0 or step == int(args.iters):
            for method in selected_methods:
                raw_state = clone_state(models[method], device=device)
                raw_val = validation_mse(
                    method,
                    models[method],
                    val_grouped_score,
                    val_anchor,
                    val_target,
                    val_raw,
                    pair_mean,
                    pair_sd,
                    grouped_batch_size=int(args.grouped_validation_batch_size),
                    pair_microbatch_size=int(args.pair_validation_microbatch_size),
                    device=device,
                )
                models[method].load_state_dict(ema[method])
                ema_val = validation_mse(
                    method,
                    models[method],
                    val_grouped_score,
                    val_anchor,
                    val_target,
                    val_raw,
                    pair_mean,
                    pair_sd,
                    grouped_batch_size=int(args.grouped_validation_batch_size),
                    pair_microbatch_size=int(args.pair_validation_microbatch_size),
                    device=device,
                )
                models[method].load_state_dict(raw_state)
                selected_kind = "raw" if raw_val <= ema_val else "ema"
                selected_value = min(raw_val, ema_val)
                if selected_value < best_value[method]:
                    best_value[method] = selected_value
                    best_state[method] = (
                        clone_state(models[method])
                        if selected_kind == "raw"
                        else {key: value.detach().cpu().clone() for key, value in ema[method].items()}
                    )
                    best_selection[method] = {
                        "step": step,
                        "source": selected_kind,
                        "validation_mse_per_coordinate": selected_value,
                    }
                row = {
                    "step": step,
                    "train_mse": train_metrics[method][0],
                    "gradient_norm_before_clip": train_metrics[method][1],
                    "val_mse_raw": raw_val,
                    "val_mse_ema": ema_val,
                }
                traces[method].append(row)
                print(method, row, flush=True)

    fixed_raw = {method: clone_state(model) for method, model in models.items()}
    fixed_ema = {
        method: {key: value.detach().cpu().clone() for key, value in state.items()}
        for method, state in ema.items()
    }
    gate_summary: dict[str, Any] = {}
    for method in selected_methods:
        if method == "linear":
            continue
        current = clone_state(models[method], device=device)
        models[method].load_state_dict(ema[method])
        gate_summary[method] = gate_diagnostics(
            method,
            models[method],
            val_raw,
            val_anchor,
            pair_mean,
            pair_sd,
            limit=int(args.gate_diagnostic_limit),
            microbatch_size=int(args.pair_validation_microbatch_size),
            device=device,
        )
        models[method].load_state_dict(current)

    diagnostics = {
        "protocol_version": SELECTED_PROTOCOL_VERSION,
        "trained_methods": list(selected_methods),
        "allow_partial_run": False,
        "prepared_dir": str(prepared),
        "storage_dtype": metadata.get("individual_pair_storage_dtype"),
        "storage_audit": {
            "train": metadata.get("train_storage_audit"),
            "validation": metadata.get("validation_storage_audit"),
        },
        "n_train_complete_datasets": len(train_target),
        "n_validation_complete_datasets": len(val_target),
        "formal_batch_size": int(args.formal_batch_size),
        "pair_microbatch_size": int(args.pair_microbatch_size),
        "row_presentations_per_arm": int(args.iters) * int(args.formal_batch_size),
        "shared_formal_minibatch_digest_sha256": batch_digest.hexdigest(),
        "fixed_steps_no_early_stopping": True,
        "fixed_final_ema_primary": True,
        "validation_best_secondary": True,
        "full_validation_for_every_arm": True,
        "identity": identity,
        "initialization": initialization,
        "parameter_count": {
            method: sum(parameter.numel() for parameter in model.parameters())
            for method, model in models.items()
        },
        "gradient_clip": {
            method: {
                "threshold": float(args.grad_clip),
                "trigger_count": clip_count[method],
                "step_count": step_count[method],
                "trigger_fraction": clip_count[method] / max(step_count[method], 1),
            }
            for method in selected_methods
        },
        "best_validation": best_selection,
        "gate_diagnostics_fixed_final_ema": gate_summary,
        "trace": traces,
        "training": asdict(training),
    }
    states_by_file = {
        "stage1_fixed_final_raw.pt": (
            fixed_raw,
            {"type": "fixed_final", "source": "raw", "step": int(args.iters)},
        ),
        "stage1_fixed_final_ema.pt": (
            fixed_ema,
            {"type": "fixed_final", "source": "ema", "step": int(args.iters)},
        ),
        "stage1_validation_best.pt": (
            best_state,
            {"type": "validation_best", "by_method": best_selection},
        ),
    }
    for filename, (states, selection) in states_by_file.items():
        save_selected_stage1_checkpoint(
            output / filename,
            states,
            stats,
            config,
            training,
            design,
            diagnostics,
            selection,
        )
    (output / "results.json").write_text(
        json.dumps(jsonable(diagnostics), indent=2), encoding="utf-8"
    )
    print(json.dumps(jsonable(diagnostics), indent=2), flush=True)


if __name__ == "__main__":
    main()
