#!/usr/bin/env python3
"""One-parameter Model-3 Stage 1: Model-2 local nCS plus temporal GRU.

The transition entry q=P01 is fixed and p=P11 is the only unknown parameter.
No analytic recovery of L_t is used by any primary arm.  Marginal and pairwise
subscores are processed exactly as local Model-2 inputs; only the outer block
aggregator is changed from an order-free map to a time-ordered GRU.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import run_hmm_common_factor_amortized_fsm_experiment as base
import run_hmm_common_factor_fsm_experiment as fixed


METHODS = ("linear", "gate", "raw")
LABELS = {
    "linear": "linear local-subscore GRU FSM",
    "gate": "positive nonlinear local-gate GRU FSM",
    "raw": "raw-sequence GRU FSM",
}


def sample_from_anchors(
    rng: np.random.Generator,
    anchor_p: np.ndarray,
    sigma_q: float,
    *,
    fixed_p01: float,
    length: int,
    block_size: int,
    tau: float,
    init_prob: float,
) -> dict[str, np.ndarray]:
    anchor_p = np.asarray(anchor_p, dtype=np.float64).reshape(-1)
    anchor_u = base.logit_np(anchor_p)
    u = anchor_u + rng.normal(size=anchor_u.shape) * float(sigma_q)
    p = base.sigmoid_np(u)
    p01_u = np.full_like(u, float(base.logit_np(float(fixed_p01))))
    transition_u = np.stack([p01_u, u], axis=1)
    y, _ = fixed.simulate_hmm_common_factor(
        rng,
        transition_u,
        int(length),
        int(block_size),
        float(tau),
        float(init_prob),
    )
    target = ((u - anchor_u) / float(sigma_q) ** 2)[:, None]
    return {
        "y": y,
        "anchor_p": anchor_p,
        "anchor_u": anchor_u[:, None],
        "u": u[:, None],
        "p": p[:, None],
        "target": target.astype(np.float32),
    }


def sample_amortized(
    rng: np.random.Generator,
    n: int,
    p_min: float,
    p_max: float,
    sigma_q: float,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    strata = (np.arange(int(n), dtype=np.float64) + rng.uniform(size=int(n))) / int(n)
    rng.shuffle(strata)
    anchors = float(p_min) + (float(p_max) - float(p_min)) * strata
    return sample_from_anchors(rng, anchors, sigma_q, **kwargs)


def sample_fixed_anchor(
    rng: np.random.Generator,
    n: int,
    anchor_p: float,
    sigma_q: float,
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    return sample_from_anchors(
        rng, np.full(int(n), float(anchor_p)), sigma_q, **kwargs
    )


def fit_features(
    raw: dict[str, np.ndarray],
    tau: float,
    pi_ref: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    s1, s2 = base.raw_subscores(raw["y"], float(tau), float(pi_ref))
    stats: dict[str, np.ndarray] = {
        "s1_mean": np.asarray(float(s1.mean()), dtype=np.float64),
        "s1_sd": np.asarray(max(float(s1.std()), 1e-8), dtype=np.float64),
        "s2_mean": np.asarray(float(s2.mean()), dtype=np.float64),
        "s2_sd": np.asarray(max(float(s2.std()), 1e-8), dtype=np.float64),
        "anchor_u_mean": np.asarray(raw["anchor_u"].mean(axis=0), dtype=np.float64),
        "anchor_u_sd": np.asarray(
            np.maximum(raw["anchor_u"].std(axis=0), 1e-8), dtype=np.float64
        ),
        "raw_y_mean": np.asarray(
            raw["y"].mean(axis=(0, 1), keepdims=True), dtype=np.float64
        ),
        "raw_y_sd": np.asarray(
            np.maximum(raw["y"].std(axis=(0, 1), keepdims=True), 1e-8),
            dtype=np.float64,
        ),
    }
    s1_z = (s1 - stats["s1_mean"]) / stats["s1_sd"]
    s2_z = (s2 - stats["s2_mean"]) / stats["s2_sd"]
    time_raw = np.stack([s1_z.mean(axis=2), s2_z.mean(axis=2)], axis=2)
    stats["time_mean"] = np.asarray(
        time_raw.mean(axis=(0, 1), keepdims=True), dtype=np.float64
    )
    stats["time_sd"] = np.asarray(
        np.maximum(time_raw.std(axis=(0, 1), keepdims=True), 1e-8),
        dtype=np.float64,
    )
    return transform_features(raw, tau, pi_ref, stats), stats


def transform_features(
    raw: dict[str, np.ndarray],
    tau: float,
    pi_ref: float,
    stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    s1, s2 = base.raw_subscores(raw["y"], float(tau), float(pi_ref))
    return {
        "s1": ((s1 - stats["s1_mean"]) / stats["s1_sd"]).astype(np.float32),
        "s2": ((s2 - stats["s2_mean"]) / stats["s2_sd"]).astype(np.float32),
        "raw_time": (
            (raw["y"] - stats["raw_y_mean"]) / stats["raw_y_sd"]
        ).astype(np.float32),
        "anchor_z": (
            (raw["anchor_u"] - stats["anchor_u_mean"])
            / stats["anchor_u_sd"]
        ).astype(np.float32),
    }


class LinearLocalGRU(nn.Module):
    def __init__(self, hidden: int, time_mean: np.ndarray, time_sd: np.ndarray):
        super().__init__()
        self.core = base.AnchorGRUCore(2, 1, int(hidden), out_dim=1)
        self.register_buffer("time_mean", torch.as_tensor(time_mean, dtype=torch.float32))
        self.register_buffer("time_sd", torch.as_tensor(time_sd, dtype=torch.float32))

    def forward(
        self, s1: torch.Tensor, s2: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        time_raw = torch.stack([s1.mean(dim=2), s2.mean(dim=2)], dim=2)
        return self.core((time_raw - self.time_mean) / self.time_sd, anchor_z)


class GatedLocalGRU(nn.Module):
    def __init__(
        self,
        hidden: int,
        gate_hidden: int,
        time_mean: np.ndarray,
        time_sd: np.ndarray,
    ):
        super().__init__()
        self.marginal_gate = fixed.PositiveMLPMultiplier(int(gate_hidden))
        self.pairwise_gate = fixed.PositiveMLPMultiplier(int(gate_hidden))
        self.core = base.AnchorGRUCore(2, 1, int(hidden), out_dim=1)
        self.register_buffer("time_mean", torch.as_tensor(time_mean, dtype=torch.float32))
        self.register_buffer("time_sd", torch.as_tensor(time_sd, dtype=torch.float32))

    def forward(
        self, s1: torch.Tensor, s2: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        g1 = s1 * self.marginal_gate(s1.unsqueeze(-1))
        g2 = s2 * self.pairwise_gate(s2.unsqueeze(-1))
        time_raw = torch.stack([g1.mean(dim=2), g2.mean(dim=2)], dim=2)
        return self.core((time_raw - self.time_mean) / self.time_sd, anchor_z)


class RawSequenceGRU(nn.Module):
    def __init__(self, hidden: int, block_size: int):
        super().__init__()
        self.core = base.AnchorGRUCore(int(block_size), 1, int(hidden), out_dim=1)

    def forward(self, raw_time: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        return self.core(raw_time, anchor_z)


def method_keys(method: str) -> tuple[str, ...]:
    if method in {"linear", "gate"}:
        return ("s1", "s2", "anchor_z")
    if method == "raw":
        return ("raw_time", "anchor_z")
    raise ValueError(method)


def forward_method(
    model: nn.Module, method: str, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    if method in {"linear", "gate"}:
        return model(batch["s1"], batch["s2"], batch["anchor_z"])
    return model(batch["raw_time"], batch["anchor_z"])


def predict(
    model: nn.Module,
    method: str,
    data: dict[str, np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    tensors = {
        key: torch.as_tensor(data[key], dtype=torch.float32, device=device)
        for key in method_keys(method)
    }
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, next(iter(tensors.values())).shape[0], int(batch_size)):
            stop = min(start + int(batch_size), next(iter(tensors.values())).shape[0])
            output.append(
                forward_method(
                    model, method, {key: value[start:stop] for key, value in tensors.items()}
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(output, axis=0).astype(np.float64)


def mse_batches(
    model: nn.Module,
    method: str,
    x: dict[str, torch.Tensor],
    target: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, target.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), target.shape[0])
            error = forward_method(
                model, method, {key: value[start:stop] for key, value in x.items()}
            ) - target[start:stop]
            total += float(error.square().sum().cpu())
            count += int(error.numel())
    model.train()
    return total / max(count, 1)


def train_model(
    model: nn.Module,
    method: str,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
    seed_offset: int,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    model = model.to(device)
    x_train = {
        key: torch.as_tensor(train[key], dtype=torch.float32, device=device)
        for key in method_keys(method)
    }
    x_val = {
        key: torch.as_tensor(val[key], dtype=torch.float32, device=device)
        for key in method_keys(method)
    }
    y_train = torch.as_tensor(train["target"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val["target"], dtype=torch.float32, device=device)
    if method == "gate":
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": list(model.marginal_gate.parameters())
                    + list(model.pairwise_gate.parameters()),
                    "lr": float(args.gate_lr),
                },
                {
                    "params": list(model.core.parameters()),
                    "lr": 0.0 if int(args.gate_only_steps) > 0 else float(args.joint_lr),
                },
            ],
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
        )
    rng = np.random.default_rng(int(args.seed) + int(seed_offset))
    best_state = base.clone_state_dict(model, cpu=True)
    ema_state = base.clone_state_dict(model, cpu=False)
    best_val = mse_batches(model, method, x_val, y_val, int(args.batch_size))
    best_step = 0
    best_source = "raw"
    bad = 0
    trace: list[dict[str, Any]] = []
    print(f"{LABELS[method]} step 0/{args.iters}: val={best_val:.6g}")
    for step in range(1, int(args.iters) + 1):
        if method == "gate" and step == int(args.gate_only_steps) + 1:
            optimizer.param_groups[1]["lr"] = float(args.joint_lr)
            print(f"{LABELS[method]}: unfreezing GRU at step {step}")
        index = torch.as_tensor(
            rng.integers(
                0, y_train.shape[0], size=min(int(args.batch_size), y_train.shape[0])
            ),
            dtype=torch.long,
            device=device,
        )
        batch = {key: value.index_select(0, index) for key, value in x_train.items()}
        target = y_train.index_select(0, index)
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_method(model, method, batch)
        loss = torch.mean((prediction - target) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss for {method} at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()
        base.update_ema_state(ema_state, model, float(args.ema_decay))
        if step == 1 or step % int(args.print_every) == 0 or step == int(args.iters):
            val_raw = mse_batches(model, method, x_val, y_val, int(args.batch_size))
            raw_state = base.clone_state_dict(model, cpu=False)
            model.load_state_dict(ema_state)
            val_ema = mse_batches(model, method, x_val, y_val, int(args.batch_size))
            model.load_state_dict(raw_state)
            if val_ema < val_raw:
                selected_val, selected_source = val_ema, "ema"
                selected_state = {
                    key: value.detach().cpu().clone() for key, value in ema_state.items()
                }
            else:
                selected_val, selected_source = val_raw, "raw"
                selected_state = base.clone_state_dict(model, cpu=True)
            stage = (
                "gate_only"
                if method == "gate" and step <= int(args.gate_only_steps)
                else "joint"
            )
            trace.append(
                {
                    "method": LABELS[method],
                    "step": step,
                    "stage": stage,
                    "train_loss": float(loss.detach().cpu()),
                    "val_loss_raw": val_raw,
                    "val_loss_ema": val_ema,
                    "best_val_loss": min(best_val, selected_val),
                    "selected_source": selected_source,
                    "grad_norm": float(grad_norm.detach().cpu()),
                }
            )
            print(
                f"{LABELS[method]} step {step}/{args.iters} [{stage}]: "
                f"train={loss.item():.6g}, val={selected_val:.6g}, "
                f"best={min(best_val, selected_val):.6g}"
            )
            if selected_val < best_val - 1e-6:
                best_val = selected_val
                best_state = selected_state
                best_step = step
                best_source = selected_source
                bad = 0
            else:
                bad += 1
            if int(args.patience) > 0 and bad >= int(args.patience):
                break
    model.load_state_dict(best_state)
    return model, trace, {
        "best_step": best_step,
        "best_val_loss": best_val,
        "best_source": best_source,
        "stopped_step": trace[-1]["step"] if trace else 0,
    }


def exact_score_u(
    y: np.ndarray,
    fixed_p01: float,
    p: float,
    tau: float,
    init_prob: float,
) -> np.ndarray:
    score = fixed.hmm_transition_score_u(
        fixed.full_emission_log_ratio(y, float(tau)),
        float(fixed_p01),
        float(p),
        float(init_prob),
    )
    return score[:, 1:2]


def evaluate(
    models: dict[str, nn.Module],
    stats: dict[str, np.ndarray],
    args: argparse.Namespace,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    fsm_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.diagnostic_seed))
    kwargs = {
        "fixed_p01": float(args.fixed_p01),
        "length": int(args.length),
        "block_size": int(args.block_size),
        "tau": float(args.tau),
        "init_prob": float(args.init_prob),
    }
    for p in base.parse_float_list(args.diagnostic_p_values):
        local = sample_fixed_anchor(
            rng,
            int(args.n_validation_per_anchor),
            float(p),
            float(args.sigma_q),
            **kwargs,
        )
        local_features = transform_features(local, args.tau, args.pi_ref, stats)
        center_u = np.full(
            (int(args.n_validation_per_anchor), 1),
            float(base.logit_np(float(p))),
            dtype=np.float64,
        )
        transition_u = np.concatenate(
            [
                np.full_like(center_u, float(base.logit_np(float(args.fixed_p01)))),
                center_u,
            ],
            axis=1,
        )
        y_center, _ = fixed.simulate_hmm_common_factor(
            rng,
            transition_u,
            int(args.length),
            int(args.block_size),
            float(args.tau),
            float(args.init_prob),
        )
        center_raw = {"y": y_center, "anchor_u": center_u}
        center_features = transform_features(
            center_raw, args.tau, args.pi_ref, stats
        )
        truth = exact_score_u(
            y_center, args.fixed_p01, p, args.tau, args.init_prob
        )
        scale = max(float(truth.std()), 1e-8)
        for method in METHODS:
            pred_local = predict(
                models[method], method, local_features, device, int(args.batch_size)
            )
            fsm_rows.append(
                {
                    "p_anchor": p,
                    "method": LABELS[method],
                    "n": int(local["target"].shape[0]),
                    "fsm_mse": float(np.mean((pred_local - local["target"]) ** 2)),
                }
            )
            prediction = predict(
                models[method], method, center_features, device, int(args.batch_size)
            )
            error = prediction - truth
            corr = float(np.corrcoef(truth[:, 0], prediction[:, 0])[0, 1])
            score_rows.append(
                {
                    "p_anchor": p,
                    "method": LABELS[method],
                    "n": int(truth.shape[0]),
                    "mse": float(np.mean(error**2)),
                    "std_mse": float(np.mean((error / scale) ** 2)),
                    "corr": corr,
                    "cosine": base.cosine_mean(truth, prediction),
                }
            )
    return score_rows, fsm_rows


def gate_summary(model: GatedLocalGRU) -> dict[str, float]:
    grid = torch.linspace(-4.0, 8.0, 241)[:, None]
    model_cpu = copy.deepcopy(model).cpu().eval()
    with torch.no_grad():
        m1 = model_cpu.marginal_gate(grid).numpy()
        m2 = model_cpu.pairwise_gate(grid).numpy()
    return {
        "marginal_min": float(m1.min()),
        "marginal_max": float(m1.max()),
        "marginal_max_abs_minus_one": float(np.max(np.abs(m1 - 1.0))),
        "pairwise_min": float(m2.min()),
        "pairwise_max": float(m2.max()),
        "pairwise_max_abs_minus_one": float(np.max(np.abs(m2 - 1.0))),
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = base.resolve_device(args.device)
    base.set_global_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    kwargs = {
        "fixed_p01": float(args.fixed_p01),
        "length": int(args.length),
        "block_size": int(args.block_size),
        "tau": float(args.tau),
        "init_prob": float(args.init_prob),
    }
    raw_train = sample_amortized(
        rng,
        int(args.n_train),
        float(args.p_min),
        float(args.p_max),
        float(args.sigma_q),
        **kwargs,
    )
    raw_val = sample_amortized(
        rng,
        int(args.n_val),
        float(args.p_min),
        float(args.p_max),
        float(args.sigma_q),
        **kwargs,
    )
    train_features, stats = fit_features(raw_train, args.tau, args.pi_ref)
    val_features = transform_features(raw_val, args.tau, args.pi_ref, stats)
    train = {**train_features, "target": raw_train["target"]}
    val = {**val_features, "target": raw_val["target"]}
    np.savez(output_dir / "feature_stats.npz", **stats)

    linear = LinearLocalGRU(args.gru_hidden, stats["time_mean"], stats["time_sd"])
    linear, linear_trace, linear_info = train_model(
        linear, "linear", train, val, args, device, 101
    )
    gate = GatedLocalGRU(
        args.gru_hidden,
        args.gate_hidden,
        stats["time_mean"],
        stats["time_sd"],
    )
    gate.core.load_state_dict(linear.core.state_dict(), strict=True)
    nested_linear = copy.deepcopy(linear).cpu().eval()
    nested_gate = copy.deepcopy(gate).cpu().eval()
    check_n = min(128, val["target"].shape[0])
    check = {key: value[:check_n] for key, value in val.items()}
    nested_error = float(
        np.max(
            np.abs(
                predict(nested_linear, "linear", check, "cpu", args.batch_size)
                - predict(nested_gate, "gate", check, "cpu", args.batch_size)
            )
        )
    )
    # Float32 CPU/GPU reductions can differ by a few 1e-7 even when the gate is
    # initialized exactly at the identity map.
    if nested_error > 1e-6:
        raise AssertionError(f"gate does not nest linear baseline: {nested_error}")
    gate, gate_trace, gate_info = train_model(gate, "gate", train, val, args, device, 202)
    raw_model = RawSequenceGRU(args.gru_hidden, args.block_size)
    raw_model, raw_trace, raw_info = train_model(
        raw_model, "raw", train, val, args, device, 303
    )
    models = {"linear": linear, "gate": gate, "raw": raw_model}
    score_rows, fsm_rows = evaluate(models, stats, args, device)
    base.write_csv(output_dir / "score_summary_by_anchor.csv", score_rows)
    base.write_csv(output_dir / "fixed_grid_fsm_loss.csv", fsm_rows)
    base.write_csv(
        output_dir / "training_trace.csv", linear_trace + gate_trace + raw_trace
    )
    diagnostics = {
        "identity_nesting_max_abs": nested_error,
        **gate_summary(gate),
    }
    (output_dir / "gate_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "unknown_parameter": "p=P11 only",
        "fixed_transition": {"P01": float(args.fixed_p01)},
        "score_coordinate": "u=logit(p)",
        "manual_L_recovery": False,
        "training_info": {
            "linear": linear_info,
            "gate": gate_info,
            "raw": raw_info,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    for method, model in models.items():
        torch.save(
            {
                "method": method,
                "label": LABELS[method],
                "state_dict": model.state_dict(),
                "config": config,
            },
            output_dir / f"model_{method}.pt",
        )
        subset = [row for row in score_rows if row["method"] == LABELS[method]]
        print(
            f"{LABELS[method]}: mean std MSE="
            f"{np.mean([row['std_mse'] for row in subset]):.8f}, "
            f"mean corr={np.mean([row['corr'] for row in subset]):.8f}"
        )
    print("gate diagnostics:", diagnostics)
    print("saved to", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=40000)
    parser.add_argument("--n-val", type=int, default=8000)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--pi-ref", type=float, default=0.3)
    parser.add_argument("--fixed-p01", type=float, default=0.06)
    parser.add_argument("--init-prob", type=float, default=0.5)
    parser.add_argument("--p-min", type=float, default=0.80)
    parser.add_argument("--p-max", type=float, default=0.99)
    parser.add_argument("--center-p", type=float, default=0.94)
    parser.add_argument("--sigma-q", type=float, default=0.25)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--gate-only-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-4)
    parser.add_argument("--joint-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--diagnostic-p-values", type=str, default="0.82,0.90,0.96,0.985")
    parser.add_argument("--n-validation-per-anchor", type=int, default=500)
    parser.add_argument("--diagnostic-seed", type=int, default=20260712)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/model3_oneparam_local_ncs_gru")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.fixed_p01 < 1.0 and 0.0 < args.p_min < args.p_max < 1.0):
        raise ValueError("invalid transition parameter bounds")
    if args.sigma_q <= 0.0:
        raise ValueError("sigma_q must be positive")
    run(args)


if __name__ == "__main__":
    main()
