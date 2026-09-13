#!/usr/bin/env python3
"""Raw-observation FSM baselines for cleaned Model 1 and Model 2.

This module deliberately lives outside the locked ``model1``/``model2``
architecture registries.  It changes only the Stage-1 representation:

    block_deepsets: raw Y -> block DeepSets -> amortized FSM score;
    direct_flat_mlp: concatenate(flatten(Y), anchor) -> one ordinary MLP score.

The FSM proposal target, anchor support, simulator, validation checkpoint rule,
data-only composite pilot, Stage-2 NPE recipe, and held-out evaluation banks are
kept aligned with the current experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from common import npe as shared_npe
from common import raw_data_npe
from common.pilot import constrained_modes_from_score_grid
from model1 import stage1 as model1_stage1
from model1 import stage2_npe as model1_stage2
from model2 import stage1 as model2_stage1
from model2 import stage2_npe as model2_stage2


METHOD = "raw_block_fsm"
METHOD_LABEL = "raw-block DeepSets FSM pilot+score NPE"
CONTEXT_DEFINITION = "(u_pilot(Y), S_raw_block_FSM(Y, u_pilot(Y)))"
DIRECT_METHOD = "direct_raw_fsm"
DIRECT_METHOD_LABEL = "direct amortized raw-FSM pilot+score NPE"
DIRECT_CONTEXT_DEFINITION = "(u_pilot(Y), S_direct_raw_FSM(Y, u_pilot(Y)))"
SUBSCORE_METHOD = "subscore_fsm"
SUBSCORE_METHOD_LABEL = "all-subscores flat MLP FSM pilot+score NPE"
SUBSCORE_CONTEXT_DEFINITION = "(u_pilot(Y), S_subscore_FSM(Y, u_pilot(Y)))"
ARCHITECTURES = ("block_deepsets", "direct_flat_mlp", "subscore_flat_mlp")
DEFAULTS: dict[str, dict[str, Any]] = {
    "model1": {
        "n_blocks": 20,
        "block_size": 20,
        "tau": 0.5,
        "sigma_q": 0.20,
        "lr_schedule": "cosine_tail",
        "diagnostic_pi_values": "0.10,0.30,0.50,0.65",
        "sbi_train_seed": 20260711,
        "sbi_seed": 52000,
        "pilot_mode": "marginal",
    },
    "model2": {
        "n_blocks": 40,
        "block_size": 40,
        "tau": 1.0,
        "sigma_q": 0.15,
        "lr_schedule": "constant",
        "diagnostic_pi_values": "0.07,0.10,0.30,0.50,0.65,0.68",
        "sbi_train_seed": 20260723,
        "sbi_seed": 54000,
        "pilot_mode": "equal_channel",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def clone_state(model: nn.Module, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device).clone()
        for key, value in model.state_dict().items()
    }


@torch.no_grad()
def update_ema(state: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    for key, value in model.state_dict().items():
        if torch.is_floating_point(value):
            state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            state[key].copy_(value.detach())


def make_mlp(in_dim: int, hidden: int, depth: int, out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(in_dim)
    for _ in range(int(depth)):
        layers.extend((nn.Linear(current, int(hidden)), nn.SiLU()))
        current = int(hidden)
    layers.append(nn.Linear(current, int(out_dim)))
    return nn.Sequential(*layers)


class RawBlockDeepSets(nn.Module):
    """Coordinate- and block-permutation-invariant score from literal raw Y."""

    def __init__(
        self,
        *,
        phi_hidden: int,
        hidden: int,
        depth: int,
        y_mean: float,
        y_sd: float,
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(2, int(phi_hidden)),
            nn.SiLU(),
            nn.Linear(int(phi_hidden), int(phi_hidden)),
            nn.SiLU(),
        )
        self.rho = make_mlp(int(phi_hidden) + 1, int(hidden), int(depth), 1)
        self.register_buffer("y_mean", torch.tensor(float(y_mean), dtype=torch.float32))
        self.register_buffer("y_sd", torch.tensor(float(y_sd), dtype=torch.float32))

    def forward(self, y: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        y_z = (y - self.y_mean) / self.y_sd
        anchor_coordinate = anchor_z.reshape(-1, 1, 1).expand_as(y_z)
        embedded = self.phi(torch.stack((y_z, anchor_coordinate), dim=-1)).mean(dim=2)
        anchor = anchor_z.reshape(-1, 1, 1).expand(-1, y.shape[1], 1)
        block_score = self.rho(torch.cat((embedded, anchor), dim=-1)).squeeze(-1)
        return block_score.sum(dim=1)


class DirectRawMLP(nn.Module):
    """Amortized Direct-FSM score from the complete, flattened observation.

    This is intentionally unstructured: there is no coordinate map, gate,
    pooling operation, blockwise readout, or permutation constraint.  The
    anchor is simply one additional input coordinate to the score network.
    """

    def __init__(
        self,
        *,
        n_blocks: int,
        block_size: int,
        hidden: int,
        depth: int,
        y_mean: float,
        y_sd: float,
    ) -> None:
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.block_size = int(block_size)
        self.net = make_mlp(
            self.n_blocks * self.block_size + 1, int(hidden), int(depth), 1
        )
        self.register_buffer("y_mean", torch.tensor(float(y_mean), dtype=torch.float32))
        self.register_buffer("y_sd", torch.tensor(float(y_sd), dtype=torch.float32))

    def forward(self, y: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        if y.ndim != 3 or y.shape[1:] != (self.n_blocks, self.block_size):
            raise ValueError(
                f"expected y shape (batch,{self.n_blocks},{self.block_size}), "
                f"got {tuple(y.shape)}"
            )
        y_flat = ((y - self.y_mean) / self.y_sd).flatten(start_dim=1)
        network_input = torch.cat((y_flat, anchor_z.reshape(-1, 1)), dim=1)
        return self.net(network_input).squeeze(-1)


class SubscoreFlatMLP(nn.Module):
    """Amortized FSM score from every local marginal score, flattened.

    The local scores ``s_kj(Y, u) = expit(u + l_kj) - expit(u)``, with ``l_kj``
    the per-observation mean-shift log-likelihood ratio, are recomputed from Y
    at the anchor inside ``forward``, so the runtime still needs only
    ``(Y, anchor)``. There is no pooling or permutation structure: this is the
    local-score summary with the aggregation removed. With the same hidden
    width it has exactly as many parameters as ``DirectRawMLP``, so the two
    unstructured Fisher-score baselines differ only in their input. Model 1 only.
    """

    def __init__(
        self,
        *,
        n_blocks: int,
        block_size: int,
        hidden: int,
        depth: int,
        tau: float,
        anchor_mean: float,
        anchor_sd: float,
        s_mean: float,
        s_sd: float,
    ) -> None:
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.block_size = int(block_size)
        self.tau = float(tau)
        self.net = make_mlp(self.n_blocks * self.block_size + 1, int(hidden), int(depth), 1)
        for key, value in (("anchor_mean", anchor_mean), ("anchor_sd", anchor_sd),
                           ("s_mean", s_mean), ("s_sd", s_sd)):
            self.register_buffer(key, torch.tensor(float(value), dtype=torch.float32))

    def local_scores(self, y: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        u = (anchor_z * self.anchor_sd + self.anchor_mean).reshape(-1, 1, 1)
        log_ratio = self.tau * y - 0.5 * self.tau**2
        return torch.sigmoid(u + log_ratio) - torch.sigmoid(u)

    def forward(self, y: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        if y.ndim != 3 or y.shape[1:] != (self.n_blocks, self.block_size):
            raise ValueError(
                f"expected y shape (batch,{self.n_blocks},{self.block_size}), "
                f"got {tuple(y.shape)}"
            )
        s_z = ((self.local_scores(y, anchor_z) - self.s_mean) / self.s_sd).flatten(start_dim=1)
        return self.net(torch.cat((s_z, anchor_z.reshape(-1, 1)), dim=1)).squeeze(-1)


def architecture_contract(architecture: str) -> tuple[str, str, str]:
    if architecture == "block_deepsets":
        return METHOD, METHOD_LABEL, CONTEXT_DEFINITION
    if architecture == "direct_flat_mlp":
        return DIRECT_METHOD, DIRECT_METHOD_LABEL, DIRECT_CONTEXT_DEFINITION
    if architecture == "subscore_flat_mlp":
        return SUBSCORE_METHOD, SUBSCORE_METHOD_LABEL, SUBSCORE_CONTEXT_DEFINITION
    raise ValueError(f"unknown raw-FSM architecture: {architecture}")


def build_score_model(
    architecture: str,
    *,
    n_blocks: int,
    block_size: int,
    phi_hidden: int,
    hidden: int,
    depth: int,
    y_mean: float,
    y_sd: float,
    subscore_stats: dict[str, float] | None = None,
) -> nn.Module:
    if architecture == "block_deepsets":
        return RawBlockDeepSets(
            phi_hidden=phi_hidden, hidden=hidden, depth=depth,
            y_mean=y_mean, y_sd=y_sd,
        )
    if architecture == "direct_flat_mlp":
        return DirectRawMLP(
            n_blocks=n_blocks, block_size=block_size, hidden=hidden, depth=depth,
            y_mean=y_mean, y_sd=y_sd,
        )
    if architecture == "subscore_flat_mlp":
        if subscore_stats is None:
            raise ValueError("subscore_flat_mlp needs tau, anchor and local-score statistics")
        return SubscoreFlatMLP(
            n_blocks=n_blocks, block_size=block_size, hidden=hidden, depth=depth,
            **subscore_stats,
        )
    raise ValueError(f"unknown raw-FSM architecture: {architecture}")


def model_module(name: str):
    return model1_stage1 if name == "model1" else model2_stage1


def model_defaults(args: argparse.Namespace) -> None:
    values = DEFAULTS[str(args.model)]
    for key in ("n_blocks", "block_size", "tau"):
        if getattr(args, key) is None:
            setattr(args, key, values[key])
    if hasattr(args, "sigma_q") and args.sigma_q is None:
        args.sigma_q = values["sigma_q"]
    if getattr(args, "lr_schedule", None) is None:
        args.lr_schedule = values["lr_schedule"]
    if getattr(args, "diagnostic_pi_values", None) is None:
        args.diagnostic_pi_values = values["diagnostic_pi_values"]
    if hasattr(args, "sbi_train_seed") and args.sbi_train_seed is None:
        args.sbi_train_seed = values["sbi_train_seed"]
    if hasattr(args, "sbi_seed") and args.sbi_seed is None:
        args.sbi_seed = values["sbi_seed"]
    if hasattr(args, "pilot_mode") and args.pilot_mode is None:
        args.pilot_mode = values["pilot_mode"]


def evaluate_mse(
    model: nn.Module,
    y: torch.Tensor,
    anchor_z: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(target), int(batch_size)):
            stop = min(start + int(batch_size), len(target))
            error = model(y[start:stop], anchor_z[start:stop]) - target[start:stop]
            total += float(torch.sum(error.square()).item())
            count += int(error.numel())
    model.train()
    return total / max(count, 1)


def predict(
    model: nn.Module,
    y: np.ndarray,
    anchor_z: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(y), int(batch_size)):
            stop = min(start + int(batch_size), len(y))
            output.append(
                model(
                    torch.as_tensor(y[start:stop], dtype=torch.float32, device=device),
                    torch.as_tensor(anchor_z[start:stop], dtype=torch.float32, device=device),
                ).cpu().numpy()
            )
    return np.concatenate(output).astype(np.float64)


def lr_multiplier(step: int, iters: int, schedule: str, start: int, floor: float) -> float:
    if schedule == "constant" or step <= start:
        return 1.0
    progress = min(max((step - start) / (iters - start), 0.0), 1.0)
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_stage1(args: argparse.Namespace) -> None:
    model_defaults(args)
    module = model_module(str(args.model))
    architecture = str(args.architecture)
    method, method_label, context_definition = architecture_contract(architecture)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = resolve(str(args.device))
    set_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    train = module.sample_anchor_batch(
        rng, int(args.n_train), float(args.anchor_pi_min), float(args.anchor_pi_max),
        float(args.sigma_q), int(args.n_blocks), int(args.block_size), float(args.tau),
    )
    val = module.sample_anchor_batch(
        rng, int(args.n_val), float(args.anchor_pi_min), float(args.anchor_pi_max),
        float(args.sigma_q), int(args.n_blocks), int(args.block_size), float(args.tau),
    )
    y_mean = float(np.mean(train["y"]))
    y_sd = max(float(np.std(train["y"])), 1e-8)
    anchor_mean = float(np.mean(train["anchor_u"]))
    anchor_sd = max(float(np.std(train["anchor_u"])), 1e-8)
    train_anchor_z = ((train["anchor_u"] - anchor_mean) / anchor_sd).astype(np.float32)
    val_anchor_z = ((val["anchor_u"] - anchor_mean) / anchor_sd).astype(np.float32)
    subscore_stats = None
    if architecture == "subscore_flat_mlp":
        if str(args.model) != "model1":
            raise ValueError("subscore_flat_mlp is implemented for model1 only")
        local = module.score_u_from_log_ratio(
            module.marginal_log_ratio(train["y"], float(args.tau)),
            module.sigmoid_np(train["anchor_u"]),
        )
        subscore_stats = {
            "tau": float(args.tau), "anchor_mean": anchor_mean, "anchor_sd": anchor_sd,
            "s_mean": float(np.mean(local)), "s_sd": max(float(np.std(local)), 1e-8),
        }
        del local

    set_seed(int(args.seed) + 33)
    model = build_score_model(
        architecture,
        n_blocks=int(args.n_blocks), block_size=int(args.block_size),
        phi_hidden=int(args.phi_hidden), hidden=int(args.hidden), depth=int(args.depth),
        y_mean=y_mean, y_sd=y_sd, subscore_stats=subscore_stats,
    ).to(device)
    trainable_parameters = int(sum(value.numel() for value in model.parameters()))
    config = {
        **vars(args),
        "output_dir": str(output),
        "method": method,
        "label": {
            "block_deepsets": "raw-observation block DeepSets anchored Direct-FSM",
            "direct_flat_mlp": "direct amortized raw-observation MLP FSM",
            "subscore_flat_mlp": "all-local-subscores flat MLP FSM",
        }[architecture],
        "method_label": method_label,
        "context_definition": context_definition,
        "input_representation": (
            "local marginal scores s_kj(Y, anchor) recomputed from Y; train-only scalar standardization"
            if architecture == "subscore_flat_mlp"
            else "literal raw Y; train-only scalar affine standardization"
        ),
        "score_network_input": {
            "block_deepsets": "shared coordinate embeddings followed by blockwise pooling",
            "direct_flat_mlp": "concatenate(flatten(Y), standardized_anchor_u)",
            "subscore_flat_mlp": "concatenate(flatten(s_kj(Y, anchor)), standardized_anchor_u)",
        }[architecture],
        "uses_phi": architecture == "block_deepsets",
        "uses_gate": False,
        "uses_pooling": architecture == "block_deepsets",
        "uses_deepsets": architecture == "block_deepsets",
        "within_block_permutation_invariant": architecture == "block_deepsets",
        "block_permutation_invariant": architecture == "block_deepsets",
        "raw_input_dimension": int(args.n_blocks) * int(args.block_size) + 1,
        "trainable_parameters": trainable_parameters,
        "anchor_sampler": "continuous_stratified_in_pi",
        "fsm_target": "(u-anchor_u)/sigma_q^2",
        "exact_score_role": "post-training diagnostics only",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez(
        output / "feature_stats.npz",
        y_mean=np.asarray(y_mean), y_sd=np.asarray(y_sd),
        anchor_u_mean=np.asarray(anchor_mean), anchor_u_sd=np.asarray(anchor_sd),
        **({"s_mean": np.asarray(subscore_stats["s_mean"]),
            "s_sd": np.asarray(subscore_stats["s_sd"])} if subscore_stats else {}),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    y_train = torch.as_tensor(train["y"], dtype=torch.float32, device=device)
    a_train = torch.as_tensor(train_anchor_z, dtype=torch.float32, device=device)
    t_train = torch.as_tensor(train["target"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val["y"], dtype=torch.float32, device=device)
    a_val = torch.as_tensor(val_anchor_z, dtype=torch.float32, device=device)
    t_val = torch.as_tensor(val["target"], dtype=torch.float32, device=device)
    train_rng = np.random.default_rng(int(args.seed) + 30_001)
    ema = clone_state(model, device=device)
    best = clone_state(model)
    best_val = evaluate_mse(model, y_val, a_val, t_val, int(args.eval_batch_size))
    best_step, best_source = 0, "raw"
    trace: list[dict[str, Any]] = []
    print(
        f"{method} parameters={trainable_parameters} "
        f"step 0/{args.iters}: val={best_val:.6g}", flush=True
    )
    for step in range(1, int(args.iters) + 1):
        multiplier = lr_multiplier(
            step, int(args.iters), str(args.lr_schedule),
            int(args.lr_decay_start_step), float(args.lr_min_ratio),
        )
        optimizer.param_groups[0]["lr"] = float(args.lr) * multiplier
        index = torch.as_tensor(
            train_rng.integers(0, len(train["target"]), size=int(args.batch_size)),
            dtype=torch.long, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(y_train.index_select(0, index), a_train.index_select(0, index))
        loss = torch.mean((prediction - t_train.index_select(0, index)).square())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        update_ema(ema, model, float(args.ema_decay))
        if step == 1 or step % int(args.print_every) == 0 or step == int(args.iters):
            raw_val = evaluate_mse(model, y_val, a_val, t_val, int(args.eval_batch_size))
            raw_state = clone_state(model, device=device)
            model.load_state_dict(ema)
            ema_val = evaluate_mse(model, y_val, a_val, t_val, int(args.eval_batch_size))
            model.load_state_dict(raw_state)
            selected_val, selected_source = (
                (ema_val, "ema")
                if str(args.checkpoint_selection) == "raw_or_ema" and ema_val < raw_val
                else (raw_val, "raw")
            )
            if selected_val < best_val - 1e-6:
                best_val, best_step, best_source = selected_val, step, selected_source
                best = clone_state(model) if selected_source == "raw" else {
                    key: value.detach().cpu().clone() for key, value in ema.items()
                }
            trace.append({
                "step": step, "train_loss": float(loss.item()),
                "val_loss_raw": raw_val, "val_loss_ema": ema_val,
                "best_val_loss": best_val, "best_step": best_step,
                "best_source": best_source, "grad_norm": float(grad_norm.item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            })
            print(
                f"{method} step {step}/{args.iters}: train={loss.item():.6g} "
                f"val_raw={raw_val:.6g} val_ema={ema_val:.6g} "
                f"best={best_val:.6g}@{best_step}/{best_source}", flush=True,
            )

    final_raw = clone_state(model)
    final_ema = {key: value.detach().cpu().clone() for key, value in ema.items()}
    info = {
        "best_step": best_step, "best_source": best_source,
        "best_val_loss": best_val, "stopped_step": int(args.iters),
        "fixed_final_step": int(args.iters), "fixed_final_source": "ema",
        "trainable_parameters": trainable_parameters,
    }
    payload_common = {
        "method": method, "label": config["label"], "config": config,
        "training_info": info,
        "input_mode": "local_subscores" if architecture == "subscore_flat_mlp" else "raw_observation",
    }
    torch.save({**payload_common, "state_dict": best, "selection": "validation_best",
                "checkpoint_source": best_source, "checkpoint_step": best_step},
               output / f"model_{method}.pt")
    torch.save({**payload_common, "state_dict": final_ema, "selection": "fixed_final",
                "checkpoint_source": "ema", "checkpoint_step": int(args.iters)},
               output / f"model_{method}_fixed_final_ema.pt")
    torch.save({**payload_common, "state_dict": final_raw, "selection": "fixed_final",
                "checkpoint_source": "raw", "checkpoint_step": int(args.iters)},
               output / f"model_{method}_fixed_final_raw.pt")
    write_csv(output / "training_trace.csv", trace)
    (output / "training_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    model.load_state_dict(best)
    if bool(args.skip_exact_diagnostics):
        print("exact score diagnostics skipped by protocol", flush=True)
        return
    diagnostics: list[dict[str, Any]] = []
    for pi in raw_data_npe.parse_list(args.diagnostic_pi_values, float):
        u = np.full(int(args.n_test), float(module.logit_np(pi)), dtype=np.float64)
        y = (
            module.simulate_mean_shift(rng, u, int(args.n_blocks), int(args.block_size), float(args.tau))
            if str(args.model) == "model1"
            else module.simulate_common_factor(rng, u, int(args.n_blocks), int(args.block_size), float(args.tau))
        )
        anchor_z = ((u - anchor_mean) / anchor_sd).astype(np.float32)
        prediction = predict(model, y, anchor_z, device=device, batch_size=int(args.eval_batch_size))
        truth = module.full_score_u(y, pi, float(args.tau))
        row = module.metric_row(truth, prediction)
        row.update({"method": config["label"], "pi": pi, "n": int(args.n_test)})
        diagnostics.append(row)
    write_csv(output / "score_summary_by_pi.csv", diagnostics)
    print(json.dumps(info, indent=2), flush=True)
    for row in diagnostics:
        print(f"pi={row['pi']:.2f} std_mse={row['std_mse']:.6g}", flush=True)


class RawFSMRuntime:
    def __init__(self, run_dir: Path, device: str = "auto", checkpoint: Path | None = None):
        self.run_dir = Path(run_dir)
        self.device = resolve(device)
        self.config = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        self.architecture = str(self.config.get("architecture", "block_deepsets"))
        self.method, self.method_label, self.context_definition = architecture_contract(
            self.architecture
        )
        if self.config.get("method", self.method) != self.method:
            raise ValueError("raw-FSM config method/architecture contract mismatch")
        self.model_name = str(self.config["model"])
        self.n_blocks = int(self.config["n_blocks"])
        self.block_size = int(self.config["block_size"])
        self.tau = float(self.config["tau"])
        self.sigma_q = float(self.config["sigma_q"])
        self.pi_min = float(self.config["anchor_pi_min"])
        self.pi_max = float(self.config["anchor_pi_max"])
        module = model_module(self.model_name)
        self.u_min = float(module.logit_np(self.pi_min))
        self.u_max = float(module.logit_np(self.pi_max))
        stats = np.load(self.run_dir / "feature_stats.npz")
        self.anchor_mean = float(stats["anchor_u_mean"])
        self.anchor_sd = float(stats["anchor_u_sd"])
        model = build_score_model(
            self.architecture,
            n_blocks=self.n_blocks, block_size=self.block_size,
            phi_hidden=int(self.config.get("phi_hidden", 32)),
            hidden=int(self.config["hidden"]), depth=int(self.config["depth"]),
            y_mean=float(stats["y_mean"]), y_sd=float(stats["y_sd"]),
            subscore_stats=(
                {"tau": self.tau, "anchor_mean": self.anchor_mean, "anchor_sd": self.anchor_sd,
                 "s_mean": float(stats["s_mean"]), "s_sd": float(stats["s_sd"])}
                if self.architecture == "subscore_flat_mlp" else None
            ),
        )
        self.checkpoint = checkpoint or self.run_dir / f"model_{self.method}.pt"
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("method") != self.method:
            raise ValueError("not a raw-FSM checkpoint")
        model.load_state_dict(payload["state_dict"], strict=True)
        self.model = model.to(self.device).eval()
        self.selection = payload.get("selection")
        self.checkpoint_source = payload.get("checkpoint_source")
        self.checkpoint_step = payload.get("checkpoint_step")
        self.training_info = payload.get("training_info", {})
        self.checkpoint_sha256 = hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir), "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256, "model": self.model_name,
            "method": self.method, "architecture": self.architecture,
            "trainable_parameters": int(sum(v.numel() for v in self.model.parameters())),
            "selection": self.selection,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_step": self.checkpoint_step, "n_blocks": self.n_blocks,
            "block_size": self.block_size, "tau": self.tau, "sigma_q": self.sigma_q,
            "pi_min": self.pi_min, "pi_max": self.pi_max,
        }

    def validate_y(self, y: np.ndarray) -> np.ndarray:
        value = np.asarray(y, dtype=np.float32)
        if value.ndim != 3 or value.shape[1:] != (self.n_blocks, self.block_size):
            raise ValueError(f"bad raw y shape {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("raw y contains non-finite values")
        return value

    def score(self, y: np.ndarray, u: float | np.ndarray, *, batch_size: int = 256) -> np.ndarray:
        value = self.validate_y(y)
        anchor = np.asarray(u, dtype=np.float64)
        if anchor.ndim == 0:
            anchor = np.full(len(value), float(anchor))
        if anchor.shape != (len(value),):
            raise ValueError("u must be scalar or one value per dataset")
        if np.any(anchor < self.u_min - 1e-12) or np.any(anchor > self.u_max + 1e-12):
            raise ValueError("u is outside the trained anchor support")
        anchor_z = ((anchor - self.anchor_mean) / self.anchor_sd).astype(np.float32)
        return predict(self.model, value, anchor_z, device=self.device, batch_size=batch_size)

    def sanity_checks(self, seed: int = 20260710) -> dict[str, Any]:
        module = model_module(self.model_name)
        rng = np.random.default_rng(seed)
        u = np.full(16, 0.5 * (self.u_min + self.u_max))
        y = (
            module.simulate_mean_shift(rng, u, self.n_blocks, self.block_size, self.tau)
            if self.model_name == "model1"
            else module.simulate_common_factor(rng, u, self.n_blocks, self.block_size, self.tau)
        )
        base = self.score(y, u)
        repeat = self.score(y.copy(), u.copy())
        block = self.score(y[:, ::-1].copy(), u)
        within = self.score(y[:, :, ::-1].copy(), u)
        finite = bool(np.all(np.isfinite(base)))
        repeat_error = float(np.max(np.abs(base - repeat)))
        block_error = float(np.max(np.abs(base - block)))
        within_error = float(np.max(np.abs(base - within)))
        permutation_invariance_required = self.architecture == "block_deepsets"
        passed = finite and repeat_error < 1e-6
        if permutation_invariance_required:
            passed = passed and block_error < 2e-5 and within_error < 2e-5
        return {
            **self.metadata(), "finite": finite,
            "deterministic_repeat_max_abs_error": repeat_error,
            "permutation_invariance_required": permutation_invariance_required,
            "block_permutation_max_abs_error": block_error,
            "within_block_permutation_max_abs_error": within_error,
            "passed": passed,
        }


def pilot_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        pi_prior_min=float(args.pi_prior_min), pi_prior_max=float(args.pi_prior_max),
        pilot_grid_size=int(args.pilot_grid_size), pilot_batch_size=int(args.pilot_batch_size),
        pilot_grid_chunk_size=int(args.pilot_grid_chunk_size),
        pilot_progress_every=int(args.pilot_progress_every),
        pilot_device=resolve(str(args.pilot_device)), pilot_backend=str(args.pilot_backend),
        pilot_mode=str(args.pilot_mode),
        # model1.stage2_npe.data_only_pilot reads the device as ``args.device``.
        device=resolve(str(args.pilot_device)),
    )


def compute_pilot(y: np.ndarray, runtime: RawFSMRuntime, args: argparse.Namespace):
    metadata = runtime.metadata()
    if runtime.model_name == "model1":
        return model1_stage2.data_only_pilot(y, metadata, pilot_args(args))

    # Model 2's main helper combines pilot construction and frozen-score
    # evaluation.  Reproduce only its pilot half here so the raw-FSM runtime is
    # the sole score provider while the analytic composite pilot stays exact.
    y_value = runtime.validate_y(y).astype(np.float64)
    u_grid = np.linspace(runtime.u_min, runtime.u_max, int(args.pilot_grid_size))
    roots: list[np.ndarray] = []
    status: list[str] = []
    mode = str(args.pilot_mode)
    backend = "torch" if str(args.pilot_backend) == "auto" else str(args.pilot_backend)
    for start in range(0, len(y_value), int(args.pilot_batch_size)):
        stop = min(start + int(args.pilot_batch_size), len(y_value))
        chunk = y_value[start:stop]
        if mode == "moment":
            raw_pi = (np.mean(chunk**2, axis=(1, 2)) - 1.0) / runtime.tau**2
            clipped = np.clip(raw_pi, runtime.pi_min, runtime.pi_max)
            batch_root = np.asarray(model2_stage1.logit_np(clipped), dtype=np.float64)
            batch_status = np.where(
                raw_pi <= runtime.pi_min, "left_boundary",
                np.where(raw_pi >= runtime.pi_max, "right_boundary", "interior"),
            ).astype("U32")
        else:
            lr1_np = model2_stage1.marginal_log_ratio(chunk, runtime.tau)
            lr2_np = model2_stage1.pairwise_log_ratio(chunk, runtime.tau)
            if backend == "torch":
                lr1 = torch.as_tensor(lr1_np, dtype=torch.float64, device=runtime.device)
                lr2 = torch.as_tensor(lr2_np, dtype=torch.float64, device=runtime.device)
                field = model2_stage2._torch_composite_pilot_score_grid(
                    lr1, lr2, u_grid,
                    grid_chunk_size=int(args.pilot_grid_chunk_size),
                    include_pairwise=mode == "equal_channel",
                )
            elif backend == "numpy":
                field = model2_stage2._numpy_composite_pilot_score_grid(
                    lr1_np, lr2_np, u_grid,
                    include_pairwise=mode == "equal_channel",
                )
            else:
                raise ValueError("pilot_backend must be auto, torch, or numpy")
            batch_root, batch_status, _, _ = constrained_modes_from_score_grid(field, u_grid)
        roots.append(np.asarray(batch_root, dtype=np.float64))
        status.extend(str(value) for value in batch_status)
        if int(args.pilot_progress_every) > 0:
            batch_index = math.ceil(stop / int(args.pilot_batch_size))
            if batch_index % int(args.pilot_progress_every) == 0 or stop == len(y_value):
                print(f"raw-FSM shared pilot: {stop}/{len(y_value)}", flush=True)
    return np.concatenate(roots), np.asarray(status, dtype="U32")


def load_or_compute_training_pilot(
    y: np.ndarray,
    pi_train: np.ndarray,
    runtime: RawFSMRuntime,
    args: argparse.Namespace,
):
    """Reuse a Model-1 Stage-2 pilot cache when it matches this training bank.

    The pilot is data-only, so one cache serves every Stage-1 checkpoint
    trained on the same NPE simulation bank.
    """
    cache = getattr(args, "pilot_cache", None)
    if cache is None or not Path(cache).exists():
        return compute_pilot(y, runtime, args)
    if runtime.model_name != "model1":
        raise ValueError("--pilot-cache is supported for model1 only")
    with np.load(cache) as stored:
        cache_pi = np.asarray(stored["pi_train"], dtype=np.float64)
        if cache_pi.shape != pi_train.shape or not np.allclose(
            cache_pi, pi_train, rtol=0.0, atol=2e-7
        ):
            raise ValueError(f"pilot cache pi_train does not match this training bank: {cache}")
        if "pilot_grid_size" in stored.files and int(stored["pilot_grid_size"]) != int(
            args.pilot_grid_size
        ):
            raise ValueError("pilot cache grid size does not match --pilot-grid-size")
        if "tau" in stored.files and not np.isclose(float(stored["tau"]), runtime.tau):
            raise ValueError("pilot cache tau does not match the Stage-1 model")
        if "pilot_mode" in stored.files and str(stored["pilot_mode"]) != "marginal":
            raise ValueError("pilot cache was not generated by the marginal pilot")
        print(f"Loaded pilot cache: {cache}", flush=True)
        return (np.asarray(stored["pilot_u"], dtype=np.float64),
                np.asarray(stored["pilot_status"], dtype="U32"))


def train_stage2(args: argparse.Namespace) -> None:
    model_defaults(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    runtime = RawFSMRuntime(Path(args.stage1_run_dir), str(args.device), args.stage1_checkpoint)
    method = runtime.method
    method_label = runtime.method_label
    context_definition = runtime.context_definition
    if runtime.model_name != str(args.model):
        raise ValueError("--model does not match the Stage-1 checkpoint")
    checks = runtime.sanity_checks(int(args.sanity_seed))
    if not checks["passed"]:
        raise AssertionError(f"raw-FSM runtime sanity failed: {checks}")
    write_csv(output / "runtime_sanity_checks.csv", [checks])
    rng = np.random.default_rng(int(args.sbi_train_seed))
    pi_train = rng.uniform(float(args.pi_prior_min), float(args.pi_prior_max), int(args.n_sbi_train))
    y_train = raw_data_npe.simulate(rng, pi_train, args)
    pilot_u, pilot_status = load_or_compute_training_pilot(y_train, pi_train, runtime, args)
    score = runtime.score(y_train, pilot_u, batch_size=int(args.score_batch_size))
    context = np.column_stack((pilot_u, score)).astype(np.float32)
    if not np.all(np.isfinite(context)):
        raise FloatingPointError("non-finite raw-FSM Stage-2 context")
    helper_args = raw_data_npe.build_npe_args(args)
    NPE, posterior_nn, BoxUniform = shared_npe.import_sbi()
    device = shared_npe.resolve_device(str(args.device))
    data_device = shared_npe.resolve_device(str(args.data_device))
    fitted = shared_npe.train_sbi_npe_method(
        NPE, posterior_nn, BoxUniform, method_label, context, pi_train,
        int(args.sbi_seed), helper_args, device, data_device,
    )
    torch.save({
        "method": method, "label": method_label,
        "state_dict": fitted["posterior"].posterior_estimator.state_dict(),
        "stage1_runtime": runtime.metadata(), "context": context_definition,
        "config": vars(args),
    }, output / f"npe_{method}_state.pt")
    np.savez_compressed(
        output / "training_contexts.npz", pi_train=pi_train.astype(np.float32),
        pilot_u=pilot_u.astype(np.float32), pilot_status=pilot_status,
        raw_fsm_score=score.astype(np.float32), context=context,
    )
    del y_train

    rows: list[dict[str, Any]] = []
    for pi_true in raw_data_npe.parse_list(args.test_pi_values, float):
        for obs_seed in raw_data_npe.parse_list(args.test_seeds, int):
            obs_rng = np.random.default_rng(obs_seed)
            y_obs = raw_data_npe.simulate(obs_rng, np.asarray([pi_true]), args)
            obs_pilot, obs_status = compute_pilot(y_obs, runtime, args)
            obs_score = runtime.score(y_obs, obs_pilot, batch_size=1)
            obs_context = np.column_stack((obs_pilot, obs_score)).astype(np.float32)[0]
            exact_grid, exact_cdf, exact = raw_data_npe.exact_posterior(y_obs[0], args)
            rows.append({
                "pi_true": pi_true, "seed": obs_seed, "method": "exact likelihood grid",
                "pilot_u_context": float(obs_pilot[0]), "pilot_status": str(obs_status[0]),
                "score_context": float("nan"), "pi_post_mean": exact["mean"],
                "pi_post_sd": exact["sd"], "exact_pi_post_sd": exact["sd"],
                "q05": exact["q05"], "q50": exact["q50"], "q95": exact["q95"],
                "pi_sq_err": float((exact["mean"] - pi_true) ** 2),
                "mean_sq_err_to_exact": 0.0, "w1_to_exact": 0.0,
                "coverage90": float(exact["q05"] <= pi_true <= exact["q95"]),
            })
            samples = shared_npe.sample_sbi_posterior(
                fitted, obs_context, int(args.posterior_n),
                int(args.posterior_seed) + obs_seed, helper_args,
            )
            mean, sd = float(np.mean(samples)), float(np.std(samples))
            q05, q50, q95 = (float(x) for x in np.quantile(samples, (0.05, 0.5, 0.95)))
            rows.append({
                "pi_true": pi_true, "seed": obs_seed, "method": method_label,
                "pilot_u_context": float(obs_pilot[0]), "pilot_status": str(obs_status[0]),
                "score_context": float(obs_score[0]), "pi_post_mean": mean,
                "pi_post_sd": sd, "exact_pi_post_sd": exact["sd"],
                "q05": q05, "q50": q50, "q95": q95,
                "pi_sq_err": float((mean - pi_true) ** 2),
                "mean_sq_err_to_exact": float((mean - exact["mean"]) ** 2),
                "w1_to_exact": shared_npe.exact_sample_w1(samples, exact_grid, exact_cdf),
                "coverage90": float(q05 <= pi_true <= q95),
            })
    summary = raw_data_npe.summarize(rows)
    write_csv(output / "posterior_by_seed.csv", rows)
    write_csv(output / "posterior_summary.csv", summary)
    protocol = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stage1_runtime": runtime.metadata(), "context_definition": context_definition,
        "pilot": "same data-only analytic composite pilot as the current model",
        "changed_component": "Stage-1 input representation only: composite scores -> raw Y",
        "true_parameter_role": "NPE target and held-out evaluation only",
    }
    (output / "config.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def add_model_geometry(
    parser: argparse.ArgumentParser,
    *,
    include_sigma_q: bool,
) -> None:
    parser.add_argument("--model", choices=tuple(DEFAULTS), required=True)
    parser.add_argument("--n-blocks", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--tau", type=float)
    if include_sigma_q:
        parser.add_argument("--sigma-q", type=float)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    one = sub.add_parser("stage1")
    add_model_geometry(one, include_sigma_q=True)
    one.add_argument("--n-train", type=int, default=40_000)
    one.add_argument("--n-val", type=int, default=8_000)
    one.add_argument("--n-test", type=int, default=5_000)
    one.add_argument("--anchor-pi-min", type=float, default=0.05)
    one.add_argument("--anchor-pi-max", type=float, default=0.70)
    one.add_argument("--seed", type=int, default=20260709)
    one.add_argument("--architecture", choices=ARCHITECTURES, default="block_deepsets")
    one.add_argument(
        "--phi-hidden",
        type=int,
        default=32,
        help="Coordinate-embedding width for block_deepsets only; ignored by direct_flat_mlp.",
    )
    one.add_argument("--hidden", type=int, default=64)
    one.add_argument("--depth", type=int, default=2)
    one.add_argument("--iters", type=int, default=20_000)
    one.add_argument("--batch-size", type=int, default=512)
    one.add_argument("--eval-batch-size", type=int, default=512)
    one.add_argument("--lr", type=float, default=1e-4)
    one.add_argument("--lr-schedule", choices=("constant", "cosine_tail"))
    one.add_argument(
        "--lr-decay-start-step",
        type=int,
        default=10_000,
        help="Cosine-tail start step; ignored when --lr-schedule constant.",
    )
    one.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.1,
        help="Final/base LR ratio for cosine_tail; ignored for constant LR.",
    )
    one.add_argument("--weight-decay", type=float, default=1e-3)
    one.add_argument("--grad-clip", type=float, default=5.0)
    one.add_argument("--ema-decay", type=float, default=0.995)
    one.add_argument("--print-every", type=int, default=100)
    one.add_argument(
        "--checkpoint-selection", choices=("raw_or_ema", "raw"), default="raw_or_ema",
        help="raw_or_ema keeps the archived rule; raw matches the NLSA launchers.",
    )
    one.add_argument(
        "--diagnostic-pi-values",
        help="Frozen exact-score diagnostics only; ignored with --skip-exact-diagnostics.",
    )
    one.add_argument(
        "--skip-exact-diagnostics", action="store_true",
        help="Train and select using FSM validation only; do not evaluate exact scores.",
    )
    one.add_argument("--device", default="auto")
    one.add_argument("--output-dir", type=Path, required=True)

    two = sub.add_parser("stage2")
    add_model_geometry(two, include_sigma_q=False)
    two.add_argument("--stage1-run-dir", type=Path, required=True)
    two.add_argument("--stage1-checkpoint", type=Path)
    two.add_argument("--pi-prior-min", type=float, default=0.05)
    two.add_argument("--pi-prior-max", type=float, default=0.70)
    two.add_argument("--n-sbi-train", type=int, default=50_000)
    two.add_argument("--sbi-train-seed", type=int)
    two.add_argument("--sbi-model", default="mdn")
    two.add_argument("--sbi-hidden-features", type=int, default=64)
    two.add_argument(
        "--sbi-num-components",
        type=int,
        default=8,
        help="Mixture components for MDN; ignored by non-MDN estimators.",
    )
    two.add_argument(
        "--sbi-num-transforms",
        type=int,
        default=5,
        help="Flow transforms for MAF/NSF; ignored by MDN and MADE.",
    )
    two.add_argument(
        "--sbi-num-bins",
        type=int,
        default=8,
        help="Spline bins for NSF only; ignored by other estimators.",
    )
    two.add_argument("--sbi-batch-size", type=int, default=256)
    two.add_argument("--sbi-lr", type=float, default=5e-4)
    two.add_argument("--max-epochs", type=int, default=300)
    two.add_argument("--validation-fraction", type=float, default=0.10)
    two.add_argument("--stop-after-epochs", type=int, default=20)
    two.add_argument("--sbi-seed", type=int)
    two.add_argument("--test-pi-values", default="0.07,0.10,0.30,0.50,0.65,0.68")
    two.add_argument("--test-seeds", default="100,101,102,103,104,105,106,107,108,109")
    two.add_argument("--posterior-n", type=int, default=5_000)
    two.add_argument("--posterior-seed", type=int, default=87_000)
    two.add_argument("--grid-size", type=int, default=5_000)
    two.add_argument("--score-batch-size", type=int, default=256)
    two.add_argument("--pilot-grid-size", type=int, default=201)
    two.add_argument("--pilot-batch-size", type=int, default=512)
    two.add_argument("--pilot-grid-chunk-size", type=int, default=16)
    two.add_argument("--pilot-progress-every", type=int, default=10)
    two.add_argument(
        "--pilot-mode",
        choices=("moment", "marginal", "equal_channel"),
        help="Model 2 pilot family; Model 1 always uses its marginal pilot.",
    )
    two.add_argument(
        "--pilot-backend",
        choices=("auto", "torch", "numpy"),
        default="auto",
        help="Model 2 pilot-grid backend; Model 1 uses its torch marginal helper.",
    )
    two.add_argument("--sanity-seed", type=int, default=20260710)
    two.add_argument(
        "--pilot-cache", type=Path, default=None,
        help="Model 1 only: reuse a Stage-2 marginal-pilot cache for the same training bank.",
    )
    two.add_argument("--device", default="auto")
    two.add_argument("--data-device", default="cpu")
    two.add_argument(
        "--pilot-device",
        default="auto",
        help="Model 1 analytic-pilot device; Model 2 reuses --device.",
    )
    two.add_argument("--output-dir", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "stage1":
        train_stage1(args)
    else:
        train_stage2(args)


if __name__ == "__main__":
    main()
