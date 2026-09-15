#!/usr/bin/env python3
"""Amortized conditional FSM for the p=3 mean-shift mixture.

Stage 1 only. We sample anchors ``a`` over a box in
``beta = (logit pi, tau, log sigma)``, draw ``theta | a ~ N(a, diag(sigma_q^2))``,
simulate ``Y ~ p_theta``, evaluate the local scores at the anchor, and regress
the standardized tube target ``(theta - a) / sigma_q`` onto the aggregated
representation. The network's score estimate is its output divided by
``sigma_q``; coordinates are standardized because the three parameters live on
very different scales.

Three local maps are compared under one shared simulation bank:

``linear``
    ILSA. Pool the identity local scores; the readout sees only group means.
``gate``
    The multiplicative map ``s (*) m(s, a)`` with ``m > 0`` elementwise, the
    p>1 reading of the elementwise gate used in the p=1 experiments.
``stacked``
    The methodology's ``phi(s, a) = (1, s, m(s, a))``, concatenating an
    unbounded learned channel beside the identity.

Both nonlinear maps nest ILSA exactly at initialization and are warm-started
from the same untrained linear readout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F

from model1.stage1 import load_matched_linear_rho, softplus_inverse
from model1_p3 import scores as sc

P = sc.N_PARAMS


@dataclass(frozen=True)
class ArchSpec:
    input_keys: tuple[str, ...]
    seed_offset: int
    label: str
    warm_start: bool = False
    needs_raw_y: bool = False


ARCHITECTURES: dict[str, ArchSpec] = {
    "linear": ArchSpec(("block", "anchor_z"), 11, "linear ILSA DeepSets FSM (p=3)"),
    "gate": ArchSpec(("s", "anchor_z"), 22, "elementwise positive gate FSM (p=3)", warm_start=True),
    "stacked": ArchSpec(("block", "s", "anchor_z"), 33, "stacked NLSA local map (p=3)", warm_start=True),
    # Reference points for the Proposition 1 decomposition. None of these nests
    # ILSA, so none is warm-started from the untrained linear readout.
    "raw": ArchSpec(("y_z", "anchor_z"), 44, "raw-data flat MLP FSM (p=3)", needs_raw_y=True),
    "substack": ArchSpec(("s", "anchor_z"), 55, "flattened all-subscore MLP FSM (p=3)"),
    "blockstack": ArchSpec(("s", "anchor_z"), 66, "per-block unpooled subscore stack FSM (p=3)"),
}

ORDERED_METHODS = ("linear", "gate", "stacked", "raw", "substack", "blockstack")

def make_rho(in_dim: int, hidden: int, depth: int, out_dim: int) -> nn.Sequential:
    """Shared block readout. Identical to the p=1 helper but vector-valued."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.SiLU())
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_vector(text: str, name: str) -> np.ndarray:
    parts = [float(p) for p in str(text).split(",") if p.strip()]
    if len(parts) != P:
        raise ValueError(f"--{name} needs {P} comma-separated values, got {len(parts)}")
    return np.asarray(parts, dtype=np.float64)


def anchor_box(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(
        [sc.logit_np(args.pi_min), args.tau_min, math.log(args.sigma_min)], dtype=np.float64
    )
    high = np.asarray(
        [sc.logit_np(args.pi_max), args.tau_max, math.log(args.sigma_max)], dtype=np.float64
    )
    if np.any(high <= low):
        raise ValueError("every anchor box coordinate must have positive width")
    return low, high


def sample_anchor_batch(
    rng: np.random.Generator,
    n: int,
    low: np.ndarray,
    high: np.ndarray,
    sigma_q: np.ndarray,
    n_blocks: int,
    block_size: int,
) -> dict[str, np.ndarray]:
    """Anchors uniform in the box, a local tube draw, and data at the draw."""
    anchor = rng.uniform(low, high, size=(n, P))
    theta = anchor + sigma_q * rng.normal(size=(n, P))
    y = sc.simulate(rng, theta, n_blocks, block_size)
    return {
        "y": y,
        "anchor": anchor,
        "theta": theta,
        # Standardized tube target; the score estimate is this divided by sigma_q.
        "target": ((theta - anchor) / sigma_q).astype(np.float32),
    }


def make_feature_stats(y: np.ndarray, anchor: np.ndarray) -> dict[str, np.ndarray]:
    s = sc.local_scores(y, anchor)
    stats: dict[str, np.ndarray] = {
        "s_mean": s.mean(axis=(0, 1, 2)),
        "s_sd": s.std(axis=(0, 1, 2)),
    }
    stats["s_sd"] = np.where(stats["s_sd"] < 1e-8, 1.0, stats["s_sd"])
    block_raw = ((s - stats["s_mean"]) / stats["s_sd"]).mean(axis=2)
    stats["block_mean"] = block_raw.mean(axis=(0, 1))
    stats["block_sd"] = block_raw.std(axis=(0, 1))
    stats["block_sd"] = np.where(stats["block_sd"] < 1e-8, 1.0, stats["block_sd"])
    stats["y_mean"] = np.asarray(y.mean(), dtype=np.float64)
    stats["y_sd"] = np.asarray(max(float(y.std()), 1e-8), dtype=np.float64)
    stats["anchor_mean"] = anchor.mean(axis=0)
    stats["anchor_sd"] = np.where(anchor.std(axis=0) < 1e-8, 1.0, anchor.std(axis=0))
    return stats


def featurize(y: np.ndarray, anchor: np.ndarray, stats: dict[str, np.ndarray],
              *, need_raw_y: bool = False) -> dict[str, np.ndarray]:
    s = sc.local_scores(y, anchor)
    s_z = ((s - stats["s_mean"]) / stats["s_sd"]).astype(np.float32)
    block_raw = s_z.mean(axis=2)
    block = ((block_raw - stats["block_mean"]) / stats["block_sd"]).astype(np.float32)
    anchor_z = ((anchor - stats["anchor_mean"]) / stats["anchor_sd"]).astype(np.float32)
    out = {"s": s_z, "block": block, "anchor_z": anchor_z}
    if need_raw_y:
        out["y_z"] = ((y - stats["y_mean"]) / stats["y_sd"]).astype(np.float32)
    return out


class PositiveGateMLP(nn.Module):
    """Elementwise positive multiplier on R^P, initialized exactly at 1."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * P, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, P),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, softplus_inverse(1.0))

    def forward(self, s: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(torch.cat([s, anchor], dim=-1)))


class LocalFeatureMLP(nn.Module):
    """Unbounded learned channel m(s, a) in R^m_dim, initially identically zero."""

    def __init__(self, hidden: int, m_dim: int):
        super().__init__()
        if m_dim < 1:
            raise ValueError("m_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(2 * P, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, m_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, anchor], dim=-1))


def expand_anchor(anchor_z: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    shape = (anchor_z.shape[0],) + (1,) * (like.ndim - 2) + (P,)
    return anchor_z.reshape(shape).expand(*like.shape[:-1], P)


class LinearDeepSets(nn.Module):
    """ILSA: pooled local map (1, s) or s alone, conditioned on the anchor."""

    def __init__(self, hidden: int, depth: int, include_constant_channel: bool = False):
        super().__init__()
        self.include_constant_channel = bool(include_constant_channel)
        self.rho = make_rho(int(self.include_constant_channel) + 2 * P, hidden, depth, out_dim=P)

    def forward(self, block: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        parts = [torch.ones_like(block[..., :1])] if self.include_constant_channel else []
        x = torch.cat([*parts, block, expand_anchor(anchor_z, block)], dim=-1)
        return self.rho(x).sum(dim=1)


class GateDeepSets(nn.Module):
    """Elementwise positive gate applied before pooling."""

    def __init__(self, hidden: int, depth: int, gate_hidden: int,
                 block_mean: np.ndarray, block_sd: np.ndarray,
                 include_constant_channel: bool = False):
        super().__init__()
        self.include_constant_channel = bool(include_constant_channel)
        self.gate = PositiveGateMLP(gate_hidden)
        self.rho = make_rho(int(self.include_constant_channel) + 2 * P, hidden, depth, out_dim=P)
        self.register_buffer("block_mean", torch.as_tensor(block_mean, dtype=torch.float32))
        self.register_buffer("block_sd", torch.as_tensor(block_sd, dtype=torch.float32))

    def forward(self, s: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        gated = s * self.gate(s, expand_anchor(anchor_z, s))
        block = (gated.mean(dim=2) - self.block_mean) / self.block_sd
        parts = [torch.ones_like(block[..., :1])] if self.include_constant_channel else []
        x = torch.cat([*parts, block, expand_anchor(anchor_z, block)], dim=-1)
        return self.rho(x).sum(dim=1)


class StackedDeepSets(nn.Module):
    """phi(s, a) = (1, s, m(s, a)); the identity channel is reused verbatim."""

    def __init__(self, hidden: int, depth: int, gate_hidden: int,
                 m_dim: int = 1, include_constant_channel: bool = True):
        super().__init__()
        self.m_dim = int(m_dim)
        self.include_constant_channel = bool(include_constant_channel)
        self.local_features = LocalFeatureMLP(gate_hidden, self.m_dim)
        width = int(self.include_constant_channel) + 2 * P + self.m_dim
        self.rho = make_rho(width, hidden, depth, out_dim=P)

    @property
    def linear_column_map(self) -> tuple[int, ...]:
        off = int(self.include_constant_channel)
        block_cols = tuple(range(off, off + P))
        anchor_cols = tuple(range(off + P + self.m_dim, off + 2 * P + self.m_dim))
        return block_cols + anchor_cols

    def readout_inputs(self, block: torch.Tensor, s: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        pooled = self.local_features(s, expand_anchor(anchor_z, s)).mean(dim=2)
        parts = [torch.ones_like(block[..., :1])] if self.include_constant_channel else []
        return torch.cat([*parts, block, pooled, expand_anchor(anchor_z, block)], dim=-1)

    def forward(self, block: torch.Tensor, s: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        return self.rho(self.readout_inputs(block, s, anchor_z)).sum(dim=1)


class RawFlatMLP(nn.Module):
    """Concatenate the whole standardized sample with the anchor; no structure.

    Sees strictly more information than any local-score method, but must learn
    the block exchangeability and the likelihood geometry from scratch.
    """

    def __init__(self, hidden: int, depth: int, n_blocks: int, block_size: int):
        super().__init__()
        self.rho = make_rho(n_blocks * block_size + P, hidden, depth, out_dim=P)

    def forward(self, y_z: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        return self.rho(torch.cat([y_z.flatten(start_dim=1), anchor_z], dim=-1))


class SubscoreStackMLP(nn.Module):
    """Flatten every local score into one vector; the local-score summary with
    no aggregation structure at all."""

    def __init__(self, hidden: int, depth: int, n_blocks: int, block_size: int):
        super().__init__()
        self.rho = make_rho(n_blocks * block_size * P + P, hidden, depth, out_dim=P)

    def forward(self, s: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        return self.rho(torch.cat([s.flatten(start_dim=1), anchor_z], dim=-1))


class BlockStackDeepSets(nn.Module):
    """Keep the additive block structure but drop the within-block pooling.

    Each block's local scores enter the readout as an ordered stack rather than
    a sum, so this is the direct ablation of the transform-sum restriction in
    Eq. (2): same blockwise additivity, no within-group aggregation.
    """

    def __init__(self, hidden: int, depth: int, block_size: int):
        super().__init__()
        self.rho = make_rho(block_size * P + P, hidden, depth, out_dim=P)

    def forward(self, s: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        stacked = s.flatten(start_dim=2)
        x = torch.cat([stacked, expand_anchor(anchor_z, stacked)], dim=-1)
        return self.rho(x).sum(dim=1)


def stacked_linear_column_map(
    stacked: StackedDeepSets, linear_constant_channel: bool,
) -> tuple[tuple[int, ...], int | None]:
    """Column map and zeroed constant column for embedding ILSA in the stacked readout."""
    if linear_constant_channel:
        if not stacked.include_constant_channel:
            raise ValueError("ILSA with a constant channel nests only in a stacked map that has one")
        return (0,) + stacked.linear_column_map, None
    return stacked.linear_column_map, (0 if stacked.include_constant_channel else None)


def build_model(method: str, config: dict[str, Any], stats: dict[str, np.ndarray]) -> nn.Module:
    hidden, depth = int(config["hidden"]), int(config["depth"])
    # Configs written before the flag existed trained ILSA on s alone.
    linear_constant = bool(config.get("linear_constant_channel", 0))
    if method == "linear":
        return LinearDeepSets(hidden, depth, include_constant_channel=linear_constant)
    if method == "gate":
        return GateDeepSets(hidden, depth, int(config["gate_hidden"]),
                            stats["block_mean"], stats["block_sd"],
                            include_constant_channel=linear_constant)
    if method == "stacked":
        return StackedDeepSets(hidden, depth, int(config["gate_hidden"]),
                               m_dim=int(config.get("m_dim", 1)),
                               include_constant_channel=bool(config.get("include_constant_channel", 1)))
    n_blocks, block_size = int(config["n_blocks"]), int(config["block_size"])
    if method == "raw":
        return RawFlatMLP(hidden, depth, n_blocks, block_size)
    if method == "substack":
        return SubscoreStackMLP(hidden, depth, n_blocks, block_size)
    if method == "blockstack":
        return BlockStackDeepSets(hidden, depth, block_size)
    raise ValueError(f"Unknown method: {method}")


def tensors_for_method(data: dict[str, np.ndarray], method: str, device: str) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(data[key], dtype=torch.float32, device=device)
        for key in ARCHITECTURES[method].input_keys
    }


def forward_method(model: nn.Module, batch: dict[str, torch.Tensor], method: str) -> torch.Tensor:
    return model(*(batch[key] for key in ARCHITECTURES[method].input_keys))


def branch_parameters(model: nn.Module, method: str) -> list[nn.Parameter] | None:
    if method == "gate":
        return list(model.gate.parameters())
    if method == "stacked":
        return list(model.local_features.parameters())
    return None


def mse_over_tensors(model, x, y, method, batch_size) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        n = int(y.shape[0])
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            err = forward_method(model, {k: v[sl] for k, v in x.items()}, method) - y[sl]
            total += float(torch.sum(err**2).item())
            count += int(err.numel())
    model.train()
    return total / max(count, 1)


def predict(model, data, method, device, batch_size) -> np.ndarray:
    x = tensors_for_method(data, method, device)
    n = int(next(iter(x.values())).shape[0])
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            out.append(forward_method(model, {k: v[sl] for k, v in x.items()}, method).cpu().numpy())
    model.train()
    return np.concatenate(out, axis=0)


def clone_state_dict(model: nn.Module, *, cpu: bool) -> dict[str, torch.Tensor]:
    return {k: (v.detach().clone().cpu() if cpu else v.detach().clone())
            for k, v in model.state_dict().items()}


@torch.no_grad()
def update_ema_state(ema_state, model, decay: float) -> None:
    for key, value in model.state_dict().items():
        if torch.is_floating_point(value):
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[key].copy_(value.detach())


def lr_multiplier(step, iters, schedule, decay_start, min_ratio) -> float:
    if schedule == "constant" or step <= decay_start:
        return 1.0
    progress = min(max((step - decay_start) / (iters - decay_start), 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_model(model, method, train, val, *, iters, batch_size, lr, branch_lr, weight_decay,
                print_every, seed, grad_clip, ema_decay, lr_schedule, lr_decay_start_step,
                lr_min_ratio, device, name) -> tuple[nn.Module, list[dict], dict]:
    started = time.perf_counter()
    model = model.to(device)
    x_train, x_val = tensors_for_method(train, method, device), tensors_for_method(val, method, device)
    y_train = torch.as_tensor(train["target"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val["target"], dtype=torch.float32, device=device)

    branch = branch_parameters(model, method)
    if branch is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        base = [lr]
    else:
        opt = torch.optim.AdamW(
            [{"params": branch, "lr": branch_lr}, {"params": list(model.rho.parameters()), "lr": lr}],
            weight_decay=weight_decay,
        )
        base = [branch_lr, lr]

    def set_lr(step: int) -> None:
        mult = lr_multiplier(step, iters, lr_schedule, lr_decay_start_step, lr_min_ratio)
        for group, value in zip(opt.param_groups, base, strict=True):
            group["lr"] = value * mult

    set_lr(0)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    best_val = mse_over_tensors(model, x_val, y_val, method, batch_size)
    if not np.isfinite(best_val):
        raise RuntimeError(f"{name} produced a non-finite initial validation loss")
    best_state, ema_state = clone_state_dict(model, cpu=True), clone_state_dict(model, cpu=False)
    best_step, best_source = 0, "raw"
    trace = [{"method": name, "step": 0, "train_loss": best_val, "val_loss_raw": best_val,
              "val_loss_ema": best_val, "best_val_loss": best_val, "best_source": best_source}]
    n_train = int(y_train.shape[0])
    print(f"{name} step 0/{iters}: val={best_val:.6g}", flush=True)

    for step in range(1, iters + 1):
        set_lr(step)
        indices = rng.integers(0, n_train, size=min(batch_size, n_train))
        digest.update(indices.tobytes())
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)
        xb = {k: v.index_select(0, idx) for k, v in x_train.items()}
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((forward_method(model, xb, method) - y_train.index_select(0, idx)) ** 2)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        update_ema_state(ema_state, model, ema_decay)

        if step % print_every == 0 or step == iters:
            raw_state = clone_state_dict(model, cpu=False)
            val_raw = mse_over_tensors(model, x_val, y_val, method, batch_size)
            model.load_state_dict(ema_state)
            val_ema = mse_over_tensors(model, x_val, y_val, method, batch_size)
            model.load_state_dict(raw_state)
            if not np.isfinite(val_raw) or not np.isfinite(val_ema):
                raise RuntimeError(f"{name} produced a non-finite validation loss at step {step}")
            # Validation-best raw only; the EMA loss is logged but cannot select.
            if val_raw < best_val:
                best_val, best_step, best_source = val_raw, step, "raw"
                best_state = clone_state_dict(model, cpu=True)
            trace.append({"method": name, "step": step, "train_loss": float(loss.item()),
                          "val_loss_raw": val_raw, "val_loss_ema": val_ema,
                          "best_val_loss": best_val, "best_source": best_source,
                          "grad_norm": float(grad_norm.item()),
                          "lr": float(opt.param_groups[0]["lr"])})
            print(f"{name} step {step}/{iters}: train={loss.item():.6g} val_raw={val_raw:.6g} "
                  f"val_ema={val_ema:.6g} best={best_val:.6g}", flush=True)

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    info = {"best_step": best_step, "best_val_loss": best_val, "best_source": best_source,
            "checkpoint_selection": "raw", "minibatch_sha256": digest.hexdigest(),
            "wall_seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in model.parameters())}
    return model, trace, info


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = sorted(set(methods) - set(ARCHITECTURES))
    if bad:
        raise ValueError(f"Unknown methods: {bad}")
    sigma_q = parse_vector(args.sigma_q, "sigma-q")
    if np.any(sigma_q <= 0):
        raise ValueError("--sigma-q entries must be positive")
    low, high = anchor_box(args)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"torch device: {device}\nOutput directory: {out_dir}", flush=True)
    print("anchor box (u, tau, log sigma):", list(np.round(low, 4)), "to", list(np.round(high, 4)), flush=True)

    rng = np.random.default_rng(args.seed)
    print("\nGenerating amortized FSM data...", flush=True)
    train_raw = sample_anchor_batch(rng, args.n_train, low, high, sigma_q, args.n_blocks, args.block_size)
    val_raw = sample_anchor_batch(rng, args.n_val, low, high, sigma_q, args.n_blocks, args.block_size)
    print("train y:", train_raw["y"].shape, "val y:", val_raw["y"].shape, flush=True)

    stats = make_feature_stats(train_raw["y"], train_raw["anchor"])
    need_raw_y = any(ARCHITECTURES[m].needs_raw_y for m in methods)
    train = {**featurize(train_raw["y"], train_raw["anchor"], stats, need_raw_y=need_raw_y),
             "target": train_raw["target"]}
    val = {**featurize(val_raw["y"], val_raw["anchor"], stats, need_raw_y=need_raw_y),
           "target": val_raw["target"]}
    print("local scores:", train["s"].shape, " block features:", train["block"].shape, flush=True)

    config = vars(args).copy()
    config["sigma_q_vector"] = sigma_q.tolist()
    config["anchor_low"], config["anchor_high"] = low.tolist(), high.tolist()
    config["device_resolved"] = device
    config["n_params"] = P
    config["param_names"] = list(sc.PARAM_NAMES)
    config["target_definition"] = "(theta - anchor) / sigma_q; score = prediction / sigma_q"
    config["exact_score_available_to_checkpoint_selection"] = False
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez(out_dir / "feature_stats.npz", **stats, sigma_q=sigma_q)

    all_trace: list[dict] = []
    training_info: dict[str, Any] = {}
    check = {k: v[: min(256, v.shape[0])] for k, v in train.items()}
    linear_rho_state = None
    linear_check = None

    ordered = [m for m in ORDERED_METHODS if m in methods]
    needs_reference = any(ARCHITECTURES[m].warm_start for m in methods)
    if needs_reference and "linear" not in methods:
        set_global_seed(args.seed + ARCHITECTURES["linear"].seed_offset)
        reference = build_model("linear", config, stats).to(device)
        linear_rho_state = clone_state_dict(reference.rho, cpu=True)
        linear_check = predict(reference, check, "linear", device, args.batch_size)

    for method in ordered:
        spec = ARCHITECTURES[method]
        set_global_seed(args.seed + spec.seed_offset)
        model = build_model(method, config, stats)
        label, nesting_error = spec.label, None
        if method == "linear":
            model = model.to(device)
            linear_rho_state = clone_state_dict(model.rho, cpu=True)
            linear_check = predict(model, check, "linear", device, args.batch_size)
        elif not spec.warm_start:
            # Reference baselines do not nest ILSA and are trained from their
            # own initialization; there is nothing to match.
            model = model.to(device)
        else:
            if linear_rho_state is None or linear_check is None:
                raise RuntimeError("matched_random requires the untrained linear readout")
            if method == "stacked":
                column_map, constant_column = stacked_linear_column_map(
                    model, bool(args.linear_constant_channel))
                load_matched_linear_rho(model.rho, linear_rho_state, column_map,
                                        constant_column=constant_column)
            else:
                model.rho.load_state_dict(linear_rho_state)
            model = model.to(device)
            nesting_error = float(np.max(np.abs(linear_check - predict(model, check, method, device, args.batch_size))))
            print(f"{method} matched-random initialization max abs diff: {nesting_error:.3e}", flush=True)
            if nesting_error > 1e-5:
                raise RuntimeError(f"{method} does not reproduce the untrained linear reference: {nesting_error:.3e}")
            label = f"{label} [matched_random]"

        print(f"\nTraining {label}...", flush=True)
        model, trace, info = train_model(
            model, method, train, val, iters=args.iters, batch_size=args.batch_size,
            lr=args.lr, branch_lr=args.branch_lr, weight_decay=args.weight_decay,
            print_every=args.print_every, seed=args.seed + 30_001, grad_clip=args.grad_clip,
            ema_decay=args.ema_decay, lr_schedule=args.lr_schedule,
            lr_decay_start_step=args.lr_decay_start_step, lr_min_ratio=args.lr_min_ratio,
            device=device, name=label)
        info.update({"method": method, "label": label})
        if nesting_error is not None:
            info["nested_initialization_max_abs_diff"] = nesting_error
        training_info[method] = info
        torch.save({"state_dict": model.state_dict(), "method": method, "label": label,
                    "config": config, "training_info": info}, out_dir / f"model_{method}.pt")
        all_trace.extend(trace)
        write_csv(out_dir / "training_trace.csv", all_trace)
        (out_dir / "training_info.json").write_text(json.dumps(training_info, indent=2), encoding="utf-8")
    print("\nSaved checkpoints to", out_dir, flush=True)
    print("Exact-score evaluation is a separate command: python -m model1_p3.evaluate_stage1", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=40_000)
    parser.add_argument("--n-val", type=int, default=8_000)
    parser.add_argument("--n-blocks", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--pi-min", type=float, default=0.05)
    parser.add_argument("--pi-max", type=float, default=0.70)
    parser.add_argument("--tau-min", type=float, default=0.50)
    parser.add_argument("--tau-max", type=float, default=2.00)
    parser.add_argument("--sigma-min", type=float, default=0.70)
    parser.add_argument("--sigma-max", type=float, default=1.40)
    parser.add_argument("--sigma-q", type=str, default="0.20,0.08,0.04")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--methods", type=str, default="linear,gate,stacked")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--m-dim", type=int, default=1)
    parser.add_argument("--include-constant-channel", type=int, choices=(0, 1), default=1)
    parser.add_argument("--linear-constant-channel", type=int, choices=(0, 1), default=1,
                        help="ILSA and gate readouts see (1, s) when 1, s alone when 0. "
                             "Use 0 to reproduce runs made before this option existed.")
    parser.add_argument("--iters", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4, help="Readout learning rate.")
    parser.add_argument("--branch-lr", type=float, default=1e-3,
                        help="Learning rate for the local gate / learned-channel MLP.")
    parser.add_argument("--lr-schedule", type=str, default="cosine_tail", choices=("constant", "cosine_tail"))
    parser.add_argument("--lr-decay-start-step", type=int, default=10_000)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="runs/model1_p3")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
