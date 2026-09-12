#!/usr/bin/env python3
"""Amortized conditional FSM for the p=3 common-factor variance mixture.

Stage 1 only. Mirrors ``model1_p3`` -- anchors uniform in a box over
``beta = (logit pi, tau, log sigma)``, a local tube draw, and the standardized
target ``(theta - a) / sigma_q`` -- but with Model 2's two local-score channel
types (marginal and within-block pairwise).

Local maps compared under one shared simulation bank:

``linear``      ILSA: pool both channels' identity scores.
``gate``        ``s (*) m(s, a)``, elementwise positive, shared by both channels.
``stacked_shared`` / ``stacked_split``
                ``phi(s, a) = (1, s, m(s, a))`` with the learned channel either
                shared by both channel types or separate per type.

All nonlinear maps nest ILSA exactly at initialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from model1.stage1 import load_matched_linear_rho, softplus_inverse
from model1_p3.stage1 import (
    clone_state_dict, lr_multiplier, make_rho, set_global_seed, update_ema_state, write_csv,
)
from model2_p3 import scores as sc

P = sc.N_PARAMS


@dataclass(frozen=True)
class ArchSpec:
    input_keys: tuple[str, ...]
    seed_offset: int
    label: str
    warm_start: bool = False


ARCHITECTURES: dict[str, ArchSpec] = {
    "linear": ArchSpec(("block", "anchor_z"), 11, "linear ILSA DeepSets FSM (M2 p=3)"),
    "gate": ArchSpec(("s1", "s2", "anchor_z"), 22, "elementwise positive gate FSM (M2 p=3)", True),
    "stacked_shared": ArchSpec(("block", "s1", "s2", "anchor_z"), 33,
                               "stacked NLSA, shared channel (M2 p=3)", True),
    "stacked_split": ArchSpec(("block", "s1", "s2", "anchor_z"), 44,
                              "stacked NLSA, per-channel (M2 p=3)", True),
}
ORDERED_METHODS = ("linear", "gate", "stacked_shared", "stacked_split")
STACKED_METHODS = {"stacked_shared", "stacked_split"}


def parse_vector(text: str, name: str) -> np.ndarray:
    parts = [float(p) for p in str(text).split(",") if p.strip()]
    if len(parts) != P:
        raise ValueError(f"--{name} needs {P} comma-separated values, got {len(parts)}")
    return np.asarray(parts, dtype=np.float64)


def anchor_box(args) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray([sc.logit_np(args.pi_min), args.tau_min, math.log(args.sigma_min)])
    high = np.asarray([sc.logit_np(args.pi_max), args.tau_max, math.log(args.sigma_max)])
    if np.any(high <= low):
        raise ValueError("every anchor box coordinate must have positive width")
    return low, high


def sample_anchor_batch(rng, n, low, high, sigma_q, n_blocks, block_size):
    anchor = rng.uniform(low, high, size=(n, P))
    theta = anchor + sigma_q * rng.normal(size=(n, P))
    return {"y": sc.simulate(rng, theta, n_blocks, block_size), "anchor": anchor,
            "theta": theta, "target": ((theta - anchor) / sigma_q).astype(np.float32)}


def make_feature_stats(y, anchor):
    s1, s2 = sc.marginal_scores(y, anchor), sc.pairwise_scores(y, anchor)
    stats = {}
    for tag, s in (("s1", s1), ("s2", s2)):
        stats[f"{tag}_mean"] = s.mean(axis=(0, 1, 2))
        sd = s.std(axis=(0, 1, 2))
        stats[f"{tag}_sd"] = np.where(sd < 1e-8, 1.0, sd)
    b1 = ((s1 - stats["s1_mean"]) / stats["s1_sd"]).mean(axis=2)
    b2 = ((s2 - stats["s2_mean"]) / stats["s2_sd"]).mean(axis=2)
    block = np.concatenate([b1, b2], axis=-1)
    stats["block_mean"] = block.mean(axis=(0, 1))
    sd = block.std(axis=(0, 1))
    stats["block_sd"] = np.where(sd < 1e-8, 1.0, sd)
    stats["anchor_mean"] = anchor.mean(axis=0)
    stats["anchor_sd"] = np.where(anchor.std(axis=0) < 1e-8, 1.0, anchor.std(axis=0))
    return stats


def featurize(y, anchor, stats):
    s1 = ((sc.marginal_scores(y, anchor) - stats["s1_mean"]) / stats["s1_sd"]).astype(np.float32)
    s2 = ((sc.pairwise_scores(y, anchor) - stats["s2_mean"]) / stats["s2_sd"]).astype(np.float32)
    block = np.concatenate([s1.mean(axis=2), s2.mean(axis=2)], axis=-1)
    block = ((block - stats["block_mean"]) / stats["block_sd"]).astype(np.float32)
    anchor_z = ((anchor - stats["anchor_mean"]) / stats["anchor_sd"]).astype(np.float32)
    return {"s1": s1, "s2": s2, "block": block, "anchor_z": anchor_z}


def expand_anchor(anchor_z, like):
    shape = (anchor_z.shape[0],) + (1,) * (like.ndim - 2) + (P,)
    return anchor_z.reshape(shape).expand(*like.shape[:-1], P)


class PositiveGateMLP(nn.Module):
    """Elementwise positive multiplier on R^P, initialized exactly at 1."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * P, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, P))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, softplus_inverse(1.0))

    def forward(self, s, anchor):
        return F.softplus(self.net(torch.cat([s, anchor], dim=-1)))


class LocalFeatureMLP(nn.Module):
    """Unbounded learned channel, identically zero at initialization."""

    def __init__(self, hidden: int, m_dim: int):
        super().__init__()
        if m_dim < 1:
            raise ValueError("m_dim must be positive")
        self.net = nn.Sequential(nn.Linear(2 * P, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, m_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, anchor):
        return self.net(torch.cat([s, anchor], dim=-1))


class LinearDeepSets(nn.Module):
    def __init__(self, hidden, depth):
        super().__init__()
        self.rho = make_rho(3 * P, hidden, depth, out_dim=P)

    def forward(self, block, anchor_z):
        return self.rho(torch.cat([block, expand_anchor(anchor_z, block)], dim=-1)).sum(dim=1)


class GateDeepSets(nn.Module):
    """One elementwise positive gate shared by both channel types, in raw coordinates."""

    def __init__(self, hidden, depth, gate_hidden, stats):
        super().__init__()
        self.gate = PositiveGateMLP(gate_hidden)
        self.rho = make_rho(3 * P, hidden, depth, out_dim=P)
        for k in ("s1_mean", "s1_sd", "s2_mean", "s2_sd", "block_mean", "block_sd"):
            self.register_buffer(k, torch.as_tensor(stats[k], dtype=torch.float32))

    def forward(self, s1, s2, anchor_z):
        pooled = []
        for s, mean, sd in ((s1, self.s1_mean, self.s1_sd), (s2, self.s2_mean, self.s2_sd)):
            raw = s * sd + mean
            multiplier = self.gate(raw, expand_anchor(anchor_z, raw))
            # Delta form keeps the m == 1 nesting numerically exact.
            pooled.append((s + raw * (multiplier - 1.0) / sd).mean(dim=2))
        block = (torch.cat(pooled, dim=-1) - self.block_mean) / self.block_sd
        return self.rho(torch.cat([block, expand_anchor(anchor_z, block)], dim=-1)).sum(dim=1)


class StackedDeepSets(nn.Module):
    """phi(s, a) = (1, s, m(s, a)) on both channel types."""

    def __init__(self, hidden, depth, gate_hidden, stats, m_dim=1,
                 include_constant_channel=True, share_local_features=True):
        super().__init__()
        self.m_dim, self.include_constant_channel = int(m_dim), bool(include_constant_channel)
        self.share_local_features = bool(share_local_features)
        self.local_features = LocalFeatureMLP(gate_hidden, self.m_dim)
        self.local_features_pairwise = (None if self.share_local_features
                                        else LocalFeatureMLP(gate_hidden, self.m_dim))
        width = int(self.include_constant_channel) + 2 * P + 2 * self.m_dim + P
        self.rho = make_rho(width, hidden, depth, out_dim=P)
        for k in ("s1_mean", "s1_sd", "s2_mean", "s2_sd"):
            self.register_buffer(k, torch.as_tensor(stats[k], dtype=torch.float32))

    @property
    def linear_column_map(self) -> tuple[int, ...]:
        off = int(self.include_constant_channel)
        block = tuple(range(off, off + 2 * P))
        anchor = tuple(range(off + 2 * P + 2 * self.m_dim, off + 3 * P + 2 * self.m_dim))
        return block + anchor

    def _pairwise(self):
        return self.local_features if self.share_local_features else self.local_features_pairwise

    def readout_inputs(self, block, s1, s2, anchor_z):
        raw1, raw2 = s1 * self.s1_sd + self.s1_mean, s2 * self.s2_sd + self.s2_mean
        p1 = self.local_features(raw1, expand_anchor(anchor_z, raw1)).mean(dim=2)
        p2 = self._pairwise()(raw2, expand_anchor(anchor_z, raw2)).mean(dim=2)
        parts = [torch.ones_like(block[..., :1])] if self.include_constant_channel else []
        return torch.cat([*parts, block, p1, p2, expand_anchor(anchor_z, block)], dim=-1)

    def forward(self, block, s1, s2, anchor_z):
        return self.rho(self.readout_inputs(block, s1, s2, anchor_z)).sum(dim=1)


def build_model(method, config, stats):
    hidden, depth, gh = int(config["hidden"]), int(config["depth"]), int(config["gate_hidden"])
    if method == "linear":
        return LinearDeepSets(hidden, depth)
    if method == "gate":
        return GateDeepSets(hidden, depth, gh, stats)
    if method in STACKED_METHODS:
        return StackedDeepSets(hidden, depth, gh, stats, m_dim=int(config.get("m_dim", 1)),
                               include_constant_channel=bool(config.get("include_constant_channel", 1)),
                               share_local_features=method == "stacked_shared")
    raise ValueError(f"Unknown method: {method}")


def tensors_for_method(data, method, device):
    return {k: torch.as_tensor(data[k], dtype=torch.float32, device=device)
            for k in ARCHITECTURES[method].input_keys}


def forward_method(model, batch, method):
    return model(*(batch[k] for k in ARCHITECTURES[method].input_keys))


def branch_parameters(model, method):
    if method == "gate":
        return list(model.gate.parameters())
    if method in STACKED_METHODS:
        return [p for mod in (model.local_features, model.local_features_pairwise)
                if mod is not None for p in mod.parameters()]
    return None


def mse_over_tensors(model, x, y, method, batch_size):
    model.eval(); total = count = 0
    with torch.no_grad():
        for start in range(0, int(y.shape[0]), batch_size):
            sl = slice(start, min(start + batch_size, int(y.shape[0])))
            err = forward_method(model, {k: v[sl] for k, v in x.items()}, method) - y[sl]
            total += float((err ** 2).sum()); count += int(err.numel())
    model.train()
    return total / max(count, 1)


def predict(model, data, method, device, batch_size):
    x = tensors_for_method(data, method, device)
    n = int(next(iter(x.values())).shape[0]); out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            out.append(forward_method(model, {k: v[sl] for k, v in x.items()}, method).cpu().numpy())
    model.train()
    return np.concatenate(out, axis=0)


def train_model(model, method, train, val, *, iters, batch_size, lr, branch_lr, weight_decay,
                print_every, seed, grad_clip, ema_decay, lr_schedule, lr_decay_start_step,
                lr_min_ratio, device, name):
    started = time.perf_counter()
    model = model.to(device)
    x_train, x_val = tensors_for_method(train, method, device), tensors_for_method(val, method, device)
    y_train = torch.as_tensor(train["target"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val["target"], dtype=torch.float32, device=device)
    branch = branch_parameters(model, method)
    if branch is None:
        opt, base = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay), [lr]
    else:
        opt = torch.optim.AdamW([{"params": branch, "lr": branch_lr},
                                 {"params": list(model.rho.parameters()), "lr": lr}],
                                weight_decay=weight_decay)
        base = [branch_lr, lr]

    def set_lr(step):
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
    best_step = 0
    trace = [{"method": name, "step": 0, "train_loss": best_val, "val_loss_raw": best_val,
              "val_loss_ema": best_val, "best_val_loss": best_val}]
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
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
                best_val, best_step = val_raw, step
                best_state = clone_state_dict(model, cpu=True)
            trace.append({"method": name, "step": step, "train_loss": float(loss.item()),
                          "val_loss_raw": val_raw, "val_loss_ema": val_ema,
                          "best_val_loss": best_val, "grad_norm": float(gn.item()),
                          "lr": float(opt.param_groups[0]["lr"])})
            print(f"{name} step {step}/{iters}: train={loss.item():.6g} "
                  f"val_raw={val_raw:.6g} val_ema={val_ema:.6g} best={best_val:.6g}", flush=True)
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    info = {"best_step": best_step, "best_val_loss": best_val, "best_source": "raw",
            "checkpoint_selection": "raw", "minibatch_sha256": digest.hexdigest(),
            "wall_seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in model.parameters())}
    return model, trace, info


def run(args) -> None:
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
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

    rng = np.random.default_rng(args.seed)
    tr = sample_anchor_batch(rng, args.n_train, low, high, sigma_q, args.n_blocks, args.block_size)
    va = sample_anchor_batch(rng, args.n_val, low, high, sigma_q, args.n_blocks, args.block_size)
    stats = make_feature_stats(tr["y"], tr["anchor"])
    train = {**featurize(tr["y"], tr["anchor"], stats), "target": tr["target"]}
    val = {**featurize(va["y"], va["anchor"], stats), "target": va["target"]}
    print("marginal:", train["s1"].shape, " pairwise:", train["s2"].shape, flush=True)

    config = vars(args).copy()
    config.update({"sigma_q_vector": sigma_q.tolist(), "anchor_low": low.tolist(),
                   "anchor_high": high.tolist(), "device_resolved": device, "n_params": P,
                   "param_names": list(sc.PARAM_NAMES),
                   "target_definition": "(theta - anchor) / sigma_q; score = prediction / sigma_q",
                   "exact_score_available_to_checkpoint_selection": False})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez(out_dir / "feature_stats.npz", **stats, sigma_q=sigma_q)

    all_trace, training_info = [], {}
    check = {k: v[: min(256, v.shape[0])] for k, v in train.items()}
    lin_rho = lin_check = None
    ordered = [m for m in ORDERED_METHODS if m in methods]
    if any(ARCHITECTURES[m].warm_start for m in methods) and "linear" not in methods:
        set_global_seed(args.seed + ARCHITECTURES["linear"].seed_offset)
        ref = build_model("linear", config, stats).to(device)
        lin_rho, lin_check = clone_state_dict(ref.rho, cpu=True), predict(ref, check, "linear", device, args.batch_size)

    for method in ordered:
        spec = ARCHITECTURES[method]
        set_global_seed(args.seed + spec.seed_offset)
        model = build_model(method, config, stats)
        label, nesting = spec.label, None
        if method == "linear":
            model = model.to(device)
            lin_rho, lin_check = clone_state_dict(model.rho, cpu=True), predict(model, check, "linear", device, args.batch_size)
        else:
            if lin_rho is None:
                raise RuntimeError("matched_random requires the untrained linear readout")
            if method in STACKED_METHODS:
                load_matched_linear_rho(model.rho, lin_rho, model.linear_column_map,
                                        constant_column=0 if model.include_constant_channel else None)
            else:
                model.rho.load_state_dict(lin_rho)
            model = model.to(device)
            nesting = float(np.max(np.abs(lin_check - predict(model, check, method, device, args.batch_size))))
            print(f"{method} matched-random initialization max abs diff: {nesting:.3e}", flush=True)
            if nesting > 1e-5:
                raise RuntimeError(f"{method} does not reproduce the untrained linear reference: {nesting:.3e}")
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
        if nesting is not None:
            info["nested_initialization_max_abs_diff"] = nesting
        training_info[method] = info
        torch.save({"state_dict": model.state_dict(), "method": method, "label": label,
                    "config": config, "training_info": info}, out_dir / f"model_{method}.pt")
        all_trace.extend(trace)
        write_csv(out_dir / "training_trace.csv", all_trace)
        (out_dir / "training_info.json").write_text(json.dumps(training_info, indent=2), encoding="utf-8")
    print("\nSaved checkpoints to", out_dir, flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=40_000)
    p.add_argument("--n-val", type=int, default=8_000)
    p.add_argument("--n-blocks", type=int, default=40)
    p.add_argument("--block-size", type=int, default=40)
    p.add_argument("--pi-min", type=float, default=0.05)
    p.add_argument("--pi-max", type=float, default=0.70)
    p.add_argument("--tau-min", type=float, default=0.50)
    p.add_argument("--tau-max", type=float, default=2.00)
    p.add_argument("--sigma-min", type=float, default=0.70)
    p.add_argument("--sigma-max", type=float, default=1.40)
    p.add_argument("--sigma-q", type=str, default="0.15,0.08,0.04")
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--methods", type=str, default="linear,gate,stacked_shared,stacked_split")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--gate-hidden", type=int, default=16)
    p.add_argument("--m-dim", type=int, default=8)
    p.add_argument("--include-constant-channel", type=int, choices=(0, 1), default=1)
    p.add_argument("--iters", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--branch-lr", type=float, default=1e-3)
    p.add_argument("--lr-schedule", type=str, default="constant", choices=("constant", "cosine_tail"))
    p.add_argument("--lr-decay-start-step", type=int, default=10_000)
    p.add_argument("--lr-min-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="runs/model2_p3")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
