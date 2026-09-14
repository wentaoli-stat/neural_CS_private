#!/usr/bin/env python3
"""Time Model 2 Stage-1 training on this machine's accelerator before a long run.

Builds each Model 2 architecture exactly as ``model2.stage1`` does, with the
same optimizer groups, gradient clipping and determinism setting, and times
real training steps on random inputs of the production 40x40 shape. It then
projects the wall time of one 20,000-step run and of the whole GPU plan, and
checks that the feature cache the trainer copies onto the device will fit.

Feature preparation runs on the CPU before training and is not timed here.

    python scripts/benchmark_model2_gpu.py                 # auto: cuda > mps > cpu
    python scripts/benchmark_model2_gpu.py --device cuda --steps 100
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model2 import stage1 as m2  # noqa: E402  (package root must be importable)

GiB = 1024**3


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build(method: str, a: argparse.Namespace) -> torch.nn.Module:
    zero, one = np.array(0.0), np.array(1.0)
    if method == "linear":
        return m2.LinearCSBetaDeepSets(a.hidden, a.depth)
    if method == "shared_radial":
        return m2.SharedRawRadialCSBetaDeepSets(
            a.hidden, a.depth, a.gate_hidden, True, zero, one, zero, one,
            np.zeros((1, 1, 2)), np.ones((1, 1, 2)),
        )
    if method in m2.STACKED_METHODS:
        return m2.StackedCSBetaDeepSets(
            a.hidden, a.depth, a.gate_hidden, zero, one, zero, one,
            m_dim=a.m_dim, share_local_features=method == "stacked_shared",
        )
    raise ValueError(f"unknown method: {method}")


def make_optimizer(model: torch.nn.Module, method: str, lr: float) -> torch.optim.Optimizer:
    """Mirror the trainer's two parameter groups for the nonlinear maps."""
    if method in m2.GATED_METHODS:
        branch = list(model.shared_gate.parameters())
    elif method in m2.STACKED_METHODS:
        branch = [p for mod in (model.local_features, model.local_features_pairwise)
                  if mod is not None for p in mod.parameters()]
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    return torch.optim.AdamW([{"params": branch, "lr": lr},
                              {"params": list(model.rho.parameters()), "lr": lr}],
                             weight_decay=1e-3)


def random_batch(n: int, a: argparse.Namespace, device: str, gen: torch.Generator) -> dict:
    k, m = a.n_blocks, a.block_size
    pairs = m * (m - 1) // 2
    draw = lambda *shape: torch.randn(*shape, generator=gen).to(device)  # noqa: E731
    return {"s1": draw(n, k, m), "s2": draw(n, k, pairs), "block": draw(n, k, 2),
            "anchor_z": draw(n), "target": draw(n)}


def time_method(method: str, a: argparse.Namespace, device: str) -> dict:
    torch.manual_seed(0)
    model = build(method, a).to(device)
    opt = make_optimizer(model, method, 1e-3)
    gen = torch.Generator().manual_seed(1)
    batch = random_batch(a.batch_size, a, device, gen)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((m2.forward_method(model, batch, method) - batch["target"]) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

    for _ in range(a.warmup):
        step()
    sync(device)
    start = time.perf_counter()
    for _ in range(a.steps):
        step()
    sync(device)
    step_s = (time.perf_counter() - start) / a.steps

    # One validation chunk, forward only; a full pass is that many chunks.
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            m2.forward_method(model, batch, method)
        sync(device)
        start = time.perf_counter()
        for _ in range(a.steps // 5 or 1):
            m2.forward_method(model, batch, method)
        sync(device)
        chunk_s = (time.perf_counter() - start) / (a.steps // 5 or 1)
    val_pass_s = chunk_s * math.ceil(a.n_val / a.batch_size)
    peak = torch.cuda.max_memory_allocated() / GiB if device.startswith("cuda") else float("nan")
    # The trainer evaluates the raw and the EMA weights at every logging step.
    run_s = a.iters * step_s + 2 * (a.iters // a.print_every) * val_pass_s
    return {"method": method, "step_s": step_s, "val_pass_s": val_pass_s,
            "run_s": run_s, "peak_gib": peak}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--methods", default="linear,shared_radial,stacked_shared,stacked_split")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n-blocks", type=int, default=40)
    ap.add_argument("--block-size", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--gate-hidden", type=int, default=16)
    ap.add_argument("--m-dim", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20_000)
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=40_000)
    ap.add_argument("--n-val", type=int, default=8_000)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()

    device = pick_device(a.device)
    torch.use_deterministic_algorithms(True, warn_only=True)  # as the trainer sets it
    name = torch.cuda.get_device_name(0) if device.startswith("cuda") else device
    print(f"device: {device} ({name}); torch {torch.__version__}")

    pairs = a.block_size * (a.block_size - 1) // 2
    per_dataset = a.n_blocks * (a.block_size + pairs + 2) * 4 + 8
    cache_gib = (a.n_train + a.n_val) * per_dataset / GiB
    print(f"feature cache the trainer copies to the device: ~{cache_gib:.1f} GiB "
          f"({a.n_train} train + {a.n_val} validation datasets)")
    if device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory / GiB
        print(f"device memory: {total:.1f} GiB")

    rows = [time_method(method, a, device) for method in a.methods.split(",")]
    print(f"\n{a.steps} timed steps after {a.warmup} warm-up, batch {a.batch_size}, "
          f"{a.n_blocks}x{a.block_size}\n")
    print(f"{'method':16s}{'s/step':>10s}{'val pass s':>12s}{'one run':>12s}{'peak GiB':>10s}")
    for r in rows:
        print(f"{r['method']:16s}{r['step_s']:10.4f}{r['val_pass_s']:12.2f}"
              f"{r['run_s'] / 60:10.1f} m{r['peak_gib']:10.2f}")

    by = {r["method"]: r["run_s"] for r in rows}
    if {"linear", "shared_radial", "stacked_shared", "stacked_split"} <= set(by):
        audit = 2 * a.seeds * (by["linear"] + by["shared_radial"])
        stacked = a.seeds * (by["stacked_shared"] + by["stacked_split"])
        print(f"\nprojected GPU plan (training + validation, excluding CPU feature preparation):")
        print(f"  learning-rate audit, linear + gate x 2 rates x {a.seeds} seeds: {audit / 3600:.1f} h")
        print(f"  stacked shared + per-channel x {a.seeds} seeds:              {stacked / 3600:.1f} h")
        print(f"  total:                                                    {(audit + stacked) / 3600:.1f} h")
    if device.startswith("cuda"):
        need = cache_gib + max(r["peak_gib"] for r in rows)
        verdict = "fits" if need < 0.9 * total else "DOES NOT FIT: use a larger GPU"
        print(f"\nmemory: cache {cache_gib:.1f} + training peak {need - cache_gib:.1f} "
              f"= {need:.1f} GiB of {total:.1f} GiB, {verdict}")


if __name__ == "__main__":
    main()
