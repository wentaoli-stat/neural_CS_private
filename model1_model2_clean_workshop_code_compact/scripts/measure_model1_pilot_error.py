#!/usr/bin/env python3
"""Measure the Model 1 data-only pilot's error variance Sigma_e by simulation.

Reports Sigma_e = E[(u_pilot(Y) - u)^2] under the Stage-2 prior pi ~ U(pi_min, pi_max),
the draft bandwidth sqrt(2 Sigma_e), and, for comparison, sqrt(2 E[I(u)^-1]) with the
exact Fisher information I(u) = E_u[g0(Y, u)^2] estimated by Monte Carlo.
Uses only simulations; no Stage-1 checkpoint, Stage-2 output or posterior is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model1 import stage1
from model1 import stage2_npe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pilot", type=int, default=50_000)
    parser.add_argument("--n-fisher-grid", type=int, default=66)
    parser.add_argument("--n-fisher-per-point", type=int, default=20_000)
    parser.add_argument("--pi-min", type=float, default=0.05)
    parser.add_argument("--pi-max", type=float, default=0.70)
    parser.add_argument("--n-blocks", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--pilot-grid-size", type=int, default=201)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--output", type=Path, default=Path("runs/pilot_error/pilot_error.json"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    metadata = {"n_blocks": args.n_blocks, "block_size": args.block_size, "tau": args.tau}
    pilot_args = argparse.Namespace(
        pi_prior_min=args.pi_min, pi_prior_max=args.pi_max,
        pilot_grid_size=args.pilot_grid_size, device="cpu",
        pilot_batch_size=512, pilot_grid_chunk_size=16, pilot_progress_every=0,
    )
    pi = rng.uniform(args.pi_min, args.pi_max, size=args.n_pilot)
    u = stage1.logit_np(pi)
    pilot_u = np.empty_like(u)
    status = np.empty(u.shape, dtype="U32")
    for start in range(0, u.size, 5_000):
        stop = min(start + 5_000, u.size)
        y = stage2_npe.simulate(rng, pi[start:stop], metadata)
        pilot_u[start:stop], status[start:stop] = stage2_npe.data_only_pilot(y, metadata, pilot_args)
    err = pilot_u - u
    boundary = np.isin(status, ["left_boundary", "right_boundary"])
    sigma_e = float(np.mean(err**2))

    edges = np.linspace(args.pi_min, args.pi_max, 6)
    by_pi = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pi >= lo) & (pi < hi) if hi < edges[-1] else (pi >= lo) & (pi <= hi)
        by_pi.append({
            "pi_lo": float(lo), "pi_hi": float(hi), "n": int(mask.sum()),
            "bias_u": float(err[mask].mean()), "mse_u": float(np.mean(err[mask] ** 2)),
            "boundary_rate": float(boundary[mask].mean()),
        })

    fisher_pi = np.linspace(args.pi_min, args.pi_max, args.n_fisher_grid)
    info = np.empty_like(fisher_pi)
    for i, value in enumerate(fisher_pi):
        u_rep = np.full(args.n_fisher_per_point, stage1.logit_np(value))
        y = stage1.simulate_mean_shift(rng, u_rep, args.n_blocks, args.block_size, args.tau)
        info[i] = float(np.mean(stage1.full_score_u(y, value, args.tau) ** 2))
    # Trapezoid average of 1/I over the uniform pi prior.
    inv_info = 1.0 / info
    mean_inv_info = float(np.sum(0.5 * (inv_info[1:] + inv_info[:-1]) * np.diff(fisher_pi))
                          / (fisher_pi[-1] - fisher_pi[0]))

    result = {
        "n_pilot": args.n_pilot,
        "seed": args.seed,
        "sigma_e_u": sigma_e,
        "sigma_e_u_interior_only": float(np.mean(err[~boundary] ** 2)),
        "pilot_bias_u": float(err.mean()),
        "pilot_boundary_rate": float(boundary.mean()),
        "draft_sigma_q_from_measured_sigma_e": float(np.sqrt(2.0 * sigma_e)),
        "mean_inverse_fisher_info_u": mean_inv_info,
        "draft_sigma_q_from_inverse_fisher": float(np.sqrt(2.0 * mean_inv_info)),
        "by_pi": by_pi,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
