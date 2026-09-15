#!/usr/bin/env python3
"""Population block-score information lost by pooling Model 1 local scores unchanged.

For one block at the generating value, with a = logit(pi),

    l_j = tau * Y_j - tau^2 / 2,   h_a(l) = expit(a + l) - expit(a),
    A = mean_j h_a(l_j)            (what ILSA's readout sees),
    B = h_a(sum_j l_j)             (the exact block score),

and D(pi, m, tau) = E{Var(B | A)} / Var(B). This is the fraction of exact-score
variance no readout of A can recover. Blocks are independent, so D is also the
fraction lost for the full score. NLSA can drive it to zero because h_a is
invertible. E(B | A) is estimated by averaging B within equal-count bins of A,
fitted on one half of the draws and evaluated on the other half.

No network, checkpoint, simulation bank or posterior is used.

    python scripts/model1_pooling_loss.py --settings 20:0.5,20:0.7,5:1.1
    python scripts/model1_pooling_loss.py --block-sizes 5,20 --tau-grid 0.3:1.6:0.1
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def expit(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def pooling_loss(pi: float, m: int, tau: float, n: int, bins: int, seed: int,
                 chunk: int = 200_000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = math.log(pi) - math.log1p(-pi)
    A = np.empty(n)
    B = np.empty(n)
    for start in range(0, n, chunk):
        size = min(chunk, n - start)
        active = rng.random(size) < pi
        y = rng.standard_normal((size, m)) + tau * active[:, None]
        l = tau * y - 0.5 * tau**2
        A[start:start + size] = (expit(a + l) - pi).mean(axis=1)
        B[start:start + size] = expit(a + l.sum(axis=1)) - pi
    half = n // 2
    edges = np.quantile(A[:half], np.linspace(0.0, 1.0, bins + 1)[1:-1])
    fit_bin = np.searchsorted(edges, A[:half])
    means = np.bincount(fit_bin, weights=B[:half], minlength=bins) / np.maximum(
        np.bincount(fit_bin, minlength=bins), 1)
    held_out = means[np.searchsorted(edges, A[half:])]
    var_b = float(B[half:].var())
    loss = float(np.mean((B[half:] - held_out) ** 2)) / var_b
    corr = float(np.corrcoef(A[half:], B[half:])[0, 1])
    return {"pi": pi, "block_size": m, "tau": tau, "D": loss, "linear_R2_loss": 1.0 - corr**2,
            # The transition-matching shift m*tau^2/2 = log((1-pi)/pi) has no
            # positive solution once pi >= 0.5.
            "var_B": var_b, "tau_rule": math.sqrt(max(2.0 * math.log((1.0 - pi) / pi) / m, 0.0))}


def parse_range(text: str) -> list[float]:
    lo, hi, step = (float(x) for x in text.split(":"))
    return [round(v, 6) for v in np.arange(lo, hi + step / 2, step)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pi-values", default="0.05,0.07,0.10,0.20,0.30,0.50,0.65")
    parser.add_argument("--settings", default="", help="Comma list of block_size:tau pairs.")
    parser.add_argument("--block-sizes", default="", help="With --tau-grid, scan tau for each block size.")
    parser.add_argument("--tau-grid", default="", help="lo:hi:step")
    parser.add_argument("--n", type=int, default=1_000_000, help="Blocks simulated per cell.")
    parser.add_argument("--bins", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    pis = [float(x) for x in args.pi_values.split(",")]
    cells: list[tuple[int, float]] = []
    if args.settings:
        cells += [(int(m), float(t)) for m, t in (item.split(":") for item in args.settings.split(","))]
    if args.block_sizes and args.tau_grid:
        cells += [(int(m), t) for m in args.block_sizes.split(",") for t in parse_range(args.tau_grid)]
    if not cells:
        parser.error("give --settings or --block-sizes with --tau-grid")

    rows = [pooling_loss(pi, m, tau, args.n, args.bins, args.seed + i)
            for i, (pi, (m, tau)) in enumerate((pi, cell) for cell in cells for pi in pis)]
    print(f"{'m':>3} {'tau':>5} {'pi':>5} {'D':>8} {'1-R2 lin':>9} {'tau rule':>9}")
    for r in rows:
        print(f"{r['block_size']:>3} {r['tau']:>5.2f} {r['pi']:>5.2f} {r['D']:>8.4f} "
              f"{r['linear_R2_loss']:>9.4f} {r['tau_rule']:>9.3f}")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print("wrote", path)


if __name__ == "__main__":
    main()
