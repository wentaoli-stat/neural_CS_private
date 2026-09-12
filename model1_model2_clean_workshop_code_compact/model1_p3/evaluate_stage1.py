#!/usr/bin/env python3
"""Frozen exact-score evaluation for the p=3 Stage-1 checkpoints.

Separate from training by design: this module generates held-out data and the
exact score only after every checkpoint is frozen, so the exact score can never
influence optimization or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model1_p3 import scores as sc
from model1_p3 import stage1


def array_sha256(value: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    d = hashlib.sha256()
    d.update(a.dtype.str.encode()); d.update(str(a.shape).encode()); d.update(a.tobytes())
    return d.hexdigest()


def checkpoint_sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def parse_beta_grid(text: str) -> list[tuple[float, float, float]]:
    """Parse 'pi:tau:sigma,pi:tau:sigma,...' into natural-parameter triples."""
    points = []
    for chunk in str(text).split(","):
        if not chunk.strip():
            continue
        pi, tau, sigma = (float(v) for v in chunk.split(":"))
        points.append((pi, tau, sigma))
    return points


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    stats = {k: np.asarray(v) for k, v in np.load(run_dir / "feature_stats.npz").items()}
    sigma_q = np.asarray(stats["sigma_q"], dtype=np.float64)
    methods = [m for m in stage1.ORDERED_METHODS if (run_dir / f"model_{m}.pt").is_file()]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    models, labels, checkpoints = {}, {}, {}
    for method in methods:
        path = run_dir / f"model_{method}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = stage1.build_model(method, config, stats)
        model.load_state_dict(payload["state_dict"], strict=True)
        models[method] = model.to(device).eval()
        labels[method] = payload.get("label", method)
        checkpoints[method] = {"path": path.name, "sha256": checkpoint_sha256(path)}

    low, high = np.asarray(config["anchor_low"]), np.asarray(config["anchor_high"])
    rng = np.random.default_rng(args.eval_seed)
    rows, datasets = [], {}
    print("==== Frozen exact-score evaluation (p=3) ====", flush=True)
    for pi, tau, sigma in parse_beta_grid(args.beta_values):
        beta = np.asarray([sc.logit_np(pi), tau, np.log(sigma)], dtype=np.float64)
        if np.any(beta < low - 1e-12) or np.any(beta > high + 1e-12):
            raise ValueError(f"evaluation point (pi={pi}, tau={tau}, sigma={sigma}) is outside the anchor box")
        beta_rep = np.repeat(beta[None, :], args.n_test, axis=0)
        y = sc.simulate(rng, beta_rep, int(config["n_blocks"]), int(config["block_size"]))
        truth = sc.exact_score(y, beta_rep)
        key = f"pi{pi:g}_tau{tau:g}_sigma{sigma:g}"
        datasets[key] = {"y_sha256": array_sha256(y), "exact_sha256": array_sha256(truth)}
        # Features are evaluated at the same beta the data were generated at.
        need_raw_y = any(stage1.ARCHITECTURES[m].needs_raw_y for m in methods)
        data = stage1.featurize(y, beta_rep, stats, need_raw_y=need_raw_y)
        truth_var = truth.var(axis=0)
        for method in methods:
            pred = stage1.predict(models[method], data, method, device, args.batch_size) / sigma_q
            err = pred - truth
            mse = (err**2).mean(axis=0)
            row = {"pi": pi, "tau": tau, "sigma": sigma, "method": method, "label": labels[method],
                   "n": args.n_test}
            for i, name in enumerate(sc.PARAM_NAMES):
                row[f"mse_{name}"] = float(mse[i])
                row[f"std_mse_{name}"] = float(mse[i] / truth_var[i])
            row["std_mse_mean"] = float(np.mean(mse / truth_var))
            rows.append(row)
            print(f"  pi={pi:<5g} tau={tau:<4g} sigma={sigma:<4g} {method:<8s} "
                  + "  ".join(f"{n}={row['std_mse_'+n]:.5f}" for n in sc.PARAM_NAMES)
                  + f"   mean={row['std_mse_mean']:.5f}", flush=True)

    stage1.write_csv(run_dir / "score_summary_by_beta.csv", rows)
    (run_dir / "exact_evaluation.json").write_text(json.dumps(
        {"eval_seed": args.eval_seed, "n_test_per_point": args.n_test,
         "beta_values": args.beta_values, "batch_size": args.batch_size,
         "datasets": datasets, "checkpoints": checkpoints,
         "standardization": "per-coordinate MSE divided by the exact score variance at that beta"},
        indent=2), encoding="utf-8")
    print("\nSaved:", run_dir / "score_summary_by_beta.csv", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument(
        "--beta-values", type=str,
        default="0.10:1.0:1.0,0.30:1.0:1.0,0.50:1.0:1.0,0.65:1.0:1.0,"
                "0.30:0.6:1.0,0.30:1.6:1.0,0.30:1.0:0.8,0.30:1.0:1.3",
        help="Comma-separated pi:tau:sigma evaluation points.")
    parser.add_argument("--eval-seed", type=int, default=20260710)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
