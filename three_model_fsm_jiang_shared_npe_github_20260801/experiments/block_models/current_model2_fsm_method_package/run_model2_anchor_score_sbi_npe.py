#!/usr/bin/env python3
"""Anchor+score NPE using a frozen amortized Model 2 FSM score field.

For every Stage-2 training simulation,

    pi_b ~ prior,
    Y_b ~ p(. | pi_b),
    x_b = (logit(pi_b), S_frozen(Y_b, logit(pi_b))).

The NPE context explicitly includes the anchor at which the frozen score is
evaluated.  In this diagnostic the anchor is the true parameter for simulations;
deployment can replace it with a cheap pilot estimate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import model2_amortized_score_runtime as score_runtime
import run_blockwise_common_factor_amortized_fsm_experiment as stage1
import run_blockwise_common_factor_two_stage_sbi_npe_experiment as npe_utils


METHOD_LABELS = {
    "linear": "linear FSM anchor+score NPE",
    "radial": "radial FSM anchor+score NPE",
}


def parse_list(text: str, cast):
    return [cast(part.strip()) for part in str(text).split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def standard_error(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0


def validate_matching_runtimes(
    runtimes: dict[str, score_runtime.AmortizedScoreRuntime],
) -> dict[str, object]:
    first = next(iter(runtimes.values()))
    reference = {
        "n_blocks": first.n_blocks,
        "block_size": first.block_size,
        "tau": first.tau,
        "pi_min": first.pi_min,
        "pi_max": first.pi_max,
    }
    for name, runtime in runtimes.items():
        current = {
            "n_blocks": runtime.n_blocks,
            "block_size": runtime.block_size,
            "tau": runtime.tau,
            "pi_min": runtime.pi_min,
            "pi_max": runtime.pi_max,
        }
        if current != reference:
            raise ValueError(f"Stage-1 runtime mismatch for {name}: {current} != {reference}")
    return reference


def simulate(
    rng: np.random.Generator,
    pi: np.ndarray,
    metadata: dict[str, object],
) -> np.ndarray:
    return stage1.simulate_common_factor(
        rng,
        stage1.logit_np(np.asarray(pi, dtype=np.float64)),
        int(metadata["n_blocks"]),
        int(metadata["block_size"]),
        float(metadata["tau"]),
    )


def anchor_score_contexts(
    runtimes: dict[str, score_runtime.AmortizedScoreRuntime],
    y: np.ndarray,
    pi_eval: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    u_eval = stage1.logit_np(np.asarray(pi_eval, dtype=np.float64).reshape(-1))
    if u_eval.shape != (y.shape[0],):
        raise ValueError(f"pi_eval must have one entry per dataset; got {u_eval.shape}")
    contexts: dict[str, np.ndarray] = {}
    for method, runtime in runtimes.items():
        score_values = runtime.score(y, u_eval, batch_size=int(batch_size)).reshape(-1)
        values = np.column_stack([u_eval, score_values])
        if values.shape != (y.shape[0], 2):
            raise AssertionError(f"{method} context has unexpected shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{method} score context contains non-finite values")
        contexts[method] = values.astype(np.float32)
    return contexts


def exact_posterior(y: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    proxy = argparse.Namespace(
        pi_prior_min=float(args.pi_prior_min),
        pi_prior_max=float(args.pi_prior_max),
        grid_size=int(args.grid_size),
        tau=float(args.tau),
    )
    grid, weights, cdf = npe_utils.exact_posterior_grid_arrays(y, proxy)
    mean = float(np.sum(weights * grid))
    sd = float(np.sqrt(np.sum(weights * (grid - mean) ** 2)))
    stats = {
        "mean": mean,
        "sd": sd,
        "q05": float(np.interp(0.05, cdf, grid)),
        "q50": float(np.interp(0.50, cdf, grid)),
        "q95": float(np.interp(0.95, cdf, grid)),
    }
    return grid, weights, cdf, stats


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((float(row["pi_true"]), str(row["method"])), []).append(row)
    out: list[dict[str, object]] = []
    for (pi_true, method), group in sorted(grouped.items()):
        sq_err = [float(row["pi_sq_err"]) for row in group]
        mean_sq_exact = [float(row["mean_sq_err_to_exact"]) for row in group]
        out.append(
            {
                "pi_true": pi_true,
                "method": method,
                "n_seeds": len(group),
                "pi_avg_mse": float(np.mean(sq_err)),
                "pi_mse_se": standard_error(sq_err),
                "rmse_mean_to_exact": float(np.sqrt(np.mean(mean_sq_exact))),
                "w1_to_exact": float(np.mean([float(row["w1_to_exact"]) for row in group])),
                "avg_post_sd": float(np.mean([float(row["pi_post_sd"]) for row in group])),
                "coverage90": float(np.mean([float(row["coverage90"]) for row in group])),
            }
        )
    return out


def build_npe_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        pi_prior_min=float(args.pi_prior_min),
        pi_prior_max=float(args.pi_prior_max),
        sbi_model=str(args.sbi_model),
        sbi_hidden_features=int(args.sbi_hidden_features),
        sbi_num_components=int(args.sbi_num_components),
        sbi_num_transforms=int(args.sbi_num_transforms),
        sbi_num_bins=int(args.sbi_num_bins),
        sbi_batch_size=int(args.sbi_batch_size),
        sbi_lr=float(args.sbi_lr),
        max_epochs=int(args.max_epochs),
        validation_fraction=float(args.validation_fraction),
        stop_after_epochs=int(args.stop_after_epochs),
    )


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_list(args.methods, str)
    bad = sorted(set(methods) - set(METHOD_LABELS))
    if bad or not methods:
        raise ValueError(f"--methods must use {sorted(METHOD_LABELS)}; got {methods}")

    device = npe_utils.resolve_device(str(args.device))
    data_device = npe_utils.resolve_device(str(args.data_device))
    runtimes = {
        method: score_runtime.AmortizedScoreRuntime(args.stage1_run_dir, method, device=device)
        for method in methods
    }
    metadata = validate_matching_runtimes(runtimes)
    args.tau = float(metadata["tau"])
    if not (
        float(metadata["pi_min"]) <= float(args.pi_prior_min)
        < float(args.pi_prior_max) <= float(metadata["pi_max"])
    ):
        raise ValueError(
            "The Stage-2 prior must lie inside the trained Stage-1 anchor support "
            f"[{metadata['pi_min']}, {metadata['pi_max']}]"
        )

    print("torch:", torch.__version__)
    print("NPE device:", device, "data device:", data_device)
    print("Stage-1 run:", args.stage1_run_dir)
    print("Context: (logit(pi_true), S(Y, logit(pi_true)))")
    print("Methods:", methods)
    print("Model metadata:", metadata)

    sanity_rows: list[dict[str, object]] = []
    for method, runtime in runtimes.items():
        checks = runtime.run_sanity_checks(seed=int(args.sanity_seed))
        sanity_rows.append(checks)
        if not bool(checks["passed"]):
            raise AssertionError(f"Frozen Stage-1 runtime sanity failed for {method}: {checks}")
    write_csv(out_dir / "runtime_sanity_checks.csv", sanity_rows)

    rng = np.random.default_rng(int(args.sbi_train_seed))
    pi_train = rng.uniform(
        float(args.pi_prior_min),
        float(args.pi_prior_max),
        size=int(args.n_sbi_train),
    )
    y_train = simulate(rng, pi_train, metadata)
    train_contexts = anchor_score_contexts(
        runtimes,
        y_train,
        pi_train,
        batch_size=int(args.score_batch_size),
    )
    del y_train

    context_rows: list[dict[str, object]] = []
    for method in methods:
        x = train_contexts[method].astype(np.float64)
        u = x[:, 0]
        z = x[:, 1]
        context_rows.append(
            {
                "method": METHOD_LABELS[method],
                "n": x.shape[0],
                "context_dim": x.shape[1],
                "anchor_u_mean": float(u.mean()),
                "anchor_u_sd": float(u.std()),
                "anchor_u_min": float(u.min()),
                "anchor_u_max": float(u.max()),
                "corr_anchor_pi": float(np.corrcoef(u, pi_train)[0, 1]),
                "score_mean": float(z.mean()),
                "score_sd": float(z.std()),
                "score_min": float(z.min()),
                "score_max": float(z.max()),
                "corr_score_pi": float(np.corrcoef(z, pi_train)[0, 1]),
            }
        )
    write_csv(out_dir / "training_context_summary.csv", context_rows)
    if bool(args.save_training_context):
        np.savez_compressed(
            out_dir / "training_contexts.npz",
            pi_train=pi_train.astype(np.float32),
            **{f"anchor_score_{method}": train_contexts[method] for method in methods},
        )

    NPE, posterior_nn, BoxUniform = npe_utils.import_sbi()
    helper_args = build_npe_args(args)
    results: dict[str, dict[str, object]] = {}
    for idx, method in enumerate(methods):
        label = METHOD_LABELS[method]
        result = npe_utils.train_sbi_npe_method(
            NPE,
            posterior_nn,
            BoxUniform,
            label,
            train_contexts[method],
            pi_train,
            int(args.sbi_seed) + 173 * idx,
            helper_args,
            device,
            data_device,
        )
        results[method] = result
        torch.save(
            {
                "method": method,
                "label": label,
                "state_dict": result["posterior"].posterior_estimator.state_dict(),
                "stage1_run_dir": str(args.stage1_run_dir),
                "context": "(logit(pi_true), S(Y, logit(pi_true)))",
                "config": vars(helper_args),
            },
            out_dir / f"npe_{method}_state.pt",
        )

    rows: list[dict[str, object]] = []
    test_pi_values = parse_list(args.test_pi_values, float)
    test_seeds = parse_list(args.test_seeds, int)
    for pi_true in test_pi_values:
        if not (float(args.pi_prior_min) <= pi_true <= float(args.pi_prior_max)):
            raise ValueError(f"test pi {pi_true} lies outside the Stage-2 prior")
        for obs_seed in test_seeds:
            obs_rng = np.random.default_rng(int(obs_seed))
            y_obs = simulate(obs_rng, np.asarray([pi_true]), metadata)
            obs_contexts = anchor_score_contexts(
                runtimes,
                y_obs,
                np.asarray([pi_true]),
                batch_size=1,
            )
            exact_grid, _, exact_cdf, exact = exact_posterior(y_obs[0], args)

            rows.append(
                {
                    "pi_true": pi_true,
                    "seed": obs_seed,
                    "method": "exact likelihood grid",
                    "anchor_u_context": float("nan"),
                    "score_context": float("nan"),
                    "pi_post_mean": exact["mean"],
                    "pi_post_sd": exact["sd"],
                    "q05": exact["q05"],
                    "q50": exact["q50"],
                    "q95": exact["q95"],
                    "pi_sq_err": float((exact["mean"] - pi_true) ** 2),
                    "mean_sq_err_to_exact": 0.0,
                    "w1_to_exact": 0.0,
                    "coverage90": float(exact["q05"] <= pi_true <= exact["q95"]),
                }
            )

            for method in methods:
                label = METHOD_LABELS[method]
                samples = npe_utils.sample_sbi_posterior(
                    results[method],
                    obs_contexts[method][0],
                    int(args.posterior_n),
                    int(args.posterior_seed) + 100_003 * methods.index(method) + obs_seed,
                    helper_args,
                )
                mean = float(np.mean(samples))
                sd = float(np.std(samples))
                q05, q50, q95 = [float(x) for x in np.quantile(samples, [0.05, 0.50, 0.95])]
                rows.append(
                    {
                        "pi_true": pi_true,
                        "seed": obs_seed,
                        "method": label,
                        "anchor_u_context": float(obs_contexts[method][0, 0]),
                        "score_context": float(obs_contexts[method][0, 1]),
                        "pi_post_mean": mean,
                        "pi_post_sd": sd,
                        "q05": q05,
                        "q50": q50,
                        "q95": q95,
                        "pi_sq_err": float((mean - pi_true) ** 2),
                        "mean_sq_err_to_exact": float((mean - exact["mean"]) ** 2),
                        "w1_to_exact": npe_utils.exact_sample_w1(samples, exact_grid, exact_cdf),
                        "coverage90": float(q05 <= pi_true <= q95),
                    }
                )

    summary = summarize(rows)
    write_csv(out_dir / "posterior_by_seed.csv", rows)
    write_csv(out_dir / "posterior_summary.csv", summary)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stage1_metadata": metadata,
        "context_definition": "(logit(pi_true), S(Y, logit(pi_true)))",
        "context_dim": 2,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n==== Anchor+score NPE posterior summary ====")
    print(f"{'pi':>6}  {'method':<34}{'MSE':>12}{'RMSE-exact':>14}{'W1-exact':>12}{'post SD':>11}{'cov90':>9}")
    print("-" * 100)
    for row in summary:
        print(
            f"{float(row['pi_true']):>6.2f}  {str(row['method']):<34}"
            f"{float(row['pi_avg_mse']):>12.4g}"
            f"{float(row['rmse_mean_to_exact']):>14.4g}"
            f"{float(row['w1_to_exact']):>12.4g}"
            f"{float(row['avg_post_sd']):>11.4g}"
            f"{float(row['coverage90']):>9.3g}"
        )
    print("\nSaved:")
    for name in (
        "config.json",
        "runtime_sanity_checks.csv",
        "training_context_summary.csv",
        "posterior_by_seed.csv",
        "posterior_summary.csv",
    ):
        print(out_dir / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-run-dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="linear,radial")
    parser.add_argument("--pi-prior-min", type=float, default=0.05)
    parser.add_argument("--pi-prior-max", type=float, default=0.70)
    parser.add_argument("--n-sbi-train", type=int, default=20_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260710)
    parser.add_argument("--sbi-model", choices=("mdn", "maf", "nsf", "made"), default="mdn")
    parser.add_argument("--sbi-hidden-features", type=int, default=64)
    parser.add_argument("--sbi-num-components", type=int, default=8)
    parser.add_argument("--sbi-num-transforms", type=int, default=5)
    parser.add_argument("--sbi-num-bins", type=int, default=8)
    parser.add_argument("--sbi-batch-size", type=int, default=256)
    parser.add_argument("--sbi-lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--sbi-seed", type=int, default=54000)
    parser.add_argument("--test-pi-values", type=str, default="0.10,0.30,0.50,0.65")
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--posterior-seed", type=int, default=87000)
    parser.add_argument("--grid-size", type=int, default=5_000)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--sanity-seed", type=int, default=20260710)
    parser.add_argument("--save-training-context", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/model2_anchor_score_sbi_npe"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.pi_prior_min < args.pi_prior_max < 1.0):
        raise ValueError("Require 0 < pi_prior_min < pi_prior_max < 1")
    if args.n_sbi_train < 100:
        raise ValueError("--n-sbi-train must be at least 100")
    run(args)


if __name__ == "__main__":
    main()
