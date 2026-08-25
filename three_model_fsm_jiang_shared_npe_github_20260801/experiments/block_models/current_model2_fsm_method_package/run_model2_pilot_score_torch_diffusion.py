#!/usr/bin/env python3
"""Conditional diffusion posterior for deployable Model 2 pilot-score contexts.

The context is built exactly as in ``run_model2_pilot_score_sbi_npe.py``:

    pilot only: u_hat(Y)
    score arms: (u_hat(Y), S_frozen(Y, u_hat(Y)))

The simulation parameter is the diffusion target only. It is never passed to
the pilot or frozen-score context builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import model2_amortized_score_runtime as score_runtime
import run_blockwise_common_factor_two_stage_torch_diffusion_experiment as diffusion
import run_model2_pilot_score_sbi_npe as common


def diffusion_label(method: str) -> str:
    return common.METHOD_LABELS[method].replace("NPE", "diffusion")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standard_error(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0


def save_diffusion_checkpoint(
    path: Path,
    method: str,
    result: dict[str, object],
    args: argparse.Namespace,
) -> None:
    posterior = result["posterior"]
    torch.save(
        {
            "method": method,
            "label": diffusion_label(method),
            "model_state_dict": posterior.model.state_dict(),
            "x_mean": posterior.x_mean,
            "x_sd": posterior.x_sd,
            "z_mean": posterior.z_mean,
            "z_sd": posterior.z_sd,
            "betas": posterior.betas.detach().cpu(),
            "best_step": int(result["best_step"]),
            "best_val_loss": float(result["best_val_loss"]),
            "validation_noise": result["validation_noise"],
            "stage1_run_dir": str(args.stage1_run_dir),
            "training_context_cache": str(args.training_context_cache),
            "context": (
                "u_pilot(Y)"
                if method == "pilot"
                else common.CONTEXT_DEFINITION
            ),
            "context_dim": int(posterior.x_mean.shape[1]),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def load_training_contexts(
    args: argparse.Namespace,
    methods: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, object]]:
    cache_path = Path(args.training_context_cache)
    with np.load(cache_path) as cache:
        required = {"pi_train", "pilot_u", "pilot_status"}
        required.update(f"pilot_score_{method}" for method in methods)
        missing = sorted(required.difference(cache.files))
        if missing:
            raise ValueError(f"training context cache is missing {missing}: {cache_path}")

        n_available = int(np.asarray(cache["pi_train"]).shape[0])
        if int(args.n_diffusion_train) > n_available:
            raise ValueError(
                f"requested {args.n_diffusion_train} contexts but cache contains {n_available}"
            )
        subset_rng = np.random.default_rng(int(args.training_subset_seed))
        subset = subset_rng.permutation(n_available)[: int(args.n_diffusion_train)]
        pi_train = np.asarray(cache["pi_train"], dtype=np.float64)[subset]
        pilot_u = np.asarray(cache["pilot_u"], dtype=np.float64)[subset]
        pilot_status = np.asarray(cache["pilot_status"], dtype="U32")[subset]
        contexts = {
            method: np.asarray(cache[f"pilot_score_{method}"], dtype=np.float32)[subset]
            for method in methods
        }

    if not np.all(
        (pi_train >= float(args.pi_prior_min))
        & (pi_train <= float(args.pi_prior_max))
    ):
        raise ValueError("cached training parameters lie outside the requested prior")

    for method, context in contexts.items():
        expected_dim = 1 if method == "pilot" else 2
        if context.shape != (pi_train.size, expected_dim):
            raise ValueError(
                f"unexpected {method} context shape {context.shape}; "
                f"expected {(pi_train.size, expected_dim)}"
            )
        if not np.all(np.isfinite(context)):
            raise ValueError(f"{method} context contains non-finite values")
        if not np.allclose(context[:, 0], pilot_u, rtol=0.0, atol=2e-6):
            raise ValueError(f"{method} first context column is not the cached data-only pilot")

    manifest = {
        "path": str(cache_path.resolve()),
        "sha256": file_sha256(cache_path),
        "n_available": n_available,
        "n_used": int(pi_train.size),
        "subset_seed": int(args.training_subset_seed),
    }
    return pi_train, contexts, pilot_u, pilot_status, manifest


def summarize_pooled(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        method = str(row["method"])
        if method != "exact likelihood grid":
            grouped.setdefault(method, []).append(row)
    out = []
    for method, group in sorted(grouped.items()):
        mean_sq_exact = [float(row["mean_sq_err_to_exact"]) for row in group]
        w1 = [float(row["w1_to_exact"]) for row in group]
        out.append(
            {
                "method": method,
                "n": len(group),
                "pi_avg_mse": float(np.mean([float(row["pi_sq_err"]) for row in group])),
                "rmse_mean_to_exact": float(np.sqrt(np.mean(mean_sq_exact))),
                "mean_w1_to_exact": float(np.mean(w1)),
                "w1_se": standard_error(w1),
                "avg_post_sd": float(np.mean([float(row["pi_post_sd"]) for row in group])),
                "coverage90": float(np.mean([float(row["coverage90"]) for row in group])),
            }
        )
    return out


def paired_w1_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values: dict[tuple[float, int, str], float] = {}
    for row in rows:
        method = str(row["method"])
        if method != "exact likelihood grid":
            values[(float(row["pi_true"]), int(row["seed"]), method)] = float(row["w1_to_exact"])

    radial = diffusion_label("radial")
    out = []
    for competitor_key in ("linear", "pilot"):
        competitor = diffusion_label(competitor_key)
        paired = [
            (radial_value, values[(pi_true, seed, competitor)])
            for (pi_true, seed, method), radial_value in values.items()
            if method == radial and (pi_true, seed, competitor) in values
        ]
        if not paired:
            continue
        radial_values = np.asarray([pair[0] for pair in paired], dtype=np.float64)
        competitor_values = np.asarray([pair[1] for pair in paired], dtype=np.float64)
        diff = radial_values - competitor_values
        sd_diff = float(diff.std(ddof=1)) if diff.size > 1 else float("nan")
        t_stat = (
            float(diff.mean() / (sd_diff / math.sqrt(diff.size)))
            if diff.size > 1 and sd_diff > 0.0
            else float("nan")
        )
        p_value = float("nan")
        try:
            from scipy.stats import t as student_t

            p_value = float(2.0 * student_t.sf(abs(t_stat), df=diff.size - 1))
        except ImportError:
            pass
        out.append(
            {
                "reference": radial,
                "competitor": competitor,
                "n_pairs": int(diff.size),
                "reference_mean_w1": float(radial_values.mean()),
                "competitor_mean_w1": float(competitor_values.mean()),
                "reference_improvement_percent": float(
                    100.0 * (competitor_values.mean() - radial_values.mean())
                    / competitor_values.mean()
                ),
                "mean_paired_difference": float(diff.mean()),
                "paired_t": t_stat,
                "paired_p": p_value,
                "reference_wins": int(np.sum(diff < 0.0)),
            }
        )
    return out


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "posterior_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    methods = common.parse_list(args.methods, str)
    bad = sorted(set(methods) - set(common.METHOD_LABELS))
    if bad or not methods:
        raise ValueError(f"--methods must use {sorted(common.METHOD_LABELS)}; got {methods}")
    score_methods = [method for method in methods if method != "pilot"]
    if not score_methods:
        raise ValueError("At least one score method is required to load Stage-1 metadata")

    device = diffusion.resolve_device(str(args.device))
    runtimes = {
        method: score_runtime.AmortizedScoreRuntime(args.stage1_run_dir, method, device=device)
        for method in score_methods
    }
    metadata = common.validate_matching_runtimes(runtimes)
    args.tau = float(metadata["tau"])
    if not (
        float(metadata["pi_min"]) <= float(args.pi_prior_min)
        < float(args.pi_prior_max) <= float(metadata["pi_max"])
    ):
        raise ValueError(
            "The diffusion prior must lie inside the trained Stage-1 anchor support "
            f"[{metadata['pi_min']}, {metadata['pi_max']}]"
        )

    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "device:", device)
    print("Stage-1 run:", args.stage1_run_dir)
    print("Training cache:", args.training_context_cache)
    print("Context:", common.CONTEXT_DEFINITION)
    print("Pilot mode:", args.pilot_mode)
    print("Methods:", methods)
    print("True parameter role: diffusion target only; never a context input")

    sanity_rows = []
    for method, runtime in runtimes.items():
        checks = runtime.run_sanity_checks(seed=int(args.sanity_seed))
        sanity_rows.append(checks)
        if not bool(checks["passed"]):
            raise AssertionError(f"Frozen Stage-1 runtime sanity failed for {method}: {checks}")
    common.write_csv(out_dir / "runtime_sanity_checks.csv", sanity_rows)

    pilot_sanity_rows = common.run_pilot_sanity_checks(runtimes, metadata, args)
    common.write_csv(out_dir / "pilot_context_sanity_checks.csv", pilot_sanity_rows)

    pi_train, train_contexts, pilot_u_train, pilot_status_train, cache_manifest = (
        load_training_contexts(args, methods)
    )
    print(
        f"Loaded {pi_train.size} cached contexts from {args.training_context_cache} "
        f"(subset_seed={args.training_subset_seed})"
    )

    true_u_train = common.stage1.logit_np(pi_train)
    boundary_train = np.isin(pilot_status_train, ["left_boundary", "right_boundary"])
    context_rows = []
    for method in methods:
        context = train_contexts[method].astype(np.float64)
        score = context[:, 1] if context.shape[1] > 1 else None
        context_rows.append(
            {
                "method": diffusion_label(method),
                "n": int(context.shape[0]),
                "context_dim": int(context.shape[1]),
                "pilot_u_mean": float(pilot_u_train.mean()),
                "pilot_u_sd": float(pilot_u_train.std()),
                "corr_pilot_true_u": float(np.corrcoef(pilot_u_train, true_u_train)[0, 1]),
                "pilot_rmse_to_true_u": float(np.sqrt(np.mean((pilot_u_train - true_u_train) ** 2))),
                "pilot_boundary_rate": float(np.mean(boundary_train)),
                "score_mean": float(score.mean()) if score is not None else float("nan"),
                "score_sd": float(score.std()) if score is not None else float("nan"),
            }
        )
    common.write_csv(out_dir / "training_context_summary.csv", context_rows)

    results: dict[str, dict[str, object]] = {}
    trace_rows: list[dict[str, object]] = []
    for method in methods:
        label = diffusion_label(method)
        result = diffusion.train_diffusion_method(
            label,
            train_contexts[method],
            pi_train,
            int(args.diffusion_seed) + 173 * common.METHOD_SEED_INDEX[method],
            args,
            device,
        )
        results[method] = result
        trace_rows.extend(result["trace"])
        save_diffusion_checkpoint(out_dir / f"diffusion_{method}.pt", method, result, args)

    rows: list[dict[str, object]] = []
    for pi_true in common.parse_list(args.test_pi_values, float):
        if not (float(args.pi_prior_min) <= pi_true <= float(args.pi_prior_max)):
            raise ValueError(f"test pi {pi_true} lies outside the Stage-2 prior")
        for obs_seed in common.parse_list(args.test_seeds, int):
            obs_rng = np.random.default_rng(int(obs_seed))
            y_obs = common.simulate(obs_rng, np.asarray([pi_true]), metadata)
            obs_contexts, obs_pilot_u, obs_pilot_status = common.pilot_score_contexts(
                runtimes,
                y_obs,
                float(metadata["tau"]),
                1,
                int(args.pilot_grid_size),
                1,
                str(args.pilot_backend),
                int(args.pilot_grid_chunk_size),
                pilot_mode=str(args.pilot_mode),
            )
            if "pilot" in methods:
                obs_contexts["pilot"] = obs_pilot_u[:, None].astype(np.float32)
            exact_grid, _, exact_cdf, exact = common.exact_posterior(y_obs[0], args)
            rows.append(
                {
                    "pi_true": pi_true,
                    "seed": obs_seed,
                    "method": "exact likelihood grid",
                    "pilot_u_context": float(obs_pilot_u[0]),
                    "pilot_pi_context": float(common.stage1.sigmoid_np(obs_pilot_u[0])),
                    "pilot_status": str(obs_pilot_status[0]),
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
                samples = results[method]["posterior"].sample(
                    obs_contexts[method][0],
                    int(args.posterior_n),
                    int(args.posterior_seed)
                    + 100_003 * common.METHOD_SEED_INDEX[method]
                    + obs_seed,
                )
                np.save(samples_dir / f"{method}_pi{pi_true:g}_seed{obs_seed}.npy", samples)
                mean = float(np.mean(samples))
                sd = float(np.std(samples))
                q05, q50, q95 = [float(value) for value in np.quantile(samples, [0.05, 0.5, 0.95])]
                rows.append(
                    {
                        "pi_true": pi_true,
                        "seed": obs_seed,
                        "method": diffusion_label(method),
                        "pilot_u_context": float(obs_contexts[method][0, 0]),
                        "pilot_pi_context": float(common.stage1.sigmoid_np(obs_contexts[method][0, 0])),
                        "pilot_status": str(obs_pilot_status[0]),
                        "score_context": (
                            float(obs_contexts[method][0, 1])
                            if obs_contexts[method].shape[1] > 1
                            else float("nan")
                        ),
                        "pi_post_mean": mean,
                        "pi_post_sd": sd,
                        "q05": q05,
                        "q50": q50,
                        "q95": q95,
                        "pi_sq_err": float((mean - pi_true) ** 2),
                        "mean_sq_err_to_exact": float((mean - exact["mean"]) ** 2),
                        "w1_to_exact": diffusion.exact_sample_w1(samples, exact_grid, exact_cdf),
                        "coverage90": float(q05 <= pi_true <= q95),
                    }
                )

    summary = common.summarize(rows)
    pooled = summarize_pooled(rows)
    paired = paired_w1_comparisons(rows)
    common.write_csv(out_dir / "posterior_by_seed.csv", rows)
    common.write_csv(out_dir / "posterior_summary.csv", summary)
    common.write_csv(out_dir / "posterior_summary_pooled.csv", pooled)
    common.write_csv(out_dir / "paired_w1_comparisons.csv", paired)
    common.write_csv(out_dir / "diffusion_training_trace.csv", trace_rows)

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stage1_metadata": metadata,
        "training_cache_manifest": cache_manifest,
        "context_definition": {
            method: ("u_pilot(Y)" if method == "pilot" else common.CONTEXT_DEFINITION)
            for method in methods
        },
        "context_dim": {method: int(train_contexts[method].shape[1]) for method in methods},
        "true_parameter_role": "diffusion target only; never passed to context builder",
        "validation_noise": "fixed per validation observation",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n==== Pilot+score diffusion pooled posterior summary ====")
    print(f"{'method':<42}{'W1-exact':>12}{'RMSE-exact':>14}{'MSE':>12}{'post SD':>11}{'cov90':>9}")
    print("-" * 100)
    for row in pooled:
        print(
            f"{str(row['method']):<42}"
            f"{float(row['mean_w1_to_exact']):>12.4g}"
            f"{float(row['rmse_mean_to_exact']):>14.4g}"
            f"{float(row['pi_avg_mse']):>12.4g}"
            f"{float(row['avg_post_sd']):>11.4g}"
            f"{float(row['coverage90']):>9.3g}"
        )
    print("\nPaired W1 comparisons (negative t favors radial):")
    for row in paired:
        print(row)
    print("\nSaved to", out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-run-dir", type=Path, required=True)
    parser.add_argument("--training-context-cache", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="pilot,linear,radial")
    parser.add_argument("--pi-prior-min", type=float, default=0.05)
    parser.add_argument("--pi-prior-max", type=float, default=0.70)
    parser.add_argument("--n-diffusion-train", type=int, default=50_000)
    parser.add_argument("--training-subset-seed", type=int, default=20260711)
    parser.add_argument("--diffusion-seed", type=int, default=64_000)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--beta-min", type=float, default=1e-4)
    parser.add_argument("--beta-max", type=float, default=0.02)
    parser.add_argument("--train-steps", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--diffusion-hidden", type=int, default=128)
    parser.add_argument("--diffusion-depth", type=int, default=3)
    parser.add_argument("--diffusion-t-embed-dim", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--test-pi-values", type=str, default="0.07,0.10,0.30,0.50,0.65,0.68")
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--posterior-seed", type=int, default=97_000)
    parser.add_argument("--grid-size", type=int, default=5_000)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--pilot-grid-size", type=int, default=201)
    parser.add_argument("--pilot-batch-size", type=int, default=512)
    parser.add_argument(
        "--pilot-mode",
        choices=("moment", "marginal", "equal_channel"),
        default="marginal",
    )
    parser.add_argument("--pilot-backend", choices=("auto", "torch", "numpy"), default="auto")
    parser.add_argument("--pilot-grid-chunk-size", type=int, default=16)
    parser.add_argument("--pilot-progress-every", type=int, default=0)
    parser.add_argument("--pilot-sanity-n", type=int, default=64)
    parser.add_argument("--min-pilot-correlation", type=float, default=0.80)
    parser.add_argument("--min-mean-abs-score", type=float, default=0.05)
    parser.add_argument("--max-mean-abs-score", type=float, default=20.0)
    parser.add_argument("--sanity-seed", type=int, default=20260710)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/model2_pilot_score_torch_diffusion"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.pi_prior_min < args.pi_prior_max < 1.0):
        raise ValueError("Require 0 < pi_prior_min < pi_prior_max < 1")
    if args.n_diffusion_train < 100:
        raise ValueError("--n-diffusion-train must be at least 100")
    if args.pilot_grid_size < 21:
        raise ValueError("--pilot-grid-size must be at least 21")
    if args.pilot_sanity_n < 16:
        raise ValueError("--pilot-sanity-n must be at least 16")
    run(args)


if __name__ == "__main__":
    main()
