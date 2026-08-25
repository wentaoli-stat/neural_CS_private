#!/usr/bin/env python3
"""Reconstruct Model-2 posteriors by integrating amortized FSM score fields.

All learned methods are likelihood-free at deployment. Exact likelihood and
exact/smoothed scores are used only as toy-model evaluation references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

import run_blockwise_common_factor_amortized_fsm_experiment as stage1
import run_blockwise_common_factor_information_precheck as precheck
from model2_amortized_score_runtime import AmortizedScoreRuntime, parse_method_list
from run_model2_mode_a_mle import (
    cumulative_trapezoid_rows,
    parse_float_list,
    resolve_run_dirs,
)


def cumulative_density_rows(density: np.ndarray, grid: np.ndarray) -> np.ndarray:
    density = np.asarray(density, dtype=np.float64)
    increments = 0.5 * (density[:, 1:] + density[:, :-1]) * np.diff(grid)[None, :]
    cdf = np.concatenate(
        [np.zeros((density.shape[0], 1), dtype=np.float64), np.cumsum(increments, axis=1)],
        axis=1,
    )
    cdf /= cdf[:, -1, None]
    cdf[:, -1] = 1.0
    return cdf


def normalize_log_density_u(
    log_density: np.ndarray,
    u_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    log_density = np.asarray(log_density, dtype=np.float64)
    shifted = log_density - np.max(log_density, axis=1, keepdims=True)
    density = np.exp(shifted)
    normalizer = np.trapezoid(density, u_grid, axis=1)
    if np.any(~np.isfinite(normalizer)) or np.any(normalizer <= 0.0):
        raise FloatingPointError("Invalid posterior normalizer")
    density /= normalizer[:, None]
    return density, cumulative_density_rows(density, u_grid)


def posterior_quantiles(cdf: np.ndarray, pi_grid: np.ndarray) -> np.ndarray:
    probs = np.asarray([0.05, 0.50, 0.95], dtype=np.float64)
    return np.asarray([np.interp(probs, row, pi_grid) for row in cdf], dtype=np.float64)


def posterior_moments(
    density_u: np.ndarray,
    u_grid: np.ndarray,
    pi_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.trapezoid(density_u * pi_grid[None, :], u_grid, axis=1)
    variance = np.trapezoid(
        density_u * (pi_grid[None, :] - mean[:, None]) ** 2,
        u_grid,
        axis=1,
    )
    return mean, np.sqrt(np.maximum(variance, 0.0))


def wasserstein_pi(cdf_a: np.ndarray, cdf_b: np.ndarray, pi_grid: np.ndarray) -> np.ndarray:
    return np.trapezoid(np.abs(cdf_a - cdf_b), pi_grid, axis=1)


def exact_loglik_grid(full_lr: np.ndarray, pi_grid: np.ndarray) -> np.ndarray:
    full_lr = np.asarray(full_lr, dtype=np.float64)
    log_pi = np.log(pi_grid)[None, None, :]
    log_one_minus_pi = np.log1p(-pi_grid)[None, None, :]
    return np.sum(
        np.logaddexp(log_one_minus_pi, log_pi + full_lr[:, :, None]),
        axis=1,
    )


def exact_score_grid(full_lr: np.ndarray, pi_grid: np.ndarray) -> np.ndarray:
    logits = stage1.logit_np(pi_grid)[None, None, :]
    active_prob = stage1.sigmoid_np(logits + full_lr[:, :, None])
    return np.sum(active_prob - pi_grid[None, None, :], axis=1)


def smoothed_score_grid(
    full_lr: np.ndarray,
    u_grid: np.ndarray,
    sigma_q: float,
    *,
    grid_size: int,
    grid_radius: float,
    chunk_size: int,
) -> np.ndarray:
    out = np.empty((full_lr.shape[0], len(u_grid)), dtype=np.float64)
    for j, u_value in enumerate(u_grid):
        out[:, j] = precheck.smoothed_fsm_score_from_full_lr(
            full_lr=full_lr,
            u0=float(u_value),
            sigma_q=float(sigma_q),
            grid_size=int(grid_size),
            grid_radius=float(grid_radius),
            chunk_size=int(chunk_size),
        )
    return out


def integrated_posterior(
    likelihood_score: np.ndarray,
    u_grid: np.ndarray,
    pi_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_likelihood = cumulative_trapezoid_rows(likelihood_score, u_grid)
    # The prior is uniform in pi. Its density in u includes d pi / d u.
    log_prior_u = np.log(pi_grid) + np.log1p(-pi_grid)
    density_u, cdf = normalize_log_density_u(log_likelihood + log_prior_u[None, :], u_grid)
    return log_likelihood, density_u, cdf


def metric_rows(
    *,
    pi_true: float,
    data_batch_seed: int,
    method: str,
    checkpoint: str,
    training_seed: int | None,
    density_u: np.ndarray,
    cdf: np.ndarray,
    exact_density_u: np.ndarray,
    exact_cdf: np.ndarray,
    smooth_cdf: np.ndarray,
    u_grid: np.ndarray,
    pi_grid: np.ndarray,
) -> list[dict[str, object]]:
    mean, sd = posterior_moments(density_u, u_grid, pi_grid)
    exact_mean, exact_sd = posterior_moments(exact_density_u, u_grid, pi_grid)
    quantiles = posterior_quantiles(cdf, pi_grid)
    exact_quantiles = posterior_quantiles(exact_cdf, pi_grid)
    w1_exact = wasserstein_pi(cdf, exact_cdf, pi_grid)
    w1_smooth = wasserstein_pi(cdf, smooth_cdf, pi_grid)
    rows: list[dict[str, object]] = []
    for i in range(len(mean)):
        rows.append(
            {
                "pi_true": float(pi_true),
                "data_index": int(i),
                "data_batch_seed": int(data_batch_seed),
                "method": method,
                "checkpoint": checkpoint,
                "training_seed": training_seed,
                "posterior_mean": float(mean[i]),
                "posterior_sd": float(sd[i]),
                "q05": float(quantiles[i, 0]),
                "q50": float(quantiles[i, 1]),
                "q95": float(quantiles[i, 2]),
                "coverage90": bool(quantiles[i, 0] <= pi_true <= quantiles[i, 2]),
                "mean_error_true": float(mean[i] - pi_true),
                "mean_sq_error_true": float((mean[i] - pi_true) ** 2),
                "w1_to_exact": float(w1_exact[i]),
                "w1_to_smoothed_oracle": float(w1_smooth[i]),
                "abs_mean_to_exact": float(abs(mean[i] - exact_mean[i])),
                "abs_sd_to_exact": float(abs(sd[i] - exact_sd[i])),
                "abs_q05_to_exact": float(abs(quantiles[i, 0] - exact_quantiles[i, 0])),
                "abs_q50_to_exact": float(abs(quantiles[i, 1] - exact_quantiles[i, 1])),
                "abs_q95_to_exact": float(abs(quantiles[i, 2] - exact_quantiles[i, 2])),
            }
        )
    return rows


def summarize_rows(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return frame.groupby(group_columns, dropna=False).agg(
        n=("posterior_mean", "size"),
        coverage90=("coverage90", "mean"),
        posterior_mean_rmse=(
            "mean_error_true",
            lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))),
        ),
        posterior_sd_mean=("posterior_sd", "mean"),
        w1_to_exact_mean=("w1_to_exact", "mean"),
        w1_to_smoothed_oracle_mean=("w1_to_smoothed_oracle", "mean"),
        abs_mean_to_exact_mean=("abs_mean_to_exact", "mean"),
        abs_sd_to_exact_mean=("abs_sd_to_exact", "mean"),
        abs_q05_to_exact_mean=("abs_q05_to_exact", "mean"),
        abs_q50_to_exact_mean=("abs_q50_to_exact", "mean"),
        abs_q95_to_exact_mean=("abs_q95_to_exact", "mean"),
    ).reset_index()


def architecture_paired_summary(by_checkpoint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        "posterior_mean_rmse",
        "w1_to_exact_mean",
        "w1_to_smoothed_oracle_mean",
        "abs_mean_to_exact_mean",
        "abs_sd_to_exact_mean",
    )
    subset = by_checkpoint[by_checkpoint["method"].isin(["linear", "radial"])]
    for metric in metrics:
        pivot = subset.pivot(index="training_seed", columns="method", values=metric).dropna()
        linear = pivot["linear"].to_numpy(dtype=np.float64)
        radial = pivot["radial"].to_numpy(dtype=np.float64)
        test = ttest_rel(linear, radial) if len(linear) >= 2 else None
        linear_mean = float(np.mean(linear))
        radial_mean = float(np.mean(radial))
        rows.append(
            {
                "metric": metric,
                "n_training_seeds": int(len(linear)),
                "linear_mean": linear_mean,
                "radial_mean": radial_mean,
                "radial_reduction_pct": 100.0 * (linear_mean - radial_mean) / linear_mean,
                "radial_wins": int(np.sum(radial < linear)),
                "paired_t": float(test.statistic) if test is not None else np.nan,
                "paired_p": float(test.pvalue) if test is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    config: dict[str, object],
    overall: pd.DataFrame,
    by_pi: pd.DataFrame,
    paired: pd.DataFrame,
    sanity: dict[str, float],
) -> None:
    columns = [
        "method",
        "posterior_mean_rmse",
        "posterior_sd_mean",
        "coverage90",
        "w1_to_exact_mean",
        "w1_to_smoothed_oracle_mean",
        "abs_mean_to_exact_mean",
    ]
    pi_columns = [
        "pi_true",
        "method",
        "posterior_mean_rmse",
        "coverage90",
        "w1_to_exact_mean",
    ]
    lines = [
        "# Model 2 Mode A Posterior Integration Report",
        "",
        "## Setting",
        "",
        f"- Stage-1 checkpoints: `{len(config['run_dirs_resolved'])}`",
        f"- True pi values: `{config['pi_true_values']}`",
        f"- Datasets per pi: `{config['n_data']}`",
        f"- Posterior grid size: `{config['posterior_grid_size']}`",
        f"- FSM sigma_q: `{config['sigma_q']}`",
        "- Uniform prior in pi; the u-space Jacobian is included explicitly.",
        "- Exact references are evaluation-only.",
        "",
        "## Integration Sanity",
        "",
        f"- Maximum exact-score-integrated W1 to exact posterior: `{sanity['max_w1']:.3e}`",
        f"- Maximum centered log-likelihood integration error: `{sanity['max_loglik_error']:.3e}`",
        "",
        "## Overall Results",
        "",
        overall[columns].to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Results by True pi",
        "",
        by_pi[pi_columns].to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Paired Radial vs Linear",
        "",
        paired.to_markdown(index=False, floatfmt=".5g"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    package_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = resolve_run_dirs(args, package_dir)
    methods = parse_method_list(args.methods)
    runtimes = [
        AmortizedScoreRuntime(run_dir, method, device=args.device)
        for run_dir in run_dirs
        for method in methods
    ]
    reference = runtimes[0]
    for runtime in runtimes:
        compatible = (
            runtime.n_blocks == reference.n_blocks
            and runtime.block_size == reference.block_size
            and np.isclose(runtime.tau, reference.tau)
            and np.isclose(runtime.sigma_q, reference.sigma_q)
            and np.isclose(runtime.u_min, reference.u_min)
            and np.isclose(runtime.u_max, reference.u_max)
        )
        if not compatible:
            raise ValueError(f"Incompatible checkpoint: {runtime.run_dir.name}/{runtime.method}")
        checks = runtime.run_sanity_checks(seed=int(args.seed) + int(runtime.config["seed"]) % 1000)
        if not checks["passed"]:
            raise RuntimeError(f"Runtime sanity failed: {runtime.run_dir.name}/{runtime.method}")
        print(f"[OK runtime] {runtime.run_dir.name}/{runtime.method}")

    u_grid = np.linspace(
        reference.u_min,
        reference.u_max,
        int(args.posterior_grid_size),
        dtype=np.float64,
    )
    pi_grid = stage1.sigmoid_np(u_grid)
    log_prior_u = np.log(pi_grid) + np.log1p(-pi_grid)
    pi_true_values = parse_float_list(args.pi_true_values)
    config = {
        **vars(args),
        "run_dirs_resolved": [str(path) for path in run_dirs],
        "device_resolved": reference.device,
        "sigma_q": float(reference.sigma_q),
        "tau": float(reference.tau),
        "n_blocks": int(reference.n_blocks),
        "block_size": int(reference.block_size),
        "pi_support": [float(pi_grid[0]), float(pi_grid[-1])],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows: list[dict[str, object]] = []
    observed_batches: list[np.ndarray] = []
    exact_density_batches: list[np.ndarray] = []
    exact_cdf_batches: list[np.ndarray] = []
    smooth_density_batches: list[np.ndarray] = []
    smooth_cdf_batches: list[np.ndarray] = []
    learned_density = np.empty(
        (
            len(runtimes),
            len(pi_true_values),
            int(args.n_data),
            len(u_grid),
        ),
        dtype=np.float32,
    )
    learned_cdf = np.empty_like(learned_density)
    learned_likelihood_score = np.empty_like(learned_density)
    max_sanity_w1 = 0.0
    max_loglik_error = 0.0

    for pi_index, pi_true in enumerate(pi_true_values):
        data_batch_seed = int(args.seed) + 100_000 * pi_index
        rng = np.random.default_rng(data_batch_seed)
        y = stage1.simulate_common_factor(
            rng,
            np.full(int(args.n_data), stage1.logit_np(pi_true), dtype=np.float64),
            n_blocks=reference.n_blocks,
            block_size=reference.block_size,
            tau=reference.tau,
        )
        observed_batches.append(y.astype(np.float32))
        full_lr = stage1.full_block_log_ratio(y, reference.tau)
        exact_loglik = exact_loglik_grid(full_lr, pi_grid)
        exact_density, exact_cdf = normalize_log_density_u(
            exact_loglik + log_prior_u[None, :], u_grid
        )
        exact_density_batches.append(exact_density.astype(np.float32))
        exact_cdf_batches.append(exact_cdf.astype(np.float32))

        exact_score = exact_score_grid(full_lr, pi_grid)
        integrated_exact_loglik, integrated_exact_density, integrated_exact_cdf = (
            integrated_posterior(exact_score, u_grid, pi_grid)
        )
        sanity_w1 = wasserstein_pi(integrated_exact_cdf, exact_cdf, pi_grid)
        max_sanity_w1 = max(max_sanity_w1, float(np.max(sanity_w1)))
        centered_exact = exact_loglik - exact_loglik[:, :1]
        max_loglik_error = max(
            max_loglik_error,
            float(np.max(np.abs(integrated_exact_loglik - centered_exact))),
        )

        smooth_score = smoothed_score_grid(
            full_lr,
            u_grid,
            reference.sigma_q,
            grid_size=int(args.smooth_grid_size),
            grid_radius=float(args.smooth_grid_radius),
            chunk_size=int(args.smooth_chunk_size),
        )
        _, smooth_density, smooth_cdf = integrated_posterior(smooth_score, u_grid, pi_grid)
        smooth_density_batches.append(smooth_density.astype(np.float32))
        smooth_cdf_batches.append(smooth_cdf.astype(np.float32))

        all_rows.extend(
            metric_rows(
                pi_true=pi_true,
                data_batch_seed=data_batch_seed,
                method="exact likelihood",
                checkpoint="reference",
                training_seed=None,
                density_u=exact_density,
                cdf=exact_cdf,
                exact_density_u=exact_density,
                exact_cdf=exact_cdf,
                smooth_cdf=smooth_cdf,
                u_grid=u_grid,
                pi_grid=pi_grid,
            )
        )
        all_rows.extend(
            metric_rows(
                pi_true=pi_true,
                data_batch_seed=data_batch_seed,
                method="exact score integrated",
                checkpoint="reference",
                training_seed=None,
                density_u=integrated_exact_density,
                cdf=integrated_exact_cdf,
                exact_density_u=exact_density,
                exact_cdf=exact_cdf,
                smooth_cdf=smooth_cdf,
                u_grid=u_grid,
                pi_grid=pi_grid,
            )
        )
        all_rows.extend(
            metric_rows(
                pi_true=pi_true,
                data_batch_seed=data_batch_seed,
                method="smoothed oracle",
                checkpoint="reference",
                training_seed=None,
                density_u=smooth_density,
                cdf=smooth_cdf,
                exact_density_u=exact_density,
                exact_cdf=exact_cdf,
                smooth_cdf=smooth_cdf,
                u_grid=u_grid,
                pi_grid=pi_grid,
            )
        )

        for runtime_index, runtime in enumerate(runtimes):
            score = runtime.score_grid(y, u_grid, batch_size=int(args.batch_size))
            _, density, cdf = integrated_posterior(score, u_grid, pi_grid)
            learned_likelihood_score[runtime_index, pi_index] = score.astype(np.float32)
            learned_density[runtime_index, pi_index] = density.astype(np.float32)
            learned_cdf[runtime_index, pi_index] = cdf.astype(np.float32)
            all_rows.extend(
                metric_rows(
                    pi_true=pi_true,
                    data_batch_seed=data_batch_seed,
                    method=runtime.method,
                    checkpoint=runtime.run_dir.name,
                    training_seed=int(runtime.config["seed"]),
                    density_u=density,
                    cdf=cdf,
                    exact_density_u=exact_density,
                    exact_cdf=exact_cdf,
                    smooth_cdf=smooth_cdf,
                    u_grid=u_grid,
                    pi_grid=pi_grid,
                )
            )
        print(f"completed pi_true={pi_true:.3g}")

    sanity = {"max_w1": max_sanity_w1, "max_loglik_error": max_loglik_error}
    (output_dir / "integration_sanity.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )
    if max_sanity_w1 > float(args.integration_sanity_w1_tol):
        raise AssertionError(
            f"Exact-score integration sanity failed: W1={max_sanity_w1:.3e}"
        )

    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_dir / "posterior_by_dataset.csv", index=False)
    by_checkpoint = summarize_rows(
        frame,
        ["method", "checkpoint", "training_seed"],
    )
    by_checkpoint.to_csv(output_dir / "posterior_summary_by_checkpoint.csv", index=False)
    overall = summarize_rows(frame, ["method"])
    overall.to_csv(output_dir / "posterior_summary.csv", index=False)
    by_pi = summarize_rows(frame, ["pi_true", "method"])
    by_pi.to_csv(output_dir / "posterior_summary_by_pi.csv", index=False)
    paired = architecture_paired_summary(by_checkpoint)
    paired.to_csv(output_dir / "radial_vs_linear_paired.csv", index=False)

    np.savez_compressed(
        output_dir / "posterior_grids.npz",
        u_grid=u_grid,
        pi_grid=pi_grid,
        pi_true_values=np.asarray(pi_true_values, dtype=np.float64),
        observed_y=np.stack(observed_batches),
        exact_density_u=np.stack(exact_density_batches),
        exact_cdf=np.stack(exact_cdf_batches),
        smoothed_density_u=np.stack(smooth_density_batches),
        smoothed_cdf=np.stack(smooth_cdf_batches),
        learned_density_u=learned_density,
        learned_cdf=learned_cdf,
        learned_likelihood_score=learned_likelihood_score,
        runtime_names=np.asarray(
            [f"{runtime.run_dir.name}/{runtime.method}" for runtime in runtimes]
        ),
    )
    write_report(
        output_dir / "MODE_A_POSTERIOR_REPORT.md",
        config,
        overall,
        by_pi,
        paired,
        sanity,
    )
    print("\n==== Posterior integration summary ====")
    print(
        overall[
            [
                "method",
                "posterior_mean_rmse",
                "posterior_sd_mean",
                "coverage90",
                "w1_to_exact_mean",
                "w1_to_smoothed_oracle_mean",
            ]
        ].to_string(index=False)
    )
    print("\n==== Paired radial vs linear ====")
    print(paired.to_string(index=False))
    print("\nSaved to", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", default="")
    parser.add_argument("--run-pattern", default="formal40k_fixed_sigma_seed202607*")
    parser.add_argument("--methods", default="linear,radial")
    parser.add_argument("--pi-true-values", default="0.10,0.30,0.50,0.65")
    parser.add_argument("--n-data", type=int, default=20)
    parser.add_argument("--posterior-grid-size", type=int, default=501)
    parser.add_argument("--smooth-grid-size", type=int, default=121)
    parser.add_argument("--smooth-grid-radius", type=float, default=5.0)
    parser.add_argument("--smooth-chunk-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--integration-sanity-w1-tol", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", default="runs/model2_mode_a_posterior_grid_smoke")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
