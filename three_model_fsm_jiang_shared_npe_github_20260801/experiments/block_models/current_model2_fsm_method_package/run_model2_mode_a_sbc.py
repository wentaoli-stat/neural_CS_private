#!/usr/bin/env python3
"""Prior-predictive SBC for Model-2 amortized FSM posterior fields.

For each replicate, pi is drawn from the same uniform prior used at deployment,
then a fresh dataset is simulated. Exact likelihood and learned-score posterior
grids are evaluation-only outputs; no true score enters Stage-1 training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kstest, ttest_rel

import run_blockwise_common_factor_amortized_fsm_experiment as stage1
from model2_amortized_score_runtime import AmortizedScoreRuntime, parse_method_list
from run_model2_mode_a_mle import parse_path_list, resolve_run_dirs
from run_model2_mode_a_posterior_grid import (
    exact_loglik_grid,
    integrated_posterior,
    normalize_log_density_u,
    posterior_moments,
    posterior_quantiles,
    wasserstein_pi,
)


def cdf_at_truth(cdf: np.ndarray, pi_grid: np.ndarray, pi_true: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.interp(value, pi_grid, row) for value, row in zip(pi_true, cdf, strict=True)],
        dtype=np.float64,
    )


def rows_from_posterior(
    *,
    method: str,
    checkpoint: str,
    training_seed: int | None,
    pi_true: np.ndarray,
    density_u: np.ndarray,
    cdf: np.ndarray,
    exact_cdf: np.ndarray,
    exact_density_u: np.ndarray,
    u_grid: np.ndarray,
    pi_grid: np.ndarray,
    rank_draws: int,
    rank_rng: np.random.Generator,
) -> list[dict[str, object]]:
    mean, sd = posterior_moments(density_u, u_grid, pi_grid)
    quantiles = posterior_quantiles(cdf, pi_grid)
    pit = cdf_at_truth(cdf, pi_grid, pi_true)
    rank = rank_rng.binomial(int(rank_draws), pit)
    exact_mean, _ = posterior_moments(exact_density_u, u_grid, pi_grid)
    w1_exact = wasserstein_pi(cdf, exact_cdf, pi_grid)
    rows: list[dict[str, object]] = []
    for i in range(len(pi_true)):
        rows.append(
            {
                "data_index": int(i),
                "pi_true": float(pi_true[i]),
                "method": method,
                "checkpoint": checkpoint,
                "training_seed": training_seed,
                "pit": float(pit[i]),
                "rank": int(rank[i]),
                "rank_draws": int(rank_draws),
                "coverage90": bool(quantiles[i, 0] <= pi_true[i] <= quantiles[i, 2]),
                "posterior_mean": float(mean[i]),
                "posterior_sd": float(sd[i]),
                "mean_error_true": float(mean[i] - pi_true[i]),
                "abs_mean_to_exact": float(abs(mean[i] - exact_mean[i])),
                "w1_to_exact": float(w1_exact[i]),
            }
        )
    return rows


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, subset in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys, strict=True))
        pit = subset["pit"].to_numpy(dtype=np.float64)
        test = kstest(pit, "uniform")
        row.update(
            {
                "n": int(len(subset)),
                "pit_mean": float(np.mean(pit)),
                "pit_sd": float(np.std(pit, ddof=1)),
                "pit_ks_stat": float(test.statistic),
                "pit_ks_p": float(test.pvalue),
                "coverage90": float(subset["coverage90"].mean()),
                "posterior_mean_rmse": float(
                    np.sqrt(np.mean(subset["mean_error_true"].to_numpy(dtype=np.float64) ** 2))
                ),
                "w1_to_exact_mean": float(subset["w1_to_exact"].mean()),
                "abs_mean_to_exact_mean": float(subset["abs_mean_to_exact"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_across_checkpoints(
    by_checkpoint: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Average learned-SBC metrics across independent Stage-1 training seeds.

    The same prior-predictive datasets are evaluated by every checkpoint.  Thus,
    pooling their 500-row SBC samples into a fictitious n=2500 KS test would
    incorrectly treat repeated datasets as independent.  This table keeps the
    valid per-checkpoint SBC calculation intact and summarizes its variation
    across Stage-1 seeds instead.
    """
    metric_columns = [
        "pit_mean",
        "pit_sd",
        "pit_ks_stat",
        "pit_ks_p",
        "coverage90",
        "posterior_mean_rmse",
        "w1_to_exact_mean",
        "abs_mean_to_exact_mean",
    ]
    rows: list[dict[str, object]] = []
    for keys, subset in by_checkpoint.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, object] = {
            "n_sbc_per_checkpoint": int(subset["n"].iloc[0]),
            "n_checkpoints": int(len(subset)),
        }
        row.update(dict(zip(group_columns, keys, strict=True)))
        for column in metric_columns:
            values = subset[column].to_numpy(dtype=np.float64)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_seed_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_true_pi_bins(
    frame: pd.DataFrame,
    *,
    pi_min: float,
    pi_max: float,
    n_bins: int,
) -> pd.DataFrame:
    if n_bins < 1:
        raise ValueError("pi_bins must be positive")
    edges = np.linspace(float(pi_min), float(pi_max), int(n_bins) + 1)
    labels = [
        f"[{edges[i]:.3g}, {edges[i + 1]:.3g}{']' if i + 1 == n_bins else ')'}"
        for i in range(int(n_bins))
    ]
    result = frame.copy()
    result["true_pi_bin"] = pd.cut(
        result["pi_true"],
        bins=edges,
        labels=labels,
        include_lowest=True,
    ).astype(str)
    return result


def compare_learned_methods(by_checkpoint: pd.DataFrame) -> pd.DataFrame:
    """Paired radial-minus-linear comparisons over Stage-1 training seeds."""
    learned = by_checkpoint[by_checkpoint["method"].isin(["linear", "radial"])]
    if learned["method"].nunique() != 2:
        return pd.DataFrame()
    wide = learned.pivot(
        index="training_seed",
        columns="method",
        values=["w1_to_exact_mean", "abs_mean_to_exact_mean", "coverage90", "pit_mean"],
    ).dropna()
    rows: list[dict[str, object]] = []
    for metric in ("w1_to_exact_mean", "abs_mean_to_exact_mean", "coverage90", "pit_mean"):
        delta = wide[(metric, "radial")].to_numpy(dtype=np.float64) - wide[
            (metric, "linear")
        ].to_numpy(dtype=np.float64)
        if len(delta) > 1:
            test = ttest_rel(wide[(metric, "radial")], wide[(metric, "linear")])
            paired_t = float(test.statistic)
            paired_p = float(test.pvalue)
        else:
            paired_t = np.nan
            paired_p = np.nan
        rows.append(
            {
                "metric": metric,
                "n_training_seeds": int(len(delta)),
                "radial_minus_linear_mean": float(np.mean(delta)),
                "radial_minus_linear_seed_sd": (
                    float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
                ),
                "paired_t": paired_t,
                "paired_t_two_sided_p": paired_p,
                "radial_better_seed_fraction": float(
                    np.mean(delta < 0.0)
                    if metric in {"w1_to_exact_mean", "abs_mean_to_exact_mean"}
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def rank_histogram(frame: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, checkpoint, training_seed), subset in frame.groupby(
        ["method", "checkpoint", "training_seed"], dropna=False
    ):
        scaled = (subset["rank"].to_numpy(dtype=np.float64) + 0.5) / (
            subset["rank_draws"].to_numpy(dtype=np.float64) + 1.0
        )
        counts, _ = np.histogram(scaled, bins=int(n_bins), range=(0.0, 1.0))
        for bin_index, count in enumerate(counts):
            rows.append(
                {
                    "method": method,
                    "checkpoint": checkpoint,
                    "training_seed": training_seed,
                    "bin": int(bin_index),
                    "count": int(count),
                    "expected_count": float(len(subset) / int(n_bins)),
                }
            )
    return pd.DataFrame(rows)


def plot_pit_histogram(frame: pd.DataFrame, path: Path, n_bins: int) -> None:
    methods = list(frame["method"].drop_duplicates())
    figure, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, method in zip(axes, methods, strict=True):
        values = frame.loc[frame["method"] == method, "pit"].to_numpy(dtype=np.float64)
        weights = np.full(len(values), 1.0 / len(values), dtype=np.float64)
        axis.hist(
            values,
            bins=int(n_bins),
            range=(0.0, 1.0),
            weights=weights,
            color="#277da1",
            edgecolor="white",
        )
        axis.axhline(1.0 / int(n_bins), color="#e76f51", linewidth=1.2)
        axis.set_title(method)
        axis.set_xlabel("posterior PIT at true pi")
    axes[0].set_ylabel("bin fraction")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(
    path: Path,
    config: dict[str, object],
    overall: pd.DataFrame,
    by_checkpoint: pd.DataFrame,
    comparison: pd.DataFrame,
    by_true_pi_bin: pd.DataFrame,
) -> None:
    columns = [
        "method",
        "n_sbc_per_checkpoint",
        "n_checkpoints",
        "pit_mean_mean",
        "pit_mean_seed_sd",
        "coverage90_mean",
        "coverage90_seed_sd",
        "w1_to_exact_mean_mean",
        "w1_to_exact_mean_seed_sd",
    ]
    checkpoint_columns = [
        "method",
        "training_seed",
        "pit_mean",
        "pit_ks_stat",
        "pit_ks_p",
        "coverage90",
        "w1_to_exact_mean",
    ]
    bin_columns = [
        "method",
        "true_pi_bin",
        "n_sbc_per_checkpoint",
        "coverage90_mean",
        "pit_mean_mean",
        "w1_to_exact_mean_mean",
        "abs_mean_to_exact_mean_mean",
    ]
    checkpoint_display = by_checkpoint[checkpoint_columns].copy()
    checkpoint_display["training_seed"] = checkpoint_display["training_seed"].map(
        lambda value: "reference" if pd.isna(value) else str(int(value))
    )
    lines = [
        "# Model 2 Prior-Predictive SBC Report",
        "",
        "## Setting",
        "",
        f"- Prior: uniform pi on `{config['pi_min']:.3g}` to `{config['pi_max']:.3g}`",
        f"- Prior-predictive datasets: `{config['n_sbc']}`",
        f"- Posterior grid points: `{config['posterior_grid_size']}`",
        f"- Standard SBC rank draws: `{config['rank_draws']}`",
        "- Exact likelihood is evaluation-only and acts as the numerical SBC sanity reference.",
        "",
        "A calibrated posterior has uniform PIT, mean PIT near 0.5, and 90% interval coverage near 0.9.",
        "The same prior-predictive datasets are reused across Stage-1 checkpoints. Therefore the overall table averages valid per-checkpoint SBC metrics; it deliberately does not report an anti-conservative pooled KS test.",
        "",
        "## Overall",
        "",
        overall[columns].to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Per Stage-1 Checkpoint",
        "",
        checkpoint_display.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Prior-Stratified Check",
        "",
        "Each row averages per-checkpoint metrics within a fixed true-pi interval. This is descriptive: conditional PIT need not be uniform within a fixed true-pi interval, so use this table to compare learned fields against the exact reference across the prior range, not as a second KS calibration test.",
        "",
        by_true_pi_bin[bin_columns].to_markdown(index=False, floatfmt=".5g"),
        "",
    ]
    if not comparison.empty:
        lines.extend(
            [
                "## Paired Radial Minus Linear Across Training Seeds",
                "",
                "For W1 and absolute posterior-mean distance, a negative difference favors radial.",
                "",
                comparison.to_markdown(index=False, floatfmt=".5g"),
                "",
            ]
        )
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

    u_grid = np.linspace(reference.u_min, reference.u_max, int(args.posterior_grid_size))
    pi_grid = stage1.sigmoid_np(u_grid)
    log_prior_u = np.log(pi_grid) + np.log1p(-pi_grid)
    rng = np.random.default_rng(int(args.seed))
    pi_true = rng.uniform(reference.pi_min, reference.pi_max, size=int(args.n_sbc))
    y = stage1.simulate_common_factor(
        rng,
        stage1.logit_np(pi_true),
        n_blocks=reference.n_blocks,
        block_size=reference.block_size,
        tau=reference.tau,
    )
    full_lr = stage1.full_block_log_ratio(y, reference.tau)
    exact_loglik = exact_loglik_grid(full_lr, pi_grid)
    exact_density, exact_cdf = normalize_log_density_u(
        exact_loglik + log_prior_u[None, :], u_grid
    )

    all_rows: list[dict[str, object]] = []
    rank_rng = np.random.default_rng(int(args.seed) + 99)
    all_rows.extend(
        rows_from_posterior(
            method="exact likelihood",
            checkpoint="reference",
            training_seed=None,
            pi_true=pi_true,
            density_u=exact_density,
            cdf=exact_cdf,
            exact_cdf=exact_cdf,
            exact_density_u=exact_density,
            u_grid=u_grid,
            pi_grid=pi_grid,
            rank_draws=int(args.rank_draws),
            rank_rng=rank_rng,
        )
    )
    for runtime in runtimes:
        print(f"evaluating {runtime.run_dir.name}/{runtime.method}")
        score = runtime.score_grid(
            y,
            u_grid,
            batch_size=int(args.batch_size),
            grid_chunk_size=int(args.grid_u_chunk_size),
        )
        _, density, cdf = integrated_posterior(score, u_grid, pi_grid)
        all_rows.extend(
            rows_from_posterior(
                method=runtime.method,
                checkpoint=runtime.run_dir.name,
                training_seed=int(runtime.config["seed"]),
                pi_true=pi_true,
                density_u=density,
                cdf=cdf,
                exact_cdf=exact_cdf,
                exact_density_u=exact_density,
                u_grid=u_grid,
                pi_grid=pi_grid,
                rank_draws=int(args.rank_draws),
                rank_rng=rank_rng,
            )
        )

    config = {
        **vars(args),
        "run_dirs_resolved": [str(path) for path in run_dirs],
        "pi_min": float(reference.pi_min),
        "pi_max": float(reference.pi_max),
        "tau": float(reference.tau),
        "sigma_q": float(reference.sigma_q),
        "n_blocks": int(reference.n_blocks),
        "block_size": int(reference.block_size),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "prior_predictive_data.npz",
        pi_true=pi_true,
        observed_y=y.astype(np.float32),
        u_grid=u_grid,
        pi_grid=pi_grid,
    )
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_dir / "sbc_by_dataset.csv", index=False)
    by_checkpoint = summarize(frame, ["method", "checkpoint", "training_seed"])
    by_checkpoint.to_csv(output_dir / "sbc_summary_by_checkpoint.csv", index=False)
    overall = summarize_across_checkpoints(by_checkpoint, ["method"])
    overall.to_csv(output_dir / "sbc_summary.csv", index=False)
    comparison = compare_learned_methods(by_checkpoint)
    comparison.to_csv(output_dir / "sbc_radial_vs_linear_by_seed.csv", index=False)
    frame_with_bins = add_true_pi_bins(
        frame,
        pi_min=float(reference.pi_min),
        pi_max=float(reference.pi_max),
        n_bins=int(args.pi_bins),
    )
    by_true_pi_bin_checkpoint = summarize(
        frame_with_bins,
        ["method", "checkpoint", "training_seed", "true_pi_bin"],
    )
    by_true_pi_bin_checkpoint.to_csv(
        output_dir / "sbc_by_true_pi_bin_by_checkpoint.csv",
        index=False,
    )
    by_true_pi_bin = summarize_across_checkpoints(
        by_true_pi_bin_checkpoint,
        ["method", "true_pi_bin"],
    )
    by_true_pi_bin.to_csv(output_dir / "sbc_by_true_pi_bin.csv", index=False)
    histogram = rank_histogram(frame, int(args.rank_bins))
    histogram.to_csv(output_dir / "sbc_rank_histogram.csv", index=False)
    plot_pit_histogram(frame, output_dir / "sbc_pit_histograms.png", int(args.rank_bins))
    write_report(
        output_dir / "SBC_REPORT.md",
        config,
        overall,
        by_checkpoint,
        comparison,
        by_true_pi_bin,
    )
    print("\n==== SBC summary ====")
    print(overall.to_string(index=False))
    print("\nSaved to", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", default="")
    parser.add_argument("--run-pattern", default="formal40k_fixed_sigma_seed202607*")
    parser.add_argument("--methods", default="linear,radial")
    parser.add_argument("--n-sbc", type=int, default=500)
    parser.add_argument("--posterior-grid-size", type=int, default=501)
    parser.add_argument("--rank-draws", type=int, default=100)
    parser.add_argument("--rank-bins", type=int, default=20)
    parser.add_argument("--pi-bins", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--grid-u-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", default="runs/model2_mode_a_sbc_smoke")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
