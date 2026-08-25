#!/usr/bin/env python3
"""Validate ULA against integrated learned-score and exact posteriors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

import run_blockwise_common_factor_amortized_fsm_experiment as stage1
from model2_amortized_score_runtime import AmortizedScoreRuntime
from run_model2_mode_a_mle import parse_float_list, parse_path_list


def reflect_to_interval(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
    width = float(upper - lower)
    if width <= 0.0:
        raise ValueError("upper must exceed lower")
    folded = np.mod(np.asarray(value, dtype=np.float64) - lower, 2.0 * width)
    return lower + np.where(folded <= width, folded, 2.0 * width - folded)


def posterior_u_variance(density_u: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    mean = np.trapezoid(density_u * u_grid[None, :], u_grid, axis=1)
    return np.trapezoid(
        density_u * (u_grid[None, :] - mean[:, None]) ** 2,
        u_grid,
        axis=1,
    )


def interpolate_score_grid(
    score_grid: np.ndarray,
    u_grid: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate one frozen score field per observed dataset."""
    score_grid = np.asarray(score_grid, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if score_grid.ndim != 2 or u.ndim != 2 or score_grid.shape[0] != u.shape[0]:
        raise ValueError("Expected score_grid=(n,G) and u=(n,n_chains)")
    left = np.searchsorted(u_grid, u, side="right") - 1
    left = np.clip(left, 0, len(u_grid) - 2)
    right = left + 1
    left_u = u_grid[left]
    fraction = (u - left_u) / (u_grid[right] - left_u)
    left_score = np.take_along_axis(score_grid, left, axis=1)
    right_score = np.take_along_axis(score_grid, right, axis=1)
    return left_score + fraction * (right_score - left_score)


def interpolation_sanity(
    runtime: AmortizedScoreRuntime,
    y: np.ndarray,
    score_grid: np.ndarray,
    u_grid: np.ndarray,
    *,
    seed: int,
    n_datasets: int = 3,
    n_points: int = 7,
) -> dict[str, float]:
    """Check cache interpolation against direct network evaluations."""
    rng = np.random.default_rng(int(seed))
    n_select = min(int(n_datasets), y.shape[0])
    indices = np.arange(n_select)
    u_values = rng.uniform(runtime.u_min, runtime.u_max, size=(n_select, int(n_points)))
    direct = runtime.score(
        np.repeat(y[indices], int(n_points), axis=0),
        u_values.reshape(-1),
        batch_size=n_select * int(n_points),
    ).reshape(n_select, int(n_points))
    interpolated = interpolate_score_grid(score_grid[indices], u_grid, u_values)
    error = direct - interpolated
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Basic multi-chain R-hat per dataset; chains shape is (n, c, draws)."""
    chains = np.asarray(chains, dtype=np.float64)
    draws = chains.shape[2]
    if chains.shape[1] < 2 or draws < 2:
        return np.full(chains.shape[0], np.nan)
    chain_means = np.mean(chains, axis=2)
    within = np.mean(np.var(chains, axis=2, ddof=1), axis=1)
    between = draws * np.var(chain_means, axis=1, ddof=1)
    variance = ((draws - 1.0) / draws) * within + between / draws
    return np.sqrt(np.divide(variance, within, out=np.ones_like(variance), where=within > 0.0))


def sample_w1(samples: np.ndarray, reference_cdf: np.ndarray, pi_grid: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    flat = np.sort(samples.reshape(samples.shape[0], -1), axis=1)
    probs = (np.arange(flat.shape[1], dtype=np.float64) + 0.5) / flat.shape[1]
    out = np.empty(flat.shape[0], dtype=np.float64)
    for i in range(flat.shape[0]):
        reference_quantiles = np.interp(probs, reference_cdf[i], pi_grid)
        out[i] = np.mean(np.abs(flat[i] - reference_quantiles))
    return out


def run_ula_batch(
    likelihood_score_grid: np.ndarray,
    surrogate_density_u: np.ndarray,
    u_grid: np.ndarray,
    *,
    u_min: float,
    u_max: float,
    step_scale: float,
    n_chains: int,
    burnin: int,
    n_samples: int,
    thin: int,
    min_epsilon: float,
    max_epsilon: float,
    initialization: str,
    seed: int,
) -> dict[str, np.ndarray]:
    n_data = likelihood_score_grid.shape[0]
    mode_index = np.argmax(surrogate_density_u, axis=1)
    mode_u = u_grid[mode_index]
    mode_pi = stage1.sigmoid_np(mode_u)
    posterior_score_grid = likelihood_score_grid + 1.0 - 2.0 * stage1.sigmoid_np(u_grid)[None, :]
    posterior_derivative_grid = np.gradient(posterior_score_grid, u_grid, axis=1)
    posterior_derivative = np.take_along_axis(
        posterior_derivative_grid, mode_index[:, None], axis=1
    ).reshape(-1)
    fallback_variance = posterior_u_variance(surrogate_density_u, u_grid)
    local_variance = np.where(
        posterior_derivative < -1e-4,
        -1.0 / posterior_derivative,
        fallback_variance,
    )
    local_variance = np.where(
        np.isfinite(local_variance) & (local_variance > 0.0),
        local_variance,
        fallback_variance,
    )
    epsilon = np.clip(float(step_scale) * local_variance, min_epsilon, max_epsilon)

    rng = np.random.default_rng(int(seed))
    if initialization == "mode":
        u = np.repeat(mode_u[:, None], int(n_chains), axis=1)
    elif initialization == "fixed_pi_0.3":
        u = np.full((n_data, int(n_chains)), stage1.logit_np(0.3), dtype=np.float64)
    else:
        raise ValueError(f"Unknown initialization {initialization!r}")
    u = reflect_to_interval(u, u_min, u_max)
    kept = np.empty((n_data, int(n_chains), int(n_samples)), dtype=np.float64)
    total_steps = int(burnin) + int(n_samples) * int(thin)
    kept_index = 0
    sqrt_noise = np.sqrt(2.0 * epsilon)[:, None]
    epsilon_b = epsilon[:, None]
    for step in range(total_steps):
        likelihood_score = interpolate_score_grid(likelihood_score_grid, u_grid, u)
        posterior_score = likelihood_score + 1.0 - 2.0 * stage1.sigmoid_np(u)
        u = reflect_to_interval(
            u
            + epsilon_b * posterior_score
            + sqrt_noise * rng.normal(size=u.shape),
            u_min,
            u_max,
        )
        if step >= int(burnin) and (step - int(burnin)) % int(thin) == 0:
            kept[:, :, kept_index] = u
            kept_index += 1
    if kept_index != int(n_samples):
        raise RuntimeError("ULA sample accounting mismatch")
    return {
        "samples_u": kept,
        "samples_pi": stage1.sigmoid_np(kept),
        "epsilon": epsilon,
        "local_variance": local_variance,
        "mode_u": mode_u,
        "mode_derivative": posterior_derivative,
    }


def metric_rows(
    *,
    pi_true: float,
    runtime: AmortizedScoreRuntime,
    step_scale: float,
    result: dict[str, np.ndarray],
    surrogate_cdf: np.ndarray,
    exact_cdf: np.ndarray,
    pi_grid: np.ndarray,
) -> list[dict[str, object]]:
    samples = np.asarray(result["samples_pi"], dtype=np.float64)
    flat = samples.reshape(samples.shape[0], -1)
    mean = np.mean(flat, axis=1)
    sd = np.std(flat, axis=1, ddof=1)
    quantiles = np.quantile(flat, [0.05, 0.50, 0.95], axis=1).T
    w1_surrogate = sample_w1(samples, surrogate_cdf, pi_grid)
    w1_exact = sample_w1(samples, exact_cdf, pi_grid)
    rhat = split_rhat(samples)
    rows: list[dict[str, object]] = []
    for i in range(samples.shape[0]):
        rows.append(
            {
                "pi_true": float(pi_true),
                "data_index": int(i),
                "method": runtime.method,
                "checkpoint": runtime.run_dir.name,
                "training_seed": int(runtime.config["seed"]),
                "step_scale": float(step_scale),
                "epsilon": float(result["epsilon"][i]),
                "local_variance_u": float(result["local_variance"][i]),
                "posterior_mode_u": float(result["mode_u"][i]),
                "mode_posterior_derivative": float(result["mode_derivative"][i]),
                "sample_mean": float(mean[i]),
                "sample_sd": float(sd[i]),
                "q05": float(quantiles[i, 0]),
                "q50": float(quantiles[i, 1]),
                "q95": float(quantiles[i, 2]),
                "coverage90": bool(quantiles[i, 0] <= pi_true <= quantiles[i, 2]),
                "mean_error_true": float(mean[i] - pi_true),
                "w1_to_surrogate": float(w1_surrogate[i]),
                "w1_to_exact": float(w1_exact[i]),
                "rhat": float(rhat[i]),
            }
        )
    return rows


def summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return frame.groupby(groups, dropna=False).agg(
        n=("sample_mean", "size"),
        coverage90=("coverage90", "mean"),
        sample_mean_rmse=(
            "mean_error_true",
            lambda x: float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))),
        ),
        sample_sd_mean=("sample_sd", "mean"),
        w1_to_surrogate_mean=("w1_to_surrogate", "mean"),
        w1_to_exact_mean=("w1_to_exact", "mean"),
        rhat_mean=("rhat", "mean"),
        rhat_max=("rhat", "max"),
        epsilon_mean=("epsilon", "mean"),
        epsilon_min=("epsilon", "min"),
        epsilon_max=("epsilon", "max"),
    ).reset_index()


def paired_radial_linear(by_checkpoint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("sample_mean_rmse", "w1_to_surrogate_mean", "w1_to_exact_mean")
    subset = by_checkpoint[by_checkpoint["method"].isin(["linear", "radial"])]
    for metric in metrics:
        pivot = subset.pivot(index="training_seed", columns="method", values=metric).dropna()
        linear = pivot["linear"].to_numpy(dtype=np.float64)
        radial = pivot["radial"].to_numpy(dtype=np.float64)
        test = ttest_rel(linear, radial) if linear.size >= 2 else None
        linear_mean = float(np.mean(linear))
        radial_mean = float(np.mean(radial))
        rows.append(
            {
                "metric": metric,
                "n_training_seeds": int(linear.size),
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
    paired: pd.DataFrame,
    interpolation: pd.DataFrame,
) -> None:
    columns = [
        "method",
        "step_scale",
        "sample_mean_rmse",
        "coverage90",
        "w1_to_surrogate_mean",
        "w1_to_exact_mean",
        "rhat_mean",
        "rhat_max",
        "epsilon_mean",
    ]
    lines = [
        "# Model 2 Mode A ULA Report",
        "",
        "## Setting",
        "",
        f"- Posterior-grid source: `{config['posterior_grid_dir']}`",
        f"- Step scales: `{config['step_scales']}`",
        f"- Chains per dataset: `{config['n_chains']}`",
        f"- Burn-in: `{config['burnin']}`",
        f"- Retained samples per chain: `{config['n_samples']}`",
        f"- Thinning: `{config['thin']}`",
        f"- Initialization: `{config['initialization']}`",
        "- Reflection is used at the trained prior-support boundaries.",
        "- W1 to surrogate isolates sampler error; W1 to exact includes field error.",
        "",
        "## Results",
        "",
        overall[columns].to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Paired Radial vs Linear",
        "",
        paired.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Cached-Field Interpolation Sanity",
        "",
        f"- Maximum direct-network versus interpolated-score error: `{interpolation['max_abs_error'].max():.3e}`",
        f"- Mean interpolation RMSE: `{interpolation['rmse'].mean():.3e}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    posterior_dir = Path(args.posterior_grid_dir).expanduser().resolve()
    source_config = json.loads((posterior_dir / "config.json").read_text(encoding="utf-8"))
    arrays = np.load(posterior_dir / "posterior_grids.npz", allow_pickle=False)
    run_dirs = parse_path_list(args.run_dirs) or [
        Path(value).resolve() for value in source_config["run_dirs_resolved"]
    ]
    methods = [part.strip() for part in str(source_config["methods"]).split(",") if part.strip()]
    runtimes = [
        AmortizedScoreRuntime(run_dir, method, device=args.device)
        for run_dir in run_dirs
        for method in methods
    ]
    runtime_names = arrays["runtime_names"].astype(str).tolist()
    resolved_names = [f"{runtime.run_dir.name}/{runtime.method}" for runtime in runtimes]
    if resolved_names != runtime_names:
        raise ValueError("Runtime order does not match posterior-grid artifact")
    for runtime in runtimes:
        checks = runtime.run_sanity_checks(seed=int(args.seed) + int(runtime.config["seed"]) % 1000)
        if not checks["passed"]:
            raise RuntimeError(f"Runtime sanity failed: {runtime.run_dir.name}/{runtime.method}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    step_scales = parse_float_list(args.step_scales)
    pi_true_values = arrays["pi_true_values"].astype(np.float64)
    u_grid = arrays["u_grid"].astype(np.float64)
    pi_grid = arrays["pi_grid"].astype(np.float64)
    observed_y = arrays["observed_y"].astype(np.float64)
    exact_cdf = arrays["exact_cdf"].astype(np.float64)
    learned_density = arrays["learned_density_u"].astype(np.float64)
    learned_cdf = arrays["learned_cdf"].astype(np.float64)
    if "learned_likelihood_score" not in arrays.files:
        raise FileNotFoundError(
            "posterior_grids.npz lacks learned_likelihood_score; rerun "
            "run_model2_mode_a_posterior_grid.py with the current code"
        )
    learned_likelihood_score = arrays["learned_likelihood_score"].astype(np.float64)
    config = {
        **vars(args),
        "posterior_grid_dir": str(posterior_dir),
        "run_dirs_resolved": [str(path) for path in run_dirs],
        "runtime_names": runtime_names,
        "pi_true_values": pi_true_values.tolist(),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows: list[dict[str, object]] = []
    interpolation_rows: list[dict[str, object]] = []
    n_pi = len(pi_true_values)
    n_per_pi = observed_y.shape[1]
    y_all = observed_y.reshape((-1,) + observed_y.shape[2:])
    exact_cdf_all = exact_cdf.reshape((-1, exact_cdf.shape[-1]))
    for runtime_index, runtime in enumerate(runtimes):
        print(f"runtime {runtime_index + 1}/{len(runtimes)}: {runtime_names[runtime_index]}")
        score_all = learned_likelihood_score[runtime_index].reshape((-1, len(u_grid)))
        density_all = learned_density[runtime_index].reshape((-1, len(u_grid)))
        cdf_all = learned_cdf[runtime_index].reshape((-1, len(u_grid)))
        sanity = interpolation_sanity(
            runtime,
            y_all,
            score_all,
            u_grid,
            seed=int(args.seed) + runtime_index,
        )
        interpolation_rows.append(
            {
                "method": runtime.method,
                "checkpoint": runtime.run_dir.name,
                "training_seed": int(runtime.config["seed"]),
                **sanity,
            }
        )
        if sanity["max_abs_error"] > float(args.interpolation_sanity_tol):
            raise AssertionError(
                f"Interpolation error too large for {runtime.run_dir.name}/{runtime.method}: "
                f"{sanity['max_abs_error']:.3e}"
            )
        for scale_index, step_scale in enumerate(step_scales):
            result = run_ula_batch(
                score_all,
                density_all,
                u_grid,
                u_min=runtime.u_min,
                u_max=runtime.u_max,
                step_scale=step_scale,
                n_chains=int(args.n_chains),
                burnin=int(args.burnin),
                n_samples=int(args.n_samples),
                thin=int(args.thin),
                min_epsilon=float(args.min_epsilon),
                max_epsilon=float(args.max_epsilon),
                initialization=str(args.initialization),
                seed=int(args.seed) + 1_000_000 * runtime_index + 100 * scale_index,
            )
            for pi_index, pi_true in enumerate(pi_true_values):
                start = pi_index * n_per_pi
                stop = start + n_per_pi
                result_slice = {
                    key: np.asarray(value)[start:stop]
                    for key, value in result.items()
                }
                all_rows.extend(
                    metric_rows(
                        pi_true=float(pi_true),
                        runtime=runtime,
                        step_scale=float(step_scale),
                        result=result_slice,
                        surrogate_cdf=cdf_all[start:stop],
                        exact_cdf=exact_cdf_all[start:stop],
                        pi_grid=pi_grid,
                    )
                )

    frame = pd.DataFrame(all_rows)
    frame.to_csv(output_dir / "ula_by_dataset.csv", index=False)
    interpolation_frame = pd.DataFrame(interpolation_rows)
    interpolation_frame.to_csv(output_dir / "interpolation_sanity.csv", index=False)
    by_checkpoint = summarize(
        frame,
        ["method", "step_scale", "checkpoint", "training_seed"],
    )
    by_checkpoint.to_csv(output_dir / "ula_summary_by_checkpoint.csv", index=False)
    overall = summarize(frame, ["method", "step_scale"])
    overall.to_csv(output_dir / "ula_summary.csv", index=False)
    by_pi = summarize(frame, ["pi_true", "method", "step_scale"])
    by_pi.to_csv(output_dir / "ula_summary_by_pi.csv", index=False)
    paired = paired_radial_linear(by_checkpoint)
    paired.to_csv(output_dir / "radial_vs_linear_paired.csv", index=False)
    write_report(
        output_dir / "MODE_A_ULA_REPORT.md",
        config,
        overall,
        paired,
        interpolation_frame,
    )
    print("\n==== ULA summary ====")
    print(overall.to_string(index=False))
    print("\nSaved to", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior-grid-dir", required=True)
    parser.add_argument("--run-dirs", default="")
    parser.add_argument("--step-scales", default="0.02,0.05,0.10")
    parser.add_argument("--n-chains", type=int, default=8)
    parser.add_argument("--burnin", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--min-epsilon", type=float, default=1e-5)
    parser.add_argument("--max-epsilon", type=float, default=0.05)
    parser.add_argument("--interpolation-sanity-tol", type=float, default=2e-3)
    parser.add_argument("--initialization", default="mode", choices=("mode", "fixed_pi_0.3"))
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", default="runs/model2_mode_a_ula_smoke")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
