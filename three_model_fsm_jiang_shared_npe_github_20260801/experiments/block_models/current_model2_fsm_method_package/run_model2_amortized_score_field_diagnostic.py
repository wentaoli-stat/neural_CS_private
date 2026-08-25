#!/usr/bin/env python3
"""Diagnose an amortized FSM score field away from the training diagonal.

For each data-generating value ``pi_true`` this script evaluates the learned
field over a full candidate-parameter grid ``pi_eval``.  It compares linear and
radial checkpoints with both the exact likelihood score and the Gaussian-
proposal-smoothed oracle score targeted by anchored FSM.

True scores are diagnostics only.  No model is trained by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_blockwise_common_factor_amortized_fsm_experiment as stage1
import run_blockwise_common_factor_information_precheck as precheck
from model2_amortized_score_runtime import (
    AmortizedScoreRuntime,
    logit_grid,
    normal_relative_density,
    parse_method_list,
)


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_path_list(text: str) -> list[Path]:
    return [Path(part.strip()).expanduser().resolve() for part in str(text).split(",") if part.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_pair(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    err = prediction - target
    mse = float(np.mean(err**2))
    target_var = float(np.var(target))
    target_rms = float(np.sqrt(np.mean(target**2)))
    pred_sd = float(np.std(prediction))
    target_sd = float(np.std(target))
    corr = (
        float(np.corrcoef(target, prediction)[0, 1])
        if target_sd > 1e-12 and pred_sd > 1e-12
        else float("nan")
    )
    denom = float(np.linalg.norm(target) * np.linalg.norm(prediction))
    cosine = float(np.dot(target, prediction) / denom) if denom > 1e-12 else float("nan")
    return {
        "mse": mse,
        "rmse": math.sqrt(max(mse, 0.0)),
        "std_mse": mse / max(target_var, 1e-12),
        "relative_rms_mse": mse / max(target_rms**2, 1e-12),
        "bias": float(np.mean(err)),
        "corr": corr,
        "cosine": cosine,
        "target_mean": float(np.mean(target)),
        "target_sd": target_sd,
        "pred_mean": float(np.mean(prediction)),
        "pred_sd": pred_sd,
    }


def sign_agreement(target: np.ndarray, prediction: np.ndarray, delta_fraction: float) -> tuple[float, int]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    threshold = float(delta_fraction) * float(np.sqrt(np.mean(target**2)))
    mask = np.isfinite(target) & np.isfinite(prediction) & (np.abs(target) > threshold)
    if not np.any(mask):
        return float("nan"), 0
    agreement = np.sign(target[mask]) == np.sign(prediction[mask])
    return float(np.mean(agreement)), int(np.sum(mask))


def oracle_roots(score: np.ndarray, u_grid: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Find an interior score root or the correct constrained boundary."""
    score = np.asarray(score, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    roots = np.empty(score.shape[0], dtype=np.float64)
    statuses: list[str] = []
    n_crossings = np.zeros(score.shape[0], dtype=np.int64)
    for i, row in enumerate(score):
        exact_zero = np.flatnonzero(np.isclose(row, 0.0, atol=1e-12))
        crossings = np.flatnonzero(row[:-1] * row[1:] < 0.0)
        n_crossings[i] = int(exact_zero.size + crossings.size)
        candidates: list[float] = [float(u_grid[j]) for j in exact_zero]
        for j in crossings:
            left_u, right_u = float(u_grid[j]), float(u_grid[j + 1])
            left_s, right_s = float(row[j]), float(row[j + 1])
            weight = -left_s / (right_s - left_s)
            candidates.append(left_u + weight * (right_u - left_u))
        if candidates:
            # Choose the crossing nearest the grid point with the smallest score.
            target_u = float(u_grid[int(np.argmin(np.abs(row)))])
            roots[i] = min(candidates, key=lambda value: abs(value - target_u))
            statuses.append("interior" if len(candidates) == 1 else "multiple")
        elif np.all(row > 0.0):
            roots[i] = float(u_grid[-1])
            statuses.append("right_boundary")
        elif np.all(row < 0.0):
            roots[i] = float(u_grid[0])
            statuses.append("left_boundary")
        else:
            roots[i] = float(u_grid[int(np.argmin(np.abs(row)))])
            statuses.append("unresolved")
    return roots, statuses, n_crossings


def root_direction_agreement(
    prediction: np.ndarray,
    roots: np.ndarray,
    u_eval: float,
    min_distance: float,
) -> tuple[float, int]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    direction = np.asarray(roots, dtype=np.float64).reshape(-1) - float(u_eval)
    mask = np.isfinite(prediction) & np.isfinite(direction) & (np.abs(direction) > float(min_distance))
    if not np.any(mask):
        return float("nan"), 0
    return float(np.mean(prediction[mask] * direction[mask] > 0.0)), int(np.sum(mask))


def region_from_distance(distance_sigma: float) -> str:
    if distance_sigma < 1e-8:
        return "diagonal"
    if distance_sigma <= 2.0:
        return "local_0_2sigma"
    if distance_sigma <= 4.0:
        return "transition_2_4sigma"
    return "far_gt_4sigma"


def plot_heatmap(
    frame: pd.DataFrame,
    method: str,
    value: str,
    output_path: Path,
    title: str,
    *,
    log10: bool,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    subset = frame[frame["method"] == method]
    table = subset.pivot_table(index="pi_true", columns="pi_eval", values=value, aggfunc="mean")
    table = table.sort_index().sort_index(axis=1)
    values = table.to_numpy(dtype=np.float64)
    label = value
    if log10:
        values = np.log10(np.maximum(values, 1e-6))
        label = f"log10({value})"
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[
            float(table.columns.min()),
            float(table.columns.max()),
            float(table.index.min()),
            float(table.index.max()),
        ],
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    lo = max(float(table.columns.min()), float(table.index.min()))
    hi = min(float(table.columns.max()), float(table.index.max()))
    ax.plot([lo, hi], [lo, hi], color="white", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.set_xlabel("candidate / evaluation pi")
    ax.set_ylabel("data-generating pi_true")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def resolve_run_dirs(args: argparse.Namespace, package_dir: Path) -> list[Path]:
    explicit = parse_path_list(args.run_dirs)
    if explicit:
        return explicit
    paths = sorted(path for path in (package_dir / "runs").glob(str(args.run_pattern)) if path.is_dir())
    if not paths:
        raise FileNotFoundError(
            f"No run directories matched {(package_dir / 'runs' / args.run_pattern)!s}"
        )
    return [path.resolve() for path in paths]


def run(args: argparse.Namespace) -> None:
    package_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    methods = parse_method_list(args.methods)
    run_dirs = resolve_run_dirs(args, package_dir)
    pi_true_values = parse_float_list(args.pi_true_values)
    if not pi_true_values:
        raise ValueError("--pi-true-values must not be empty")

    runtimes: list[AmortizedScoreRuntime] = []
    runtime_checks: list[dict[str, object]] = []
    for run_dir in run_dirs:
        for method in methods:
            runtime = AmortizedScoreRuntime(run_dir, method, device=args.device)
            checks = runtime.run_sanity_checks(seed=int(args.seed) + len(runtimes))
            runtime_checks.append(checks)
            if not checks["passed"]:
                raise RuntimeError(f"Runtime checks failed for {run_dir.name}/{method}: {checks}")
            runtimes.append(runtime)
            print(
                f"[OK runtime] {run_dir.name}/{method} "
                f"best={runtime.best_source} forward={checks['forward_max_abs']:.3e} "
                f"d/du={checks['derivative_fd_max_abs']:.3e}"
            )

    reference = runtimes[0]
    for runtime in runtimes[1:]:
        for key in ("n_blocks", "block_size", "tau", "sigma_q", "pi_min", "pi_max"):
            if not math.isclose(float(getattr(runtime, key)), float(getattr(reference, key)), rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"All checkpoints must share {key}")

    pi_eval_base, _ = logit_grid(reference.pi_min, reference.pi_max, int(args.pi_eval_grid_size))
    pi_eval = np.unique(np.concatenate([pi_eval_base, np.asarray(pi_true_values, dtype=np.float64)]))
    pi_eval.sort()
    u_eval = stage1.logit_np(pi_eval)
    if np.any(np.asarray(pi_true_values) < reference.pi_min) or np.any(
        np.asarray(pi_true_values) > reference.pi_max
    ):
        raise ValueError("pi_true values must lie inside the trained anchor support")

    config = {
        **vars(args),
        "run_dirs_resolved": [str(path) for path in run_dirs],
        "pi_eval_values": pi_eval.tolist(),
        "device_resolved": reference.device,
        "n_blocks": reference.n_blocks,
        "block_size": reference.block_size,
        "tau": reference.tau,
        "sigma_q": reference.sigma_q,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "runtime_checks.json").write_text(
        json.dumps(runtime_checks, indent=2), encoding="utf-8"
    )

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, object]] = []
    root_rows: list[dict[str, object]] = []
    oracle_cache: dict[float, dict[str, np.ndarray]] = {}

    print(
        f"Evaluating {len(pi_true_values)} pi_true values x {len(pi_eval)} pi_eval values "
        f"with n={args.n_data} per row..."
    )
    for pi_true in pi_true_values:
        true_u = float(stage1.logit_np(pi_true))
        y = stage1.simulate_common_factor(
            rng,
            np.full(int(args.n_data), true_u, dtype=np.float64),
            n_blocks=reference.n_blocks,
            block_size=reference.block_size,
            tau=reference.tau,
        )
        full_lr = stage1.full_block_log_ratio(y, reference.tau)
        exact = np.empty((int(args.n_data), len(pi_eval)), dtype=np.float64)
        smooth = np.empty_like(exact)
        for j, (pi_value, u_value) in enumerate(zip(pi_eval, u_eval, strict=True)):
            exact[:, j] = stage1.full_score_u(y, float(pi_value), reference.tau)
            smooth[:, j] = precheck.smoothed_fsm_score_from_full_lr(
                full_lr=full_lr,
                u0=float(u_value),
                sigma_q=reference.sigma_q,
                grid_size=int(args.smooth_grid_size),
                grid_radius=float(args.smooth_grid_radius),
                chunk_size=int(args.smooth_chunk_size),
            )
        roots, root_statuses, n_crossings = oracle_roots(smooth, u_eval)
        oracle_cache[float(pi_true)] = {"y": y, "exact": exact, "smooth": smooth, "roots": roots}
        for data_index, (root, status, crossings) in enumerate(
            zip(roots, root_statuses, n_crossings, strict=True)
        ):
            root_rows.append(
                {
                    "pi_true": float(pi_true),
                    "data_index": int(data_index),
                    "oracle_root_u": float(root),
                    "oracle_root_pi": float(stage1.sigmoid_np(root)),
                    "oracle_root_status": status,
                    "n_crossings": int(crossings),
                }
            )

        floor_by_eval = [metric_pair(exact[:, j], smooth[:, j]) for j in range(len(pi_eval))]
        min_root_distance = 0.25 * float(np.median(np.diff(u_eval)))
        for runtime in runtimes:
            learned = runtime.score_grid(y, u_eval, batch_size=int(args.batch_size))
            for j, (pi_value, u_value) in enumerate(zip(pi_eval, u_eval, strict=True)):
                exact_metrics = metric_pair(exact[:, j], learned[:, j])
                smooth_metrics = metric_pair(smooth[:, j], learned[:, j])
                sign_value, sign_n = sign_agreement(
                    smooth[:, j], learned[:, j], float(args.sign_delta_fraction)
                )
                root_value, root_n = root_direction_agreement(
                    learned[:, j], roots, float(u_value), min_root_distance
                )
                distance_sigma = abs(float(u_value) - true_u) / reference.sigma_q
                floor = floor_by_eval[j]
                rows.append(
                    {
                        "checkpoint": runtime.run_dir.name,
                        "training_seed": runtime.config.get("seed"),
                        "best_source": runtime.best_source,
                        "method": runtime.method,
                        "label": runtime.label,
                        "pi_true": float(pi_true),
                        "pi_eval": float(pi_value),
                        "u_true": true_u,
                        "u_eval": float(u_value),
                        "distance_sigma": distance_sigma,
                        "local_relative_density": normal_relative_density(distance_sigma),
                        "region": region_from_distance(distance_sigma),
                        "n_data": int(args.n_data),
                        "model_to_exact_mse": exact_metrics["mse"],
                        "model_to_exact_std_mse": exact_metrics["std_mse"],
                        "model_to_exact_corr": exact_metrics["corr"],
                        "model_to_exact_cosine": exact_metrics["cosine"],
                        "model_to_smoothed_mse": smooth_metrics["mse"],
                        "model_to_smoothed_std_mse": smooth_metrics["std_mse"],
                        "model_to_smoothed_relative_rms_mse": smooth_metrics["relative_rms_mse"],
                        "model_to_smoothed_corr": smooth_metrics["corr"],
                        "model_to_smoothed_cosine": smooth_metrics["cosine"],
                        "smoothing_floor_mse": floor["mse"],
                        "smoothing_floor_std_mse": floor["std_mse"],
                        "smoothed_exact_corr": floor["corr"],
                        "sign_agreement_smoothed": sign_value,
                        "sign_agreement_n": sign_n,
                        "root_direction_agreement": root_value,
                        "root_direction_n": root_n,
                        "learned_mean": float(np.mean(learned[:, j])),
                        "learned_sd": float(np.std(learned[:, j])),
                        "smoothed_mean": float(np.mean(smooth[:, j])),
                        "smoothed_sd": float(np.std(smooth[:, j])),
                        "exact_mean": float(np.mean(exact[:, j])),
                        "exact_sd": float(np.std(exact[:, j])),
                    }
                )
        print(f"  completed pi_true={pi_true:.3g}")

    write_csv(output_dir / "field_metrics_by_checkpoint.csv", rows)
    write_csv(output_dir / "smoothed_oracle_roots.csv", root_rows)
    frame = pd.DataFrame(rows)

    per_checkpoint = (
        frame.groupby(["checkpoint", "training_seed", "method", "region"], dropna=False)
        .agg(
            n_cells=("pi_eval", "size"),
            model_to_exact_std_mse=("model_to_exact_std_mse", "mean"),
            model_to_smoothed_std_mse=("model_to_smoothed_std_mse", "mean"),
            model_to_smoothed_relative_rms_mse=(
                "model_to_smoothed_relative_rms_mse",
                "mean",
            ),
            smoothing_floor_std_mse=("smoothing_floor_std_mse", "mean"),
            sign_agreement_smoothed=("sign_agreement_smoothed", "mean"),
            root_direction_agreement=("root_direction_agreement", "mean"),
        )
        .reset_index()
    )
    per_checkpoint.to_csv(output_dir / "field_region_by_checkpoint.csv", index=False)
    summary = (
        per_checkpoint.groupby(["method", "region"], dropna=False)
        .agg(
            n_checkpoints=("checkpoint", "nunique"),
            model_to_exact_std_mse_mean=("model_to_exact_std_mse", "mean"),
            model_to_exact_std_mse_sd=("model_to_exact_std_mse", "std"),
            model_to_smoothed_std_mse_mean=("model_to_smoothed_std_mse", "mean"),
            model_to_smoothed_std_mse_sd=("model_to_smoothed_std_mse", "std"),
            model_to_smoothed_relative_rms_mse_mean=(
                "model_to_smoothed_relative_rms_mse",
                "mean",
            ),
            model_to_smoothed_relative_rms_mse_sd=(
                "model_to_smoothed_relative_rms_mse",
                "std",
            ),
            smoothing_floor_std_mse_mean=("smoothing_floor_std_mse", "mean"),
            sign_agreement_mean=("sign_agreement_smoothed", "mean"),
            sign_agreement_sd=("sign_agreement_smoothed", "std"),
            root_direction_agreement_mean=("root_direction_agreement", "mean"),
            root_direction_agreement_sd=("root_direction_agreement", "std"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "field_region_summary.csv", index=False)

    cell_means = (
        frame.groupby(["method", "pi_true", "pi_eval"], dropna=False)
        .agg(
            model_to_smoothed_std_mse=("model_to_smoothed_std_mse", "mean"),
            model_to_smoothed_relative_rms_mse=(
                "model_to_smoothed_relative_rms_mse",
                "mean",
            ),
            sign_agreement_smoothed=("sign_agreement_smoothed", "mean"),
            root_direction_agreement=("root_direction_agreement", "mean"),
        )
        .reset_index()
    )
    comparison_rows: list[dict[str, object]] = []
    if {"linear", "radial"}.issubset(set(methods)):
        linear_cells = cell_means[cell_means["method"] == "linear"].drop(columns="method")
        radial_cells = cell_means[cell_means["method"] == "radial"].drop(columns="method")
        paired = linear_cells.merge(
            radial_cells,
            on=["pi_true", "pi_eval"],
            suffixes=("_linear", "_radial"),
            validate="one_to_one",
        )
        for row in paired.to_dict(orient="records"):
            linear_value = float(row["model_to_smoothed_std_mse_linear"])
            radial_value = float(row["model_to_smoothed_std_mse_radial"])
            comparison_rows.append(
                {
                    **row,
                    "radial_improvement_pct": 100.0 * (linear_value - radial_value) / linear_value,
                    "radial_wins": bool(radial_value < linear_value),
                }
            )
        write_csv(output_dir / "radial_vs_linear_surface.csv", comparison_rows)

    for method in methods:
        plot_heatmap(
            frame,
            method,
            "model_to_smoothed_std_mse",
            figure_dir / f"{method}_model_to_smoothed_std_mse.png",
            f"{method}: learned field vs smoothed oracle",
            log10=True,
        )
        plot_heatmap(
            frame,
            method,
            "sign_agreement_smoothed",
            figure_dir / f"{method}_sign_agreement_smoothed.png",
            f"{method}: score sign agreement with smoothed oracle",
            log10=False,
            vmin=0.0,
            vmax=1.0,
        )
        plot_heatmap(
            frame,
            method,
            "root_direction_agreement",
            figure_dir / f"{method}_root_direction_agreement.png",
            f"{method}: drift toward smoothed-oracle root",
            log10=False,
            vmin=0.0,
            vmax=1.0,
        )

    go_rows: list[dict[str, object]] = []
    for method in methods:
        method_summary = summary[summary["method"] == method]
        far = method_summary[method_summary["region"] == "far_gt_4sigma"]
        diagonal = method_summary[method_summary["region"] == "diagonal"]
        far_sign = float(far["sign_agreement_mean"].iloc[0]) if not far.empty else float("nan")
        far_root = (
            float(far["root_direction_agreement_mean"].iloc[0]) if not far.empty else float("nan")
        )
        diagonal_error = (
            float(diagonal["model_to_smoothed_std_mse_mean"].iloc[0])
            if not diagonal.empty
            else float("nan")
        )
        far_relative_rms = (
            float(far["model_to_smoothed_relative_rms_mse_mean"].iloc[0])
            if not far.empty
            else float("nan")
        )
        passed = bool(
            np.isfinite(far_sign)
            and np.isfinite(far_root)
            and far_sign >= float(args.go_min_far_sign)
            and far_root >= float(args.go_min_far_root)
        )
        go_rows.append(
            {
                "method": method,
                "diagonal_model_to_smoothed_std_mse": diagonal_error,
                "far_sign_agreement": far_sign,
                "far_root_direction_agreement": far_root,
                "far_model_to_smoothed_relative_rms_mse": far_relative_rms,
                "required_far_sign": float(args.go_min_far_sign),
                "required_far_root": float(args.go_min_far_root),
                "mode_a_field_go": passed,
            }
        )
    write_csv(output_dir / "mode_a_go_summary.csv", go_rows)

    print("\n==== Region summary ====")
    print(summary.to_string(index=False))
    print("\n==== Mode A field GO ====")
    for row in go_rows:
        print(
            f"{row['method']:<8} far_sign={row['far_sign_agreement']:.3f} "
            f"far_root={row['far_root_direction_agreement']:.3f} "
            f"far_rel_rms_mse={row['far_model_to_smoothed_relative_rms_mse']:.4f} "
            f"GO={row['mode_a_field_go']}"
        )
    if comparison_rows:
        wins = sum(bool(row["radial_wins"]) for row in comparison_rows)
        improvements = np.asarray(
            [float(row["radial_improvement_pct"]) for row in comparison_rows], dtype=np.float64
        )
        print(
            f"radial vs linear surface: {wins}/{len(comparison_rows)} cells won; "
            f"mean improvement={np.mean(improvements):.2f}%"
        )

    report_lines = [
        "# Model 2 amortized score-field diagnostic",
        "",
        "## Setting",
        "",
        f"- Stage-1 checkpoints: {len(run_dirs)} training seeds",
        f"- Methods: `{','.join(methods)}`",
        f"- Data-generating pi: `{','.join(str(value) for value in pi_true_values)}`",
        f"- Candidate pi grid points: {len(pi_eval)}",
        f"- Datasets per pi_true: {int(args.n_data)}",
        f"- Model: K={reference.n_blocks}, m={reference.block_size}, tau={reference.tau}",
        f"- Anchored FSM sigma_q={reference.sigma_q}",
        "- Exact and smoothed-oracle scores are evaluation-only.",
        "",
        "## Runtime checks",
        "",
        f"All {len(runtime_checks)} checkpoint/method runtimes passed forward, derivative, finiteness, and permutation checks.",
        "",
        "## Region summary",
        "",
        "| method | region | model-to-smoothed std-MSE | relative RMS MSE | sign agreement | root-direction agreement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        report_lines.append(
            f"| {row['method']} | {row['region']} | "
            f"{float(row['model_to_smoothed_std_mse_mean']):.4f} | "
            f"{float(row['model_to_smoothed_relative_rms_mse_mean']):.4f} | "
            f"{float(row['sign_agreement_mean']):.4f} | "
            f"{float(row['root_direction_agreement_mean']):.4f} |"
        )
    report_lines.extend(["", "## GO decision", ""])
    for row in go_rows:
        report_lines.append(
            f"- **{row['method']}**: far sign={float(row['far_sign_agreement']):.4f}, "
            f"far root direction={float(row['far_root_direction_agreement']):.4f}, "
            f"far relative RMS MSE={float(row['far_model_to_smoothed_relative_rms_mse']):.4f}; "
            f"GO=`{row['mode_a_field_go']}`."
        )
    if comparison_rows:
        report_lines.extend(
            [
                "",
                "## Paired radial-vs-linear surface result",
                "",
                f"- Radial wins {wins}/{len(comparison_rows)} pi_true/pi_eval cells.",
                f"- Mean cell-wise reduction in model-to-smoothed std-MSE: {float(np.mean(improvements)):.2f}%.",
                f"- Median reduction: {float(np.median(improvements)):.2f}%.",
                f"- Range: {float(np.min(improvements)):.2f}% to {float(np.max(improvements)):.2f}%.",
            ]
        )
    report_lines.extend(
        [
            "",
            "This diagnostic certifies field direction over the tested support. It does not yet certify posterior tails or ULA discretization; those are the next Mode-A layers.",
            "",
        ]
    )
    (output_dir / "SCORE_FIELD_DIAGNOSTIC_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print("\nSaved to", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", default="")
    parser.add_argument("--run-pattern", default="formal40k_fixed_sigma_seed202607*" )
    parser.add_argument("--methods", default="linear,radial")
    parser.add_argument("--pi-true-values", default="0.07,0.10,0.30,0.50,0.65,0.68")
    parser.add_argument("--pi-eval-grid-size", type=int, default=21)
    parser.add_argument("--n-data", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smooth-grid-size", type=int, default=121)
    parser.add_argument("--smooth-grid-radius", type=float, default=5.0)
    parser.add_argument("--smooth-chunk-size", type=int, default=64)
    parser.add_argument("--sign-delta-fraction", type=float, default=0.05)
    parser.add_argument("--go-min-far-sign", type=float, default=0.90)
    parser.add_argument("--go-min-far-root", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--output-dir",
        default="runs/model2_amortized_score_field_diagnostic",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
