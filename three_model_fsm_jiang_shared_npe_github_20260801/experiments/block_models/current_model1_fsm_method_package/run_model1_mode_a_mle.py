#!/usr/bin/env python3
"""Mode A MLE for the amortized Model-2 FSM score field.

The script separates three questions:

1. Does the frozen learned field have the correct constrained global root?
2. Can safeguarded Newton reach that root from a cheap non-oracle pilot?
3. How much of the remaining gap is already present in the smoothed-oracle
   field targeted by anchored FSM?

Exact likelihoods and smoothed-oracle scores are evaluation-only references.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import ttest_rel

import run_blockwise_mean_shift_amortized_fsm_experiment as stage1
import model1_mode_a_reference as mode_a_reference
from model1_amortized_score_runtime import AmortizedScoreRuntime, parse_method_list


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


def resolve_run_dirs(args: argparse.Namespace, package_dir: Path) -> list[Path]:
    explicit = parse_path_list(args.run_dirs)
    if explicit:
        return explicit
    paths = sorted(path for path in (package_dir / "runs").glob(args.run_pattern) if path.is_dir())
    if not paths:
        raise FileNotFoundError(f"No run directories matched {args.run_pattern!r}")
    return [path.resolve() for path in paths]


def cumulative_trapezoid_rows(score: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    increments = 0.5 * (score[:, 1:] + score[:, :-1]) * np.diff(u_grid)[None, :]
    return np.concatenate(
        [np.zeros((score.shape[0], 1), dtype=np.float64), np.cumsum(increments, axis=1)],
        axis=1,
    )


def constrained_modes_from_score_grid(
    score: np.ndarray,
    u_grid: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Select the global constrained mode implied by a one-dimensional field."""
    score = np.asarray(score, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    if score.ndim != 2 or score.shape[1] != u_grid.size:
        raise ValueError("score must have shape (n, len(u_grid))")
    potential = cumulative_trapezoid_rows(score, u_grid)
    roots = np.empty(score.shape[0], dtype=np.float64)
    statuses: list[str] = []
    n_stationary = np.zeros(score.shape[0], dtype=np.int64)
    residual = np.empty(score.shape[0], dtype=np.float64)

    for i, (row, row_potential) in enumerate(zip(score, potential, strict=True)):
        candidates: list[tuple[float, float, float, str]] = [
            (float(u_grid[0]), float(row_potential[0]), abs(float(row[0])), "left_boundary"),
            (float(u_grid[-1]), float(row_potential[-1]), abs(float(row[-1])), "right_boundary"),
        ]
        zero_mask = np.isclose(row, 0.0, atol=1e-12)
        crossings = np.flatnonzero(
            (row[:-1] * row[1:] < 0.0) & ~zero_mask[:-1] & ~zero_mask[1:]
        )
        exact_zero = np.flatnonzero(zero_mask)
        n_stationary[i] = int(crossings.size + exact_zero.size)
        for j in exact_zero:
            candidates.append((float(u_grid[j]), float(row_potential[j]), 0.0, "interior"))
        for j in crossings:
            left_u, right_u = float(u_grid[j]), float(u_grid[j + 1])
            left_s, right_s = float(row[j]), float(row[j + 1])
            weight = float(np.clip(-left_s / (right_s - left_s), 0.0, 1.0))
            root_u = left_u + weight * (right_u - left_u)
            root_potential = float(
                row_potential[j]
                + 0.5 * left_s * (root_u - left_u)
            )
            candidates.append((root_u, root_potential, 0.0, "interior"))

        # Highest integrated potential wins; residual and interior status break ties.
        best = max(candidates, key=lambda item: (item[1], -item[2], item[3] == "interior"))
        roots[i] = best[0]
        residual[i] = best[2]
        if best[3] == "interior" and n_stationary[i] > 1:
            statuses.append("multiple_selected_global")
        else:
            statuses.append(best[3])
    return roots, statuses, n_stationary, residual


def exact_loglik_from_full_lr(full_lr: np.ndarray, u: float) -> np.ndarray:
    pi = float(stage1.sigmoid_np(u))
    return np.sum(
        np.logaddexp(math.log1p(-pi), math.log(pi) + np.asarray(full_lr, dtype=np.float64)),
        axis=1,
    )


def exact_constrained_mle(
    full_lr: np.ndarray,
    u_min: float,
    u_max: float,
    xatol: float,
) -> tuple[np.ndarray, list[str]]:
    full_lr = np.asarray(full_lr, dtype=np.float64)
    estimates = np.empty(full_lr.shape[0], dtype=np.float64)
    statuses: list[str] = []
    for i, row in enumerate(full_lr):
        def objective(u: float) -> float:
            return -float(exact_loglik_from_full_lr(row[None, :], u)[0])

        result = minimize_scalar(
            objective,
            bounds=(float(u_min), float(u_max)),
            method="bounded",
            options={"xatol": float(xatol), "maxiter": 500},
        )
        candidates = [
            (float(u_min), objective(float(u_min))),
            (float(u_max), objective(float(u_max))),
            (float(result.x), float(result.fun)),
        ]
        best_u, _ = min(candidates, key=lambda item: item[1])
        estimates[i] = best_u
        boundary_tol = max(10.0 * float(xatol), 1e-7)
        if abs(best_u - u_min) <= boundary_tol:
            statuses.append("left_boundary")
        elif abs(best_u - u_max) <= boundary_tol:
            statuses.append("right_boundary")
        else:
            statuses.append("interior")
    return estimates, statuses


def composite_pilot_score_grid(
    y: np.ndarray,
    u_grid: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Marginal composite estimating function used as a cheap pilot."""
    lr1 = stage1.marginal_log_ratio(y, float(tau))
    out = np.empty((y.shape[0], len(u_grid)), dtype=np.float64)
    for j, u in enumerate(np.asarray(u_grid, dtype=np.float64)):
        pi = float(stage1.sigmoid_np(u))
        s1 = stage1.score_u_from_log_ratio(lr1, pi)
        out[:, j] = s1.mean(axis=2).sum(axis=1)
    return out


def status_matches_exact(estimate_status: str, exact_status: str) -> bool:
    if exact_status == "interior":
        return estimate_status in {"interior", "multiple_selected_global", "converged_interior"}
    return estimate_status == exact_status or estimate_status == f"converged_{exact_status}"


def safeguarded_newton(
    runtime: AmortizedScoreRuntime,
    y: np.ndarray,
    initial_u: np.ndarray,
    *,
    max_iters: int,
    score_tol: float,
    step_tol: float,
    min_curvature: float,
    max_step: float,
    fallback_lr: float,
    max_backtracks: int,
    batch_size: int,
) -> dict[str, object]:
    u = np.clip(np.asarray(initial_u, dtype=np.float64), runtime.u_min, runtime.u_max)
    n = y.shape[0]
    done = np.zeros(n, dtype=bool)
    iterations = np.zeros(n, dtype=np.int64)
    statuses = np.full(n, "max_iter", dtype=object)

    for iteration in range(1, int(max_iters) + 1):
        score, derivative = runtime.score_and_derivative(y, u, batch_size=batch_size)
        at_left = u <= runtime.u_min + step_tol
        at_right = u >= runtime.u_max - step_tol
        left_solution = at_left & (score <= 0.0)
        right_solution = at_right & (score >= 0.0)
        # A zero with positive curvature is a local minimum of the implied
        # likelihood, not an MLE. Only accept locally concave interior roots.
        interior_solution = (np.abs(score) <= score_tol) & (
            derivative < -float(min_curvature)
        )
        newly_done = (~done) & (left_solution | right_solution | interior_solution)
        statuses[newly_done & left_solution] = "converged_left_boundary"
        statuses[newly_done & right_solution] = "converged_right_boundary"
        statuses[newly_done & interior_solution & ~left_solution & ~right_solution] = "converged_interior"
        iterations[newly_done] = iteration - 1
        done |= newly_done
        if np.all(done):
            break

        valid_newton = derivative < -float(min_curvature)
        delta = np.where(valid_newton, -score / derivative, float(fallback_lr) * score)
        delta = np.clip(delta, -float(max_step), float(max_step))
        delta[done] = 0.0
        candidate = np.clip(u + delta, runtime.u_min, runtime.u_max)
        candidate_score = runtime.score(y, candidate, batch_size=batch_size)

        for _ in range(int(max_backtracks)):
            improves = np.abs(candidate_score) <= np.abs(score) + 1e-8
            candidate_left = candidate <= runtime.u_min + step_tol
            candidate_right = candidate >= runtime.u_max - step_tol
            valid_boundary = (candidate_left & (candidate_score <= 0.0)) | (
                candidate_right & (candidate_score >= 0.0)
            )
            bad = (~done) & ~(improves | valid_boundary)
            if not np.any(bad):
                break
            delta[bad] *= 0.5
            candidate[bad] = np.clip(u[bad] + delta[bad], runtime.u_min, runtime.u_max)
            candidate_score = runtime.score(y, candidate, batch_size=batch_size)

        moved = np.abs(candidate - u)
        stalled = (~done) & (moved <= step_tol)
        statuses[stalled] = "stalled"
        iterations[stalled] = iteration
        done |= stalled
        u = candidate
        iterations[~done] = iteration

    final_score = runtime.score(y, u, batch_size=batch_size)
    # Reclassify valid constrained boundary solutions reached on the final step.
    left = (u <= runtime.u_min + step_tol) & (final_score <= 0.0)
    right = (u >= runtime.u_max - step_tol) & (final_score >= 0.0)
    _, final_derivative = runtime.score_and_derivative(y, u, batch_size=batch_size)
    interior = (np.abs(final_score) <= score_tol) & (
        final_derivative < -float(min_curvature)
    )
    statuses[left] = "converged_left_boundary"
    statuses[right] = "converged_right_boundary"
    statuses[interior & ~left & ~right] = "converged_interior"
    converged = np.asarray([str(value).startswith("converged_") for value in statuses])
    return {
        "u": u,
        "score": final_score,
        "iterations": iterations,
        "status": statuses.tolist(),
        "converged": converged,
    }


def estimate_rows(
    *,
    pi_true: float,
    data_batch_seed: int,
    exact_u: np.ndarray,
    exact_status: list[str],
    smooth_u: np.ndarray,
    smooth_status: list[str],
    pilot_u: np.ndarray,
    pilot_status: list[str],
    method: str,
    solver: str,
    initialization: str,
    checkpoint: str,
    training_seed: int | None,
    estimate_u: np.ndarray,
    estimate_status: list[str],
    converged: np.ndarray | None = None,
    iterations: np.ndarray | None = None,
    residual: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(len(estimate_u)):
        pi_hat = float(stage1.sigmoid_np(estimate_u[i]))
        exact_pi = float(stage1.sigmoid_np(exact_u[i]))
        smooth_pi = float(stage1.sigmoid_np(smooth_u[i]))
        pilot_pi = float(stage1.sigmoid_np(pilot_u[i]))
        status = str(estimate_status[i])
        rows.append(
            {
                "pi_true": float(pi_true),
                "data_index": i,
                "data_batch_seed": int(data_batch_seed),
                "checkpoint": checkpoint,
                "training_seed": training_seed,
                "method": method,
                "solver": solver,
                "initialization": initialization,
                "estimate_u": float(estimate_u[i]),
                "estimate_pi": pi_hat,
                "estimate_status": status,
                "converged": bool(converged[i]) if converged is not None else True,
                "iterations": int(iterations[i]) if iterations is not None else 0,
                "score_residual": abs(float(residual[i])) if residual is not None else 0.0,
                "exact_mle_u": float(exact_u[i]),
                "exact_mle_pi": exact_pi,
                "exact_mle_status": exact_status[i],
                "smoothed_oracle_u": float(smooth_u[i]),
                "smoothed_oracle_pi": smooth_pi,
                "smoothed_oracle_status": smooth_status[i],
                "composite_pilot_u": float(pilot_u[i]),
                "composite_pilot_pi": pilot_pi,
                "composite_pilot_status": pilot_status[i],
                "error_true_pi": pi_hat - float(pi_true),
                "distance_exact_mle_pi": abs(pi_hat - exact_pi),
                "distance_smoothed_oracle_pi": abs(pi_hat - smooth_pi),
                "distance_composite_pilot_pi": abs(pi_hat - pilot_pi),
                "boundary_status_matches_exact": status_matches_exact(status, exact_status[i]),
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(
        ["method", "solver", "initialization", "checkpoint", "training_seed"],
        dropna=False,
    )
    per_checkpoint = grouped.agg(
        n=("estimate_pi", "size"),
        convergence_rate=("converged", "mean"),
        pi_rmse=("error_true_pi", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
        pi_mae=("error_true_pi", lambda x: float(np.mean(np.abs(np.asarray(x))))),
        rmse_to_exact_mle=(
            "distance_exact_mle_pi",
            lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))),
        ),
        mae_to_exact_mle=("distance_exact_mle_pi", "mean"),
        rmse_to_smoothed_oracle=(
            "distance_smoothed_oracle_pi",
            lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))),
        ),
        boundary_match_rate=("boundary_status_matches_exact", "mean"),
        mean_iterations=("iterations", "mean"),
        mean_score_residual=("score_residual", "mean"),
    ).reset_index()
    return per_checkpoint


def aggregate_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Average checkpoint-level metrics without treating repeated rows as seeds."""
    return frame.groupby(group_columns, dropna=False).agg(
        n_checkpoints=("checkpoint", "nunique"),
        convergence_rate=("convergence_rate", "mean"),
        pi_rmse_mean=("pi_rmse", "mean"),
        pi_rmse_sd=("pi_rmse", "std"),
        rmse_to_exact_mle_mean=("rmse_to_exact_mle", "mean"),
        rmse_to_exact_mle_sd=("rmse_to_exact_mle", "std"),
        rmse_to_smoothed_oracle_mean=("rmse_to_smoothed_oracle", "mean"),
        boundary_match_rate=("boundary_match_rate", "mean"),
        mean_iterations=("mean_iterations", "mean"),
        mean_score_residual=("mean_score_residual", "mean"),
    ).reset_index()


def write_report(
    path: Path,
    config: dict[str, object],
    formal: pd.DataFrame,
    by_pi: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    columns = [
        "method",
        "solver",
        "initialization",
        "convergence_rate",
        "pi_rmse_mean",
        "rmse_to_exact_mle_mean",
        "rmse_to_smoothed_oracle_mean",
        "boundary_match_rate",
        "mean_iterations",
    ]
    pi_columns = [
        "pi_true",
        "method",
        "solver",
        "initialization",
        "convergence_rate",
        "pi_rmse_mean",
        "rmse_to_exact_mle_mean",
    ]
    lines = [
        "# Model 1 Mode A MLE Report",
        "",
        "## Setting",
        "",
        f"- Stage-1 checkpoints: `{len(config['run_dirs_resolved'])}`",
        f"- Methods: `{config['methods']}`",
        f"- True pi values: `{config['pi_true_values']}`",
        f"- Datasets per pi: `{config['n_data']}`",
        f"- Root grid size: `{config['root_grid_size']}`",
        f"- FSM proposal sigma_q: `{config['sigma_q']}`",
        "- Exact likelihood and smoothed-oracle fields are evaluation-only references.",
        "- The cheap pilot is the equally weighted marginal composite score.",
        "",
        "## Overall Results",
        "",
        formal[columns].to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Results by True pi",
        "",
        by_pi[pi_columns].to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Paired Radial vs Linear Comparison",
        "",
        paired.to_markdown(index=False, floatfmt=".5g"),
        "",
        "## Interpretation Rules",
        "",
        "- `score_grid` is the global constrained mode of the integrated learned score field.",
        "- `safeguarded_newton` tests whether that field is usable as an iterative Mode A optimizer.",
        "- Boundary optima are valid constrained MLEs and are tracked separately from failures.",
        "- Compare learned methods first with the smoothed oracle, then with the exact MLE.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def paired_radial_linear_summary(per_checkpoint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("pi_rmse", "rmse_to_exact_mle", "rmse_to_smoothed_oracle")
    for (solver, initialization), subset in per_checkpoint.groupby(
        ["solver", "initialization"], dropna=False
    ):
        if not {"linear", "radial"}.issubset(set(subset["method"])):
            continue
        paired_subset = subset[subset["method"].isin(["linear", "radial"])]
        for metric in metrics:
            pivot = paired_subset.pivot(
                index="training_seed", columns="method", values=metric
            ).dropna(subset=["linear", "radial"])
            linear = pivot["linear"].to_numpy(dtype=np.float64)
            radial = pivot["radial"].to_numpy(dtype=np.float64)
            if linear.size < 2:
                t_stat, p_value = np.nan, np.nan
            else:
                test = ttest_rel(linear, radial)
                t_stat, p_value = float(test.statistic), float(test.pvalue)
            linear_mean = float(np.mean(linear))
            radial_mean = float(np.mean(radial))
            rows.append(
                {
                    "solver": solver,
                    "initialization": initialization,
                    "metric": metric,
                    "n_training_seeds": int(linear.size),
                    "linear_mean": linear_mean,
                    "radial_mean": radial_mean,
                    "radial_reduction_pct": 100.0 * (linear_mean - radial_mean) / linear_mean,
                    "radial_wins": int(np.sum(radial < linear)),
                    "paired_t": t_stat,
                    "paired_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    package_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = parse_method_list(args.methods)
    run_dirs = resolve_run_dirs(args, package_dir)
    runtimes = [
        AmortizedScoreRuntime(run_dir, method, device=args.device)
        for run_dir in run_dirs
        for method in methods
    ]
    reference = runtimes[0]
    for runtime in runtimes:
        compatibility = (
            runtime.n_blocks == reference.n_blocks
            and runtime.block_size == reference.block_size
            and np.isclose(runtime.tau, reference.tau)
            and np.isclose(runtime.sigma_q, reference.sigma_q)
            and np.isclose(runtime.u_min, reference.u_min)
            and np.isclose(runtime.u_max, reference.u_max)
        )
        if not compatibility:
            raise ValueError(
                f"Incompatible checkpoint setting: {runtime.run_dir.name}/{runtime.method}"
            )
        checks = runtime.run_sanity_checks(seed=int(args.seed) + int(runtime.config["seed"]) % 1000)
        if not checks["passed"]:
            raise RuntimeError(f"Runtime sanity failed for {runtime.run_dir.name}/{runtime.method}")
        print(f"[OK runtime] {runtime.run_dir.name}/{runtime.method}, best={runtime.best_source}")

    pi_true_values = parse_float_list(args.pi_true_values)
    u_grid = np.linspace(reference.u_min, reference.u_max, int(args.root_grid_size), dtype=np.float64)
    pi_grid = stage1.sigmoid_np(u_grid)
    config = {
        **vars(args),
        "run_dirs_resolved": [str(path) for path in run_dirs],
        "device_resolved": reference.device,
        "u_grid_min": float(u_grid[0]),
        "u_grid_max": float(u_grid[-1]),
        "pi_grid_min": float(pi_grid[0]),
        "pi_grid_max": float(pi_grid[-1]),
        "sigma_q": float(reference.sigma_q),
        "tau": float(reference.tau),
        "n_blocks": int(reference.n_blocks),
        "block_size": int(reference.block_size),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_rows: list[dict[str, object]] = []
    for pi_index, pi_true in enumerate(pi_true_values):
        data_batch_seed = int(args.seed) + 100_000 * pi_index
        rng = np.random.default_rng(data_batch_seed)
        true_u = float(stage1.logit_np(pi_true))
        y = stage1.simulate_mean_shift(
            rng,
            np.full(int(args.n_data), true_u, dtype=np.float64),
            n_blocks=reference.n_blocks,
            block_size=reference.block_size,
            tau=reference.tau,
        )
        full_lr = stage1.full_block_log_ratio(y, reference.tau)
        exact_u, exact_status = exact_constrained_mle(
            full_lr,
            reference.u_min,
            reference.u_max,
            float(args.exact_xatol),
        )

        smooth_score = np.empty((int(args.n_data), len(u_grid)), dtype=np.float64)
        for j, u_value in enumerate(u_grid):
            smooth_score[:, j] = mode_a_reference.smoothed_fsm_score_from_full_lr(
                full_lr=full_lr,
                u0=float(u_value),
                sigma_q=reference.sigma_q,
                grid_size=int(args.smooth_grid_size),
                grid_radius=float(args.smooth_grid_radius),
                chunk_size=int(args.smooth_chunk_size),
            )
        smooth_u, smooth_status, _, smooth_residual = constrained_modes_from_score_grid(
            smooth_score, u_grid
        )

        pilot_score = composite_pilot_score_grid(y, u_grid, reference.tau)
        pilot_u, pilot_status, _, pilot_residual = constrained_modes_from_score_grid(
            pilot_score, u_grid
        )
        all_rows.extend(
            estimate_rows(
                pi_true=pi_true,
                data_batch_seed=data_batch_seed,
                exact_u=exact_u,
                exact_status=exact_status,
                smooth_u=smooth_u,
                smooth_status=smooth_status,
                pilot_u=pilot_u,
                pilot_status=pilot_status,
                method="exact MLE",
                solver="exact_likelihood",
                initialization="none",
                checkpoint="reference",
                training_seed=None,
                estimate_u=exact_u,
                estimate_status=exact_status,
            )
        )
        all_rows.extend(
            estimate_rows(
                pi_true=pi_true,
                data_batch_seed=data_batch_seed,
                exact_u=exact_u,
                exact_status=exact_status,
                smooth_u=smooth_u,
                smooth_status=smooth_status,
                pilot_u=pilot_u,
                pilot_status=pilot_status,
                method="smoothed oracle",
                solver="score_grid",
                initialization="none",
                checkpoint="reference",
                training_seed=None,
                estimate_u=smooth_u,
                estimate_status=smooth_status,
                residual=smooth_residual,
            )
        )
        all_rows.extend(
            estimate_rows(
                pi_true=pi_true,
                data_batch_seed=data_batch_seed,
                exact_u=exact_u,
                exact_status=exact_status,
                smooth_u=smooth_u,
                smooth_status=smooth_status,
                pilot_u=pilot_u,
                pilot_status=pilot_status,
                method="marginal composite pilot",
                solver="score_grid",
                initialization="none",
                checkpoint="reference",
                training_seed=None,
                estimate_u=pilot_u,
                estimate_status=pilot_status,
                residual=pilot_residual,
            )
        )

        for runtime in runtimes:
            learned_score = runtime.score_grid(y, u_grid, batch_size=int(args.batch_size))
            learned_u, learned_status, _, learned_residual = constrained_modes_from_score_grid(
                learned_score, u_grid
            )
            training_seed = int(runtime.config["seed"])
            checkpoint = runtime.run_dir.name
            all_rows.extend(
                estimate_rows(
                    pi_true=pi_true,
                    data_batch_seed=data_batch_seed,
                    exact_u=exact_u,
                    exact_status=exact_status,
                    smooth_u=smooth_u,
                    smooth_status=smooth_status,
                    pilot_u=pilot_u,
                    pilot_status=pilot_status,
                    method=runtime.method,
                    solver="score_grid",
                    initialization="none",
                    checkpoint=checkpoint,
                    training_seed=training_seed,
                    estimate_u=learned_u,
                    estimate_status=learned_status,
                    residual=learned_residual,
                )
            )

            for initialization, initial_u in (
                ("composite_pilot", pilot_u),
                ("fixed_pi_0.3", np.full(int(args.n_data), stage1.logit_np(0.3))),
            ):
                newton = safeguarded_newton(
                    runtime,
                    y,
                    initial_u,
                    max_iters=int(args.newton_max_iters),
                    score_tol=float(args.newton_score_tol),
                    step_tol=float(args.newton_step_tol),
                    min_curvature=float(args.newton_min_curvature),
                    max_step=float(args.newton_max_step),
                    fallback_lr=float(args.newton_fallback_lr),
                    max_backtracks=int(args.newton_max_backtracks),
                    batch_size=int(args.batch_size),
                )
                all_rows.extend(
                    estimate_rows(
                        pi_true=pi_true,
                        data_batch_seed=data_batch_seed,
                        exact_u=exact_u,
                        exact_status=exact_status,
                        smooth_u=smooth_u,
                        smooth_status=smooth_status,
                        pilot_u=pilot_u,
                        pilot_status=pilot_status,
                        method=runtime.method,
                        solver="safeguarded_newton",
                        initialization=initialization,
                        checkpoint=checkpoint,
                        training_seed=training_seed,
                        estimate_u=np.asarray(newton["u"]),
                        estimate_status=list(newton["status"]),
                        converged=np.asarray(newton["converged"]),
                        iterations=np.asarray(newton["iterations"]),
                        residual=np.asarray(newton["score"]),
                    )
                )
        print(f"completed pi_true={pi_true:.3g}")

    write_csv(out_dir / "mle_by_dataset.csv", all_rows)
    per_checkpoint = summarize(all_rows)
    per_checkpoint.to_csv(out_dir / "mle_summary_by_checkpoint.csv", index=False)
    formal = aggregate_summary(
        per_checkpoint,
        ["method", "solver", "initialization"],
    )
    formal.to_csv(out_dir / "mle_summary.csv", index=False)

    raw_frame = pd.DataFrame(all_rows)
    by_pi_checkpoint_rows: list[pd.DataFrame] = []
    for pi_true, subset in raw_frame.groupby("pi_true", sort=True):
        summarized = summarize(subset.to_dict(orient="records"))
        summarized.insert(0, "pi_true", float(pi_true))
        by_pi_checkpoint_rows.append(summarized)
    by_pi_checkpoint = pd.concat(by_pi_checkpoint_rows, ignore_index=True)
    by_pi_checkpoint.to_csv(out_dir / "mle_summary_by_pi_checkpoint.csv", index=False)
    by_pi = aggregate_summary(
        by_pi_checkpoint,
        ["pi_true", "method", "solver", "initialization"],
    )
    by_pi.to_csv(out_dir / "mle_summary_by_pi.csv", index=False)
    paired = paired_radial_linear_summary(per_checkpoint)
    paired.to_csv(out_dir / "radial_vs_linear_paired.csv", index=False)
    write_report(out_dir / "MODE_A_MLE_REPORT.md", config, formal, by_pi, paired)

    print("\n==== Mode A MLE summary ====")
    print(formal.to_string(index=False))
    print("\nSaved to", out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", default="")
    parser.add_argument("--run-pattern", default="model1_amortized_seed202607*")
    parser.add_argument("--methods", default="linear,radial")
    parser.add_argument("--pi-true-values", default="0.10,0.30,0.50,0.65")
    parser.add_argument("--n-data", type=int, default=20)
    parser.add_argument("--root-grid-size", type=int, default=201)
    parser.add_argument("--smooth-grid-size", type=int, default=121)
    parser.add_argument("--smooth-grid-radius", type=float, default=5.0)
    parser.add_argument("--smooth-chunk-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--exact-xatol", type=float, default=1e-10)
    parser.add_argument("--newton-max-iters", type=int, default=30)
    parser.add_argument("--newton-score-tol", type=float, default=1e-3)
    parser.add_argument("--newton-step-tol", type=float, default=1e-6)
    parser.add_argument("--newton-min-curvature", type=float, default=1e-3)
    parser.add_argument("--newton-max-step", type=float, default=0.5)
    parser.add_argument("--newton-fallback-lr", type=float, default=0.05)
    parser.add_argument("--newton-max-backtracks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", default="runs/model1_mode_a_mle_smoke")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
