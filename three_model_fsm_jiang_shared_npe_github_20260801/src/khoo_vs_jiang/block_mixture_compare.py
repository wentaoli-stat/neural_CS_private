"""Paired full-dataset comparison for the original Model 1/2 experiment.

This evaluator does not retrain or modify either method.  It loads the frozen
formal linear/radial FSM checkpoints from the original project and the frozen
paper-native Jiang checkpoints produced by :mod:`block_mixture_jiang`.

All score errors are evaluated on the same simulated *full datasets* and in
the same logit-parameter coordinate.  Round-1 roots are compared on the same
100 datasets used by the Jiang repeated-data evaluation.  Jiang Round 2 is
reported only as a data-local diagnostic because its proposal depends on the
particular observed dataset used to train that round.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .block_mixture_jiang import (
    BlockMixtureConfig,
    exact_mle,
    exact_score_p,
    simulate_raw_blocks,
)
from .jiang_official import debiased_scores, load_fitted
from .upstreams import load_upstreams


TRAINING_SEEDS = (20260709, 20260710, 20260711, 20260712, 20260713)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sigmoid(u: np.ndarray | float) -> np.ndarray:
    value = np.asarray(u, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-value))


def _logit(p: float) -> float:
    return math.log(float(p)) - math.log1p(-float(p))


def _config(model: str) -> BlockMixtureConfig:
    if model == "model1":
        return BlockMixtureConfig(model, 20, 0.5, 0.05, 0.70, 0.30, 20)
    if model == "model2":
        return BlockMixtureConfig(model, 20, 1.0, 0.05, 0.70, 0.30, 40)
    raise ValueError(model)


def _load_runtime_class(original_root: Path, model: str) -> type:
    package = original_root / f"current_{model}_fsm_method_package"
    if not package.is_dir():
        raise FileNotFoundError(package)
    text = str(package.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)
    module = importlib.import_module(f"{model}_amortized_score_runtime")
    return module.AmortizedScoreRuntime


def _fsm_run_dirs(original_root: Path, model: str) -> list[Path]:
    del original_root  # Kept in the signature for compatibility with older scripts.
    fallback = Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints" / "ours"
    checkpoint_root = Path(os.environ.get("FSM_CHECKPOINT_ROOT", fallback)).expanduser()
    base = checkpoint_root / model
    pattern = "seed{}"
    paths = [base / pattern.format(seed) for seed in TRAINING_SEEDS]
    for path in paths:
        for name in ("config.json", "feature_stats.npz", "model_linear.pt", "model_radial.pt"):
            if not (path / name).is_file():
                raise FileNotFoundError(path / name)
    return paths


def _simulate_full_datasets(
    model: str,
    p: float,
    n_datasets: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    config = _config(model)
    blocks = simulate_raw_blocks(
        np.random.default_rng(int(seed)),
        float(p),
        config,
        n=int(n_datasets) * config.n_observations,
    )
    container = blocks.reshape(
        int(n_datasets), config.n_observations, 2, config.block_size // 2
    )
    raw = container.reshape(int(n_datasets), config.n_observations, config.block_size)
    return raw.astype(np.float64), container.astype(np.float32)


def _simulate_repeated_full_datasets(
    model: str,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    config = _config(model)
    rows = []
    for replicate in range(int(replicates)):
        replicate_seed = int(seed) + 10_000 * replicate
        block = simulate_raw_blocks(
            np.random.default_rng(replicate_seed),
            config.p_true,
            config,
            n=config.n_observations,
        )
        rows.append(block)
    container = np.stack(rows, axis=0)
    raw = container.reshape(int(replicates), config.n_observations, config.block_size)
    return raw.astype(np.float64), container.astype(np.float32)


def _exact_full_u_score(raw: np.ndarray, p: float, model: str) -> np.ndarray:
    config = _config(model)
    n_datasets = int(raw.shape[0])
    block_score_p = exact_score_p(raw.reshape(-1, config.block_size), p, config)
    score_p = block_score_p.reshape(n_datasets, config.n_observations).sum(axis=1)
    return float(p) * (1.0 - float(p)) * score_p


def _jiang_full_u_score(fitted: Any, container: np.ndarray, p: float) -> np.ndarray:
    n_datasets, n_blocks = container.shape[:2]
    single_p = debiased_scores(
        fitted,
        float(p),
        container.reshape(-1, *container.shape[2:]),
        device="auto",
        physical_coordinate=True,
    ).reshape(n_datasets, n_blocks)
    return float(p) * (1.0 - float(p)) * single_p.sum(axis=1)


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    variance = float(np.var(truth))
    error = prediction - truth
    return {
        "mse": float(np.mean(np.square(error))),
        "std_mse": float(np.mean(np.square(error)) / max(variance, 1e-15)),
        "corr": float(np.corrcoef(prediction, truth)[0, 1]),
        "mean_error": float(np.mean(error)),
    }


def _cumulative_trapezoid(score: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    increments = 0.5 * (score[:, 1:] + score[:, :-1]) * np.diff(u_grid)[None, :]
    return np.concatenate(
        (np.zeros((score.shape[0], 1), dtype=np.float64), np.cumsum(increments, axis=1)),
        axis=1,
    )


def constrained_modes_from_score_grid(
    score: np.ndarray,
    u_grid: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Existing Mode-A global-potential root rule, factored for paired use."""
    score = np.asarray(score, dtype=np.float64)
    u_grid = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    potential = _cumulative_trapezoid(score, u_grid)
    roots = np.empty(score.shape[0], dtype=np.float64)
    statuses: list[str] = []
    for index, (row, row_potential) in enumerate(zip(score, potential, strict=True)):
        candidates: list[tuple[float, float, float, str]] = [
            (float(u_grid[0]), float(row_potential[0]), abs(float(row[0])), "left_boundary"),
            (float(u_grid[-1]), float(row_potential[-1]), abs(float(row[-1])), "right_boundary"),
        ]
        crossings = np.flatnonzero(row[:-1] * row[1:] < 0.0)
        for location in crossings:
            left_s, right_s = float(row[location]), float(row[location + 1])
            weight = float(np.clip(-left_s / (right_s - left_s), 0.0, 1.0))
            root = float(u_grid[location] + weight * (u_grid[location + 1] - u_grid[location]))
            root_potential = float(
                row_potential[location]
                + 0.5 * left_s * (root - float(u_grid[location]))
            )
            candidates.append((root, root_potential, 0.0, "interior"))
        best = max(candidates, key=lambda item: (item[1], -item[2], item[3] == "interior"))
        roots[index] = best[0]
        statuses.append(best[3] if len(crossings) <= 1 else "multiple_selected_global")
    return roots, statuses


def _root_metrics(
    estimate: np.ndarray,
    exact: np.ndarray,
    p_true: float,
    statuses: list[str],
) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    return {
        "bias_to_true": float(np.mean(estimate - float(p_true))),
        "rmse_to_true": float(np.sqrt(np.mean(np.square(estimate - float(p_true))))),
        "rmse_to_exact_mle": float(np.sqrt(np.mean(np.square(estimate - exact)))),
        "mae_to_exact_mle": float(np.mean(np.abs(estimate - exact))),
        "boundary_rate": float(np.mean(["boundary" in value for value in statuses])),
    }


def compare_model(
    model: str,
    original_root: Path,
    jiang_dir: Path,
    *,
    diagnostic_n: int,
    replicates: int,
    diagnostic_seed: int,
    repeated_seed: int,
    grid_size: int,
) -> dict[str, Any]:
    config = _config(model)
    runtime_class = _load_runtime_class(original_root, model)
    run_dirs = _fsm_run_dirs(original_root, model)
    runtimes = {
        method: [runtime_class(path, method, device="auto") for path in run_dirs]
        for method in ("linear", "radial")
    }
    upstreams = load_upstreams(require_frozen_jiang_commit=True)
    jiang_round1 = load_fitted(jiang_dir / "models/jiang_round1.pt", upstreams, device="auto")
    jiang_round2 = load_fitted(jiang_dir / "models/jiang_round2.pt", upstreams, device="auto")

    score_rows: list[dict[str, Any]] = []
    for p_index, p in enumerate((0.10, 0.30, 0.50, 0.65)):
        raw, container = _simulate_full_datasets(
            model, p, diagnostic_n, int(diagnostic_seed) + 1_000_000 * p_index
        )
        truth = _exact_full_u_score(raw, p, model)
        for round_name, fitted in (("jiang_round1", jiang_round1), ("jiang_round2_local", jiang_round2)):
            score_rows.append(
                {
                    "model": model,
                    "p": p,
                    "method": round_name,
                    "training_seed": int(jiang_round1.diagnostics["seed"]),
                    **_metrics(_jiang_full_u_score(fitted, container, p), truth),
                }
            )
        u = _logit(p)
        for method, method_runtimes in runtimes.items():
            for seed, runtime in zip(TRAINING_SEEDS, method_runtimes, strict=True):
                score_rows.append(
                    {
                        "model": model,
                        "p": p,
                        "method": f"fsm_{method}",
                        "training_seed": seed,
                        **_metrics(runtime.score(raw, u), truth),
                    }
                )

    raw_repeated, container_repeated = _simulate_repeated_full_datasets(
        model, replicates, repeated_seed
    )
    exact = np.asarray(
        [exact_mle(container_repeated[index], config) for index in range(replicates)],
        dtype=np.float64,
    )
    repeated_path = jiang_dir / "round1_repeated.json"
    repeated = json.loads(repeated_path.read_text(encoding="utf-8"))
    if int(repeated["summary"]["replicates"]) != int(replicates):
        raise ValueError("Jiang repeated result uses a different replicate count")
    jiang_estimate = np.asarray([row["p_hat"] for row in repeated["rows"]], dtype=np.float64)
    jiang_exact = np.asarray([row["exact_mle"] for row in repeated["rows"]], dtype=np.float64)
    # Checkpoints were produced under a different SciPy build; bounded scalar
    # optimization can differ by a few 1e-8 at the same optimum.
    if not np.allclose(jiang_exact, exact, rtol=0.0, atol=1e-6):
        raise ValueError("paired simulated datasets do not reproduce Jiang's exact MLE rows")
    jiang_error = np.abs(jiang_estimate - exact)

    u_grid = np.linspace(_logit(config.p_min), _logit(config.p_max), int(grid_size))
    root_rows: list[dict[str, Any]] = []
    for method, method_runtimes in runtimes.items():
        for seed, runtime in zip(TRAINING_SEEDS, method_runtimes, strict=True):
            score_grid = runtime.score_grid(
                raw_repeated,
                u_grid,
                batch_size=min(100, replicates),
                grid_chunk_size=4,
            )
            root_u, statuses = constrained_modes_from_score_grid(score_grid, u_grid)
            estimate = _sigmoid(root_u)
            root_rows.append(
                {
                    "model": model,
                    "method": f"fsm_{method}",
                    "training_seed": seed,
                    **_root_metrics(estimate, exact, config.p_true, statuses),
                    "paired_win_rate_abs_error_vs_jiang_round1": float(
                        np.mean(np.abs(estimate - exact) < jiang_error)
                    ),
                }
            )

    jiang_status = [
        "boundary" if bool(row["boundary"]) else "interior" for row in repeated["rows"]
    ]
    root_rows.append(
        {
            "model": model,
            "method": "jiang_round1",
            "training_seed": int(jiang_round1.diagnostics["seed"]),
            **_root_metrics(jiang_estimate, exact, config.p_true, jiang_status),
            "convergence_rate": float(repeated["summary"]["convergence_rate"]),
        }
    )

    observed_container = np.load(jiang_dir / "observed_raw_iid_blocks.npy").reshape(
        1, config.n_observations, 2, config.block_size // 2
    )
    observed_raw = observed_container.reshape(
        1, config.n_observations, config.block_size
    ).astype(np.float64)
    observed_exact = exact_mle(observed_container[0], config)
    fixed_fsm: dict[str, list[dict[str, float]]] = {}
    for method, method_runtimes in runtimes.items():
        rows = []
        for seed, runtime in zip(TRAINING_SEEDS, method_runtimes, strict=True):
            score_grid = runtime.score_grid(
                observed_raw,
                u_grid,
                batch_size=1,
                grid_chunk_size=4,
            )
            root_u, _ = constrained_modes_from_score_grid(score_grid, u_grid)
            estimate = float(_sigmoid(root_u)[0])
            rows.append(
                {
                    "training_seed": seed,
                    "p_hat": estimate,
                    "absolute_error_to_exact_mle": abs(estimate - observed_exact),
                }
            )
        fixed_fsm[f"fsm_{method}"] = rows
    jiang_summary = json.loads((jiang_dir / "summary.json").read_text(encoding="utf-8"))

    config_rows = [json.loads((path / "config.json").read_text(encoding="utf-8")) for path in run_dirs]
    return {
        "model": model,
        "experiment": {
            "block_size": config.block_size,
            "tau": config.tau,
            "n_observations": config.n_observations,
            "p_true": config.p_true,
            "diagnostic_n_full_datasets_per_p": int(diagnostic_n),
            "root_replicates": int(replicates),
            "root_grid_size": int(grid_size),
        },
        "score_rows": score_rows,
        "root_rows": root_rows,
        "fixed_observed_dataset": {
            "exact_mle": observed_exact,
            **fixed_fsm,
            "jiang_round1": jiang_summary["round1"],
            "jiang_round2_local": jiang_summary["round2"],
            "interpretation": "one data-dependent local Round-2 fit; descriptive, not repeated-data evidence",
        },
        "budgets": {
            "fsm_n_train_full_datasets": [int(row["n_train"]) for row in config_rows],
            "fsm_raw_training_blocks": [
                int(row["n_train"]) * int(row["n_blocks"]) for row in config_rows
            ],
            "jiang_config": json.loads((jiang_dir / "config.json").read_text(encoding="utf-8")),
        },
        "scope": {
            "jiang_round1": "global proposal; valid matched full-range and repeated-data comparison",
            "jiang_round2_local": "proposal depends on one observed dataset; score diagnostic only, not repeated-data coverage",
            "posterior": "not compared; Jiang confidence sets are not posterior distributions",
        },
    }


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    score_rows = result["score_rows"]
    methods = sorted({row["method"] for row in score_rows})
    score_summary = {}
    for method in methods:
        selected = [row for row in score_rows if row["method"] == method]
        score_summary[method] = {
            "mean_std_mse": float(np.mean([row["std_mse"] for row in selected])),
            "mean_corr": float(np.mean([row["corr"] for row in selected])),
            "n_rows": len(selected),
        }
    root_summary = {}
    for method in sorted({row["method"] for row in result["root_rows"]}):
        selected = [row for row in result["root_rows"] if row["method"] == method]
        root_summary[method] = {
            "mean_rmse_to_true": float(np.mean([row["rmse_to_true"] for row in selected])),
            "mean_rmse_to_exact_mle": float(
                np.mean([row["rmse_to_exact_mle"] for row in selected])
            ),
            "mean_boundary_rate": float(np.mean([row["boundary_rate"] for row in selected])),
            "mean_paired_win_rate_vs_jiang": (
                None
                if method == "jiang_round1"
                else float(
                    np.mean(
                        [row["paired_win_rate_abs_error_vs_jiang_round1"] for row in selected]
                    )
                )
            ),
            "n_training_seeds": len(selected),
        }
    fixed = result["fixed_observed_dataset"]
    fixed_summary = {
        method: {
            "mean_p_hat": float(np.mean([row["p_hat"] for row in fixed[method]])),
            "mean_absolute_error_to_exact_mle": float(
                np.mean([row["absolute_error_to_exact_mle"] for row in fixed[method]])
            ),
            "n_training_seeds": len(fixed[method]),
        }
        for method in ("fsm_linear", "fsm_radial")
    }
    for method in ("jiang_round1", "jiang_round2_local"):
        fixed_summary[method] = {
            "p_hat": float(fixed[method]["p_hat"]),
            "absolute_error_to_exact_mle": float(
                fixed[method]["absolute_error_to_exact_mle"]
            ),
            "n_training_seeds": 1,
        }
    return {"score": score_summary, "root": root_summary, "fixed_observed": fixed_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--jiang-model1-dir", required=True)
    parser.add_argument("--jiang-model2-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diagnostic-n", type=int, default=1_000)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--diagnostic-seed", type=int, default=20260801)
    parser.add_argument("--repeated-seed", type=int, default=20260734)
    parser.add_argument("--grid-size", type=int, default=301)
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    original_root = Path(args.original_root).expanduser().resolve()
    combined = {}
    for model, jiang_path in (
        ("model1", args.jiang_model1_dir),
        ("model2", args.jiang_model2_dir),
    ):
        result = compare_model(
            model,
            original_root,
            Path(jiang_path).expanduser().resolve(),
            diagnostic_n=int(args.diagnostic_n),
            replicates=int(args.replicates),
            diagnostic_seed=int(args.diagnostic_seed),
            repeated_seed=int(args.repeated_seed),
            grid_size=int(args.grid_size),
        )
        result["summary"] = _summarize(result)
        _write_json(output / f"{model}_paired.json", result)
        combined[model] = result["summary"]
        print(model, json.dumps(result["summary"], indent=2))
    _write_json(output / "summary.json", combined)


if __name__ == "__main__":
    main()
