"""Matched full-dataset Stage-2 NPE experiment for block Models 1 and 2.

The comparison is deliberately restricted to globally amortized Stage-1
estimators.  Jiang contributes the authors' observation-independent Round-1
single-block score.  That score is summed over every iid block in a complete
dataset before it is passed to NPE.  The original frozen linear and nonlinear
FSM fields already return a complete-dataset score.

Every learned-score arm receives exactly the same

* Stage-2 simulated parameters and complete datasets,
* data-only complete-dataset pilot,
* context definition ``(pilot_u, full_u_score_at_pilot)``,
* NPE architecture, optimizer settings, and random seed, and
* 100 (by default) held-out complete datasets.

Jiang Round 2 is intentionally excluded: its proposal is conditioned on one
particular observed dataset, so it is not a globally amortized feature map for
new test datasets.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

from .block_mixture_compare import TRAINING_SEEDS, _config, _fsm_run_dirs
from .block_mixture_jiang import (
    BlockMixtureConfig,
    block_log_likelihood_ratio,
    simulate_raw_blocks,
)
from .jiang_official import (
    JiangFitted,
    debiased_scores_tensor,
    load_fitted as load_jiang,
    resolve_device,
)
from .nonlinear_emission_stage2 import _stage2_upstream
from .upstreams import load_upstreams


PILOT_LABEL = "shared data-only pilot + NPE"
JIANG_LABEL = "Jiang official Round 1 summed-block score + shared NPE"
LINEAR_PREFIX = "ours linear full-dataset score"
NONLINEAR_PREFIX = "ours nonlinear-gate full-dataset score"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _logit_array(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return np.log(value) - np.log1p(-value)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_original_root() -> Path:
    return _workspace_root() / "experiments" / "block_models"


def default_jiang_dir(model: str) -> Path:
    return _workspace_root() / "artifacts" / "checkpoints" / "jiang" / model


def _parse_seeds(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("at least one FSM training seed is required")
    unknown = sorted(set(values).difference(TRAINING_SEEDS))
    if unknown:
        raise ValueError(
            f"unknown FSM seeds {unknown}; frozen seeds are {list(TRAINING_SEEDS)}"
        )
    if len(values) != len(set(values)):
        raise ValueError("FSM seeds must not be duplicated")
    return values


def _load_legacy_modules(original_root: Path, model: str) -> tuple[type, Any]:
    package = Path(original_root) / f"current_{model}_fsm_method_package"
    if not package.is_dir():
        raise FileNotFoundError(package)
    package_text = str(package.resolve())
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    runtime_module = importlib.import_module(f"{model}_amortized_score_runtime")
    runner_name = (
        "run_model1_pilot_score_sbi_npe"
        if model == "model1"
        else "run_model2_pilot_score_sbi_npe"
    )
    runner = importlib.import_module(runner_name)
    return runtime_module.AmortizedScoreRuntime, runner


def _selected_fsm_run_dirs(
    original_root: Path,
    model: str,
    seeds: list[int],
) -> dict[int, Path]:
    all_paths = dict(
        zip(
            TRAINING_SEEDS,
            _fsm_run_dirs(Path(original_root), model),
            strict=True,
        )
    )
    return {seed: all_paths[seed] for seed in seeds}


def _simulate_full_datasets(
    rng: np.random.Generator,
    p: np.ndarray,
    config: BlockMixtureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one complete iid-block dataset for every parameter row."""
    probability = np.asarray(p, dtype=np.float64).reshape(-1)
    repeated = np.repeat(probability, config.n_observations)
    container = simulate_raw_blocks(rng, repeated, config).reshape(
        probability.size,
        config.n_observations,
        2,
        config.block_size // 2,
    )
    raw = container.reshape(
        probability.size,
        config.n_observations,
        config.block_size,
    )
    return raw.astype(np.float64), container.astype(np.float32)


def stratified_prior(
    rng: np.random.Generator,
    n: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    if int(n) < 1:
        raise ValueError("n must be positive")
    strata = (np.arange(int(n), dtype=np.float64) + rng.uniform(size=int(n))) / int(n)
    rng.shuffle(strata)
    return lower + (upper - lower) * strata


def exact_posterior(
    y: np.ndarray,
    config: BlockMixtureConfig,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Evaluation-only exact posterior under the uniform Stage-2 prior."""
    axis = np.linspace(config.p_min, config.p_max, int(grid_size), dtype=np.float64)
    log_ratio = block_log_likelihood_ratio(y, config).reshape(-1)
    log_weight = np.sum(
        np.logaddexp(
            np.log1p(-axis)[:, None],
            np.log(axis)[:, None] + log_ratio[None, :],
        ),
        axis=1,
    )
    log_weight -= float(np.max(log_weight))
    weight = np.exp(log_weight)
    weight /= float(np.sum(weight))
    cdf = np.cumsum(weight)
    mean = float(np.sum(weight * axis))
    sd = float(np.sqrt(np.sum(weight * np.square(axis - mean))))
    return axis, cdf, {
        "posterior_mean": mean,
        "posterior_sd": sd,
        "q05": float(np.interp(0.05, cdf, axis)),
        "q50": float(np.interp(0.50, cdf, axis)),
        "q95": float(np.interp(0.95, cdf, axis)),
    }


def jiang_full_u_score_at_pilot(
    fitted: JiangFitted,
    container: np.ndarray,
    pilot_u: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Sum official debiased single-block scores over each complete dataset."""
    device = resolve_device(device)
    container = np.asarray(container, dtype=np.float32)
    pilot_u = np.asarray(pilot_u, dtype=np.float64).reshape(-1)
    if container.ndim != 4 or container.shape[0] != pilot_u.size:
        raise ValueError("container and pilot_u batch sizes do not agree")
    n_datasets, n_blocks = container.shape[:2]
    pilot_p = _sigmoid(pilot_u)
    block_p = np.repeat(pilot_p, n_blocks)
    flat = np.ascontiguousarray(container.reshape(n_datasets * n_blocks, -1))

    output: list[np.ndarray] = []
    fitted.score_model.eval()
    fitted.debias_model.eval()
    for start in range(0, flat.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), flat.shape[0])
        x_tensor = torch.as_tensor(
            flat[start:stop], dtype=torch.float32, device=device
        )
        p_tensor = torch.as_tensor(
            block_p[start:stop, None], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            score_v = debiased_scores_tensor(fitted, p_tensor, x_tensor).reshape(-1)
            score_p = score_v * float(fitted.scale_theta)
        output.append(score_p.cpu().numpy().astype(np.float64))

    single_p = np.concatenate(output).reshape(n_datasets, n_blocks)
    full_p = single_p.sum(axis=1)
    full_u = pilot_p * (1.0 - pilot_p) * full_p
    if not np.all(np.isfinite(full_u)):
        raise RuntimeError("non-finite Jiang full-dataset score")
    return full_u.astype(np.float32)


def _pilot_args(args: argparse.Namespace, device: str) -> argparse.Namespace:
    return argparse.Namespace(
        pi_prior_min=float(args.p_prior_min),
        pi_prior_max=float(args.p_prior_max),
        pilot_grid_size=int(args.pilot_grid_size),
        pilot_device=device,
        pilot_batch_size=int(args.pilot_batch_size),
        pilot_grid_chunk_size=int(args.pilot_grid_chunk_size),
        pilot_progress_every=int(args.pilot_progress_every),
    )


def shared_pilot_and_fsm_contexts(
    model: str,
    runner: Any,
    runtimes: dict[str, Any],
    y: np.ndarray,
    args: argparse.Namespace,
    *,
    pilot_device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build one data-only pilot and all FSM contexts from the same datasets."""
    if not runtimes:
        raise ValueError("at least one frozen FSM runtime is required")
    first = next(iter(runtimes.values()))
    if model == "model1":
        metadata = {
            "n_blocks": first.n_blocks,
            "block_size": first.block_size,
            "tau": first.tau,
            "pi_min": first.pi_min,
            "pi_max": first.pi_max,
            "sigma_q": first.sigma_q,
        }
        pilot_u, status = runner.data_only_pilot(
            y,
            metadata,
            _pilot_args(args, pilot_device),
        )
        contexts = runner.score_contexts(
            runtimes,
            y,
            pilot_u,
            int(args.score_batch_size),
        )
    else:
        contexts, pilot_u, status = runner.pilot_score_contexts(
            runtimes,
            y,
            float(first.tau),
            int(args.score_batch_size),
            int(args.pilot_grid_size),
            int(args.pilot_batch_size),
            str(args.pilot_backend),
            int(args.pilot_grid_chunk_size),
            int(args.pilot_progress_every),
            pilot_mode=str(args.model2_pilot_mode),
        )
    pilot_u = np.asarray(pilot_u, dtype=np.float64).reshape(-1)
    status = np.asarray(status, dtype="U32").reshape(-1)
    if pilot_u.shape != (y.shape[0],) or not np.all(np.isfinite(pilot_u)):
        raise RuntimeError("shared data-only pilot has an invalid shape or value")
    for label, context in contexts.items():
        context = np.asarray(context, dtype=np.float32)
        if context.shape != (y.shape[0], 2):
            raise RuntimeError(f"bad FSM context shape for {label}: {context.shape}")
        if not np.allclose(context[:, 0], pilot_u, rtol=0.0, atol=2e-6):
            raise RuntimeError(f"{label} did not receive the shared pilot")
    return pilot_u, status, contexts


def build_all_contexts(
    model: str,
    runner: Any,
    runtimes: dict[str, Any],
    jiang: JiangFitted,
    raw: np.ndarray,
    container: np.ndarray,
    args: argparse.Namespace,
    *,
    device: str,
    pilot_device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    pilot_u, status, contexts = shared_pilot_and_fsm_contexts(
        model,
        runner,
        runtimes,
        raw,
        args,
        pilot_device=pilot_device,
    )
    jiang_score = jiang_full_u_score_at_pilot(
        jiang,
        container,
        pilot_u,
        device=device,
        batch_size=int(args.jiang_score_batch_size),
    )
    output = {
        PILOT_LABEL: pilot_u[:, None].astype(np.float32),
        JIANG_LABEL: np.column_stack([pilot_u, jiang_score]).astype(np.float32),
    }
    output.update(contexts)
    return pilot_u, status, output


def _runtime_labels(
    run_dirs: dict[int, Path],
    runtime_class: type,
    *,
    device: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    runtimes: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for seed, run_dir in run_dirs.items():
        for method, prefix in (
            ("linear", LINEAR_PREFIX),
            ("radial", NONLINEAR_PREFIX),
        ):
            label = f"{prefix} [stage1 seed {seed}] + shared NPE"
            runtime = runtime_class(run_dir, method, device=device)
            runtimes[label] = runtime
            metadata[label] = {
                "family": "ours_linear" if method == "linear" else "ours_nonlinear",
                "stage1_seed": int(seed),
                "stage1_run_dir": str(run_dir.resolve()),
                "stage1_method": method,
            }
    return runtimes, metadata


def _summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        squared = np.asarray(
            [row["posterior_mean_sq_error_to_exact"] for row in selected],
            dtype=np.float64,
        )
        output.append(
            {
                "method": method,
                "family": selected[0]["family"],
                "stage1_seed": selected[0]["stage1_seed"],
                "n_test": len(selected),
                "posterior_mean_rmse_to_exact": float(np.sqrt(squared.mean())),
                "posterior_mean_mae_to_exact": float(
                    np.mean([row["abs_mean_error_to_exact"] for row in selected])
                ),
                "posterior_sd_mae_to_exact": float(
                    np.mean([row["abs_sd_error_to_exact"] for row in selected])
                ),
                "interval_endpoint_mae_to_exact": float(
                    np.mean([row["interval_endpoint_mae_to_exact"] for row in selected])
                ),
                "interval_width_mae_to_exact": float(
                    np.mean([row["abs_width_error_to_exact"] for row in selected])
                ),
                "w1_to_exact": float(
                    np.mean([row["w1_to_exact"] for row in selected])
                ),
            }
        )
    return output


def _summarize_families(
    method_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "posterior_mean_rmse_to_exact",
        "posterior_mean_mae_to_exact",
        "posterior_sd_mae_to_exact",
        "interval_endpoint_mae_to_exact",
        "interval_width_mae_to_exact",
        "w1_to_exact",
    )
    output: list[dict[str, Any]] = []
    for family in ("pilot_only", "jiang_round1", "ours_linear", "ours_nonlinear"):
        selected = [row for row in method_summary if row["family"] == family]
        if not selected:
            continue
        record: dict[str, Any] = {
            "family": family,
            "n_stage1_fits": len(selected),
            "n_test_per_fit": int(selected[0]["n_test"]),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            record[metric] = float(values.mean())
            record[f"{metric}_stage1_sd"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        output.append(record)
    return output


def _save_npe_state(
    output: Path,
    result: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    safe = (
        str(metadata["family"])
        + (
            ""
            if metadata.get("stage1_seed") is None
            else f"_s{int(metadata['stage1_seed'])}"
        )
    )
    torch.save(
        {
            "method": result["method"],
            "metadata": metadata,
            "state_dict": result["posterior"].posterior_estimator.state_dict(),
        },
        output / f"npe_{safe}.pt",
    )


def _metadata_for_contexts(
    runtime_metadata: dict[str, dict[str, Any]],
    jiang: JiangFitted,
) -> dict[str, dict[str, Any]]:
    return {
        PILOT_LABEL: {
            "family": "pilot_only",
            "stage1_seed": None,
            "stage1_method": "none",
        },
        JIANG_LABEL: {
            "family": "jiang_round1",
            "stage1_seed": int(jiang.diagnostics["seed"]),
            "stage1_method": "official_jiang_round1",
        },
        **runtime_metadata,
    }


def _validate_protocol(
    contexts: dict[str, np.ndarray],
    pilot_u: np.ndarray,
    jiang: JiangFitted,
) -> None:
    if int(jiang.round_id) != 1:
        raise ValueError("the shared NPE comparison accepts Jiang Round 1 only")
    if jiang.score_architecture != "official_mlp":
        raise ValueError("the Jiang checkpoint is not the authors' official MLP")
    for method, values in contexts.items():
        expected = 1 if method == PILOT_LABEL else 2
        if values.shape != (pilot_u.size, expected):
            raise RuntimeError(f"{method} has context shape {values.shape}, expected dim {expected}")
        if not np.allclose(values[:, 0], pilot_u, rtol=0.0, atol=2e-6):
            raise RuntimeError(f"{method} did not use the identical shared pilot")
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{method} context contains non-finite values")


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = str(args.model)
    config = _config(model)
    args.p_prior_min = config.p_min
    args.p_prior_max = config.p_max

    original_root = Path(args.original_root).resolve()
    jiang_dir = (
        default_jiang_dir(model).resolve()
        if args.jiang_dir is None
        else Path(args.jiang_dir).resolve()
    )
    seeds = _parse_seeds(args.fsm_seeds)
    device = resolve_device(args.device)
    pilot_device = resolve_device(args.pilot_device)
    data_device = resolve_device(args.data_device)

    upstreams = load_upstreams(require_frozen_jiang_commit=True)
    runtime_class, runner = _load_legacy_modules(original_root, model)
    run_dirs = _selected_fsm_run_dirs(original_root, model, seeds)
    runtimes, runtime_metadata = _runtime_labels(
        run_dirs,
        runtime_class,
        device=device,
    )
    jiang = load_jiang(
        jiang_dir / "models" / "jiang_round1.pt",
        upstreams,
        device=device,
    )

    rng = np.random.default_rng(int(args.sbi_train_seed))
    p_train = stratified_prior(
        rng,
        int(args.n_sbi_train),
        config.p_min,
        config.p_max,
    )
    print(
        f"{model}: simulating {p_train.size} complete Stage-2 datasets "
        f"({config.n_observations} iid blocks each)",
        flush=True,
    )
    raw_train, container_train = _simulate_full_datasets(rng, p_train, config)
    context_started = time.time()
    pilot_u_train, pilot_status_train, train_contexts = build_all_contexts(
        model,
        runner,
        runtimes,
        jiang,
        raw_train,
        container_train,
        args,
        device=device,
        pilot_device=pilot_device,
    )
    _validate_protocol(train_contexts, pilot_u_train, jiang)
    del raw_train, container_train

    test_rng = np.random.default_rng(int(args.test_seed))
    p_test = stratified_prior(
        test_rng,
        int(args.n_test),
        config.p_min,
        config.p_max,
    )
    raw_test, container_test = _simulate_full_datasets(test_rng, p_test, config)
    pilot_u_test, pilot_status_test, test_contexts = build_all_contexts(
        model,
        runner,
        runtimes,
        jiang,
        raw_test,
        container_test,
        args,
        device=device,
        pilot_device=pilot_device,
    )
    _validate_protocol(test_contexts, pilot_u_test, jiang)
    print(
        f"context construction finished in {time.time() - context_started:.1f}s; "
        f"pilot corr={np.corrcoef(_logit_array(p_train), pilot_u_train)[0, 1]:.4f}",
        flush=True,
    )

    context_payload: dict[str, np.ndarray] = {
        "p_train": p_train.astype(np.float32),
        "pilot_u_train": pilot_u_train.astype(np.float32),
        "pilot_status_train": pilot_status_train,
        "p_test": p_test.astype(np.float32),
        "pilot_u_test": pilot_u_test.astype(np.float32),
        "pilot_status_test": pilot_status_test,
    }
    labels = list(train_contexts)
    for index, label in enumerate(labels):
        context_payload[f"train_context_{index}"] = train_contexts[label]
        context_payload[f"test_context_{index}"] = test_contexts[label]
    np.savez_compressed(output / "shared_contexts.npz", **context_payload)
    (output / "context_labels.json").write_text(
        json.dumps({str(index): label for index, label in enumerate(labels)}, indent=2),
        encoding="utf-8",
    )

    protocol = {
        "model": model,
        "n_iid_blocks_per_complete_dataset": config.n_observations,
        "block_size": config.block_size,
        "n_sbi_train_complete_datasets": int(args.n_sbi_train),
        "n_test_complete_datasets": int(args.n_test),
        "jiang_checkpoint": str(
            (jiang_dir / "models" / "jiang_round1.pt").resolve()
        ),
        "jiang_round": int(jiang.round_id),
        "jiang_round2_used": False,
        "fsm_run_dirs": {str(seed): str(path.resolve()) for seed, path in run_dirs.items()},
        "shared_data_only_pilot": True,
        "shared_stage2_simulations": True,
        "shared_npe_architecture_optimizer_and_seed": True,
        "score_context": "(pilot_u(D), estimated full-dataset u-score at pilot_u(D))",
        "jiang_full_score": "sum of official debiased Round-1 single-block scores",
        "true_parameter_role": "Stage-2 NPE target only; never a context input",
        "exact_likelihood_role": "held-out posterior evaluation only",
        "pilot_corr_true_u_train": float(
            np.corrcoef(_logit_array(p_train), pilot_u_train)[0, 1]
        ),
        "pilot_boundary_rate_train": float(
            np.mean(np.char.find(pilot_status_train, "boundary") >= 0)
        ),
        "pilot_boundary_rate_test": float(
            np.mean(np.char.find(pilot_status_test, "boundary") >= 0)
        ),
    }
    if bool(args.context_only):
        (output / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
        print("context-only run saved to", output, flush=True)
        return

    stage2 = _stage2_upstream()
    NPE, posterior_nn, BoxUniform = stage2.import_sbi()
    context_metadata = _metadata_for_contexts(runtime_metadata, jiang)
    npe_results: dict[str, dict[str, Any]] = {}
    for method in labels:
        result = stage2.train_npe(
            NPE,
            posterior_nn,
            BoxUniform,
            method,
            train_contexts[method],
            p_train,
            int(args.sbi_seed),
            args,
            device,
            data_device,
        )
        npe_results[method] = result
        _save_npe_state(output, result, context_metadata[method])

    exact_rows: list[tuple[np.ndarray, np.ndarray, dict[str, float]]] = [
        exact_posterior(raw_test[index], config, int(args.grid_size))
        for index in range(int(args.n_test))
    ]
    rows: list[dict[str, Any]] = []
    for index in range(int(args.n_test)):
        axis, cdf, exact = exact_rows[index]
        evaluation_seed = int(args.posterior_seed) + index
        for method in labels:
            result = npe_results[method]
            samples = stage2.posterior_samples(
                result,
                test_contexts[method][index],
                int(args.posterior_n),
                evaluation_seed,
                args,
            )
            mean = float(np.mean(samples))
            sd = float(np.std(samples))
            q05, q50, q95 = np.quantile(samples, (0.05, 0.50, 0.95))
            metadata = context_metadata[method]
            rows.append(
                {
                    "model": model,
                    "test_index": index,
                    "p_simulated": float(p_test[index]),
                    "method": method,
                    "family": metadata["family"],
                    "stage1_seed": metadata.get("stage1_seed"),
                    "posterior_mean": mean,
                    "posterior_sd": sd,
                    "q05": float(q05),
                    "q50": float(q50),
                    "q95": float(q95),
                    "exact_posterior_mean": exact["posterior_mean"],
                    "exact_posterior_sd": exact["posterior_sd"],
                    "exact_q05": exact["q05"],
                    "exact_q50": exact["q50"],
                    "exact_q95": exact["q95"],
                    "posterior_mean_sq_error_to_exact": float(
                        np.square(mean - exact["posterior_mean"])
                    ),
                    "abs_mean_error_to_exact": abs(
                        mean - exact["posterior_mean"]
                    ),
                    "abs_sd_error_to_exact": abs(sd - exact["posterior_sd"]),
                    "interval_endpoint_mae_to_exact": 0.5
                    * (
                        abs(float(q05) - exact["q05"])
                        + abs(float(q95) - exact["q95"])
                    ),
                    "abs_width_error_to_exact": abs(
                        float(q95 - q05) - (exact["q95"] - exact["q05"])
                    ),
                    "w1_to_exact": float(stage2.exact_w1(samples, axis, cdf)),
                }
            )

    method_summary = _summarize_methods(rows)
    family_summary = _summarize_families(method_summary)
    _write_csv(output / "posterior_by_dataset.csv", rows)
    _write_csv(output / "posterior_summary_by_stage1_fit.csv", method_summary)
    _write_csv(output / "posterior_summary_by_family.csv", family_summary)
    protocol["npe"] = {
        "model": args.sbi_model,
        "hidden_features": int(args.sbi_hidden_features),
        "num_components": int(args.sbi_num_components),
        "batch_size": int(args.sbi_batch_size),
        "learning_rate": float(args.sbi_lr),
        "maximum_epochs": int(args.max_epochs),
        "early_stop_patience": int(args.stop_after_epochs),
        "seed_shared_by_every_method": int(args.sbi_seed),
    }
    protocol["posterior_evaluation"] = {
        "posterior_samples_per_dataset": int(args.posterior_n),
        "exact_grid_size": int(args.grid_size),
        "headline_target": "exact posterior, never the simulated parameter",
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(family_summary, indent=2), flush=True)
    print("saved to", output, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("model1", "model2"), required=True)
    parser.add_argument("--original-root", type=Path, default=default_original_root())
    parser.add_argument("--jiang-dir", type=Path)
    parser.add_argument(
        "--fsm-seeds",
        default=",".join(str(seed) for seed in TRAINING_SEEDS),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-sbi-train", type=int, default=50_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260723)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=20260724)
    parser.add_argument("--pilot-grid-size", type=int, default=201)
    parser.add_argument("--pilot-batch-size", type=int, default=512)
    parser.add_argument("--pilot-grid-chunk-size", type=int, default=16)
    parser.add_argument("--pilot-progress-every", type=int, default=10)
    parser.add_argument(
        "--pilot-backend",
        choices=("auto", "torch", "numpy"),
        default="auto",
    )
    parser.add_argument(
        "--model2-pilot-mode",
        choices=("moment", "marginal", "equal_channel"),
        default="equal_channel",
    )
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--jiang-score-batch-size", type=int, default=1024)
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
    parser.add_argument("--sbi-seed", type=int, default=54_000)
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--posterior-seed", type=int, default=87_000)
    parser.add_argument("--grid-size", type=int, default=5_000)
    parser.add_argument("--context-only", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pilot-device", default="auto")
    parser.add_argument("--data-device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if int(args.n_sbi_train) < 1:
        raise ValueError("--n-sbi-train must be positive")
    if int(args.n_test) < 1:
        raise ValueError("--n-test must be positive")
    if int(args.pilot_grid_size) < 21:
        raise ValueError("--pilot-grid-size must be at least 21")
    run(args)


if __name__ == "__main__":
    main()
