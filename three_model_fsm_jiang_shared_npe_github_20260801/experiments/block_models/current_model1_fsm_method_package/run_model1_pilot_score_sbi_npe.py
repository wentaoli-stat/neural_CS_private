#!/usr/bin/env python3
"""Deployable pilot+score NPE for the amortized Model 1 FSM field.

For every Stage-2 simulation,

    pi ~ prior,  Y ~ p(. | pi),
    u_hat = marginal_composite_pilot(Y),
    context = (u_hat, S_frozen(Y, u_hat)).

The simulated parameter is used only as the NPE target. It is not exposed to
the pilot or frozen-score context builder.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import model1_amortized_score_runtime as score_runtime
import run_blockwise_mean_shift_amortized_fsm_experiment as stage1
import run_blockwise_mean_shift_two_stage_sbi_npe_experiment as npe_utils
from run_model1_mode_a_mle import constrained_modes_from_score_grid


METHOD_LABELS = {
    "pilot": "pilot-only NPE",
    "linear": "linear FSM pilot+score NPE",
    "radial": "radial FSM pilot+score NPE",
    "khoo_block": "Khoo raw-block DeepSets pilot+score NPE",
    "jiang_m4": "Jiang M4 pilot+score NPE",
}

METHOD_SEED_INDEX = {
    "linear": 0,
    "radial": 1,
    "pilot": 2,
    "khoo_block": 3,
    "jiang_m4": 4,
}

CONTEXT_DEFINITION = "(u_pilot(Y), S_frozen(Y, u_pilot(Y)))"


def parse_list(text: str, cast):
    return [cast(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_method_run_dirs(text: str | None) -> dict[str, Path]:
    """Parse optional ``method=/path`` Stage-1 runtime overrides."""
    if text is None or not str(text).strip():
        return {}
    overrides: dict[str, Path] = {}
    for item in str(text).split(","):
        if not item.strip():
            continue
        method, separator, path = item.partition("=")
        method = method.strip()
        path = path.strip()
        if not separator or not method or not path:
            raise ValueError(
                "--stage1-run-dirs must be comma-separated method=/path entries"
            )
        if method not in METHOD_LABELS or method == "pilot":
            raise ValueError(f"Invalid Stage-1 runtime override method: {method!r}")
        if method in overrides:
            raise ValueError(f"Duplicate Stage-1 runtime override: {method!r}")
        overrides[method] = Path(path)
    return overrides


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def standard_error(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0


def validate_matching_runtimes(
    runtimes: dict[str, object],
) -> dict[str, object]:
    first = next(iter(runtimes.values()))
    reference = {
        "n_blocks": first.n_blocks,
        "block_size": first.block_size,
        "tau": first.tau,
        "pi_min": first.pi_min,
        "pi_max": first.pi_max,
        "sigma_q": first.sigma_q,
    }
    for method, runtime in runtimes.items():
        current = {
            "n_blocks": runtime.n_blocks,
            "block_size": runtime.block_size,
            "tau": runtime.tau,
            "pi_min": runtime.pi_min,
            "pi_max": runtime.pi_max,
            "sigma_q": runtime.sigma_q,
        }
        if current != reference:
            raise ValueError(f"Stage-1 runtime mismatch for {method}: {current} != {reference}")
    return reference


def simulate(
    rng: np.random.Generator,
    pi: np.ndarray,
    metadata: dict[str, object],
) -> np.ndarray:
    return stage1.simulate_mean_shift(
        rng,
        stage1.logit_np(np.asarray(pi, dtype=np.float64)),
        int(metadata["n_blocks"]),
        int(metadata["block_size"]),
        float(metadata["tau"]),
    )


def marginal_pilot_score_grid(
    y: np.ndarray,
    u_grid: np.ndarray,
    tau: float,
    *,
    device: str,
    data_batch_size: int,
    grid_chunk_size: int,
    progress_every: int = 0,
) -> np.ndarray:
    """Evaluate the marginal-composite pilot grid without any true parameter."""
    y_arr = np.asarray(y, dtype=np.float64)
    u_values = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    out = np.empty((y_arr.shape[0], u_values.size), dtype=np.float64)
    torch_device = torch.device(device)
    n_batches = math.ceil(y_arr.shape[0] / int(data_batch_size))
    with torch.inference_mode():
        for batch_idx, start in enumerate(range(0, y_arr.shape[0], int(data_batch_size)), start=1):
            stop = min(start + int(data_batch_size), y_arr.shape[0])
            lr = torch.as_tensor(
                stage1.marginal_log_ratio(y_arr[start:stop], float(tau)),
                dtype=torch.float32,
                device=torch_device,
            )
            for grid_start in range(0, u_values.size, int(grid_chunk_size)):
                grid_stop = min(grid_start + int(grid_chunk_size), u_values.size)
                u = torch.as_tensor(
                    u_values[grid_start:grid_stop],
                    dtype=torch.float32,
                    device=torch_device,
                ).reshape(1, -1, 1, 1)
                pi = torch.sigmoid(u)
                score = (torch.sigmoid(lr[:, None] + u) - pi).mean(dim=3).sum(dim=2)
                out[start:stop, grid_start:grid_stop] = (
                    score.detach().cpu().numpy().astype(np.float64)
                )
            if int(progress_every) > 0 and (
                batch_idx % int(progress_every) == 0 or batch_idx == n_batches
            ):
                print(f"pilot batches: {batch_idx}/{n_batches}", flush=True)
    return out


def data_only_pilot(
    y: np.ndarray,
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    pi_grid = np.linspace(
        float(args.pi_prior_min),
        float(args.pi_prior_max),
        int(args.pilot_grid_size),
        dtype=np.float64,
    )
    u_grid = stage1.logit_np(pi_grid)
    score_grid = marginal_pilot_score_grid(
        y,
        u_grid,
        float(metadata["tau"]),
        device=str(args.pilot_device),
        data_batch_size=int(args.pilot_batch_size),
        grid_chunk_size=int(args.pilot_grid_chunk_size),
        progress_every=int(args.pilot_progress_every),
    )
    pilot_u, status, _, _ = constrained_modes_from_score_grid(score_grid, u_grid)
    return pilot_u.astype(np.float64), np.asarray(status, dtype="U32")


def score_contexts(
    runtimes: dict[str, object],
    y: np.ndarray,
    pilot_u: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    contexts: dict[str, np.ndarray] = {}
    for method, runtime in runtimes.items():
        score = runtime.score(y, pilot_u, batch_size=int(batch_size)).reshape(-1)
        contexts[method] = np.column_stack([pilot_u, score]).astype(np.float32)
    return contexts


def run_pilot_sanity_checks(
    runtimes: dict[str, object],
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    signature = inspect.signature(data_only_pilot)
    forbidden = {"pi", "pi_true", "theta", "u_true"}
    leaked = forbidden.intersection(signature.parameters)
    if leaked:
        raise AssertionError(f"data_only_pilot exposes oracle parameters: {sorted(leaked)}")

    rng = np.random.default_rng(int(args.sanity_seed) + 7919)
    pi_check = rng.uniform(
        float(args.pi_prior_min),
        float(args.pi_prior_max),
        size=int(args.pilot_sanity_n),
    )
    y_check = simulate(rng, pi_check, metadata)
    first_u, first_status = data_only_pilot(y_check, metadata, args)
    second_u, second_status = data_only_pilot(y_check, metadata, args)
    if not np.array_equal(first_u, second_u) or not np.array_equal(first_status, second_status):
        raise AssertionError("data-only pilot is not deterministic")

    true_u = stage1.logit_np(pi_check)
    corr = float(np.corrcoef(first_u, true_u)[0, 1])
    if not np.isfinite(corr) or corr < float(args.min_pilot_correlation):
        raise AssertionError(f"pilot correlation is too low: {corr:.4f}")
    contexts = score_contexts(runtimes, y_check, first_u, int(args.score_batch_size))
    boundary = np.isin(first_status, ["left_boundary", "right_boundary"])
    rows = []
    for method, context in contexts.items():
        score = context[:, 1].astype(np.float64)
        mean_abs = float(np.mean(np.abs(score)))
        if not (float(args.min_mean_abs_score) < mean_abs < float(args.max_mean_abs_score)):
            raise AssertionError(f"{method} score failed alive check: mean_abs={mean_abs:.4g}")
        rows.append(
            {
                "method": method,
                "n": int(pi_check.size),
                "context_dim": 2,
                "no_oracle_parameter": True,
                "pilot_deterministic": True,
                "corr_pilot_true_u": corr,
                "pilot_boundary_rate": float(np.mean(boundary)),
                "mean_abs_score": mean_abs,
                "passed": True,
            }
        )
    return rows


def exact_posterior(
    y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    grid, weights, cdf = npe_utils.exact_posterior_grid_arrays(y, args)
    mean = float(np.sum(weights * grid))
    sd = float(np.sqrt(np.sum(weights * (grid - mean) ** 2)))
    return grid, weights, cdf, {
        "mean": mean,
        "sd": sd,
        "q05": float(np.interp(0.05, cdf, grid)),
        "q50": float(np.interp(0.50, cdf, grid)),
        "q95": float(np.interp(0.95, cdf, grid)),
    }


def summarize_pooled(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        method = str(row["method"])
        if method != "exact likelihood grid":
            groups.setdefault(method, []).append(row)
    out = []
    for method, group in sorted(groups.items()):
        w1 = [float(row["w1_to_exact"]) for row in group]
        out.append(
            {
                "method": method,
                "n": len(group),
                "pi_avg_mse": float(np.mean([float(row["pi_sq_err"]) for row in group])),
                "rmse_mean_to_exact": float(
                    np.sqrt(np.mean([float(row["mean_sq_err_to_exact"]) for row in group]))
                ),
                "mean_w1_to_exact": float(np.mean(w1)),
                "w1_se": standard_error(w1),
                "avg_post_sd": float(np.mean([float(row["pi_post_std"]) for row in group])),
                "coverage90": float(np.mean([float(row["pi_coverage90"]) for row in group])),
            }
        )
    return out


def paired_w1_comparisons(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {
        (float(row["pi_true"]), int(row["seed"]), str(row["method"])): float(row["w1_to_exact"])
        for row in rows
        if str(row["method"]) != "exact likelihood grid"
    }
    radial = METHOD_LABELS["radial"]
    out = []
    if radial not in {method for _, _, method in values}:
        return out
    competitors = sorted({method for _, _, method in values} - {radial})
    for competitor in competitors:
        paired = [
            (radial_value, values[(pi, seed, competitor)])
            for (pi, seed, method), radial_value in values.items()
            if method == radial and (pi, seed, competitor) in values
        ]
        radial_values = np.asarray([item[0] for item in paired], dtype=np.float64)
        competitor_values = np.asarray([item[1] for item in paired], dtype=np.float64)
        difference = radial_values - competitor_values
        t_stat = float("nan")
        p_value = float("nan")
        if difference.size > 1 and difference.std(ddof=1) > 0:
            t_stat = float(difference.mean() / (difference.std(ddof=1) / math.sqrt(difference.size)))
            try:
                from scipy.stats import t as student_t

                p_value = float(2.0 * student_t.sf(abs(t_stat), df=difference.size - 1))
            except ImportError:
                pass
        out.append(
            {
                "reference": radial,
                "competitor": competitor,
                "n_pairs": int(difference.size),
                "reference_mean_w1": float(radial_values.mean()),
                "competitor_mean_w1": float(competitor_values.mean()),
                "reference_improvement_percent": float(
                    100.0 * (competitor_values.mean() - radial_values.mean())
                    / competitor_values.mean()
                ),
                "mean_paired_difference": float(difference.mean()),
                "paired_t": t_stat,
                "paired_p": p_value,
                "reference_wins": int(np.sum(difference < 0.0)),
            }
        )
    return out


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "posterior_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    methods = parse_list(args.methods, str)
    bad = sorted(set(methods) - set(METHOD_LABELS))
    if bad or not methods:
        raise ValueError(f"--methods must use {sorted(METHOD_LABELS)}; got {methods}")
    score_methods = [method for method in methods if method != "pilot"]
    if not score_methods:
        raise ValueError("At least one score method is required")

    device = npe_utils.resolve_device(str(args.device))
    data_device = npe_utils.resolve_device(str(args.data_device))
    args.pilot_device = npe_utils.resolve_device(str(args.pilot_device))
    runtime_overrides = parse_method_run_dirs(args.stage1_run_dirs)
    unknown_overrides = sorted(set(runtime_overrides) - set(score_methods))
    if unknown_overrides:
        raise ValueError(
            f"Stage-1 runtime overrides were supplied for inactive methods: {unknown_overrides}"
        )
    runtime_dirs = {
        method: runtime_overrides.get(method, args.stage1_run_dir)
        for method in score_methods
    }
    runtimes = {}
    for method in score_methods:
        runtimes[method] = score_runtime.AmortizedScoreRuntime(
            runtime_dirs[method], method, device=device
        )
    metadata = validate_matching_runtimes(runtimes)
    args.tau = float(metadata["tau"])
    if not (
        float(metadata["pi_min"]) <= float(args.pi_prior_min)
        < float(args.pi_prior_max) <= float(metadata["pi_max"])
    ):
        raise ValueError("Stage-2 prior lies outside the trained Stage-1 anchor support")

    print("torch:", torch.__version__, "NPE device:", device, "data device:", data_device)
    print("Stage-1 run:", args.stage1_run_dir)
    print("Stage-1 runtime directories:", runtime_dirs)
    print("Context:", CONTEXT_DEFINITION)
    print("True parameter role: NPE target only; never a context input")
    print("Metadata:", metadata)

    runtime_rows = []
    for method, runtime in runtimes.items():
        checks = runtime.run_sanity_checks(seed=int(args.sanity_seed))
        runtime_rows.append(checks)
        if not bool(checks["passed"]):
            raise AssertionError(f"Stage-1 runtime sanity failed for {method}: {checks}")
    write_csv(out_dir / "runtime_sanity_checks.csv", runtime_rows)
    pilot_sanity = run_pilot_sanity_checks(runtimes, metadata, args)
    write_csv(out_dir / "pilot_context_sanity_checks.csv", pilot_sanity)
    print("Pilot sanity:", pilot_sanity)

    if args.training_context_cache is not None:
        cache_path = Path(args.training_context_cache)
        with np.load(cache_path) as cache:
            required = {"pi_train", "pilot_u", "pilot_status"}
            required.update(f"pilot_score_{method}" for method in methods)
            missing = sorted(required.difference(cache.files))
            if missing:
                raise ValueError(f"training cache missing {missing}: {cache_path}")
            n_available = int(np.asarray(cache["pi_train"]).shape[0])
            if int(args.n_sbi_train) > n_available:
                raise ValueError(f"requested {args.n_sbi_train}, cache has {n_available}")
            rng_subset = np.random.default_rng(int(args.training_subset_seed))
            subset = rng_subset.permutation(n_available)[: int(args.n_sbi_train)]
            pi_train = np.asarray(cache["pi_train"], dtype=np.float64)[subset]
            pilot_u_train = np.asarray(cache["pilot_u"], dtype=np.float64)[subset]
            pilot_status_train = np.asarray(cache["pilot_status"], dtype="U32")[subset]
            contexts = {
                method: np.asarray(cache[f"pilot_score_{method}"], dtype=np.float32)[subset]
                for method in methods
            }
        print(f"Loaded {pi_train.size}/{n_available} cached contexts from {cache_path}")
    else:
        rng = np.random.default_rng(int(args.sbi_train_seed))
        pi_train = rng.uniform(
            float(args.pi_prior_min),
            float(args.pi_prior_max),
            size=int(args.n_sbi_train),
        )
        y_train = simulate(rng, pi_train, metadata)
        started = time.time()
        pilot_u_train, pilot_status_train = data_only_pilot(y_train, metadata, args)
        print("training pilot seconds:", round(time.time() - started, 2))
        contexts = score_contexts(runtimes, y_train, pilot_u_train, int(args.score_batch_size))
        if "pilot" in methods:
            contexts["pilot"] = pilot_u_train[:, None].astype(np.float32)
        del y_train

    for method, context in contexts.items():
        expected_dim = 1 if method == "pilot" else 2
        if context.shape != (pi_train.size, expected_dim):
            raise ValueError(f"bad {method} context shape: {context.shape}")
        if not np.allclose(context[:, 0], pilot_u_train, rtol=0.0, atol=2e-6):
            raise ValueError(f"{method} context does not use the shared data-only pilot")

    true_u_train = stage1.logit_np(pi_train)
    boundary = np.isin(pilot_status_train, ["left_boundary", "right_boundary"])
    context_rows = []
    for method in methods:
        context = contexts[method].astype(np.float64)
        score = context[:, 1] if context.shape[1] > 1 else None
        context_rows.append(
            {
                "method": METHOD_LABELS[method],
                "n": int(context.shape[0]),
                "context_dim": int(context.shape[1]),
                "corr_pilot_true_u": float(np.corrcoef(pilot_u_train, true_u_train)[0, 1]),
                "pilot_rmse_to_true_u": float(np.sqrt(np.mean((pilot_u_train - true_u_train) ** 2))),
                "pilot_boundary_rate": float(np.mean(boundary)),
                "score_mean": float(score.mean()) if score is not None else float("nan"),
                "score_sd": float(score.std()) if score is not None else float("nan"),
            }
        )
    write_csv(out_dir / "training_context_summary.csv", context_rows)
    if bool(args.save_training_context):
        np.savez_compressed(
            out_dir / "training_contexts.npz",
            pi_train=pi_train.astype(np.float32),
            true_u_train=true_u_train.astype(np.float32),
            pilot_u=pilot_u_train.astype(np.float32),
            pilot_status=pilot_status_train,
            **{f"pilot_score_{method}": contexts[method] for method in methods},
        )

    NPE, posterior_nn, BoxUniform = npe_utils.import_sbi()
    results: dict[str, dict[str, object]] = {}
    for method in methods:
        label = METHOD_LABELS[method]
        result = npe_utils.train_sbi_npe_method(
            NPE,
            posterior_nn,
            BoxUniform,
            label,
            contexts[method],
            pi_train,
            int(args.sbi_seed) + 173 * METHOD_SEED_INDEX[method],
            args,
            device,
            data_device,
        )
        results[method] = result
        torch.save(
            {
                "method": method,
                "label": label,
                "state_dict": result["posterior"].posterior_estimator.state_dict(),
                "stage1_run_dir": str(runtime_dirs.get(method, args.stage1_run_dir)),
                "context": "u_pilot(Y)" if method == "pilot" else CONTEXT_DEFINITION,
                "context_dim": int(contexts[method].shape[1]),
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            },
            out_dir / f"npe_{method}_state.pt",
        )

    rows: list[dict[str, object]] = []
    for pi_true in parse_list(args.test_pi_values, float):
        for obs_seed in parse_list(args.test_seeds, int):
            rng = np.random.default_rng(int(obs_seed))
            y_obs = simulate(rng, np.asarray([pi_true]), metadata)
            pilot_u, pilot_status = data_only_pilot(y_obs, metadata, args)
            obs_contexts = score_contexts(runtimes, y_obs, pilot_u, 1)
            if "pilot" in methods:
                obs_contexts["pilot"] = pilot_u[:, None].astype(np.float32)
            exact_grid, _, exact_cdf, exact = exact_posterior(y_obs[0], args)
            exact_cdf_true = npe_utils.exact_cdf_at(pi_true, exact_grid, exact_cdf)
            rows.append(
                {
                    "method": "exact likelihood grid",
                    "seed": int(obs_seed),
                    "pi_true": pi_true,
                    "pilot_u_context": float(pilot_u[0]),
                    "pilot_pi_context": float(stage1.sigmoid_np(pilot_u[0])),
                    "pilot_status": str(pilot_status[0]),
                    "score_context": float("nan"),
                    "pi_post_mean": exact["mean"],
                    "pi_post_std": exact["sd"],
                    "pi_q05": exact["q05"],
                    "pi_q50": exact["q50"],
                    "pi_q95": exact["q95"],
                    "pi_sq_err": float((exact["mean"] - pi_true) ** 2),
                    "pi_coverage90": float(exact["q05"] <= pi_true <= exact["q95"]),
                    "exact_pi_post_mean": exact["mean"],
                    "exact_pi_post_std": exact["sd"],
                    "exact_pi_q05": exact["q05"],
                    "exact_pi_q50": exact["q50"],
                    "exact_pi_q95": exact["q95"],
                    "mean_abs_err_to_exact": 0.0,
                    "mean_sq_err_to_exact": 0.0,
                    "sd_abs_err_to_exact": 0.0,
                    "q05_abs_err_to_exact": 0.0,
                    "q50_abs_err_to_exact": 0.0,
                    "q95_abs_err_to_exact": 0.0,
                    "w1_to_exact": 0.0,
                    "cdf_at_true": exact_cdf_true,
                    "exact_cdf_at_true": exact_cdf_true,
                    "cdf_at_true_abs_err_to_exact": 0.0,
                }
            )

            for method in methods:
                samples = npe_utils.sample_sbi_posterior(
                    results[method],
                    obs_contexts[method][0],
                    int(args.posterior_n),
                    int(args.posterior_seed)
                    + 100_003 * METHOD_SEED_INDEX[method]
                    + int(obs_seed),
                    args,
                )
                np.save(samples_dir / f"{method}_pi{pi_true:g}_seed{obs_seed}.npy", samples)
                mean = float(np.mean(samples))
                sd = float(np.std(samples))
                q05, q50, q95 = [float(value) for value in np.quantile(samples, [0.05, 0.5, 0.95])]
                cdf_at_true = float(np.mean(samples <= pi_true))
                rows.append(
                    {
                        "method": METHOD_LABELS[method],
                        "seed": int(obs_seed),
                        "pi_true": pi_true,
                        "pilot_u_context": float(pilot_u[0]),
                        "pilot_pi_context": float(stage1.sigmoid_np(pilot_u[0])),
                        "pilot_status": str(pilot_status[0]),
                        "score_context": (
                            float(obs_contexts[method][0, 1])
                            if obs_contexts[method].shape[1] > 1
                            else float("nan")
                        ),
                        "pi_post_mean": mean,
                        "pi_post_std": sd,
                        "pi_q05": q05,
                        "pi_q50": q50,
                        "pi_q95": q95,
                        "pi_sq_err": float((mean - pi_true) ** 2),
                        "pi_coverage90": float(q05 <= pi_true <= q95),
                        "exact_pi_post_mean": exact["mean"],
                        "exact_pi_post_std": exact["sd"],
                        "exact_pi_q05": exact["q05"],
                        "exact_pi_q50": exact["q50"],
                        "exact_pi_q95": exact["q95"],
                        "mean_abs_err_to_exact": abs(mean - exact["mean"]),
                        "mean_sq_err_to_exact": float((mean - exact["mean"]) ** 2),
                        "sd_abs_err_to_exact": abs(sd - exact["sd"]),
                        "q05_abs_err_to_exact": abs(q05 - exact["q05"]),
                        "q50_abs_err_to_exact": abs(q50 - exact["q50"]),
                        "q95_abs_err_to_exact": abs(q95 - exact["q95"]),
                        "w1_to_exact": npe_utils.exact_sample_w1(samples, exact_grid, exact_cdf),
                        "cdf_at_true": cdf_at_true,
                        "exact_cdf_at_true": exact_cdf_true,
                        "cdf_at_true_abs_err_to_exact": abs(cdf_at_true - exact_cdf_true),
                    }
                )

    per_pi = npe_utils.summarize_full_posterior(rows)
    pooled = summarize_pooled(rows)
    paired = paired_w1_comparisons(rows)
    write_csv(out_dir / "posterior_by_seed.csv", rows)
    write_csv(out_dir / "posterior_full_metrics_by_pi.csv", per_pi)
    write_csv(out_dir / "posterior_summary_pooled.csv", pooled)
    write_csv(out_dir / "paired_w1_comparisons.csv", paired)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stage1_metadata": metadata,
        "stage1_runtime_dirs": {method: str(path) for method, path in runtime_dirs.items()},
        "context_definition": {
            method: "u_pilot(Y)" if method == "pilot" else CONTEXT_DEFINITION
            for method in methods
        },
        "true_parameter_role": "NPE target only; never passed to context builder",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n==== Model 1 pilot+score NPE pooled summary ====")
    print(f"{'method':<36}{'W1-exact':>12}{'RMSE-exact':>14}{'MSE':>12}{'post SD':>11}{'cov90':>9}")
    print("-" * 94)
    for row in pooled:
        print(
            f"{str(row['method']):<36}"
            f"{float(row['mean_w1_to_exact']):>12.4g}"
            f"{float(row['rmse_mean_to_exact']):>14.4g}"
            f"{float(row['pi_avg_mse']):>12.4g}"
            f"{float(row['avg_post_sd']):>11.4g}"
            f"{float(row['coverage90']):>9.3g}"
        )
    print("\nPaired W1 comparisons:")
    for row in paired:
        print(row)
    print("\nSaved to", out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage1-run-dirs",
        type=str,
        default=None,
        help=(
            "Optional comma-separated method=/path overrides. Methods without an "
            "override use --stage1-run-dir."
        ),
    )
    parser.add_argument("--methods", type=str, default="pilot,linear,radial")
    parser.add_argument("--pi-prior-min", type=float, default=0.05)
    parser.add_argument("--pi-prior-max", type=float, default=0.70)
    parser.add_argument("--n-sbi-train", type=int, default=50_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260711)
    parser.add_argument("--training-context-cache", type=Path, default=None)
    parser.add_argument("--training-subset-seed", type=int, default=20260711)
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
    parser.add_argument("--sbi-seed", type=int, default=52_000)
    parser.add_argument("--test-pi-values", type=str, default="0.07,0.10,0.30,0.50,0.65,0.68")
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--posterior-seed", type=int, default=87_000)
    parser.add_argument("--grid-size", type=int, default=5_000)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--pilot-grid-size", type=int, default=201)
    parser.add_argument("--pilot-batch-size", type=int, default=512)
    parser.add_argument("--pilot-grid-chunk-size", type=int, default=16)
    parser.add_argument("--pilot-progress-every", type=int, default=10)
    parser.add_argument("--pilot-sanity-n", type=int, default=64)
    parser.add_argument("--min-pilot-correlation", type=float, default=0.70)
    parser.add_argument("--min-mean-abs-score", type=float, default=0.02)
    parser.add_argument("--max-mean-abs-score", type=float, default=20.0)
    parser.add_argument("--sanity-seed", type=int, default=20260710)
    parser.add_argument("--save-training-context", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-device", type=str, default="cpu")
    parser.add_argument("--pilot-device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/model1_pilot_score_sbi_npe"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.pi_prior_min < args.pi_prior_max < 1.0):
        raise ValueError("Require 0 < pi_prior_min < pi_prior_max < 1")
    if args.n_sbi_train < 100:
        raise ValueError("--n-sbi-train must be at least 100")
    if args.pilot_grid_size < 21:
        raise ValueError("--pilot-grid-size must be at least 21")
    if args.pilot_sanity_n < 16:
        raise ValueError("--pilot-sanity-n must be at least 16")
    run(args)


if __name__ == "__main__":
    main()
