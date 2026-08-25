#!/usr/bin/env python3
"""Pilot+score NPE using a frozen amortized Model 2 FSM score field.

For every Stage-2 training simulation,

    pi_b ~ prior,
    Y_b ~ p(. | pi_b),
    u_hat_b = cheap_pilot(Y_b),
    x_b = (u_hat_b, S_frozen(Y_b, u_hat_b)).

The same pilot is computed from data during training and deployment.  The true
simulation parameter is used only as the NPE target and never enters the context.
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

from common.pilot import constrained_modes_from_score_grid
from model2 import npe_utils
from model2 import runtime as score_runtime
from model2 import stage1


METHOD_LABELS = {
    "pilot": "pilot-only NPE",
    "linear": "linear FSM pilot+score NPE",
    "shared_radial": "shared-raw-gate FSM pilot+score NPE",
}

METHOD_SEED_INDEX = {
    "linear": 0,
    "shared_radial": 1,
    "pilot": 2,
}

SELECTED_PILOT_MODE = "equal_channel"
SELECTED_PILOT_BACKEND = "torch"
CONTEXT_DEFINITION = "(u_pilot(Y), S(Y, u_pilot(Y))); equal-channel pilot"


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
        "sigma_q": first.sigma_q,
    }
    for name, runtime in runtimes.items():
        current = {
            "n_blocks": runtime.n_blocks,
            "block_size": runtime.block_size,
            "tau": runtime.tau,
            "pi_min": runtime.pi_min,
            "pi_max": runtime.pi_max,
            "sigma_q": runtime.sigma_q,
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


def pilot_score_contexts(
    runtimes: dict[str, score_runtime.AmortizedScoreRuntime],
    y: np.ndarray,
    tau: float,
    pilot_grid_size: int = 201,
    pilot_batch_size: int = 512,
    pilot_grid_chunk_size: int = 16,
    context_score_batch_size: int = 256,
    pilot_progress_every: int = 0,
    cached_pilot_u: np.ndarray | None = None,
    cached_pilot_status: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Build an oracle-free context while reusing per-chunk log-ratios.

    The selected equal-channel torch pilot evaluates several grid locations
    together on the Stage-1 runtime device. A cached pilot skips the grid sweep
    but still evaluates the frozen scores from the active checkpoints.
    """
    if not runtimes:
        raise ValueError("At least one frozen runtime is required")
    if int(pilot_grid_size) < 21:
        raise ValueError("pilot_grid_size must be at least 21")
    if int(pilot_batch_size) < 1:
        raise ValueError("pilot_batch_size must be positive")
    if int(pilot_grid_chunk_size) < 1:
        raise ValueError("pilot_grid_chunk_size must be positive")
    if int(context_score_batch_size) < 1:
        raise ValueError("context_score_batch_size must be positive")
    reference = next(iter(runtimes.values()))
    for method, runtime in runtimes.items():
        if runtime.device != reference.device:
            raise ValueError(
                f"All runtimes must share one device; {method} uses {runtime.device}, "
                f"reference uses {reference.device}"
            )
    y_arr = reference.validate_y(y)
    u_grid = np.linspace(reference.u_min, reference.u_max, int(pilot_grid_size))
    if (cached_pilot_u is None) != (cached_pilot_status is None):
        raise ValueError("cached_pilot_u and cached_pilot_status must be supplied together")
    if cached_pilot_u is not None:
        cached_pilot_u = np.asarray(cached_pilot_u, dtype=np.float64).reshape(-1)
        cached_pilot_status = np.asarray(cached_pilot_status, dtype="U32").reshape(-1)
        if cached_pilot_u.shape != (y_arr.shape[0],):
            raise ValueError("cached pilot length does not match y")
        if cached_pilot_status.shape != (y_arr.shape[0],):
            raise ValueError("cached pilot status length does not match y")

    roots: list[np.ndarray] = []
    statuses: list[str] = []
    context_parts: dict[str, list[np.ndarray]] = {method: [] for method in runtimes}
    n_chunks = math.ceil(y_arr.shape[0] / int(pilot_batch_size))
    started = time.perf_counter()

    for chunk_index, start in enumerate(
        range(0, y_arr.shape[0], int(pilot_batch_size)),
        start=1,
    ):
        stop = min(start + int(pilot_batch_size), y_arr.shape[0])
        y_chunk = y_arr[start:stop]

        # Compute the expensive local evidence exactly once for this data chunk.
        lr1_np = stage1.marginal_log_ratio(y_chunk, float(tau))
        lr2_np = stage1.pairwise_log_ratio(y_chunk, float(tau))
        lr1 = torch.as_tensor(lr1_np, dtype=torch.float64, device=reference.device)
        lr2 = torch.as_tensor(lr2_np, dtype=torch.float64, device=reference.device)

        if cached_pilot_u is None:
            pilot_field = _torch_composite_pilot_score_grid(
                lr1,
                lr2,
                u_grid,
                grid_chunk_size=int(pilot_grid_chunk_size),
                include_pairwise=True,
            )
            batch_roots, batch_status, _, _ = constrained_modes_from_score_grid(
                pilot_field,
                u_grid,
            )
            batch_u = np.asarray(batch_roots, dtype=np.float64)
            batch_status_arr = np.asarray(batch_status, dtype="U32")
        else:
            batch_u = cached_pilot_u[start:stop]
            batch_status_arr = cached_pilot_status[start:stop]

        roots.append(batch_u)
        statuses.extend(str(value) for value in batch_status_arr)
        with torch.inference_mode():
            for score_start in range(0, stop - start, int(context_score_batch_size)):
                score_stop = min(score_start + int(context_score_batch_size), stop - start)
                score_slice = slice(score_start, score_stop)
                for method, runtime in runtimes.items():
                    score, _ = runtime.score_from_log_ratios_tensor(
                        lr1[score_slice], lr2[score_slice], batch_u[score_slice]
                    )
                    values = np.column_stack(
                        [
                            batch_u[score_slice],
                            score.detach().cpu().numpy().astype(np.float64),
                        ]
                    ).astype(np.float32)
                    context_parts[method].append(values)

        should_report = (
            int(pilot_progress_every) > 0
            and (
                chunk_index == 1
                or chunk_index == n_chunks
                or chunk_index % int(pilot_progress_every) == 0
            )
        )
        if should_report:
            elapsed = time.perf_counter() - started
            rate = stop / max(elapsed, 1e-9)
            eta = (y_arr.shape[0] - stop) / max(rate, 1e-9)
            source = "cache" if cached_pilot_u is not None else "equal_channel/torch"
            print(
                f"pilot contexts [{source}]: {stop}/{y_arr.shape[0]} "
                f"({chunk_index}/{n_chunks}), elapsed={elapsed:.1f}s, eta={eta:.1f}s",
                flush=True,
            )

    u_pilot = np.concatenate(roots, axis=0)
    pilot_status = np.asarray(statuses, dtype="U32")
    if u_pilot.shape != (y_arr.shape[0],) or pilot_status.shape != (y_arr.shape[0],):
        raise AssertionError("pilot builder returned an unexpected shape")
    if not np.all(np.isfinite(u_pilot)):
        raise FloatingPointError("pilot contains non-finite values")

    contexts: dict[str, np.ndarray] = {}
    for method in runtimes:
        values = np.concatenate(context_parts[method], axis=0)
        if values.shape != (y_arr.shape[0], 2):
            raise AssertionError(f"{method} context has unexpected shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{method} score context contains non-finite values")
        contexts[method] = values.astype(np.float32)
    return contexts, u_pilot, pilot_status


def _numpy_composite_pilot_score_grid(
    lr1: np.ndarray,
    lr2: np.ndarray,
    u_grid: np.ndarray,
    *,
    include_pairwise: bool,
) -> np.ndarray:
    """Reference equal-channel pilot grid from precomputed log-ratios."""
    out = np.empty((lr1.shape[0], len(u_grid)), dtype=np.float64)
    for j, u in enumerate(np.asarray(u_grid, dtype=np.float64)):
        pi = float(stage1.sigmoid_np(u))
        s1 = stage1.score_u_from_log_ratio(lr1, pi)
        out[:, j] = s1.mean(axis=2).sum(axis=1)
        if include_pairwise:
            s2 = stage1.score_u_from_log_ratio(lr2, pi)
            out[:, j] += s2.mean(axis=2).sum(axis=1)
    return out


def _torch_composite_pilot_score_grid(
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    u_grid: np.ndarray,
    *,
    grid_chunk_size: int,
    include_pairwise: bool,
) -> np.ndarray:
    """GPU-friendly equal-channel pilot grid using bounded grid chunks."""
    n = int(lr1.shape[0])
    u_values = np.asarray(u_grid, dtype=np.float64).reshape(-1)
    out = np.empty((n, u_values.size), dtype=np.float64)
    # Pilot root selection does not need float64 elementwise sigmoid accuracy.
    # Frozen-score evaluation below still uses the original float64 tensors.
    lr1_pilot = lr1.to(dtype=torch.float32)
    lr2_pilot = lr2.to(dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, u_values.size, int(grid_chunk_size)):
            stop = min(start + int(grid_chunk_size), u_values.size)
            u = torch.as_tensor(
                u_values[start:stop],
                dtype=torch.float32,
                device=lr1.device,
            )
            u_b = u.reshape(1, -1, 1, 1)
            pi_b = torch.sigmoid(u_b)
            score1 = (torch.sigmoid(lr1_pilot[:, None] + u_b) - pi_b).mean(dim=3).sum(dim=2)
            if include_pairwise:
                score2 = (
                    torch.sigmoid(lr2_pilot[:, None] + u_b) - pi_b
                ).mean(dim=3).sum(dim=2)
                score1.add_(score2)
            out[:, start:stop] = score1.detach().cpu().numpy().astype(np.float64)
    return out


def run_pilot_sanity_checks(
    runtimes: dict[str, score_runtime.AmortizedScoreRuntime],
    metadata: dict[str, object],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Abort before NPE training if the context leaks labels or loses location."""
    signature = inspect.signature(pilot_score_contexts)
    forbidden = {"pi", "pi_eval", "pi_true", "theta", "u_true"}
    leaked = forbidden.intersection(signature.parameters)
    if leaked:
        raise AssertionError(f"pilot_score_contexts exposes oracle parameters: {sorted(leaked)}")

    rng = np.random.default_rng(int(args.sanity_seed) + 7919)
    pi_check = rng.uniform(
        float(args.pi_prior_min),
        float(args.pi_prior_max),
        size=int(args.pilot_sanity_n),
    )
    y_check = simulate(rng, pi_check, metadata)
    first, first_u, first_status = pilot_score_contexts(
        runtimes,
        y_check,
        float(metadata["tau"]),
        pilot_grid_size=int(args.pilot_grid_size),
        pilot_batch_size=int(args.pilot_batch_size),
        pilot_grid_chunk_size=int(args.pilot_grid_chunk_size),
        context_score_batch_size=int(args.context_score_batch_size),
    )
    second, second_u, second_status = pilot_score_contexts(
        runtimes,
        y_check,
        float(metadata["tau"]),
        pilot_grid_size=int(args.pilot_grid_size),
        pilot_batch_size=int(args.pilot_batch_size),
        pilot_grid_chunk_size=int(args.pilot_grid_chunk_size),
        context_score_batch_size=int(args.context_score_batch_size),
    )
    if not np.array_equal(first_u, second_u) or not np.array_equal(first_status, second_status):
        raise AssertionError("data-only pilot is not deterministic")

    true_u = stage1.logit_np(pi_check)
    corr = float(np.corrcoef(first_u, true_u)[0, 1])
    if not np.isfinite(corr) or corr < float(args.min_pilot_correlation):
        raise AssertionError(
            f"pilot is not informative enough: corr(u_pilot, u_true)={corr:.4f}"
        )
    boundary = np.isin(first_status, ["left_boundary", "right_boundary"])
    rows: list[dict[str, object]] = []
    for method, values in first.items():
        if not np.array_equal(values, second[method]):
            raise AssertionError(f"{method} context is not deterministic")
        mean_abs_score = float(np.mean(np.abs(values[:, 1])))
        if not np.isfinite(mean_abs_score) or mean_abs_score >= float(args.max_mean_abs_score):
            raise AssertionError(
                f"{method} score column failed finite/explosion check: "
                f"mean_abs={mean_abs_score:.4g}"
            )
        rows.append(
            {
                "method": method,
                "n": int(pi_check.size),
                "context_dim": 2,
                "no_oracle_parameter": True,
                "pilot_deterministic": True,
                "corr_pilot_true_u": corr,
                "pilot_boundary_rate": float(np.mean(boundary)),
                "mean_abs_score": mean_abs_score,
                "passed": True,
            }
        )
    return rows


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
    score_methods = [method for method in methods if method != "pilot"]
    if not score_methods:
        raise ValueError("At least one score method is required to load Stage-1 metadata")

    device = npe_utils.resolve_device(str(args.device))
    data_device = npe_utils.resolve_device(str(args.data_device))
    stage1_run_dir = Path(args.stage1_run_dir)
    runtimes = {
        method: score_runtime.AmortizedScoreRuntime(
            stage1_run_dir,
            method,
            device=device,
        )
        for method in score_methods
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
    print("Stage-1 run:", stage1_run_dir)
    print("Context:", CONTEXT_DEFINITION)
    print("Pilot:", SELECTED_PILOT_MODE, SELECTED_PILOT_BACKEND)
    print("Methods:", methods)
    print("Model metadata:", metadata)

    sanity_rows: list[dict[str, object]] = []
    for method, runtime in runtimes.items():
        checks = runtime.run_sanity_checks(seed=int(args.sanity_seed))
        sanity_rows.append(checks)
        if not bool(checks["passed"]):
            raise AssertionError(f"Frozen Stage-1 runtime sanity failed for {method}: {checks}")
    write_csv(out_dir / "runtime_sanity_checks.csv", sanity_rows)

    pilot_sanity_rows = run_pilot_sanity_checks(runtimes, metadata, args)
    write_csv(out_dir / "pilot_context_sanity_checks.csv", pilot_sanity_rows)
    print("Pilot sanity:", pilot_sanity_rows)

    if args.training_context_cache is not None:
        context_cache_path = Path(args.training_context_cache)
        with np.load(context_cache_path) as cache:
            required = {"pi_train", "pilot_u", "pilot_status"}
            required.update(f"pilot_score_{method}" for method in methods)
            missing = sorted(required.difference(cache.files))
            if missing:
                raise ValueError(
                    f"training context cache is missing {missing}: {context_cache_path}"
                )
            n_available = int(np.asarray(cache["pi_train"]).shape[0])
            if int(args.n_sbi_train) > n_available:
                raise ValueError(
                    f"requested {args.n_sbi_train} contexts but cache contains {n_available}"
                )
            subset_rng = np.random.default_rng(int(args.training_subset_seed))
            subset = subset_rng.permutation(n_available)[: int(args.n_sbi_train)]
            pi_train = np.asarray(cache["pi_train"], dtype=np.float64)[subset]
            pilot_u_train = np.asarray(cache["pilot_u"], dtype=np.float64)[subset]
            pilot_status_train = np.asarray(cache["pilot_status"], dtype="U32")[subset]
            train_contexts = {
                method: np.asarray(cache[f"pilot_score_{method}"], dtype=np.float32)[subset]
                for method in methods
            }
        if not np.all(
            (pi_train >= float(args.pi_prior_min))
            & (pi_train <= float(args.pi_prior_max))
        ):
            raise ValueError("cached training parameters lie outside the requested prior")
        print(
            f"Loaded nested training subset: {args.n_sbi_train}/{n_available} from "
            f"{context_cache_path} (subset_seed={args.training_subset_seed})",
            flush=True,
        )
    else:
        rng = np.random.default_rng(int(args.sbi_train_seed))
        pi_train = rng.uniform(
            float(args.pi_prior_min),
            float(args.pi_prior_max),
            size=int(args.n_sbi_train),
        )
        y_train = simulate(rng, pi_train, metadata)
        pilot_cache_path = (
            Path(args.pilot_cache)
            if args.pilot_cache is not None
            else out_dir / "pilot_cache.npz"
        )
        cached_pilot_u: np.ndarray | None = None
        cached_pilot_status: np.ndarray | None = None
        if pilot_cache_path.exists():
            with np.load(pilot_cache_path) as cache:
                if "pi_train" not in cache or "pilot_u" not in cache or "pilot_status" not in cache:
                    raise ValueError(f"pilot cache is missing required arrays: {pilot_cache_path}")
                cache_pi = np.asarray(cache["pi_train"], dtype=np.float64)
                if cache_pi.shape != pi_train.shape or not np.allclose(
                    cache_pi,
                    pi_train,
                    rtol=0.0,
                    atol=2e-7,
                ):
                    raise ValueError(
                        f"pilot cache pi_train does not match this Stage-2 training set: {pilot_cache_path}"
                    )
                if "pilot_grid_size" in cache and int(cache["pilot_grid_size"]) != int(
                    args.pilot_grid_size
                ):
                    raise ValueError("pilot cache grid size does not match --pilot-grid-size")
                if "tau" in cache and not np.isclose(float(cache["tau"]), float(metadata["tau"])):
                    raise ValueError("pilot cache tau does not match the Stage-1 model")
                if "pilot_mode" in cache and str(cache["pilot_mode"]) != SELECTED_PILOT_MODE:
                    raise ValueError("pilot cache was not generated by the equal-channel pilot")
                cached_pilot_u = np.asarray(cache["pilot_u"], dtype=np.float64)
                cached_pilot_status = np.asarray(cache["pilot_status"], dtype="U32")
            print(f"Loaded pilot cache: {pilot_cache_path}", flush=True)

        train_contexts, pilot_u_train, pilot_status_train = pilot_score_contexts(
            runtimes,
            y_train,
            float(metadata["tau"]),
            pilot_grid_size=int(args.pilot_grid_size),
            pilot_batch_size=int(args.pilot_batch_size),
            pilot_grid_chunk_size=int(args.pilot_grid_chunk_size),
            context_score_batch_size=int(args.context_score_batch_size),
            pilot_progress_every=int(args.pilot_progress_every),
            cached_pilot_u=cached_pilot_u,
            cached_pilot_status=cached_pilot_status,
        )
        if "pilot" in methods:
            train_contexts["pilot"] = pilot_u_train[:, None].astype(np.float32)
        if cached_pilot_u is None:
            pilot_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pilot_cache_path,
                pi_train=pi_train.astype(np.float32),
                pilot_u=pilot_u_train.astype(np.float64),
                pilot_status=pilot_status_train,
                pilot_grid_size=np.asarray(int(args.pilot_grid_size)),
                tau=np.asarray(float(metadata["tau"])),
                sbi_train_seed=np.asarray(int(args.sbi_train_seed)),
                pilot_mode=np.asarray(SELECTED_PILOT_MODE),
                pilot_space=np.asarray("uniform_u"),
            )
            print(f"Saved pilot cache: {pilot_cache_path}", flush=True)
        del y_train

    context_rows: list[dict[str, object]] = []
    true_u_train = stage1.logit_np(pi_train)
    boundary_train = np.isin(pilot_status_train, ["left_boundary", "right_boundary"])
    for method in methods:
        x = train_contexts[method].astype(np.float64)
        u = x[:, 0]
        z = x[:, 1] if x.shape[1] > 1 else None
        context_rows.append(
            {
                "method": METHOD_LABELS[method],
                "n": x.shape[0],
                "context_dim": x.shape[1],
                "pilot_u_mean": float(u.mean()),
                "pilot_u_sd": float(u.std()),
                "pilot_u_min": float(u.min()),
                "pilot_u_max": float(u.max()),
                "corr_pilot_true_u": float(np.corrcoef(u, true_u_train)[0, 1]),
                "pilot_rmse_to_true_u": float(np.sqrt(np.mean((u - true_u_train) ** 2))),
                "n_pilot_boundary": int(np.sum(boundary_train)),
                "pilot_boundary_rate": float(np.mean(boundary_train)),
                "score_mean": float(z.mean()) if z is not None else float("nan"),
                "score_sd": float(z.std()) if z is not None else float("nan"),
                "score_min": float(z.min()) if z is not None else float("nan"),
                "score_max": float(z.max()) if z is not None else float("nan"),
                "corr_score_true_u": (
                    float(np.corrcoef(z, true_u_train)[0, 1])
                    if z is not None
                    else float("nan")
                ),
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
            **{f"pilot_score_{method}": train_contexts[method] for method in methods},
        )

    NPE, posterior_nn, BoxUniform = npe_utils.import_sbi()
    helper_args = build_npe_args(args)
    results: dict[str, dict[str, object]] = {}
    for method in methods:
        label = METHOD_LABELS[method]
        method_seed = (
            int(args.sbi_seed)
            if bool(args.shared_npe_seed)
            else int(args.sbi_seed) + 173 * METHOD_SEED_INDEX[method]
        )
        result = npe_utils.train_sbi_npe_method(
            NPE,
            posterior_nn,
            BoxUniform,
            label,
            train_contexts[method],
            pi_train,
            method_seed,
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
                "stage1_run_dir": str(stage1_run_dir),
                "stage1_checkpoint_path": (
                    str(runtimes[method].checkpoint_path)
                    if method in runtimes
                    else None
                ),
                "stage1_runtime_metadata": (
                    runtimes[method].metadata() if method in runtimes else None
                ),
                "context": "u_pilot(Y)" if method == "pilot" else CONTEXT_DEFINITION,
                "context_dim": int(train_contexts[method].shape[1]),
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
            obs_contexts, obs_pilot_u, obs_pilot_status = pilot_score_contexts(
                runtimes,
                y_obs,
                float(metadata["tau"]),
                pilot_grid_size=int(args.pilot_grid_size),
                pilot_batch_size=1,
                pilot_grid_chunk_size=int(args.pilot_grid_chunk_size),
                context_score_batch_size=1,
            )
            if "pilot" in methods:
                obs_contexts["pilot"] = obs_pilot_u[:, None].astype(np.float32)
            exact_grid, _, exact_cdf, exact = exact_posterior(y_obs[0], args)

            rows.append(
                {
                    "pi_true": pi_true,
                    "seed": obs_seed,
                    "method": "exact likelihood grid",
                    "pilot_u_context": float(obs_pilot_u[0]),
                    "pilot_pi_context": float(stage1.sigmoid_np(obs_pilot_u[0])),
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
                label = METHOD_LABELS[method]
                samples = npe_utils.sample_sbi_posterior(
                    results[method],
                    obs_contexts[method][0],
                    int(args.posterior_n),
                    int(args.posterior_seed)
                    + (
                        0
                        if bool(args.shared_posterior_seed)
                        else 100_003 * METHOD_SEED_INDEX[method]
                    )
                    + obs_seed,
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
                        "pilot_u_context": float(obs_contexts[method][0, 0]),
                        "pilot_pi_context": float(stage1.sigmoid_np(obs_contexts[method][0, 0])),
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
        "stage1_runtime_metadata": {
            method: runtime.metadata() for method, runtime in runtimes.items()
        },
        "context_definition": {
            method: ("u_pilot(Y)" if method == "pilot" else CONTEXT_DEFINITION)
            for method in methods
        },
        "context_dim": {method: int(train_contexts[method].shape[1]) for method in methods},
        "pilot_mode": SELECTED_PILOT_MODE,
        "pilot_backend": SELECTED_PILOT_BACKEND,
        "true_parameter_role": "NPE target only; never passed to context builder",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n==== Pilot+score NPE posterior summary ====")
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
        "pilot_context_sanity_checks.csv",
        "training_context_summary.csv",
        "posterior_by_seed.csv",
        "posterior_summary.csv",
    ):
        print(out_dir / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-run-dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="pilot,linear,shared_radial")
    parser.add_argument("--pi-prior-min", type=float, default=0.05)
    parser.add_argument("--pi-prior-max", type=float, default=0.70)
    parser.add_argument("--n-sbi-train", type=int, default=50_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260723)
    parser.add_argument("--sbi-model", choices=("mdn", "maf", "nsf", "made"), default="mdn")
    parser.add_argument("--sbi-hidden-features", type=int, default=64)
    parser.add_argument(
        "--sbi-num-components", type=int, default=8, help="MDN-only component count."
    )
    parser.add_argument(
        "--sbi-num-transforms",
        type=int,
        default=5,
        help="Flow-only transform count; ignored when --sbi-model=mdn.",
    )
    parser.add_argument(
        "--sbi-num-bins",
        type=int,
        default=8,
        help="NSF-only spline-bin count; ignored by MDN/MAF/MADE.",
    )
    parser.add_argument("--sbi-batch-size", type=int, default=256)
    parser.add_argument("--sbi-lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--sbi-seed", type=int, default=54000)
    parser.add_argument(
        "--shared-npe-seed",
        type=int,
        choices=(0, 1),
        default=1,
        help="Use the same NPE initialization/split seed for every method.",
    )
    parser.add_argument("--test-pi-values", type=str, default="0.07,0.10,0.30,0.50,0.65,0.68")
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--posterior-seed", type=int, default=87000)
    parser.add_argument(
        "--shared-posterior-seed",
        type=int,
        choices=(0, 1),
        default=1,
        help="Use paired posterior Monte Carlo seeds for every method.",
    )
    parser.add_argument("--grid-size", type=int, default=5_000)
    parser.add_argument("--pilot-grid-size", type=int, default=201)
    parser.add_argument("--pilot-batch-size", type=int, default=512)
    parser.add_argument("--context-score-batch-size", type=int, default=256)
    parser.add_argument(
        "--pilot-grid-chunk-size",
        type=int,
        default=16,
        help="Number of u-grid points evaluated together by the torch pilot backend.",
    )
    parser.add_argument(
        "--pilot-progress-every",
        type=int,
        default=5,
        help="Print pilot progress every this many data chunks; 0 disables progress.",
    )
    parser.add_argument(
        "--pilot-cache",
        type=Path,
        default=None,
        help="Optional reusable NPZ containing pi_train, pilot_u, and pilot_status.",
    )
    parser.add_argument(
        "--training-context-cache",
        type=Path,
        default=None,
        help="Reuse a saved training_contexts.npz instead of resimulating data.",
    )
    parser.add_argument(
        "--training-subset-seed",
        type=int,
        default=20260711,
        help="Fixed permutation seed used to form nested training-budget subsets.",
    )
    parser.add_argument("--pilot-sanity-n", type=int, default=64)
    parser.add_argument("--min-pilot-correlation", type=float, default=0.80)
    parser.add_argument("--max-mean-abs-score", type=float, default=20.0)
    parser.add_argument("--sanity-seed", type=int, default=20260710)
    parser.add_argument("--save-training-context", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/model2_pilot_score_sbi_npe"),
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
    if args.pilot_batch_size < 1:
        raise ValueError("--pilot-batch-size must be positive")
    if args.context_score_batch_size < 1:
        raise ValueError("--context-score-batch-size must be positive")
    if args.pilot_grid_chunk_size < 1:
        raise ValueError("--pilot-grid-chunk-size must be positive")
    if args.pilot_progress_every < 0:
        raise ValueError("--pilot-progress-every must be non-negative")
    if args.pilot_sanity_n < 16:
        raise ValueError("--pilot-sanity-n must be at least 16")
    run(args)


if __name__ == "__main__":
    main()
