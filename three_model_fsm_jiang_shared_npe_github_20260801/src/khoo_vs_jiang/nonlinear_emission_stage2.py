"""Frozen Stage-2 NPE for the nonlinear-emission gate ablation.

This is the same pilot-plus-score construction as the existing Model-3 Stage 2:
the context is ``(u_hat(Y), S_hat(Y; u_hat(Y)))``.  Stage-1 weights are frozen,
and the linear and gated arms receive the same simulations and NPE architecture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .dgp import logit
from .jiang_official import resolve_device
from .nonlinear_emission_screen import (
    NonlinearEmissionFitted,
    full_emission_log_ratio,
    load_fitted,
    simulate_trajectories,
    transform_features,
)
from .upstreams import UpstreamModules, load_upstreams


LINEAR_LABEL = "linear local-subscore GRU FSM"
GATE_LABEL = "positive nonlinear local-gate GRU FSM"
EXACT_LABEL = "exact likelihood grid"


def _stage2_upstream():
    return importlib.import_module("shared_npe_backend")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def temporal_pairwise_score_grid(
    y: np.ndarray,
    u_grid: np.ndarray,
    fitted: NonlinearEmissionFitted,
) -> np.ndarray:
    """Adjacent-time composite score used as the data-only Stage-2 pilot."""
    emission = full_emission_log_ratio(y, fitted.emission)
    ratio = np.exp(np.clip(emission, -60.0, 60.0))
    p = (1.0 / (1.0 + np.exp(-np.asarray(u_grid, dtype=np.float64))))[None, :]
    p01 = float(fitted.hmm.fixed_p01)
    state_one = np.full((1, p.shape[1]), float(fitted.hmm.init_prob))
    dstate_dp = np.zeros_like(state_one)
    score = np.zeros((ratio.shape[0], p.shape[1]), dtype=np.float64)

    for time in range(1, ratio.shape[1]):
        previous_ratio = ratio[:, time - 1 : time]
        current_ratio = ratio[:, time : time + 1]
        from_zero = (1.0 - p01) + p01 * current_ratio
        from_one = previous_ratio * ((1.0 - p) + p * current_ratio)
        pair_ratio = from_zero + state_one * (from_one - from_zero)
        derivative_p = (
            dstate_dp * (from_one - from_zero)
            + state_one * previous_ratio * (current_ratio - 1.0)
        )
        score += p * (1.0 - p) * derivative_p / np.maximum(pair_ratio, 1e-300)
        next_derivative = state_one + (p - p01) * dstate_dp
        state_one = p01 + (p - p01) * state_one
        dstate_dp = next_derivative
    return score


def data_only_pilot(
    y: np.ndarray,
    *,
    fitted: NonlinearEmissionFitted,
    grid_size: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the composite pilot without access to the generating parameter."""
    stage2 = _stage2_upstream()
    u_grid = np.linspace(
        float(logit(fitted.hmm.p_min)),
        float(logit(fitted.hmm.p_max)),
        int(grid_size),
    )
    roots: list[np.ndarray] = []
    statuses: list[str] = []
    stationary: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for start in range(0, y.shape[0], int(batch_size)):
        field = temporal_pairwise_score_grid(
            y[start : start + int(batch_size)], u_grid, fitted
        )
        root, status, n_stationary, residual = stage2.constrained_modes_from_score_grid(
            field, u_grid
        )
        roots.append(root)
        statuses.extend(status)
        stationary.append(n_stationary)
        residuals.append(residual)
    return (
        np.concatenate(roots),
        np.asarray(statuses, dtype="U32"),
        np.concatenate(stationary),
        np.concatenate(residuals),
    )


def frozen_features(
    fitted: NonlinearEmissionFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    pilot_u: np.ndarray,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    pilot_u = np.asarray(pilot_u, dtype=np.float64).reshape(-1)
    features = transform_features(
        {"y": y, "anchor_u": pilot_u[:, None]},
        fitted.emission,
        fitted.training.pi_ref,
        fitted.stats,
        upstreams,
    )
    one = upstreams.khoo_oneparam
    linear = one.predict(
        fitted.linear_model,
        "linear",
        features,
        device,
        fitted.training.batch_size,
    ).reshape(-1)
    gate = one.predict(
        fitted.gate_model,
        "gate",
        features,
        device,
        fitted.training.batch_size,
    ).reshape(-1)
    output = {
        LINEAR_LABEL: np.column_stack([pilot_u, linear]).astype(np.float32),
        GATE_LABEL: np.column_stack([pilot_u, gate]).astype(np.float32),
    }
    if not all(np.all(np.isfinite(value)) for value in output.values()):
        raise RuntimeError("non-finite frozen Stage-2 feature")
    return output


def hmm_loglik_axis(
    y: np.ndarray, p_axis: np.ndarray, fitted: NonlinearEmissionFitted
) -> np.ndarray:
    emission = full_emission_log_ratio(y[None, :, :], fitted.emission)[0]
    p = np.asarray(p_axis, dtype=np.float64)
    p01 = float(fitted.hmm.fixed_p01)
    init = float(fitted.hmm.init_prob)
    alpha0 = np.full_like(p, np.log1p(-init))
    alpha1 = np.full_like(p, np.log(init) + emission[0])
    for time in range(1, emission.shape[0]):
        next0 = np.logaddexp(
            alpha0 + np.log1p(-p01), alpha1 + np.log1p(-p)
        )
        next1 = emission[time] + np.logaddexp(
            alpha0 + np.log(p01), alpha1 + np.log(p)
        )
        alpha0, alpha1 = next0, next1
    return np.logaddexp(alpha0, alpha1)


def exact_posterior(
    y: np.ndarray,
    fitted: NonlinearEmissionFitted,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    axis = np.linspace(fitted.hmm.p_min, fitted.hmm.p_max, int(grid_size))
    log_weight = hmm_loglik_axis(y, axis, fitted)
    weight = np.exp(log_weight - np.max(log_weight))
    weight /= weight.sum()
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


def _result_seed(method: str, observation_seed: int) -> int:
    digest = hashlib.blake2b(method.encode("utf-8"), digest_size=4).hexdigest()
    return int(observation_seed) + 10_000 + int(digest, 16) % 1_000_000


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in (EXACT_LABEL, LINEAR_LABEL, GATE_LABEL):
        selected = [row for row in rows if row["method"] == method]
        squared_error_to_exact = np.asarray(
            [row["posterior_mean_sq_error_to_exact"] for row in selected],
            dtype=np.float64,
        )
        output.append(
            {
                "method": method,
                "n_test": len(selected),
                "posterior_mean_rmse_to_exact": float(
                    np.sqrt(squared_error_to_exact.mean())
                ),
                "mean_abs_mean_error_to_exact": float(
                    np.mean([row["abs_mean_error_to_exact"] for row in selected])
                ),
                "mean_abs_sd_error_to_exact": float(
                    np.mean([row["abs_sd_error_to_exact"] for row in selected])
                ),
                "mean_interval_endpoint_mae_to_exact": float(
                    np.mean([row["interval_endpoint_mae_to_exact"] for row in selected])
                ),
                "mean_abs_width_error_to_exact": float(
                    np.mean([row["abs_width_error_to_exact"] for row in selected])
                ),
                "mean_w1_to_exact": float(
                    np.mean([row["w1_to_exact"] for row in selected])
                ),
            }
        )
    return output


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstreams = load_upstreams(require_frozen_jiang_commit=True)
    device = resolve_device(args.device)
    data_device = resolve_device(args.data_device)
    fitted = load_fitted(Path(args.checkpoint), upstreams, device=device)
    stage2 = _stage2_upstream()
    NPE, posterior_nn, BoxUniform = stage2.import_sbi()

    rng = np.random.default_rng(int(args.sbi_train_seed))
    strata = (
        np.arange(int(args.n_sbi_train), dtype=np.float64)
        + rng.uniform(size=int(args.n_sbi_train))
    ) / int(args.n_sbi_train)
    rng.shuffle(strata)
    p_train = fitted.hmm.p_min + (fitted.hmm.p_max - fitted.hmm.p_min) * strata
    y_train = simulate_trajectories(rng, p_train, fitted.hmm, fitted.emission)
    pilot_u, pilot_status, _, _ = data_only_pilot(
        y_train,
        fitted=fitted,
        grid_size=int(args.pilot_grid_size),
        batch_size=int(args.pilot_batch_size),
    )
    features = frozen_features(
        fitted, upstreams, y_train, pilot_u, device=device
    )
    pilot_p = 1.0 / (1.0 + np.exp(-pilot_u))
    pilot_corr = float(np.corrcoef(logit(p_train), pilot_u)[0, 1])
    print(
        "pilot corr/RMSE:",
        pilot_corr,
        float(np.sqrt(np.mean(np.square(pilot_p - p_train)))),
    )

    # The upstream NPE helpers read these conventional argument names.
    args.p_prior_min = fitted.hmm.p_min
    args.p_prior_max = fitted.hmm.p_max
    npe_results = [
        stage2.train_npe(
            NPE,
            posterior_nn,
            BoxUniform,
            method,
            features[method],
            p_train,
            int(args.sbi_seed),
            args,
            device,
            data_device,
        )
        for method in (LINEAR_LABEL, GATE_LABEL)
    ]

    rows: list[dict[str, Any]] = []
    p_values = [float(value) for value in args.test_p_values.split(",")]
    observation_seeds = [int(value) for value in args.test_seeds.split(",")]
    for p_true in p_values:
        for observation_seed in observation_seeds:
            y_obs = simulate_trajectories(
                np.random.default_rng(observation_seed),
                p_true,
                fitted.hmm,
                fitted.emission,
                n=1,
            )
            obs_pilot_u, _, _, _ = data_only_pilot(
                y_obs,
                fitted=fitted,
                grid_size=int(args.pilot_grid_size),
                batch_size=int(args.pilot_batch_size),
            )
            obs_features = frozen_features(
                fitted, upstreams, y_obs, obs_pilot_u, device=device
            )
            axis, cdf, exact = exact_posterior(
                y_obs[0], fitted, int(args.grid_size)
            )
            rows.append(
                {
                    "method": EXACT_LABEL,
                    "p_true": p_true,
                    "observation_seed": observation_seed,
                    **exact,
                    "exact_posterior_mean": exact["posterior_mean"],
                    "exact_posterior_sd": exact["posterior_sd"],
                    "exact_q05": exact["q05"],
                    "exact_q50": exact["q50"],
                    "exact_q95": exact["q95"],
                    "posterior_mean_sq_error_to_exact": 0.0,
                    "abs_mean_error_to_exact": 0.0,
                    "abs_sd_error_to_exact": 0.0,
                    "interval_endpoint_mae_to_exact": 0.0,
                    "abs_width_error_to_exact": 0.0,
                    "w1_to_exact": 0.0,
                }
            )
            for result in npe_results:
                method = str(result["method"])
                samples = stage2.posterior_samples(
                    result,
                    obs_features[method][0],
                    int(args.posterior_n),
                    _result_seed(method, observation_seed),
                    args,
                )
                mean = float(samples.mean())
                q05, q50, q95 = np.quantile(samples, [0.05, 0.50, 0.95])
                rows.append(
                    {
                        "method": method,
                        "p_true": p_true,
                        "observation_seed": observation_seed,
                        "posterior_mean": mean,
                        "posterior_sd": float(samples.std()),
                        "q05": float(q05),
                        "q50": float(q50),
                        "q95": float(q95),
                        "exact_posterior_mean": exact["posterior_mean"],
                        "exact_posterior_sd": exact["posterior_sd"],
                        "exact_q05": exact["q05"],
                        "exact_q50": exact["q50"],
                        "exact_q95": exact["q95"],
                        "posterior_mean_sq_error_to_exact": (
                            mean - exact["posterior_mean"]
                        ) ** 2,
                        "abs_mean_error_to_exact": abs(
                            mean - exact["posterior_mean"]
                        ),
                        "abs_sd_error_to_exact": abs(
                            float(samples.std()) - exact["posterior_sd"]
                        ),
                        "interval_endpoint_mae_to_exact": 0.5
                        * (
                            abs(float(q05) - exact["q05"])
                            + abs(float(q95) - exact["q95"])
                        ),
                        "abs_width_error_to_exact": abs(
                            float(q95 - q05) - (exact["q95"] - exact["q05"])
                        ),
                        "w1_to_exact": float(
                            stage2.exact_w1(samples, axis, cdf)
                        ),
                    }
                )

    summary = _summarize(rows)
    _write_csv(output / "posterior_by_seed.csv", rows)
    _write_csv(output / "posterior_summary.csv", summary)
    config = {
        **vars(args),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stage1_frozen": True,
        "stage2_context": "(data-only pilot u_hat, frozen score at u_hat)",
        "training_simulations_shared": True,
        "npe_architecture_and_seed_shared": True,
        "pilot_corr_true_u": pilot_corr,
        "pilot_status_fractions": {
            str(status): float(np.mean(pilot_status == status))
            for status in np.unique(pilot_status)
        },
        "exact_score_used_for_training": False,
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("saved to", output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-sbi-train", type=int, default=50_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260725)
    parser.add_argument("--pilot-grid-size", type=int, default=201)
    parser.add_argument("--pilot-batch-size", type=int, default=512)
    parser.add_argument("--sbi-model", choices=("mdn", "maf", "nsf", "made"), default="mdn")
    parser.add_argument("--sbi-hidden-features", type=int, default=64)
    parser.add_argument("--sbi-num-components", type=int, default=5)
    parser.add_argument("--sbi-num-transforms", type=int, default=5)
    parser.add_argument("--sbi-num-bins", type=int, default=8)
    parser.add_argument("--sbi-batch-size", type=int, default=256)
    parser.add_argument("--sbi-lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--sbi-seed", type=int, default=54000)
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--grid-size", type=int, default=5_000)
    parser.add_argument("--test-p-values", default="0.84,0.90,0.94,0.97")
    parser.add_argument(
        "--test-seeds",
        default=",".join(str(value) for value in range(100, 125)),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-device", default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
