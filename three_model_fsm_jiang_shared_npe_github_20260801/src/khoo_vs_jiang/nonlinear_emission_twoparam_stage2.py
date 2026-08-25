"""Shared frozen-score NPE for the two-transition-parameter experiment.

The Stage-1 linear and gated networks are frozen.  Every arm receives the same
simulations, the same data-only soft-transition pilot, the same four-dimensional
context ``(pilot_u01, pilot_u11, score_u01, score_u11)``, and the same NPE
configuration.  The exact two-dimensional HMM posterior is evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dgp import logit
from .jiang_official import resolve_device
from .nonlinear_emission_screen import full_emission_log_ratio
from .nonlinear_emission_twoparam import (
    PARAMETER_NAMES,
    TwoParameterFitted,
    load_fitted_2d,
    simulate_trajectories_2d,
    transform_features_2d,
)
from .upstreams import UpstreamModules, load_upstreams


LINEAR_LABEL = "linear local-subscore GRU FSM"
GATE_LABEL = "positive nonlinear local-gate GRU FSM"
PILOT_LABEL = "adjacent-time pairwise composite pilot"
EXACT_LABEL = "exact likelihood grid"
METHODS = (PILOT_LABEL, LINEAR_LABEL, GATE_LABEL)


def _stage2_upstream():
    return importlib.import_module(
        "run_hmm_common_factor_two_stage_sbi_npe_experiment"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def soft_transition_pilot(
    y: np.ndarray,
    fitted: TwoParameterFitted,
) -> tuple[np.ndarray, np.ndarray]:
    """Data-only transition pilot from soft local state evidence.

    No generating parameter, latent state, full likelihood, or exact HMM score
    is used.  The known emission ratio converts each time block to a soft state
    probability under a neutral 1/2 state prior, after which adjacent soft
    transition counts give a two-parameter pilot.
    """
    evidence = full_emission_log_ratio(y, fitted.emission)
    state_one = 1.0 / (1.0 + np.exp(-np.clip(evidence, -60.0, 60.0)))
    previous = state_one[:, :-1]
    current = state_one[:, 1:]
    p01 = np.sum((1.0 - previous) * current, axis=1) / np.maximum(
        np.sum(1.0 - previous, axis=1), 1e-12
    )
    p11 = np.sum(previous * current, axis=1) / np.maximum(
        np.sum(previous, axis=1), 1e-12
    )
    theta = np.column_stack([p01, p11])
    theta = np.clip(theta, fitted.hmm.lower[None, :], fitted.hmm.upper[None, :])
    status = np.where(
        np.any(
            np.isclose(theta, fitted.hmm.lower[None, :], atol=1e-10)
            | np.isclose(theta, fitted.hmm.upper[None, :], atol=1e-10),
            axis=1,
        ),
        "clipped",
        "interior",
    )
    return logit(theta), status


def temporal_pairwise_composite_pilot_2d(
    y: np.ndarray,
    fitted: TwoParameterFitted,
    *,
    grid_size: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximize an adjacent-time pairwise composite likelihood on a 2D grid.

    Each factor is the unconditional bivariate marginal density of
    ``(Y[t-1], Y[t])``.  Overlapping factors are multiplied as a composite
    likelihood.  This deliberately does not run the full HMM forward filter.
    """
    if int(grid_size) < 3:
        raise ValueError("pilot grid_size must be at least three")
    device = resolve_device(device)
    p01_axis = np.linspace(fitted.hmm.p01_min, fitted.hmm.p01_max, int(grid_size))
    p11_axis = np.linspace(fitted.hmm.p11_min, fitted.hmm.p11_max, int(grid_size))
    p01_grid, p11_grid = np.meshgrid(p01_axis, p11_axis, indexing="ij")
    theta_grid = np.column_stack([p01_grid.reshape(-1), p11_grid.reshape(-1)])
    p01 = torch.as_tensor(
        theta_grid[:, 0], dtype=torch.float64, device=device
    ).reshape(1, -1)
    p11 = torch.as_tensor(
        theta_grid[:, 1], dtype=torch.float64, device=device
    ).reshape(1, -1)
    evidence = full_emission_log_ratio(y, fitted.emission)
    ratios = np.exp(np.clip(evidence, -60.0, 60.0))
    estimates: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, ratios.shape[0], int(batch_size)):
            value = torch.as_tensor(
                ratios[start : start + int(batch_size)],
                dtype=torch.float64,
                device=device,
            )
            loglik = torch.zeros(
                value.shape[0], theta_grid.shape[0], dtype=torch.float64, device=device
            )
            state_one = torch.full_like(p01, float(fitted.hmm.init_prob))
            for time in range(1, fitted.hmm.length):
                previous_ratio = value[:, time - 1 : time]
                current_ratio = value[:, time : time + 1]
                pair_ratio = (1.0 - state_one) * (
                    (1.0 - p01) + p01 * current_ratio
                ) + state_one * previous_ratio * (
                    (1.0 - p11) + p11 * current_ratio
                )
                loglik += torch.log(torch.clamp(pair_ratio, min=1e-300))
                state_one = p01 + (p11 - p01) * state_one
            index = torch.argmax(loglik, dim=1).cpu().numpy()
            estimates.append(theta_grid[index])
    theta = np.concatenate(estimates, axis=0)
    boundary = np.any(
        np.isclose(theta, fitted.hmm.lower[None, :], atol=1e-12)
        | np.isclose(theta, fitted.hmm.upper[None, :], atol=1e-12),
        axis=1,
    )
    return logit(theta), np.where(boundary, "boundary", "interior")


def frozen_features_2d(
    fitted: TwoParameterFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    pilot_u: np.ndarray,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    pilot_u = np.asarray(pilot_u, dtype=np.float64).reshape(-1, 2)
    features = transform_features_2d(
        {"y": y, "anchor_u": pilot_u},
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
    )
    gate = one.predict(
        fitted.gate_model,
        "gate",
        features,
        device,
        fitted.training.batch_size,
    )
    output = {
        PILOT_LABEL: pilot_u.astype(np.float32),
        LINEAR_LABEL: np.column_stack([pilot_u, linear]).astype(np.float32),
        GATE_LABEL: np.column_stack([pilot_u, gate]).astype(np.float32),
    }
    if output[PILOT_LABEL].shape != (y.shape[0], 2):
        raise AssertionError("pilot-only context must have width two")
    if not all(
        output[method].shape == (y.shape[0], 4)
        for method in (LINEAR_LABEL, GATE_LABEL)
    ):
        raise AssertionError("pilot-plus-score contexts must have width four")
    if not all(np.all(np.isfinite(value)) for value in output.values()):
        raise RuntimeError("non-finite frozen Stage-2 feature")
    return output


def exact_posterior_grid_2d(
    y: np.ndarray,
    fitted: TwoParameterFitted,
    grid_size: int,
) -> dict[str, Any]:
    """Exact uniform-prior posterior grid, used only after inference."""
    p01_axis = np.linspace(fitted.hmm.p01_min, fitted.hmm.p01_max, int(grid_size))
    p11_axis = np.linspace(fitted.hmm.p11_min, fitted.hmm.p11_max, int(grid_size))
    p01_grid, p11_grid = np.meshgrid(p01_axis, p11_axis, indexing="ij")
    p01 = p01_grid.reshape(-1)
    p11 = p11_grid.reshape(-1)
    evidence = full_emission_log_ratio(y[None, :, :], fitted.emission)[0]
    alpha0 = np.full_like(p01, np.log1p(-fitted.hmm.init_prob))
    alpha1 = np.full_like(p01, np.log(fitted.hmm.init_prob) + evidence[0])
    for time in range(1, fitted.hmm.length):
        next0 = np.logaddexp(
            alpha0 + np.log1p(-p01), alpha1 + np.log1p(-p11)
        )
        next1 = evidence[time] + np.logaddexp(
            alpha0 + np.log(p01), alpha1 + np.log(p11)
        )
        alpha0, alpha1 = next0, next1
    log_weight = np.logaddexp(alpha0, alpha1)
    weight = np.exp(log_weight - np.max(log_weight))
    weight /= weight.sum()
    weight = weight.reshape(p01_grid.shape)
    mean = np.asarray(
        [np.sum(weight * p01_grid), np.sum(weight * p11_grid)],
        dtype=np.float64,
    )
    centered0 = p01_grid - mean[0]
    centered1 = p11_grid - mean[1]
    covariance = np.asarray(
        [
            [np.sum(weight * centered0**2), np.sum(weight * centered0 * centered1)],
            [np.sum(weight * centered0 * centered1), np.sum(weight * centered1**2)],
        ],
        dtype=np.float64,
    )
    marginal0 = weight.sum(axis=1)
    marginal1 = weight.sum(axis=0)
    cdf0 = np.cumsum(marginal0)
    cdf1 = np.cumsum(marginal1)
    quantiles = np.asarray(
        [
            [np.interp(q, cdf0, p01_axis), np.interp(q, cdf1, p11_axis)]
            for q in (0.05, 0.50, 0.95)
        ],
        dtype=np.float64,
    )
    return {
        "axes": (p01_axis, p11_axis),
        "weight": weight,
        "cdfs": (cdf0, cdf1),
        "mean": mean,
        "covariance": covariance,
        "sd": np.sqrt(np.diag(covariance)),
        "quantiles": quantiles,
    }


def marginal_w1(samples: np.ndarray, axis: np.ndarray, cdf: np.ndarray) -> float:
    samples = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    probabilities = (np.arange(samples.size, dtype=np.float64) + 0.5) / samples.size
    return float(np.mean(np.abs(samples - np.interp(probabilities, cdf, axis))))


def _test_theta_values(value: str) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for item in value.split(","):
        coordinates = [float(part) for part in item.split(":")]
        if len(coordinates) != 2:
            raise ValueError("test theta values require p01:p11 syntax")
        output.append(np.asarray(coordinates, dtype=np.float64))
    return output


def _summarize(rows: list[dict[str, Any]], fitted: TwoParameterFitted) -> list[dict[str, Any]]:
    scale = fitted.hmm.upper - fitted.hmm.lower
    output: list[dict[str, Any]] = []
    for method in (EXACT_LABEL, *METHODS):
        selected = [row for row in rows if row["method"] == method]
        result: dict[str, Any] = {"method": method, "n_test": len(selected)}
        if method == EXACT_LABEL:
            result.update(
                {
                    "standardized_posterior_mean_rmse_to_exact": 0.0,
                    "mean_marginal_w1_to_exact": 0.0,
                    "mean_standardized_covariance_error_to_exact": 0.0,
                }
            )
        else:
            standardized_sq = np.asarray(
                [
                    sum(
                        (row[f"{name}_mean_error_to_exact"] / scale[index]) ** 2
                        for index, name in enumerate(PARAMETER_NAMES)
                    )
                    for row in selected
                ]
            )
            result.update(
                {
                    "standardized_posterior_mean_rmse_to_exact": float(
                        np.sqrt(standardized_sq.mean())
                    ),
                    "mean_marginal_w1_to_exact": float(
                        np.mean(
                            [
                                0.5 * (row["p01_w1_to_exact"] + row["p11_w1_to_exact"])
                                for row in selected
                            ]
                        )
                    ),
                    "mean_standardized_covariance_error_to_exact": float(
                        np.mean(
                            [row["standardized_covariance_error_to_exact"] for row in selected]
                        )
                    ),
                }
            )
            for name in PARAMETER_NAMES:
                result[f"{name}_posterior_mean_rmse_to_exact"] = float(
                    np.sqrt(
                        np.mean(
                            [row[f"{name}_mean_error_to_exact"] ** 2 for row in selected]
                        )
                    )
                )
                result[f"{name}_mean_abs_sd_error_to_exact"] = float(
                    np.mean([abs(row[f"{name}_sd_error_to_exact"]) for row in selected])
                )
                result[f"{name}_mean_endpoint_mae_to_exact"] = float(
                    np.mean([row[f"{name}_endpoint_mae_to_exact"] for row in selected])
                )
        output.append(result)
    return output


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstreams = load_upstreams(require_frozen_jiang_commit=True)
    device = resolve_device(args.device)
    data_device = resolve_device(args.data_device)
    fitted = load_fitted_2d(Path(args.checkpoint), upstreams, device=device)
    stage2 = _stage2_upstream()
    NPE, posterior_nn, BoxUniform = stage2.import_sbi()

    rng = np.random.default_rng(int(args.sbi_train_seed))
    n_train = int(args.n_sbi_train)
    strata = (
        np.arange(n_train, dtype=np.float64)[:, None] + rng.uniform(size=(n_train, 2))
    ) / n_train
    for coordinate in range(2):
        rng.shuffle(strata[:, coordinate])
    theta_train = fitted.hmm.lower[None, :] + (
        fitted.hmm.upper - fitted.hmm.lower
    )[None, :] * strata
    y_train = simulate_trajectories_2d(
        rng, theta_train, fitted.hmm, fitted.emission
    )
    pilot_u, pilot_status = temporal_pairwise_composite_pilot_2d(
        y_train,
        fitted,
        grid_size=int(args.pilot_grid_size),
        batch_size=int(args.pilot_batch_size),
        device=device,
    )
    features = frozen_features_2d(
        fitted, upstreams, y_train, pilot_u, device=device
    )
    pilot_theta = 1.0 / (1.0 + np.exp(-pilot_u))
    print(
        "pilot RMSE:",
        np.sqrt(np.mean(np.square(pilot_theta - theta_train), axis=0)),
        flush=True,
    )

    args.p01_prior_min = fitted.hmm.p01_min
    args.p01_prior_max = fitted.hmm.p01_max
    args.p11_prior_min = fitted.hmm.p11_min
    args.p11_prior_max = fitted.hmm.p11_max
    npe_results = [
        stage2.train_sbi_npe_method(
            NPE,
            posterior_nn,
            BoxUniform,
            method,
            features[method],
            theta_train,
            int(args.sbi_seed),
            args,
            device,
            data_device,
        )
        for method in METHODS
    ]

    rows: list[dict[str, Any]] = []
    theta_values = _test_theta_values(args.test_theta_values)
    observation_seeds = [int(value) for value in args.test_seeds.split(",")]
    scale = fitted.hmm.upper - fitted.hmm.lower
    for theta_true in theta_values:
        for observation_seed in observation_seeds:
            y_obs = simulate_trajectories_2d(
                np.random.default_rng(observation_seed),
                theta_true,
                fitted.hmm,
                fitted.emission,
                n=1,
            )
            obs_pilot_u, _ = temporal_pairwise_composite_pilot_2d(
                y_obs,
                fitted,
                grid_size=int(args.pilot_grid_size),
                batch_size=int(args.pilot_batch_size),
                device=device,
            )
            obs_features = frozen_features_2d(
                fitted, upstreams, y_obs, obs_pilot_u, device=device
            )
            exact = exact_posterior_grid_2d(
                y_obs[0], fitted, int(args.grid_size)
            )
            exact_row: dict[str, Any] = {
                "method": EXACT_LABEL,
                "p01_true": float(theta_true[0]),
                "p11_true": float(theta_true[1]),
                "observation_seed": observation_seed,
            }
            for index, name in enumerate(PARAMETER_NAMES):
                exact_row[f"{name}_posterior_mean"] = float(exact["mean"][index])
                exact_row[f"{name}_posterior_sd"] = float(exact["sd"][index])
                exact_row[f"{name}_q05"] = float(exact["quantiles"][0, index])
                exact_row[f"{name}_q50"] = float(exact["quantiles"][1, index])
                exact_row[f"{name}_q95"] = float(exact["quantiles"][2, index])
            rows.append(exact_row)

            evaluation_seed = int(observation_seed) + 91_000
            for result in npe_results:
                method = str(result["method"])
                samples = stage2.sample_sbi_posterior(
                    result,
                    obs_features[method][0],
                    int(args.posterior_n),
                    evaluation_seed,
                    args,
                )
                mean = samples.mean(axis=0)
                covariance = np.cov(samples, rowvar=False, ddof=0)
                sd = np.sqrt(np.diag(covariance))
                quantiles = np.quantile(samples, [0.05, 0.50, 0.95], axis=0)
                normalized_covariance_error = (
                    np.diag(1.0 / scale)
                    @ (covariance - exact["covariance"])
                    @ np.diag(1.0 / scale)
                )
                row = {
                    "method": method,
                    "p01_true": float(theta_true[0]),
                    "p11_true": float(theta_true[1]),
                    "observation_seed": observation_seed,
                    "standardized_covariance_error_to_exact": float(
                        np.linalg.norm(normalized_covariance_error, ord="fro")
                    ),
                }
                for index, name in enumerate(PARAMETER_NAMES):
                    row[f"{name}_posterior_mean"] = float(mean[index])
                    row[f"{name}_posterior_sd"] = float(sd[index])
                    row[f"{name}_q05"] = float(quantiles[0, index])
                    row[f"{name}_q50"] = float(quantiles[1, index])
                    row[f"{name}_q95"] = float(quantiles[2, index])
                    row[f"{name}_exact_mean"] = float(exact["mean"][index])
                    row[f"{name}_exact_sd"] = float(exact["sd"][index])
                    row[f"{name}_mean_error_to_exact"] = float(
                        mean[index] - exact["mean"][index]
                    )
                    row[f"{name}_sd_error_to_exact"] = float(
                        sd[index] - exact["sd"][index]
                    )
                    row[f"{name}_endpoint_mae_to_exact"] = float(
                        0.5
                        * (
                            abs(quantiles[0, index] - exact["quantiles"][0, index])
                            + abs(
                                quantiles[2, index]
                                - exact["quantiles"][2, index]
                            )
                        )
                    )
                    row[f"{name}_w1_to_exact"] = marginal_w1(
                        samples[:, index], exact["axes"][index], exact["cdfs"][index]
                    )
                rows.append(row)

    summary = _summarize(rows, fitted)
    _write_csv(output / "posterior_by_seed.csv", rows)
    _write_csv(output / "posterior_summary.csv", summary)
    config = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "n_sbi_train": n_train,
        "stage2_context": "pilot-only or (shared adjacent-time pairwise-composite pilot u_hat, frozen 2D score at u_hat)",
        "pilot_status_fractions": {
            str(status): float(np.mean(pilot_status == status))
            for status in np.unique(pilot_status)
        },
        "exact_score_used_for_training": False,
        "exact_posterior_used_for_training": False,
        "true_test_parameter_used_for_inference": False,
        "analytic_local_emission_ratio_used_by_shared_pilot": True,
        "summary": summary,
    }
    (output / "config_and_summary.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("saved to", output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-device", default="cpu")
    parser.add_argument("--n-sbi-train", type=int, default=50_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260722)
    parser.add_argument("--sbi-seed", type=int, default=20260723)
    parser.add_argument("--sbi-model", default="mdn")
    parser.add_argument("--sbi-hidden-features", type=int, default=64)
    parser.add_argument("--sbi-num-transforms", type=int, default=5)
    parser.add_argument("--sbi-num-bins", type=int, default=10)
    parser.add_argument("--sbi-num-components", type=int, default=10)
    parser.add_argument("--sbi-batch-size", type=int, default=512)
    parser.add_argument("--sbi-lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--grid-size", type=int, default=151)
    parser.add_argument("--pilot-grid-size", type=int, default=41)
    parser.add_argument("--pilot-batch-size", type=int, default=256)
    parser.add_argument(
        "--test-theta-values",
        default="0.04:0.88,0.06:0.94,0.10:0.94,0.06:0.97",
    )
    parser.add_argument(
        "--test-seeds",
        default=",".join(str(20260800 + index) for index in range(25)),
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
