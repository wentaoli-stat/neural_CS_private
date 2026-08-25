"""Root and joint-confidence evaluation for the two-parameter FSM models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.stats import chi2, norm
import torch

from .dgp import logit
from .jiang_official import resolve_device
from .nonlinear_emission_screen import full_emission_log_ratio, raw_subscores
from .nonlinear_emission_twoparam import (
    PARAMETER_NAMES,
    TwoParameterFitted,
    load_fitted_2d,
    simulate_trajectories_2d,
)
from .upstreams import UpstreamModules, load_upstreams


def _precompute_inputs(
    fitted: TwoParameterFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    s1, s2 = raw_subscores(
        y, fitted.emission, fitted.training.pi_ref, upstreams
    )
    s1 = (s1 - fitted.stats["s1_mean"]) / fitted.stats["s1_sd"]
    s2 = (s2 - fitted.stats["s2_mean"]) / fitted.stats["s2_sd"]
    return (
        torch.as_tensor(s1, dtype=torch.float32, device=device),
        torch.as_tensor(s2, dtype=torch.float32, device=device),
    )


def _scores_tensor(
    fitted: TwoParameterFitted,
    model: torch.nn.Module,
    u: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
) -> torch.Tensor:
    if u.ndim == 1:
        u = u.reshape(1, 2)
    if u.shape == (1, 2):
        u = u.expand(s1.shape[0], 2)
    if u.shape != (s1.shape[0], 2):
        raise ValueError("u must be shared (1,2) or observation-specific (n,2)")
    mean = torch.as_tensor(
        fitted.stats["anchor_u_mean"], dtype=torch.float32, device=u.device
    )
    sd = torch.as_tensor(
        fitted.stats["anchor_u_sd"], dtype=torch.float32, device=u.device
    )
    return model(s1, s2, (u - mean) / sd)


def solve_score_root_2d(
    fitted: TwoParameterFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    method: str,
    *,
    device: str = "auto",
    seed: int = 0,
    max_nfev: int = 300,
    score_tol: float = 1e-3,
    max_initializations: int = 8,
) -> dict[str, Any]:
    """Find a bounded zero of the mean learned score with multiple starts."""
    device = resolve_device(device)
    model = fitted.gate_model if method == "gate" else fitted.linear_model
    model.eval()
    s1, s2 = _precompute_inputs(fitted, upstreams, y, device)
    lower = logit(fitted.hmm.lower)
    upper = logit(fitted.hmm.upper)
    rng = np.random.default_rng(int(seed))
    starts = [0.5 * (lower + upper)]
    starts.extend(
        rng.uniform(lower, upper, size=(max(int(max_initializations) - 1, 0), 2))
    )

    def residual(u_value: np.ndarray) -> np.ndarray:
        u_tensor = torch.as_tensor(
            u_value, dtype=torch.float32, device=device
        ).reshape(1, 2)
        with torch.no_grad():
            value = _scores_tensor(fitted, model, u_tensor, s1, s2).mean(dim=0)
        return value.detach().cpu().numpy().astype(np.float64)

    candidates: list[tuple[float, Any]] = []
    for start in starts:
        result = least_squares(
            residual,
            np.asarray(start, dtype=np.float64),
            bounds=(lower, upper),
            max_nfev=int(max_nfev),
            # The learned network is float32.  SciPy's default finite-
            # difference perturbation is too small to change its output and
            # can falsely report convergence at the initialization.
            diff_step=1e-3,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        candidates.append((float(np.linalg.norm(residual(result.x))), result))
        if candidates[-1][0] < float(score_tol):
            break
    residual_norm, best = min(candidates, key=lambda item: item[0])
    theta_hat = 1.0 / (1.0 + np.exp(-best.x))
    boundary = np.isclose(theta_hat, fitted.hmm.lower, atol=1e-6) | np.isclose(
        theta_hat, fitted.hmm.upper, atol=1e-6
    )
    return {
        "theta_hat": theta_hat,
        "u_hat": np.asarray(best.x, dtype=np.float64),
        "normalized_score_norm": residual_norm,
        "score_converged": bool(residual_norm < float(score_tol)),
        "optimizer_success": bool(best.success),
        "nfev": int(best.nfev),
        "at_boundary": boundary,
        "attempts": len(candidates),
    }


def covariance_estimates_2d(
    fitted: TwoParameterFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    theta_hat: np.ndarray,
    method: str,
    *,
    device: str = "auto",
) -> dict[str, np.ndarray | float]:
    """Estimate A, B and physical-coordinate sandwich covariance per observation."""
    device = resolve_device(device)
    model = fitted.gate_model if method == "gate" else fitted.linear_model
    model.eval()
    s1, s2 = _precompute_inputs(fitted, upstreams, y, device)
    u = torch.as_tensor(
        logit(np.asarray(theta_hat, dtype=np.float64)),
        dtype=torch.float32,
        device=device,
    ).requires_grad_(True)

    def mean_score(value: torch.Tensor) -> torch.Tensor:
        return _scores_tensor(fitted, model, value, s1, s2).mean(dim=0)

    # cuDNN exposes GRU backward only in training mode.  These GRUs contain no
    # dropout or batch normalization, so switching modes changes no numerical
    # forward behavior; it only enables the Jacobian calculation.
    was_training = model.training
    model.train()
    try:
        scores = _scores_tensor(fitted, model, u, s1, s2)
        jacobian = torch.autograd.functional.jacobian(mean_score, u)
        b_u = torch.einsum("ni,nj->ij", scores, scores) / scores.shape[0]
    finally:
        model.train(was_training)
    a_u = -jacobian
    a_np = a_u.detach().cpu().numpy().astype(np.float64)
    b_np = b_u.detach().cpu().numpy().astype(np.float64)
    inverse_a = np.linalg.pinv(a_np, rcond=1e-8)
    sandwich_u = inverse_a @ b_np @ inverse_a.T
    theta_hat = np.asarray(theta_hat, dtype=np.float64).reshape(2)
    delta = np.diag(theta_hat * (1.0 - theta_hat))
    sandwich_theta = delta @ sandwich_u @ delta
    return {
        "A_u": a_np,
        "B_u": b_np,
        "Cov_sand_theta": sandwich_theta,
        "A_condition": float(np.linalg.cond(a_np)),
    }


def exact_loglik_relative_2d(
    fitted: TwoParameterFitted,
    y: np.ndarray,
    theta: np.ndarray,
) -> float:
    """Exact HMM likelihood used only as an evaluation reference."""
    p01, p11 = np.asarray(theta, dtype=np.float64).reshape(2)
    emission_lr = full_emission_log_ratio(y, fitted.emission)
    init = float(fitted.hmm.init_prob)
    log_alpha0 = np.full(y.shape[0], np.log1p(-init), dtype=np.float64)
    log_alpha1 = np.log(init) + emission_lr[:, 0]
    for time in range(1, fitted.hmm.length):
        next0 = np.logaddexp(
            log_alpha0 + np.log1p(-p01),
            log_alpha1 + np.log1p(-p11),
        )
        next1 = emission_lr[:, time] + np.logaddexp(
            log_alpha0 + np.log(p01), log_alpha1 + np.log(p11)
        )
        log_alpha0, log_alpha1 = next0, next1
    return float(np.sum(np.logaddexp(log_alpha0, log_alpha1)))


def exact_mle_2d(
    fitted: TwoParameterFitted,
    y: np.ndarray,
) -> np.ndarray:
    """Bounded multistart exact MLE, for evaluation only."""
    lower = fitted.hmm.lower
    upper = fitted.hmm.upper
    axes = [
        np.linspace(lower[index], upper[index], 5, dtype=np.float64)
        for index in range(2)
    ]
    starts = np.asarray(np.meshgrid(*axes, indexing="ij")).reshape(2, -1).T
    values = np.asarray(
        [exact_loglik_relative_2d(fitted, y, value) for value in starts]
    )
    ordered = starts[np.argsort(values)[-5:]]
    candidates: list[tuple[float, np.ndarray]] = []
    for start in ordered:
        result = minimize(
            lambda theta: -exact_loglik_relative_2d(fitted, y, theta),
            start,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper)),
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
        )
        candidates.append((float(result.fun), np.asarray(result.x, dtype=np.float64)))
    return min(candidates, key=lambda item: item[0])[1]


def confidence_summary(
    theta_hat: np.ndarray,
    covariance_per_observation: np.ndarray,
    n_observations: int,
    truth: np.ndarray,
    level: float = 0.95,
) -> dict[str, Any]:
    covariance = np.asarray(covariance_per_observation, dtype=np.float64) / int(
        n_observations
    )
    theta_hat = np.asarray(theta_hat, dtype=np.float64).reshape(2)
    truth = np.asarray(truth, dtype=np.float64).reshape(2)
    z = float(norm.ppf(0.5 + level / 2.0))
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    lower = theta_hat - z * standard_error
    upper = theta_hat + z * standard_error
    delta = truth - theta_hat
    mahalanobis = float(delta @ np.linalg.pinv(covariance, rcond=1e-10) @ delta)
    cutoff = float(chi2.ppf(level, df=2))
    determinant = max(float(np.linalg.det(covariance)), 0.0)
    return {
        "covariance": covariance,
        "lower": lower,
        "upper": upper,
        "marginal_coverage": (lower <= truth) & (truth <= upper),
        "joint_coverage": bool(mahalanobis <= cutoff),
        "mahalanobis": mahalanobis,
        "ellipse_area": float(np.pi * cutoff * np.sqrt(determinant)),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summarize(
    rows: list[dict[str, Any]], fitted: TwoParameterFitted
) -> dict[str, Any]:
    truth = fitted.hmm.truth
    scale = fitted.hmm.upper - fitted.hmm.lower
    output: dict[str, Any] = {}
    for method in ("exact", "linear", "gate"):
        estimates = np.asarray(
            [
                [row[f"{method}_p01_hat"], row[f"{method}_p11_hat"]]
                for row in rows
            ],
            dtype=np.float64,
        )
        error = estimates - truth[None, :]
        item: dict[str, Any] = {
            "standardized_vector_rmse": float(
                np.sqrt(np.mean(np.sum(np.square(error / scale[None, :]), axis=1)))
            )
        }
        for coordinate, name in enumerate(PARAMETER_NAMES):
            item[f"{name}_bias"] = float(error[:, coordinate].mean())
            item[f"{name}_mae"] = float(np.abs(error[:, coordinate]).mean())
            item[f"{name}_rmse"] = float(
                np.sqrt(np.square(error[:, coordinate]).mean())
            )
        if method != "exact":
            item["root_convergence"] = float(
                np.mean([row[f"{method}_converged"] for row in rows])
            )
            item["boundary_rate"] = float(
                np.mean([row[f"{method}_boundary"] for row in rows])
            )
            item["joint_coverage"] = float(
                np.mean([row[f"{method}_joint_coverage"] for row in rows])
            )
            item["mean_ellipse_area"] = float(
                np.nanmean([row[f"{method}_ellipse_area"] for row in rows])
            )
            for name in PARAMETER_NAMES:
                item[f"{name}_marginal_coverage"] = float(
                    np.mean(
                        [row[f"{method}_{name}_marginal_coverage"] for row in rows]
                    )
                )
                item[f"{name}_mean_interval_width"] = float(
                    np.nanmean(
                        [row[f"{method}_{name}_interval_width"] for row in rows]
                    )
                )
        output[method] = item
    return output


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstreams = load_upstreams(require_frozen_jiang_commit=True)
    fitted = load_fitted_2d(Path(args.checkpoint), upstreams, device=args.device)
    rows: list[dict[str, Any]] = []
    for replicate in range(int(args.replicates)):
        seed = int(args.seed) + 10_000 * replicate
        y = simulate_trajectories_2d(
            np.random.default_rng(seed),
            fitted.hmm.truth,
            fitted.hmm,
            fitted.emission,
            n=int(args.n_observations),
        )
        exact = exact_mle_2d(fitted, y)
        row: dict[str, Any] = {
            "replicate": replicate,
            "seed": seed,
            "exact_p01_hat": float(exact[0]),
            "exact_p11_hat": float(exact[1]),
        }
        for method in ("linear", "gate"):
            root = solve_score_root_2d(
                fitted,
                upstreams,
                y,
                method,
                device=args.device,
                seed=seed + (1 if method == "linear" else 2),
                max_nfev=int(args.max_nfev),
                max_initializations=int(args.max_initializations),
            )
            covariance = covariance_estimates_2d(
                fitted,
                upstreams,
                y,
                root["theta_hat"],
                method,
                device=args.device,
            )
            confidence = confidence_summary(
                root["theta_hat"],
                covariance["Cov_sand_theta"],
                int(args.n_observations),
                fitted.hmm.truth,
            )
            row[f"{method}_p01_hat"] = float(root["theta_hat"][0])
            row[f"{method}_p11_hat"] = float(root["theta_hat"][1])
            row[f"{method}_converged"] = bool(root["score_converged"])
            row[f"{method}_boundary"] = bool(np.any(root["at_boundary"]))
            row[f"{method}_score_norm"] = float(root["normalized_score_norm"])
            row[f"{method}_A_condition"] = float(covariance["A_condition"])
            row[f"{method}_joint_coverage"] = bool(confidence["joint_coverage"])
            row[f"{method}_ellipse_area"] = float(confidence["ellipse_area"])
            for coordinate, name in enumerate(PARAMETER_NAMES):
                row[f"{method}_{name}_marginal_coverage"] = bool(
                    confidence["marginal_coverage"][coordinate]
                )
                row[f"{method}_{name}_interval_width"] = float(
                    confidence["upper"][coordinate]
                    - confidence["lower"][coordinate]
                )
        rows.append(row)
        if (replicate + 1) % int(args.print_every) == 0:
            print(f"completed {replicate + 1}/{args.replicates}", flush=True)

    with (output / "replicates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "replicates": int(args.replicates),
        "n_observations": int(args.n_observations),
        "hmm": fitted.hmm.to_dict(),
        "emission": asdict(fitted.emission),
        "exact_likelihood_role": "evaluation-only MLE",
        "results": _summarize(rows, fitted),
    }
    (output / "summary.json").write_text(
        json.dumps(_json_value(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps(_json_value(summary["results"]), indent=2))
    print("saved to", output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--n-observations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--max-initializations", type=int, default=8)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
