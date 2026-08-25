"""Repeated-dataset inference for the nonlinear-emission gate ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
import torch

from .dgp import logit
from .jiang_official import normal_confidence_intervals, resolve_device
from .nonlinear_emission_screen import (
    NonlinearEmissionFitted,
    exact_score_u,
    full_emission_log_ratio,
    load_fitted,
    raw_subscores,
    simulate_trajectories,
)
from .upstreams import UpstreamModules, load_upstreams


def _precompute_inputs(
    fitted: NonlinearEmissionFitted,
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
    fitted: NonlinearEmissionFitted,
    model: torch.nn.Module,
    u: torch.Tensor,
    s1: torch.Tensor,
    s2: torch.Tensor,
) -> torch.Tensor:
    if u.ndim == 0:
        u = u.reshape(1, 1)
    elif u.ndim == 1:
        u = u.reshape(-1, 1)
    if u.shape[0] == 1:
        u = u.expand(s1.shape[0], 1)
    mean = torch.as_tensor(
        fitted.stats["anchor_u_mean"], dtype=torch.float32, device=u.device
    )
    sd = torch.as_tensor(
        fitted.stats["anchor_u_sd"], dtype=torch.float32, device=u.device
    )
    return model(s1, s2, (u - mean) / sd)


def solve_score_root(
    fitted: NonlinearEmissionFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    method: str,
    *,
    device: str = "auto",
    initial_p: float | None = None,
    seed: int = 0,
    maxiter: int = 3_000,
    score_tol: float = 1e-3,
    step_tol: float = 1e-6,
    max_initializations: int = 10,
) -> dict[str, Any]:
    device = resolve_device(device)
    model = fitted.gate_model if method == "gate" else fitted.linear_model
    model.eval()
    s1, s2 = _precompute_inputs(fitted, upstreams, y, device)
    n = s1.shape[0]
    lower = float(logit(fitted.hmm.p_min))
    upper = float(logit(fitted.hmm.p_max))
    rng = np.random.default_rng(int(seed))
    attempts = 1 if initial_p is not None else int(max_initializations)
    last: dict[str, Any] | None = None
    with torch.no_grad():
        for attempt in range(1, attempts + 1):
            p_start = (
                float(initial_p)
                if initial_p is not None
                else float(rng.uniform(fitted.hmm.p_min, fitted.hmm.p_max))
            )
            u = torch.tensor([[float(logit(p_start))]], device=device)
            step_converged = False
            iteration = 0
            for iteration in range(int(maxiter)):
                score = torch.sum(_scores_tensor(fitted, model, u, s1, s2))
                updated = torch.clamp(u + (0.1 / n) * score, lower, upper)
                if torch.linalg.vector_norm(updated - u) < float(step_tol):
                    u = updated
                    step_converged = True
                    break
                u = updated
            score = torch.sum(_scores_tensor(fitted, model, u, s1, s2))
            normalized = float((score.abs() / n).cpu())
            p_hat = float(torch.sigmoid(u).cpu())
            last = {
                "p_hat": p_hat,
                "normalized_score_abs": normalized,
                "iterations": int(iteration),
                "attempts": int(attempt),
                "step_converged": step_converged,
                "score_converged": normalized < score_tol,
                "at_lower_bound": abs(p_hat - fitted.hmm.p_min) < 1e-7,
                "at_upper_bound": abs(p_hat - fitted.hmm.p_max) < 1e-7,
            }
            if normalized < score_tol:
                return last
    if last is None:
        raise RuntimeError("root solver made no attempts")
    return last


def covariance_estimates(
    fitted: NonlinearEmissionFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    p_hat: float,
    method: str,
    *,
    device: str = "auto",
) -> dict[str, float]:
    device = resolve_device(device)
    model = fitted.gate_model if method == "gate" else fitted.linear_model
    s1, s2 = _precompute_inputs(fitted, upstreams, y, device)
    u = torch.tensor(
        [[float(logit(p_hat))]], dtype=torch.float32, device=device, requires_grad=True
    )
    was_training = model.training
    model.train()
    try:
        score = _scores_tensor(fitted, model, u, s1, s2)
        i_ss = torch.mean(score.square())
        derivative = torch.autograd.grad(torch.mean(score), u)[0].reshape(())
    finally:
        model.train(was_training)
    i_ss_value = float(i_ss.detach().cpu())
    i_curv_value = float((-derivative).detach().cpu())
    delta_sq = (p_hat * (1.0 - p_hat)) ** 2
    return {
        "I_ss_u": i_ss_value,
        "I_curv_u": i_curv_value,
        "Cov_ss": delta_sq / i_ss_value if i_ss_value > 0.0 else float("nan"),
        "Cov_curv": delta_sq / i_curv_value
        if i_curv_value > 0.0
        else float("nan"),
        "Cov_sand": delta_sq * i_ss_value / i_curv_value**2
        if i_curv_value != 0.0
        else float("nan"),
    }


def exact_loglik_relative(
    fitted: NonlinearEmissionFitted, y: np.ndarray, p11: float
) -> float:
    """p-dependent HMM log likelihood, omitting the common state-zero density."""
    emission_lr = full_emission_log_ratio(y, fitted.emission)
    p01 = float(fitted.hmm.fixed_p01)
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


def exact_mle(fitted: NonlinearEmissionFitted, y: np.ndarray) -> float:
    result = minimize_scalar(
        lambda p: -exact_loglik_relative(fitted, y, float(p)),
        bounds=(fitted.hmm.p_min, fitted.hmm.p_max),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x)


def _summarize(rows: list[dict[str, Any]], p_true: float) -> dict[str, Any]:
    output: dict[str, Any] = {}
    exact = np.asarray([row["exact_p_hat"] for row in rows])
    output["exact"] = {
        "bias": float(np.mean(exact - p_true)),
        "mae": float(np.mean(np.abs(exact - p_true))),
        "rmse": float(np.sqrt(np.mean(np.square(exact - p_true)))),
    }
    for method in ("linear", "gate"):
        estimates = np.asarray([row[f"{method}_p_hat"] for row in rows])
        item: dict[str, Any] = {
            "bias": float(np.mean(estimates - p_true)),
            "mae": float(np.mean(np.abs(estimates - p_true))),
            "rmse": float(np.sqrt(np.mean(np.square(estimates - p_true)))),
            "root_convergence": float(
                np.mean([row[f"{method}_converged"] for row in rows])
            ),
            "boundary_rate": float(
                np.mean([row[f"{method}_boundary"] for row in rows])
            ),
        }
        for key in ("Cov_ss", "Cov_curv", "Cov_sand"):
            lower = np.asarray([row[f"{method}_{key}_lower"] for row in rows])
            upper = np.asarray([row[f"{method}_{key}_upper"] for row in rows])
            finite = np.isfinite(lower) & np.isfinite(upper)
            item[key] = {
                "finite_rate": float(np.mean(finite)),
                "coverage": float(
                    np.mean((lower[finite] <= p_true) & (p_true <= upper[finite]))
                )
                if np.any(finite)
                else float("nan"),
                "mean_width": float(np.mean(upper[finite] - lower[finite]))
                if np.any(finite)
                else float("nan"),
            }
        output[method] = item
    return output


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstreams = load_upstreams(require_frozen_jiang_commit=True)
    fitted = load_fitted(Path(args.checkpoint), upstreams, device=args.device)
    rows: list[dict[str, Any]] = []
    for replicate in range(int(args.replicates)):
        seed = int(args.seed) + 10_000 * replicate
        y = simulate_trajectories(
            np.random.default_rng(seed),
            fitted.hmm.p_true,
            fitted.hmm,
            fitted.emission,
            n=int(args.n_observations),
        )
        row: dict[str, Any] = {
            "replicate": replicate,
            "seed": seed,
            "exact_p_hat": exact_mle(fitted, y),
        }
        for method in ("linear", "gate"):
            root = solve_score_root(
                fitted,
                upstreams,
                y,
                method,
                device=args.device,
                seed=seed + (1 if method == "linear" else 2),
                maxiter=int(args.maxiter),
            )
            covariance = covariance_estimates(
                fitted,
                upstreams,
                y,
                float(root["p_hat"]),
                method,
                device=args.device,
            )
            intervals = normal_confidence_intervals(
                float(root["p_hat"]), covariance, int(args.n_observations)
            )
            row[f"{method}_p_hat"] = root["p_hat"]
            row[f"{method}_converged"] = root["score_converged"]
            row[f"{method}_boundary"] = bool(
                root["at_lower_bound"] or root["at_upper_bound"]
            )
            for key, interval in intervals.items():
                row[f"{method}_{key}_lower"] = interval["lower"]
                row[f"{method}_{key}_upper"] = interval["upper"]
        rows.append(row)
        if (replicate + 1) % int(args.print_every) == 0:
            print(f"completed {replicate + 1}/{args.replicates}")

    with (output / "replicates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "replicates": int(args.replicates),
        "n_observations": int(args.n_observations),
        "seed_rule": f"{args.seed} + 10000 * replicate",
        "hmm": fitted.hmm.to_dict(),
        "emission": {
            "rho": fitted.emission.rho,
            "scale": fitted.emission.scale,
        },
        "results": _summarize(rows, fitted.hmm.p_true),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["results"], indent=2))
    print("saved to", output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--n-observations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--maxiter", type=int, default=3_000)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
