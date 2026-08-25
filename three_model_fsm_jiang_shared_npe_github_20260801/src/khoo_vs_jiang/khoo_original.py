"""Read-only adapter for the existing composite-score -> gate -> GRU method."""

from __future__ import annotations

from argparse import Namespace
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dgp import HMMConfig, logit
from .jiang_official import normal_confidence_intervals, resolve_device
from .upstreams import UpstreamModules


@dataclass(frozen=True)
class KhooTrainingConfig:
    n_train: int = 40_000
    n_val: int = 8_000
    sigma_q: float = 0.25
    pi_ref: float = 0.30
    gru_hidden: int = 64
    gate_hidden: int = 16
    iters: int = 3_000
    gate_only_steps: int = 400
    batch_size: int = 512
    lr: float = 3e-4
    gate_lr: float = 1e-4
    joint_lr: float = 5e-5
    weight_decay: float = 1e-3
    grad_clip: float = 5.0
    ema_decay: float = 0.995
    patience: int = 20
    print_every: int = 100

    @classmethod
    def smoke(cls) -> "KhooTrainingConfig":
        return cls(
            n_train=256,
            n_val=64,
            iters=20,
            gate_only_steps=5,
            batch_size=64,
            patience=5,
            print_every=5,
        )


@dataclass
class KhooFitted:
    linear_model: torch.nn.Module
    gate_model: torch.nn.Module
    stats: dict[str, np.ndarray]
    hmm: HMMConfig
    training: KhooTrainingConfig
    diagnostics: dict[str, Any]


def _training_namespace(config: KhooTrainingConfig, seed: int) -> Namespace:
    return Namespace(**asdict(config), seed=int(seed))


def fit_khoo_original(
    *,
    upstreams: UpstreamModules,
    hmm: HMMConfig,
    config: KhooTrainingConfig,
    seed: int,
    device: str = "auto",
) -> KhooFitted:
    """Train only the upstream linear warm start and original gated method."""
    device = resolve_device(device)
    base = upstreams.khoo_amortized
    one = upstreams.khoo_oneparam
    base.set_global_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    model_kwargs = {
        "fixed_p01": float(hmm.fixed_p01),
        "length": int(hmm.length),
        "block_size": int(hmm.block_size),
        "tau": float(hmm.tau),
        "init_prob": float(hmm.init_prob),
    }
    raw_train = one.sample_amortized(
        rng,
        int(config.n_train),
        float(hmm.p_min),
        float(hmm.p_max),
        float(config.sigma_q),
        **model_kwargs,
    )
    raw_val = one.sample_amortized(
        rng,
        int(config.n_val),
        float(hmm.p_min),
        float(hmm.p_max),
        float(config.sigma_q),
        **model_kwargs,
    )
    train_features, stats = one.fit_features(
        raw_train, float(hmm.tau), float(config.pi_ref)
    )
    val_features = one.transform_features(
        raw_val, float(hmm.tau), float(config.pi_ref), stats
    )
    train = {**train_features, "target": raw_train["target"]}
    val = {**val_features, "target": raw_val["target"]}
    args = _training_namespace(config, seed)

    linear = one.LinearLocalGRU(
        config.gru_hidden, stats["time_mean"], stats["time_sd"]
    )
    linear, linear_trace, linear_info = one.train_model(
        linear, "linear", train, val, args, device, 101
    )
    gate = one.GatedLocalGRU(
        config.gru_hidden,
        config.gate_hidden,
        stats["time_mean"],
        stats["time_sd"],
    )
    gate.core.load_state_dict(linear.core.state_dict(), strict=True)
    check_n = min(128, val["target"].shape[0])
    check = {key: value[:check_n] for key, value in val.items()}
    nesting_error = float(
        np.max(
            np.abs(
                one.predict(copy.deepcopy(linear).cpu().eval(), "linear", check, "cpu", config.batch_size)
                - one.predict(copy.deepcopy(gate).cpu().eval(), "gate", check, "cpu", config.batch_size)
            )
        )
    )
    if nesting_error > 1e-6:
        raise AssertionError(f"upstream gated model failed strict nesting: {nesting_error}")
    gate, gate_trace, gate_info = one.train_model(
        gate, "gate", train, val, args, device, 202
    )
    diagnostics = {
        "identity_nesting_max_abs": nesting_error,
        "linear_training": linear_info,
        "gate_training": gate_info,
        "gate_summary": one.gate_summary(gate),
        "upstream_module": str(Path(str(one.__file__)).resolve()),
        "training_trace": linear_trace + gate_trace,
    }
    return KhooFitted(
        linear_model=linear,
        gate_model=gate,
        stats=stats,
        hmm=hmm,
        training=config,
        diagnostics=diagnostics,
    )


def save_fitted(path: Path, fitted: KhooFitted) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "linear_state_dict": fitted.linear_model.state_dict(),
            "gate_state_dict": fitted.gate_model.state_dict(),
            "stats": fitted.stats,
            "hmm": fitted.hmm.to_dict(),
            "training": asdict(fitted.training),
            "diagnostics": fitted.diagnostics,
            "method": "upstream_positive_nonlinear_local_gate_GRU_FSM",
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "hmm": fitted.hmm.to_dict(),
                "training": asdict(fitted.training),
                "diagnostics": fitted.diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_fitted(
    path: Path,
    upstreams: UpstreamModules,
    *,
    device: str = "auto",
) -> KhooFitted:
    device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    hmm = HMMConfig(**checkpoint["hmm"])
    training = KhooTrainingConfig(**checkpoint["training"])
    stats = checkpoint["stats"]
    one = upstreams.khoo_oneparam
    linear = one.LinearLocalGRU(
        training.gru_hidden, stats["time_mean"], stats["time_sd"]
    ).to(device)
    linear.load_state_dict(checkpoint["linear_state_dict"])
    gate = one.GatedLocalGRU(
        training.gru_hidden,
        training.gate_hidden,
        stats["time_mean"],
        stats["time_sd"],
    ).to(device)
    gate.load_state_dict(checkpoint["gate_state_dict"])
    return KhooFitted(
        linear_model=linear,
        gate_model=gate,
        stats=stats,
        hmm=hmm,
        training=training,
        diagnostics=checkpoint["diagnostics"],
    )


def _features(
    fitted: KhooFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    p: float,
) -> dict[str, np.ndarray]:
    raw = {
        "y": np.asarray(y, dtype=np.float64),
        "anchor_u": np.full(
            (np.asarray(y).shape[0], 1), float(logit(float(p))), dtype=np.float64
        ),
    }
    return upstreams.khoo_oneparam.transform_features(
        raw, fitted.hmm.tau, fitted.training.pi_ref, fitted.stats
    )


@torch.no_grad()
def scores_u(
    fitted: KhooFitted,
    upstreams: UpstreamModules,
    p: float,
    y: np.ndarray,
    *,
    device: str = "auto",
) -> np.ndarray:
    device = resolve_device(device)
    features = _features(fitted, upstreams, y, p)
    return upstreams.khoo_oneparam.predict(
        fitted.gate_model,
        "gate",
        features,
        device,
        fitted.training.batch_size,
    )


def _precompute_score_inputs(
    fitted: KhooFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    base = upstreams.khoo_amortized
    s1, s2 = base.raw_subscores(
        np.asarray(y, dtype=np.float64), fitted.hmm.tau, fitted.training.pi_ref
    )
    s1 = (s1 - fitted.stats["s1_mean"]) / fitted.stats["s1_sd"]
    s2 = (s2 - fitted.stats["s2_mean"]) / fitted.stats["s2_sd"]
    return (
        torch.as_tensor(s1, dtype=torch.float32, device=device),
        torch.as_tensor(s2, dtype=torch.float32, device=device),
    )


def _scores_tensor(
    fitted: KhooFitted,
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
    anchor_z = (u - mean) / sd
    return fitted.gate_model(s1, s2, anchor_z)


def solve_score_root(
    fitted: KhooFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    *,
    device: str = "auto",
    initial_p: float | None = None,
    weights: np.ndarray | None = None,
    seed: int = 0,
    maxiter: int = 10_000,
    score_tol: float = 1e-3,
    step_tol: float = 1e-6,
    max_initializations: int = 10,
) -> dict[str, Any]:
    """Projected score ascent in the method's native u=logit(p) coordinate."""
    device = resolve_device(device)
    s1, s2 = _precompute_score_inputs(fitted, upstreams, y, device)
    n = s1.shape[0]
    weight = (
        torch.ones(n, 1, device=device)
        if weights is None
        else torch.as_tensor(
            np.asarray(weights, dtype=np.float32).reshape(n, 1), device=device
        )
    )
    lower = float(logit(fitted.hmm.p_min))
    upper = float(logit(fitted.hmm.p_max))
    rng = np.random.default_rng(int(seed))
    attempts = 1 if initial_p is not None else int(max_initializations)
    fitted.gate_model.eval()
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
                score = torch.sum(_scores_tensor(fitted, u, s1, s2) * weight)
                updated = torch.clamp(u + (0.1 / n) * score, lower, upper)
                if torch.linalg.vector_norm(updated - u) < float(step_tol):
                    u = updated
                    step_converged = True
                    break
                u = updated
            score = torch.sum(_scores_tensor(fitted, u, s1, s2) * weight)
            normalized = float((score.abs() / n).cpu())
            p_hat = float(torch.sigmoid(u).cpu())
            last = {
                "p_hat": p_hat,
                "u_hat": float(u.cpu()),
                "score_sum_u": float(score.cpu()),
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
    fitted: KhooFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    p_hat: float,
    *,
    device: str = "auto",
) -> dict[str, float]:
    device = resolve_device(device)
    s1, s2 = _precompute_score_inputs(fitted, upstreams, y, device)
    u = torch.tensor(
        [[float(logit(p_hat))]], dtype=torch.float32, device=device, requires_grad=True
    )
    # cuDNN intentionally disables RNN backward in eval mode.  We need a
    # derivative with respect to the anchor input for the curvature estimate,
    # not a parameter update.  The imported one-layer GRU has neither dropout
    # nor batch normalization, so train/eval compute exactly the same function.
    was_training = fitted.gate_model.training
    fitted.gate_model.train()
    try:
        score = _scores_tensor(fitted, u, s1, s2)
        i_ss = torch.mean(score.square())
        mean_score = torch.mean(score)
        derivative = torch.autograd.grad(mean_score, u)[0].reshape(())
    finally:
        fitted.gate_model.train(was_training)
    i_curv = -derivative
    i_ss_value = float(i_ss.detach().cpu())
    i_curv_value = float(i_curv.detach().cpu())
    delta_sq = (p_hat * (1.0 - p_hat)) ** 2
    cov_ss_u = 1.0 / i_ss_value if i_ss_value > 0.0 else float("nan")
    cov_curv_u = 1.0 / i_curv_value if i_curv_value > 0.0 else float("nan")
    cov_sand_u = (
        i_ss_value / i_curv_value**2 if i_curv_value != 0.0 else float("nan")
    )
    return {
        "I_ss_u": i_ss_value,
        "I_curv_u": i_curv_value,
        "Cov_ss": delta_sq * cov_ss_u,
        "Cov_curv": delta_sq * cov_curv_u,
        "Cov_sand": delta_sq * cov_sand_u,
        "mean_score_u": float(mean_score.detach().cpu()),
    }


def infer_dataset(
    fitted: KhooFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    *,
    num_bootstrap: int = 0,
    seed: int = 0,
    device: str = "auto",
    maxiter: int = 10_000,
) -> dict[str, Any]:
    root = solve_score_root(
        fitted, upstreams, y, device=device, seed=seed, maxiter=maxiter
    )
    p_hat = float(root["p_hat"])
    covariance = covariance_estimates(
        fitted, upstreams, y, p_hat, device=device
    )
    n = np.asarray(y).shape[0]
    output: dict[str, Any] = {
        "root": root,
        "covariance": covariance,
        "normal_intervals": normal_confidence_intervals(p_hat, covariance, n),
    }
    if int(num_bootstrap) > 0:
        rng = np.random.default_rng(int(seed) + 100_000)
        roots = np.empty(int(num_bootstrap), dtype=np.float64)
        converged = np.zeros(int(num_bootstrap), dtype=bool)
        for index in range(int(num_bootstrap)):
            result = solve_score_root(
                fitted,
                upstreams,
                y,
                device=device,
                initial_p=p_hat,
                weights=rng.exponential(size=n),
                seed=seed + index + 1,
                maxiter=maxiter,
            )
            roots[index] = result["p_hat"]
            converged[index] = result["score_converged"]
        q025, q975 = np.quantile(roots - p_hat, [0.025, 0.975])
        lower, upper = p_hat - q975, p_hat - q025
        output["bootstrap"] = {
            "method": "bootstrap_repo_basic",
            "lower": float(lower),
            "upper": float(upper),
            "width": float(upper - lower),
            "convergence_rate": float(np.mean(converged)),
        }
        output["bootstrap_roots"] = roots
    return output


def score_diagnostic(
    fitted: KhooFitted,
    upstreams: UpstreamModules,
    p_values: list[float],
    *,
    n_per_value: int = 500,
    seed: int = 0,
    device: str = "auto",
) -> list[dict[str, float]]:
    from .dgp import exact_score_u, simulate_trajectories

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float]] = []
    for p in p_values:
        y = simulate_trajectories(rng, p, fitted.hmm, upstreams, n=n_per_value)
        prediction = scores_u(fitted, upstreams, p, y, device=device)
        truth = exact_score_u(y, p, fitted.hmm, upstreams)
        truth_sd = max(float(truth.std()), 1e-12)
        error = prediction - truth
        rows.append(
            {
                "p": float(p),
                "n": float(n_per_value),
                "mse": float(np.mean(error**2)),
                "std_mse": float(np.mean((error / truth_sd) ** 2)),
                "corr": float(np.corrcoef(prediction[:, 0], truth[:, 0])[0, 1]),
                "prediction_mean": float(prediction.mean()),
                "truth_mean": float(truth.mean()),
                "truth_sd": truth_sd,
            }
        )
    return rows
