"""Two-transition-parameter nonlinear-emission Direct-FSM experiment.

This is the clean two-dimensional extension of ``nonlinear_emission_screen``.
Both transition probabilities are unknown,

    theta = (p01, p11),

while the sparse scale-mixture emission is kept fixed and known.  The linear
and gated arms receive identical analytic marginal/pairwise emission
subscores, identical proposal simulations, and identical GRU capacity.  Their
only difference is the positive nonlinear local gate.  Exact HMM scores are
computed after training for diagnostics only.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .dgp import logit
from .jiang_official import resolve_device
from .khoo_original import KhooTrainingConfig
from .nonlinear_emission_screen import (
    SparseScaleMixtureEmission,
    full_emission_log_ratio,
    raw_subscores,
)
from .upstreams import UpstreamModules, load_upstreams


PARAMETER_NAMES = ("p01", "p11")


@dataclass(frozen=True)
class TwoTransitionHMMConfig:
    """Bounds and truth for a two-transition-parameter HMM."""

    length: int = 50
    block_size: int = 10
    init_prob: float = 0.5
    p01_min: float = 0.02
    p01_max: float = 0.20
    p01_true: float = 0.06
    p11_min: float = 0.80
    p11_max: float = 0.99
    p11_true: float = 0.94

    def validate(self) -> None:
        if self.length < 2 or self.block_size < 2:
            raise ValueError("length and block_size must both be at least two")
        if not 0.0 < self.init_prob < 1.0:
            raise ValueError("init_prob must lie in (0,1)")
        if not 0.0 < self.p01_min < self.p01_true < self.p01_max < 1.0:
            raise ValueError("require p01_min < p01_true < p01_max in (0,1)")
        if not 0.0 < self.p11_min < self.p11_true < self.p11_max < 1.0:
            raise ValueError("require p11_min < p11_true < p11_max in (0,1)")

    @property
    def lower(self) -> np.ndarray:
        return np.asarray([self.p01_min, self.p11_min], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray([self.p01_max, self.p11_max], dtype=np.float64)

    @property
    def truth(self) -> np.ndarray:
        return np.asarray([self.p01_true, self.p11_true], dtype=np.float64)

    @property
    def x_dim(self) -> int:
        return int(self.length * self.block_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LinearLocalGRU2D(nn.Module):
    """Linear marginal/pairwise reduction followed by a two-output GRU."""

    def __init__(
        self,
        hidden: int,
        time_mean: np.ndarray,
        time_sd: np.ndarray,
        upstreams: UpstreamModules,
    ) -> None:
        super().__init__()
        self.core = upstreams.khoo_amortized.AnchorGRUCore(
            2, 2, int(hidden), out_dim=2
        )
        self.register_buffer(
            "time_mean", torch.as_tensor(time_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "time_sd", torch.as_tensor(time_sd, dtype=torch.float32)
        )

    def forward(
        self, s1: torch.Tensor, s2: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        time_raw = torch.stack([s1.mean(dim=2), s2.mean(dim=2)], dim=2)
        return self.core((time_raw - self.time_mean) / self.time_sd, anchor_z)


class GatedLocalGRU2D(nn.Module):
    """Positive nonlinear local gate followed by a two-output GRU."""

    def __init__(
        self,
        hidden: int,
        gate_hidden: int,
        time_mean: np.ndarray,
        time_sd: np.ndarray,
        upstreams: UpstreamModules,
    ) -> None:
        super().__init__()
        self.marginal_gate = upstreams.khoo_fixed.PositiveMLPMultiplier(
            int(gate_hidden)
        )
        self.pairwise_gate = upstreams.khoo_fixed.PositiveMLPMultiplier(
            int(gate_hidden)
        )
        self.core = upstreams.khoo_amortized.AnchorGRUCore(
            2, 2, int(hidden), out_dim=2
        )
        self.register_buffer(
            "time_mean", torch.as_tensor(time_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "time_sd", torch.as_tensor(time_sd, dtype=torch.float32)
        )

    def forward(
        self, s1: torch.Tensor, s2: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        gated1 = s1 * self.marginal_gate(s1.unsqueeze(-1))
        gated2 = s2 * self.pairwise_gate(s2.unsqueeze(-1))
        time_raw = torch.stack(
            [gated1.mean(dim=2), gated2.mean(dim=2)], dim=2
        )
        return self.core((time_raw - self.time_mean) / self.time_sd, anchor_z)


@dataclass
class TwoParameterFitted:
    linear_model: nn.Module
    gate_model: nn.Module
    stats: dict[str, np.ndarray]
    hmm: TwoTransitionHMMConfig
    emission: SparseScaleMixtureEmission
    training: KhooTrainingConfig
    diagnostics: dict[str, Any]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=np.float64)))


def simulate_trajectories_2d(
    rng: np.random.Generator,
    theta: np.ndarray,
    hmm: TwoTransitionHMMConfig,
    emission: SparseScaleMixtureEmission,
    *,
    n: int | None = None,
    return_states: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate independent trajectories at physical theta=(p01,p11)."""
    hmm.validate()
    emission.validate()
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim == 1:
        if values.shape != (2,):
            raise ValueError("one theta must have shape (2,)")
        count = 1 if n is None else int(n)
        values = np.repeat(values[None, :], count, axis=0)
    elif values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("theta must have shape (2,) or (n,2)")
    if n is not None and values.shape[0] != int(n):
        raise ValueError("theta and n imply inconsistent sample sizes")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("transition probabilities must lie in (0,1)")

    states = np.zeros((values.shape[0], hmm.length), dtype=np.int64)
    states[:, 0] = rng.binomial(1, hmm.init_prob, size=values.shape[0])
    for time in range(1, hmm.length):
        probability = np.where(
            states[:, time - 1] == 1, values[:, 1], values[:, 0]
        )
        states[:, time] = rng.binomial(1, probability)

    tail = rng.binomial(
        1,
        emission.rho,
        size=(values.shape[0], hmm.length, hmm.block_size),
    )
    standard_deviation = np.where(tail == 1, emission.scale, 1.0)
    standard_deviation = np.where(
        states[:, :, None] == 1, standard_deviation, 1.0
    )
    y = (rng.normal(size=standard_deviation.shape) * standard_deviation).astype(
        np.float32
    )
    if return_states:
        return y, states
    return y


def sample_from_anchors_2d(
    rng: np.random.Generator,
    anchor_theta: np.ndarray,
    sigma_q: np.ndarray | float,
    hmm: TwoTransitionHMMConfig,
    emission: SparseScaleMixtureEmission,
) -> dict[str, np.ndarray]:
    anchor_theta = np.asarray(anchor_theta, dtype=np.float64)
    if anchor_theta.ndim != 2 or anchor_theta.shape[1] != 2:
        raise ValueError("anchor_theta must have shape (n,2)")
    sigma = np.broadcast_to(np.asarray(sigma_q, dtype=np.float64), (2,))
    if np.any(sigma <= 0.0):
        raise ValueError("proposal standard deviations must be positive")
    anchor_u = logit(anchor_theta)
    u = anchor_u + rng.normal(size=anchor_u.shape) * sigma[None, :]
    theta = _sigmoid(u)
    y = simulate_trajectories_2d(rng, theta, hmm, emission)
    target = (u - anchor_u) / np.square(sigma)[None, :]
    return {
        "y": y,
        "anchor_theta": anchor_theta,
        "anchor_u": anchor_u,
        "u": u,
        "theta": theta,
        "target": target.astype(np.float32),
    }


def sample_amortized_2d(
    rng: np.random.Generator,
    n: int,
    hmm: TwoTransitionHMMConfig,
    emission: SparseScaleMixtureEmission,
    sigma_q: np.ndarray | float,
) -> dict[str, np.ndarray]:
    """Two independent randomized Latin-hypercube anchor coordinates."""
    strata = (
        np.arange(int(n), dtype=np.float64)[:, None]
        + rng.uniform(size=(int(n), 2))
    ) / int(n)
    for coordinate in range(2):
        rng.shuffle(strata[:, coordinate])
    anchors = hmm.lower[None, :] + (hmm.upper - hmm.lower)[None, :] * strata
    return sample_from_anchors_2d(rng, anchors, sigma_q, hmm, emission)


def fit_features_2d(
    raw: dict[str, np.ndarray],
    emission: SparseScaleMixtureEmission,
    pi_ref: float,
    upstreams: UpstreamModules,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    s1, s2 = raw_subscores(raw["y"], emission, pi_ref, upstreams)
    stats: dict[str, np.ndarray] = {
        "s1_mean": np.asarray(float(s1.mean()), dtype=np.float64),
        "s1_sd": np.asarray(max(float(s1.std()), 1e-8), dtype=np.float64),
        "s2_mean": np.asarray(float(s2.mean()), dtype=np.float64),
        "s2_sd": np.asarray(max(float(s2.std()), 1e-8), dtype=np.float64),
        "anchor_u_mean": np.asarray(raw["anchor_u"].mean(axis=0), dtype=np.float64),
        "anchor_u_sd": np.asarray(
            np.maximum(raw["anchor_u"].std(axis=0), 1e-8), dtype=np.float64
        ),
    }
    s1_z = (s1 - stats["s1_mean"]) / stats["s1_sd"]
    s2_z = (s2 - stats["s2_mean"]) / stats["s2_sd"]
    time_raw = np.stack([s1_z.mean(axis=2), s2_z.mean(axis=2)], axis=2)
    stats["time_mean"] = np.asarray(
        time_raw.mean(axis=(0, 1), keepdims=True), dtype=np.float64
    )
    stats["time_sd"] = np.asarray(
        np.maximum(time_raw.std(axis=(0, 1), keepdims=True), 1e-8),
        dtype=np.float64,
    )
    return transform_features_2d(raw, emission, pi_ref, stats, upstreams), stats


def transform_features_2d(
    raw: dict[str, np.ndarray],
    emission: SparseScaleMixtureEmission,
    pi_ref: float,
    stats: dict[str, np.ndarray],
    upstreams: UpstreamModules,
) -> dict[str, np.ndarray]:
    s1, s2 = raw_subscores(raw["y"], emission, pi_ref, upstreams)
    anchor_u = np.asarray(raw["anchor_u"], dtype=np.float64)
    if anchor_u.ndim != 2 or anchor_u.shape[1] != 2:
        raise ValueError("anchor_u must have shape (n,2)")
    return {
        "s1": ((s1 - stats["s1_mean"]) / stats["s1_sd"]).astype(np.float32),
        "s2": ((s2 - stats["s2_mean"]) / stats["s2_sd"]).astype(np.float32),
        "anchor_z": (
            (anchor_u - stats["anchor_u_mean"]) / stats["anchor_u_sd"]
        ).astype(np.float32),
    }


def exact_score_u_2d(
    y: np.ndarray,
    theta: np.ndarray,
    hmm: TwoTransitionHMMConfig,
    emission: SparseScaleMixtureEmission,
    upstreams: UpstreamModules,
) -> np.ndarray:
    """Exact two-dimensional HMM score, for diagnostics only."""
    theta = np.asarray(theta, dtype=np.float64).reshape(2)
    score = upstreams.khoo_fixed.hmm_transition_score_u(
        full_emission_log_ratio(y, emission),
        float(theta[0]),
        float(theta[1]),
        float(hmm.init_prob),
    )
    return np.asarray(score, dtype=np.float64)


def fit_models_2d(
    *,
    upstreams: UpstreamModules,
    hmm: TwoTransitionHMMConfig,
    emission: SparseScaleMixtureEmission,
    training: KhooTrainingConfig,
    sigma_q: np.ndarray | float,
    seed: int,
    device: str,
) -> TwoParameterFitted:
    device = resolve_device(device)
    one = upstreams.khoo_oneparam
    one.base.set_global_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    raw_train = sample_amortized_2d(
        rng, training.n_train, hmm, emission, sigma_q
    )
    raw_val = sample_amortized_2d(rng, training.n_val, hmm, emission, sigma_q)
    train_features, stats = fit_features_2d(
        raw_train, emission, training.pi_ref, upstreams
    )
    val_features = transform_features_2d(
        raw_val, emission, training.pi_ref, stats, upstreams
    )
    train = {**train_features, "target": raw_train["target"]}
    val = {**val_features, "target": raw_val["target"]}
    args = argparse.Namespace(**asdict(training), seed=int(seed))

    linear = LinearLocalGRU2D(
        training.gru_hidden,
        stats["time_mean"],
        stats["time_sd"],
        upstreams,
    )
    linear, linear_trace, linear_info = one.train_model(
        linear, "linear", train, val, args, device, 301
    )
    gate = GatedLocalGRU2D(
        training.gru_hidden,
        training.gate_hidden,
        stats["time_mean"],
        stats["time_sd"],
        upstreams,
    )
    gate.core.load_state_dict(linear.core.state_dict(), strict=True)
    check_n = min(128, raw_val["target"].shape[0])
    check = {key: value[:check_n] for key, value in val.items()}
    nesting_error = float(
        np.max(
            np.abs(
                one.predict(
                    copy.deepcopy(linear).cpu().eval(),
                    "linear",
                    check,
                    "cpu",
                    training.batch_size,
                )
                - one.predict(
                    copy.deepcopy(gate).cpu().eval(),
                    "gate",
                    check,
                    "cpu",
                    training.batch_size,
                )
            )
        )
    )
    if nesting_error > 1e-6:
        raise AssertionError(f"gated model failed strict nesting: {nesting_error}")
    gate, gate_trace, gate_info = one.train_model(
        gate, "gate", train, val, args, device, 302
    )
    return TwoParameterFitted(
        linear_model=linear,
        gate_model=gate,
        stats=stats,
        hmm=hmm,
        emission=emission,
        training=training,
        diagnostics={
            "identity_nesting_max_abs": nesting_error,
            "linear_training": linear_info,
            "gate_training": gate_info,
            "gate_summary": one.gate_summary(gate),
            "training_trace": linear_trace + gate_trace,
        },
    )


def load_fitted_2d(
    path: Path,
    upstreams: UpstreamModules,
    *,
    device: str = "auto",
) -> TwoParameterFitted:
    device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    hmm = TwoTransitionHMMConfig(**checkpoint["hmm"])
    emission = SparseScaleMixtureEmission(**checkpoint["emission"])
    training = KhooTrainingConfig(**checkpoint["training"])
    stats = checkpoint["stats"]
    linear = LinearLocalGRU2D(
        training.gru_hidden,
        stats["time_mean"],
        stats["time_sd"],
        upstreams,
    ).to(device)
    gate = GatedLocalGRU2D(
        training.gru_hidden,
        training.gate_hidden,
        stats["time_mean"],
        stats["time_sd"],
        upstreams,
    ).to(device)
    linear.load_state_dict(checkpoint["linear_state_dict"])
    gate.load_state_dict(checkpoint["gate_state_dict"])
    return TwoParameterFitted(
        linear_model=linear,
        gate_model=gate,
        stats=stats,
        hmm=hmm,
        emission=emission,
        training=training,
        diagnostics={},
    )


def _parse_theta_values(value: str) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for item in value.split(","):
        coordinates = [float(part) for part in item.split(":")]
        if len(coordinates) != 2:
            raise ValueError("theta values must use p01:p11 comma-separated syntax")
        output.append(np.asarray(coordinates, dtype=np.float64))
    return output


def score_diagnostic_2d(
    fitted: TwoParameterFitted,
    upstreams: UpstreamModules,
    theta_values: list[np.ndarray],
    *,
    n_per_value: int,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    device = resolve_device(device)
    one = upstreams.khoo_oneparam
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for theta in theta_values:
        theta = np.asarray(theta, dtype=np.float64).reshape(2)
        y = simulate_trajectories_2d(
            rng, theta, fitted.hmm, fitted.emission, n=n_per_value
        )
        raw = {
            "y": y,
            "anchor_u": np.repeat(logit(theta)[None, :], n_per_value, axis=0),
        }
        features = transform_features_2d(
            raw,
            fitted.emission,
            fitted.training.pi_ref,
            fitted.stats,
            upstreams,
        )
        truth = exact_score_u_2d(
            y, theta, fitted.hmm, fitted.emission, upstreams
        )
        truth_sd = np.maximum(truth.std(axis=0), 1e-12)
        for method, model in (
            ("linear", fitted.linear_model),
            ("gate", fitted.gate_model),
        ):
            prediction = one.predict(
                model, method, features, device, fitted.training.batch_size
            )
            error = prediction - truth
            row: dict[str, Any] = {
                "method": method,
                "p01": float(theta[0]),
                "p11": float(theta[1]),
                "n": int(n_per_value),
                "std_mse": float(
                    np.mean(np.square(error / truth_sd[None, :]))
                ),
                "mse": float(np.mean(np.square(error))),
                "mean_corr": float(
                    np.mean(
                        [
                            np.corrcoef(prediction[:, j], truth[:, j])[0, 1]
                            for j in range(2)
                        ]
                    )
                ),
            }
            for coordinate, name in enumerate(PARAMETER_NAMES):
                row[f"{name}_std_mse"] = float(
                    np.mean(np.square(error[:, coordinate] / truth_sd[coordinate]))
                )
                row[f"{name}_corr"] = float(
                    np.corrcoef(
                        prediction[:, coordinate], truth[:, coordinate]
                    )[0, 1]
                )
                row[f"{name}_prediction_mean"] = float(
                    prediction[:, coordinate].mean()
                )
                row[f"{name}_truth_mean"] = float(truth[:, coordinate].mean())
                row[f"{name}_truth_sd"] = float(truth_sd[coordinate])
            rows.append(row)
    return rows


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


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstreams = load_upstreams(require_frozen_jiang_commit=True)
    hmm = TwoTransitionHMMConfig(
        length=int(args.length),
        block_size=int(args.block_size),
        init_prob=float(args.init_prob),
        p01_min=float(args.p01_min),
        p01_max=float(args.p01_max),
        p01_true=float(args.p01_true),
        p11_min=float(args.p11_min),
        p11_max=float(args.p11_max),
        p11_true=float(args.p11_true),
    )
    hmm.validate()
    emission = SparseScaleMixtureEmission(
        rho=float(args.rho), scale=float(args.scale)
    )
    sigma_q = np.asarray([args.sigma_q_p01, args.sigma_q_p11], dtype=np.float64)
    training = KhooTrainingConfig(
        n_train=int(args.n_train),
        n_val=int(args.n_val),
        sigma_q=float(np.mean(sigma_q)),
        pi_ref=float(args.pi_ref),
        gru_hidden=int(args.gru_hidden),
        gate_hidden=int(args.gate_hidden),
        iters=int(args.iters),
        gate_only_steps=int(args.gate_only_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        gate_lr=float(args.gate_lr),
        joint_lr=float(args.joint_lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        ema_decay=float(args.ema_decay),
        patience=int(args.patience),
        print_every=int(args.print_every),
    )
    fitted = fit_models_2d(
        upstreams=upstreams,
        hmm=hmm,
        emission=emission,
        training=training,
        sigma_q=sigma_q,
        seed=int(args.seed),
        device=args.device,
    )
    theta_values = _parse_theta_values(args.theta_values)
    rows = score_diagnostic_2d(
        fitted,
        upstreams,
        theta_values,
        n_per_value=int(args.diagnostic_n),
        seed=int(args.seed) + 10_000,
        device=args.device,
    )
    aggregate: dict[str, Any] = {}
    for method in ("linear", "gate"):
        selected = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            "mean_std_mse": float(np.mean([row["std_mse"] for row in selected])),
            "mean_corr": float(np.mean([row["mean_corr"] for row in selected])),
            **{
                f"{name}_mean_std_mse": float(
                    np.mean([row[f"{name}_std_mse"] for row in selected])
                )
                for name in PARAMETER_NAMES
            },
            **{
                f"{name}_mean_corr": float(
                    np.mean([row[f"{name}_corr"] for row in selected])
                )
                for name in PARAMETER_NAMES
            },
        }
    aggregate["gate_relative_std_mse_improvement"] = float(
        (aggregate["linear"]["mean_std_mse"] - aggregate["gate"]["mean_std_mse"])
        / aggregate["linear"]["mean_std_mse"]
    )
    result = {
        "hmm": hmm.to_dict(),
        "emission": asdict(emission),
        "training": asdict(training),
        "sigma_q": sigma_q.tolist(),
        "seed": int(args.seed),
        "training_diagnostics": fitted.diagnostics,
        "score_rows": rows,
        "aggregate": aggregate,
        "training_target": "two-dimensional Direct-FSM proposal target only",
        "analytic_local_emission_ratio_used": True,
        "exact_full_hmm_score_role": "evaluation only",
        "true_parameter_used_for_training": False,
    }
    torch.save(
        {
            "linear_state_dict": fitted.linear_model.state_dict(),
            "gate_state_dict": fitted.gate_model.state_dict(),
            "stats": fitted.stats,
            "hmm": hmm.to_dict(),
            "emission": asdict(emission),
            "training": asdict(training),
            "sigma_q": sigma_q,
        },
        output / "models.pt",
    )
    (output / "summary.json").write_text(
        json.dumps(_json_value(result), indent=2), encoding="utf-8"
    )
    print(json.dumps(_json_value(aggregate), indent=2))
    print("saved to", output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--init-prob", type=float, default=0.5)
    parser.add_argument("--p01-min", type=float, default=0.02)
    parser.add_argument("--p01-max", type=float, default=0.20)
    parser.add_argument("--p01-true", type=float, default=0.06)
    parser.add_argument("--p11-min", type=float, default=0.80)
    parser.add_argument("--p11-max", type=float, default=0.99)
    parser.add_argument("--p11-true", type=float, default=0.94)
    parser.add_argument("--rho", type=float, default=0.20)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--n-train", type=int, default=15_000)
    parser.add_argument("--n-val", type=int, default=3_000)
    parser.add_argument("--sigma-q-p01", type=float, default=0.25)
    parser.add_argument("--sigma-q-p11", type=float, default=0.25)
    parser.add_argument("--pi-ref", type=float, default=0.30)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--iters", type=int, default=1_500)
    parser.add_argument("--gate-only-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gate-lr", type=float, default=1e-4)
    parser.add_argument("--joint-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--diagnostic-n", type=int, default=1_000)
    parser.add_argument(
        "--theta-values",
        default="0.04:0.88,0.06:0.94,0.10:0.94,0.06:0.97,0.14:0.90",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
