"""Matched linear-vs-gated FSM screen under a nonlinear HMM emission.

The transition model is unchanged from the main benchmark.  The only change is
the conditional emission distribution.  Under hidden state zero, coordinates
are standard normal.  Under hidden state one, each coordinate independently
comes from a sparse scale mixture

    (1-rho) N(0, 1) + rho N(0, scale**2).

The state-one/state-zero log likelihood ratio is available in closed form, so
the exact HMM transition score remains available for evaluation.  It is never
used as a training target: both learned arms use the same Direct-FSM proposal
target.  The arms differ only by the positive nonlinear local gate copied from
the existing Khoo implementation.
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

from .dgp import HMMConfig, logit
from .jiang_official import resolve_device
from .khoo_original import KhooTrainingConfig
from .upstreams import UpstreamModules, load_upstreams


@dataclass(frozen=True)
class SparseScaleMixtureEmission:
    """Independent-coordinate, heavy-tailed state-one emission."""

    rho: float = 0.10
    scale: float = 5.0

    def validate(self) -> None:
        if not 0.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (0,1)")
        if self.scale <= 1.0:
            raise ValueError("scale must exceed one")


@dataclass
class NonlinearEmissionFitted:
    linear_model: torch.nn.Module
    gate_model: torch.nn.Module
    stats: dict[str, np.ndarray]
    hmm: HMMConfig
    emission: SparseScaleMixtureEmission
    training: KhooTrainingConfig
    diagnostics: dict[str, Any]


def load_fitted(
    path: Path,
    upstreams: UpstreamModules,
    *,
    device: str = "auto",
) -> NonlinearEmissionFitted:
    """Load a fitted matched-ablation checkpoint without touching upstreams."""
    device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    hmm = HMMConfig(**checkpoint["hmm"])
    emission = SparseScaleMixtureEmission(**checkpoint["emission"])
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
    return NonlinearEmissionFitted(
        linear_model=linear,
        gate_model=gate,
        stats=stats,
        hmm=hmm,
        emission=emission,
        training=training,
        diagnostics={},
    )


def coordinate_log_ratio(
    y: np.ndarray, emission: SparseScaleMixtureEmission
) -> np.ndarray:
    """log f_1(y)/f_0(y) for one or more scalar coordinates."""
    emission.validate()
    y = np.asarray(y, dtype=np.float64)
    log_base = np.log1p(-float(emission.rho))
    log_tail = (
        np.log(float(emission.rho))
        - np.log(float(emission.scale))
        + 0.5 * np.square(y) * (1.0 - 1.0 / float(emission.scale) ** 2)
    )
    return np.logaddexp(log_base, log_tail)


def full_emission_log_ratio(
    y: np.ndarray, emission: SparseScaleMixtureEmission
) -> np.ndarray:
    """Exact per-time state-one/state-zero log likelihood ratio."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError("y must have shape (n,length,block_size)")
    return coordinate_log_ratio(y, emission).sum(axis=2)


def pairwise_emission_log_ratio(
    coordinate_lr: np.ndarray,
) -> np.ndarray:
    """Exact bivariate log ratios for every unordered coordinate pair."""
    coordinate_lr = np.asarray(coordinate_lr, dtype=np.float64)
    if coordinate_lr.ndim != 3:
        raise ValueError("coordinate_lr must have shape (n,length,block_size)")
    pairs = [
        coordinate_lr[:, :, i] + coordinate_lr[:, :, j]
        for i in range(coordinate_lr.shape[2])
        for j in range(i + 1, coordinate_lr.shape[2])
    ]
    if not pairs:
        raise ValueError("block_size must be at least two")
    return np.stack(pairs, axis=2)


def raw_subscores(
    y: np.ndarray,
    emission: SparseScaleMixtureEmission,
    pi_ref: float,
    upstreams: UpstreamModules,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginal and pairwise local mixture scores used by both learned arms."""
    coordinate_lr = coordinate_log_ratio(y, emission)
    pair_lr = pairwise_emission_log_ratio(coordinate_lr)
    mapping = upstreams.khoo_fixed.local_score_u_from_log_ratio
    return mapping(coordinate_lr, float(pi_ref)), mapping(pair_lr, float(pi_ref))


def simulate_trajectories(
    rng: np.random.Generator,
    p11: np.ndarray | float,
    hmm: HMMConfig,
    emission: SparseScaleMixtureEmission,
    *,
    n: int | None = None,
    return_states: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate independent HMM trajectories from the scale-mixture model."""
    hmm.validate()
    emission.validate()
    values = np.asarray(p11, dtype=np.float64).reshape(-1)
    if values.size == 1 and n is not None:
        values = np.full(int(n), float(values[0]), dtype=np.float64)
    elif n is not None and values.size != int(n):
        raise ValueError("p11 and n imply inconsistent sample sizes")
    if values.size < 1 or np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("all p11 values must lie in (0,1)")

    states = np.zeros((values.size, hmm.length), dtype=np.int64)
    states[:, 0] = rng.binomial(1, hmm.init_prob, size=values.size)
    for time in range(1, hmm.length):
        probability = np.where(
            states[:, time - 1] == 1, values, float(hmm.fixed_p01)
        )
        states[:, time] = rng.binomial(1, probability)

    tail = rng.binomial(
        1, emission.rho, size=(values.size, hmm.length, hmm.block_size)
    )
    standard_deviation = np.where(tail == 1, emission.scale, 1.0)
    standard_deviation = np.where(states[:, :, None] == 1, standard_deviation, 1.0)
    y = (rng.normal(size=standard_deviation.shape) * standard_deviation).astype(
        np.float32
    )
    if return_states:
        return y, states
    return y


def sample_from_anchors(
    rng: np.random.Generator,
    anchor_p: np.ndarray,
    sigma_q: float,
    hmm: HMMConfig,
    emission: SparseScaleMixtureEmission,
) -> dict[str, np.ndarray]:
    anchor_p = np.asarray(anchor_p, dtype=np.float64).reshape(-1)
    anchor_u = logit(anchor_p)
    u = anchor_u + rng.normal(size=anchor_u.shape) * float(sigma_q)
    p = 1.0 / (1.0 + np.exp(-u))
    y = simulate_trajectories(rng, p, hmm, emission)
    target = ((u - anchor_u) / float(sigma_q) ** 2)[:, None]
    return {
        "y": y,
        "anchor_p": anchor_p[:, None],
        "anchor_u": anchor_u[:, None],
        "u": u[:, None],
        "p": p[:, None],
        "target": target.astype(np.float32),
    }


def sample_amortized(
    rng: np.random.Generator,
    n: int,
    hmm: HMMConfig,
    emission: SparseScaleMixtureEmission,
    sigma_q: float,
) -> dict[str, np.ndarray]:
    strata = (np.arange(int(n), dtype=np.float64) + rng.uniform(size=int(n))) / int(n)
    rng.shuffle(strata)
    anchors = hmm.p_min + (hmm.p_max - hmm.p_min) * strata
    return sample_from_anchors(rng, anchors, sigma_q, hmm, emission)


def fit_features(
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
    return transform_features(raw, emission, pi_ref, stats, upstreams), stats


def transform_features(
    raw: dict[str, np.ndarray],
    emission: SparseScaleMixtureEmission,
    pi_ref: float,
    stats: dict[str, np.ndarray],
    upstreams: UpstreamModules,
) -> dict[str, np.ndarray]:
    s1, s2 = raw_subscores(raw["y"], emission, pi_ref, upstreams)
    return {
        "s1": ((s1 - stats["s1_mean"]) / stats["s1_sd"]).astype(np.float32),
        "s2": ((s2 - stats["s2_mean"]) / stats["s2_sd"]).astype(np.float32),
        "anchor_z": (
            (raw["anchor_u"] - stats["anchor_u_mean"])
            / stats["anchor_u_sd"]
        ).astype(np.float32),
    }


def exact_score_u(
    y: np.ndarray,
    p11: float,
    hmm: HMMConfig,
    emission: SparseScaleMixtureEmission,
    upstreams: UpstreamModules,
) -> np.ndarray:
    score = upstreams.khoo_fixed.hmm_transition_score_u(
        full_emission_log_ratio(y, emission),
        float(hmm.fixed_p01),
        float(p11),
        float(hmm.init_prob),
    )
    return np.asarray(score[:, 1:2], dtype=np.float64)


def fit_models(
    *,
    upstreams: UpstreamModules,
    hmm: HMMConfig,
    emission: SparseScaleMixtureEmission,
    training: KhooTrainingConfig,
    seed: int,
    device: str,
) -> NonlinearEmissionFitted:
    device = resolve_device(device)
    one = upstreams.khoo_oneparam
    one.base.set_global_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    raw_train = sample_amortized(
        rng, training.n_train, hmm, emission, training.sigma_q
    )
    raw_val = sample_amortized(rng, training.n_val, hmm, emission, training.sigma_q)
    train_features, stats = fit_features(
        raw_train, emission, training.pi_ref, upstreams
    )
    val_features = transform_features(
        raw_val, emission, training.pi_ref, stats, upstreams
    )
    train = {**train_features, "target": raw_train["target"]}
    val = {**val_features, "target": raw_val["target"]}
    args = argparse.Namespace(**asdict(training), seed=int(seed))

    linear = one.LinearLocalGRU(
        training.gru_hidden, stats["time_mean"], stats["time_sd"]
    )
    linear, linear_trace, linear_info = one.train_model(
        linear, "linear", train, val, args, device, 101
    )
    gate = one.GatedLocalGRU(
        training.gru_hidden,
        training.gate_hidden,
        stats["time_mean"],
        stats["time_sd"],
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
        gate, "gate", train, val, args, device, 202
    )
    return NonlinearEmissionFitted(
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


def score_diagnostic(
    fitted: NonlinearEmissionFitted,
    upstreams: UpstreamModules,
    p_values: list[float],
    *,
    n_per_value: int,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    device = resolve_device(device)
    one = upstreams.khoo_oneparam
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for p in p_values:
        y = simulate_trajectories(
            rng, p, fitted.hmm, fitted.emission, n=n_per_value
        )
        raw = {
            "y": y,
            "anchor_u": np.full((n_per_value, 1), float(logit(p))),
        }
        features = transform_features(
            raw,
            fitted.emission,
            fitted.training.pi_ref,
            fitted.stats,
            upstreams,
        )
        truth = exact_score_u(y, p, fitted.hmm, fitted.emission, upstreams)
        truth_sd = max(float(truth.std()), 1e-12)
        full_lr = full_emission_log_ratio(y, fitted.emission)
        linear_time = np.stack(
            [features["s1"].mean(axis=2), features["s2"].mean(axis=2)], axis=2
        )
        for method, model in (
            ("linear", fitted.linear_model),
            ("gate", fitted.gate_model),
        ):
            prediction = one.predict(
                model, method, features, device, fitted.training.batch_size
            )
            error = prediction - truth
            rows.append(
                {
                    "method": method,
                    "p": float(p),
                    "n": int(n_per_value),
                    "mse": float(np.mean(np.square(error))),
                    "std_mse": float(np.mean(np.square(error / truth_sd))),
                    "corr": float(
                        np.corrcoef(prediction[:, 0], truth[:, 0])[0, 1]
                    ),
                    "prediction_mean": float(prediction.mean()),
                    "truth_mean": float(truth.mean()),
                    "truth_sd": truth_sd,
                    "linear_time_to_exact_emission_lr_corr": float(
                        np.corrcoef(
                            linear_time.reshape(-1, 2)[:, 0], full_lr.reshape(-1)
                        )[0, 1]
                    ),
                }
            )
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
    hmm = HMMConfig(
        length=int(args.length),
        block_size=int(args.block_size),
        fixed_p01=float(args.fixed_p01),
        init_prob=float(args.init_prob),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        p_true=float(args.p_true),
    )
    emission = SparseScaleMixtureEmission(rho=float(args.rho), scale=float(args.scale))
    training = KhooTrainingConfig(
        n_train=int(args.n_train),
        n_val=int(args.n_val),
        sigma_q=float(args.sigma_q),
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
    fitted = fit_models(
        upstreams=upstreams,
        hmm=hmm,
        emission=emission,
        training=training,
        seed=int(args.seed),
        device=args.device,
    )
    p_values = [float(value) for value in args.p_values.split(",")]
    rows = score_diagnostic(
        fitted,
        upstreams,
        p_values,
        n_per_value=int(args.diagnostic_n),
        seed=int(args.seed) + 10_000,
        device=args.device,
    )
    aggregate: dict[str, Any] = {}
    for method in ("linear", "gate"):
        selected = [row for row in rows if row["method"] == method]
        aggregate[method] = {
            "mean_std_mse": float(np.mean([row["std_mse"] for row in selected])),
            "mean_corr": float(np.mean([row["corr"] for row in selected])),
        }
    aggregate["gate_relative_std_mse_improvement"] = float(
        (aggregate["linear"]["mean_std_mse"] - aggregate["gate"]["mean_std_mse"])
        / aggregate["linear"]["mean_std_mse"]
    )
    result = {
        "hmm": hmm.to_dict(),
        "emission": asdict(emission),
        "training": asdict(training),
        "seed": int(args.seed),
        "training_diagnostics": fitted.diagnostics,
        "score_rows": rows,
        "aggregate": aggregate,
        "training_target": "Direct-FSM proposal target only",
        "exact_score_role": "evaluation only",
    }
    torch.save(
        {
            "linear_state_dict": fitted.linear_model.state_dict(),
            "gate_state_dict": fitted.gate_model.state_dict(),
            "stats": fitted.stats,
            "hmm": hmm.to_dict(),
            "emission": asdict(emission),
            "training": asdict(training),
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
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--fixed-p01", type=float, default=0.06)
    parser.add_argument("--init-prob", type=float, default=0.5)
    parser.add_argument("--p-min", type=float, default=0.80)
    parser.add_argument("--p-max", type=float, default=0.99)
    parser.add_argument("--p-true", type=float, default=0.94)
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--sigma-q", type=float, default=0.25)
    parser.add_argument("--pi-ref", type=float, default=0.30)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--iters", type=int, default=1_000)
    parser.add_argument("--gate-only-steps", type=int, default=250)
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
    parser.add_argument("--p-values", default="0.84,0.90,0.94,0.97")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
