"""Paper-native Jiang baseline for the original block-mixture Models 1 and 2.

The only experiment-specific component is the simulator.  Each raw block is
one outer iid observation and is passed directly to the authors' ELU MLP.
Training, Fisher/Bartlett regularization, conditional debiasing, two-round
proposal refinement, score-root inference, and confidence intervals are
provided by :mod:`khoo_vs_jiang.jiang_official`, which imports the authors'
frozen ``MLE.utils_sm`` implementation.

No composite score, likelihood-ratio feature, GRU, or NPE is supplied to the
Jiang estimator.  Exact likelihood ratios are used only after fitting for
diagnostic scores and an evaluation-only exact MLE.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from .dgp import HMMConfig
from .jiang_official import (
    GaussianProposal,
    JiangTrainingConfig,
    debiased_scores,
    fit_jiang_round,
    infer_dataset,
    save_fitted,
)
from .upstreams import load_upstreams


@dataclass(frozen=True)
class BlockMixtureConfig:
    model: str
    block_size: int
    tau: float
    p_min: float
    p_max: float
    p_true: float
    n_observations: int

    def validate(self) -> None:
        if self.model not in {"model1", "model2"}:
            raise ValueError("model must be 'model1' or 'model2'")
        if self.block_size < 2 or self.block_size % 2:
            raise ValueError("block_size must be an even integer of at least two")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if not 0.0 < self.p_min < self.p_true < self.p_max < 1.0:
            raise ValueError("require 0 < p_min < p_true < p_max < 1")
        if self.n_observations < 2:
            raise ValueError("n_observations must be at least two")

    @property
    def label(self) -> str:
        return (
            "blockwise mean-shift mixture"
            if self.model == "model1"
            else "blockwise common-factor variance mixture"
        )


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), indent=2), encoding="utf-8")


def _container_config(config: BlockMixtureConfig) -> HMMConfig:
    """Return the geometry/parameter container expected by Jiang's adapter.

    ``jiang_official`` only requires ``x_dim``, parameter bounds, and metadata
    from this object when a custom simulator is supplied.  Splitting a raw
    length-m block into a 2 by (m/2) array preserves the flattened observation
    exactly and does not introduce a sequence model.
    """

    config.validate()
    return HMMConfig(
        length=2,
        block_size=config.block_size // 2,
        tau=config.tau,
        fixed_p01=0.1,
        init_prob=0.5,
        p_min=config.p_min,
        p_max=config.p_max,
        p_true=config.p_true,
    )


def simulate_raw_blocks(
    rng: np.random.Generator,
    p: np.ndarray | float,
    config: BlockMixtureConfig,
    *,
    n: int | None = None,
) -> np.ndarray:
    """Simulate iid raw blocks, returned in the adapter's 3-D container."""

    config.validate()
    probability = np.asarray(p, dtype=np.float64).reshape(-1)
    if probability.size == 1 and n is not None:
        probability = np.full(int(n), float(probability[0]), dtype=np.float64)
    elif n is not None and probability.size != int(n):
        raise ValueError("p and n imply inconsistent sample sizes")
    if probability.size < 1 or np.any(probability <= 0.0) or np.any(probability >= 1.0):
        raise ValueError("all mixture probabilities must lie in (0,1)")

    latent = rng.uniform(size=probability.size) < probability
    noise = rng.standard_normal((probability.size, config.block_size))
    if config.model == "model1":
        raw = noise + config.tau * latent[:, None]
    else:
        factor = rng.standard_normal(probability.size)
        raw = noise + config.tau * (latent * factor)[:, None]
    return raw.reshape(probability.size, 2, config.block_size // 2).astype(np.float32)


def block_log_likelihood_ratio(y: np.ndarray, config: BlockMixtureConfig) -> np.ndarray:
    """Evaluation-only log f1(y)/f0(y) for one raw block."""

    raw = np.asarray(y, dtype=np.float64).reshape(-1, config.block_size)
    if config.model == "model1":
        return config.tau * raw.sum(axis=1) - 0.5 * config.block_size * config.tau**2
    denominator = 1.0 + config.block_size * config.tau**2
    return (
        -0.5 * np.log(denominator)
        + config.tau**2 * np.square(raw.sum(axis=1)) / (2.0 * denominator)
    )


def exact_score_p(
    y: np.ndarray,
    p: float,
    config: BlockMixtureConfig,
) -> np.ndarray:
    """Evaluation-only single-block likelihood score in physical p."""

    log_ratio = block_log_likelihood_ratio(y, config)
    logit_p = math.log(float(p)) - math.log1p(-float(p))
    posterior = 1.0 / (1.0 + np.exp(-(logit_p + log_ratio)))
    score_u = posterior - float(p)
    return (score_u / (float(p) * (1.0 - float(p)))).reshape(-1, 1)


def exact_mle(y: np.ndarray, config: BlockMixtureConfig) -> float:
    """Evaluation-only exact MLE from the tractable benchmark likelihood."""

    log_ratio = block_log_likelihood_ratio(y, config)

    def negative_log_likelihood(p: float) -> float:
        return -float(
            np.sum(
                np.logaddexp(math.log1p(-float(p)), math.log(float(p)) + log_ratio)
            )
        )

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(config.p_min, config.p_max),
        method="bounded",
        options={"xatol": 1e-10},
    )
    candidates = [
        (float(result.x), float(result.fun)),
        (config.p_min, negative_log_likelihood(config.p_min)),
        (config.p_max, negative_log_likelihood(config.p_max)),
    ]
    return min(candidates, key=lambda item: item[1])[0]


def _training_configs(profile: str) -> tuple[JiangTrainingConfig, JiangTrainingConfig]:
    if profile == "smoke":
        first = JiangTrainingConfig.smoke()
        return first, replace(first, lambda_fisher=1e-3)
    if profile == "pilot":
        first = JiangTrainingConfig(
            training_size=3_000,
            batch_size=32,
            coord_epochs=30,
            joint_epochs=30,
            fisher_epochs=15,
            extra_sample_size=200,
            extra_obs_size=100,
            extra_batch_size=10,
            debias_epochs=100,
            debias_fisher_epochs=15,
        )
        return first, replace(
            first,
            fisher_epochs=10,
            fisher_lr=1e-5,
            lambda_fisher=1e-3,
        )
    if profile != "paper":
        raise ValueError(profile)
    first = JiangTrainingConfig()
    return first, replace(
        first,
        fisher_epochs=20,
        fisher_lr=1e-5,
        lambda_fisher=1e-3,
    )


def _proposal(
    round1: dict[str, Any],
    config: BlockMixtureConfig,
) -> GaussianProposal:
    covariance = float(round1["covariance"]["Cov_sand"])
    if not np.isfinite(covariance) or covariance <= 0.0:
        raise RuntimeError("round-one sandwich covariance is not positive and finite")
    return GaussianProposal(
        mean=float(
            np.clip(round1["root"]["p_hat"], config.p_min, config.p_max)
        ),
        sd=max(
            6.0 * np.sqrt(covariance / int(config.n_observations)),
            1e-6,
        ),
    )


def _score_diagnostic(
    fitted: Any,
    config: BlockMixtureConfig,
    p_values: list[float],
    *,
    n_per_value: int,
    seed: int,
    device: str,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float]] = []
    for p in p_values:
        y = simulate_raw_blocks(rng, p, config, n=int(n_per_value))
        prediction = debiased_scores(
            fitted,
            p,
            y,
            device=device,
            physical_coordinate=True,
        )
        truth = exact_score_p(y, p, config)
        truth_sd = max(float(np.std(truth)), 1e-12)
        error = prediction - truth
        rows.append(
            {
                "p": float(p),
                "n": int(n_per_value),
                "mse": float(np.mean(np.square(error))),
                "std_mse": float(np.mean(np.square(error / truth_sd))),
                "corr": float(np.corrcoef(prediction[:, 0], truth[:, 0])[0, 1]),
                "prediction_mean": float(np.mean(prediction)),
                "truth_mean": float(np.mean(truth)),
                "truth_sd": truth_sd,
            }
        )
    return rows


def _repeated_round1(
    fitted: Any,
    config: BlockMixtureConfig,
    *,
    replicates: int,
    seed: int,
    root_maxiter: int,
    device: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for replicate in range(int(replicates)):
        replicate_seed = int(seed) + 10_000 * replicate
        y = simulate_raw_blocks(
            np.random.default_rng(replicate_seed),
            config.p_true,
            config,
            n=config.n_observations,
        )
        inference = infer_dataset(
            fitted,
            y,
            num_bootstrap=0,
            seed=replicate_seed + 1,
            device=device,
            maxiter=int(root_maxiter),
        )
        row: dict[str, Any] = {
            "replicate": replicate,
            "seed": replicate_seed,
            "p_hat": inference["root"]["p_hat"],
            "score_converged": inference["root"]["score_converged"],
            "boundary": bool(
                inference["root"]["at_lower_bound"]
                or inference["root"]["at_upper_bound"]
            ),
            "exact_mle": exact_mle(y, config),
        }
        for covariance, interval in inference["normal_intervals"].items():
            row[f"{covariance}_lower"] = interval["lower"]
            row[f"{covariance}_upper"] = interval["upper"]
            row[f"{covariance}_width"] = interval["width"]
        rows.append(row)

    estimates = np.asarray([row["p_hat"] for row in rows], dtype=np.float64)
    exact = np.asarray([row["exact_mle"] for row in rows], dtype=np.float64)
    summary: dict[str, Any] = {
        "replicates": int(replicates),
        "p_true": config.p_true,
        "bias": float(np.mean(estimates - config.p_true)),
        "rmse_to_true": float(np.sqrt(np.mean(np.square(estimates - config.p_true)))),
        "rmse_to_exact_mle": float(np.sqrt(np.mean(np.square(estimates - exact)))),
        "exact_mle_rmse_to_true": float(
            np.sqrt(np.mean(np.square(exact - config.p_true)))
        ),
        "convergence_rate": float(np.mean([row["score_converged"] for row in rows])),
        "boundary_rate": float(np.mean([row["boundary"] for row in rows])),
    }
    for covariance in ("Cov_ss", "Cov_curv", "Cov_sand"):
        lower = np.asarray([row[f"{covariance}_lower"] for row in rows])
        upper = np.asarray([row[f"{covariance}_upper"] for row in rows])
        finite = np.isfinite(lower) & np.isfinite(upper)
        summary[covariance] = {
            "finite_rate": float(np.mean(finite)),
            "coverage": float(
                np.mean(
                    (lower[finite] <= config.p_true)
                    & (config.p_true <= upper[finite])
                )
            )
            if np.any(finite)
            else float("nan"),
            "mean_width": float(np.mean(upper[finite] - lower[finite]))
            if np.any(finite)
            else float("nan"),
        }
    return {"summary": summary, "rows": rows}


def run(args: argparse.Namespace) -> None:
    defaults = {
        "model1": {"tau": 0.5, "n_observations": 20},
        "model2": {"tau": 1.0, "n_observations": 40},
    }[args.model]
    config = BlockMixtureConfig(
        model=args.model,
        block_size=int(args.block_size),
        tau=float(defaults["tau"] if args.tau is None else args.tau),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        p_true=float(args.p_true),
        n_observations=int(
            defaults["n_observations"]
            if args.n_observations is None
            else args.n_observations
        ),
    )
    config.validate()
    container = _container_config(config)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstreams = load_upstreams(require_frozen_jiang_commit=True)
    first_config, second_config = _training_configs(args.profile)

    override_values = {
        name: value
        for name, value in {
            "training_size": args.training_size_override,
            "batch_size": args.batch_size_override,
            "coord_epochs": args.coord_epochs_override,
            "joint_epochs": args.joint_epochs_override,
            "fisher_epochs": args.fisher_epochs_override,
            "extra_sample_size": args.extra_sample_size_override,
            "extra_obs_size": args.extra_obs_size_override,
            "debias_epochs": args.debias_epochs_override,
            "debias_fisher_epochs": args.debias_fisher_epochs_override,
        }.items()
        if value is not None
    }
    if override_values:
        first_config = replace(first_config, **override_values)
        second_config = replace(second_config, **override_values)

    def simulator(rng, p, ignored_container, ignored_upstreams):
        del ignored_container, ignored_upstreams
        return simulate_raw_blocks(rng, p, config)

    observed = simulate_raw_blocks(
        np.random.default_rng(int(args.observed_seed)),
        config.p_true,
        config,
        n=config.n_observations,
    )
    np.save(output / "observed_raw_iid_blocks.npy", observed)
    _write_json(
        output / "config.json",
        {
            "experiment": asdict(config),
            "model_label": config.label,
            "profile": args.profile,
            "training_seed": int(args.training_seed),
            "observed_seed": int(args.observed_seed),
            "round1": asdict(first_config),
            "round2": asdict(second_config),
            "score_network_input": "[physical parameter after official scaling, raw block]",
            "outer_iid_unit": f"one raw R^{config.block_size} block",
            "full_data_score": "sum over raw iid blocks",
            "official_jiang_architecture": "ELU MLP",
            "official_jiang_loss_and_debias_imported": True,
            "two_round": True,
            "npe_for_jiang": False,
            "composite_score_features_for_jiang": False,
            "exact_likelihood_role": "evaluation only",
            "overrides": override_values,
        },
    )

    print(f"=== {config.model}: official Jiang round 1 ===")
    round1 = fit_jiang_round(
        upstreams=upstreams,
        hmm=container,
        config=first_config,
        proposal=None,
        round_id=1,
        seed=int(args.training_seed),
        device=args.device,
        score_architecture="official_mlp",
        simulator=simulator,
    )
    save_fitted(output / "models/jiang_round1.pt", round1)
    inference1 = infer_dataset(
        round1,
        observed,
        num_bootstrap=int(args.num_bootstrap),
        seed=int(args.training_seed) + 1,
        device=args.device,
        maxiter=int(args.root_maxiter),
    )
    roots = inference1.pop("bootstrap_roots", None)
    if roots is not None:
        np.save(output / "inference/round1_bootstrap_roots.npy", roots)
    _write_json(output / "inference/round1.json", inference1)

    proposal = _proposal(inference1, config)
    _write_json(output / "inference/round2_proposal.json", asdict(proposal))
    print(f"=== {config.model}: official Jiang round 2 ===")
    round2 = fit_jiang_round(
        upstreams=upstreams,
        hmm=container,
        config=second_config,
        proposal=proposal,
        round_id=2,
        seed=int(args.training_seed) + 10_000,
        device=args.device,
        score_architecture="official_mlp",
        simulator=simulator,
    )
    save_fitted(output / "models/jiang_round2.pt", round2)
    inference2 = infer_dataset(
        round2,
        observed,
        previous_round_p=float(inference1["root"]["p_hat"]),
        num_bootstrap=int(args.num_bootstrap),
        seed=int(args.training_seed) + 10_001,
        device=args.device,
        maxiter=int(args.root_maxiter),
    )
    roots = inference2.pop("bootstrap_roots", None)
    if roots is not None:
        np.save(output / "inference/round2_bootstrap_roots.npy", roots)
    _write_json(output / "inference/round2.json", inference2)

    p_values = [float(value) for value in args.p_values.split(",")]
    diagnostic1 = _score_diagnostic(
        round1,
        config,
        p_values,
        n_per_value=int(args.diagnostic_n),
        seed=int(args.diagnostic_seed),
        device=args.device,
    )
    diagnostic2 = _score_diagnostic(
        round2,
        config,
        p_values,
        n_per_value=int(args.diagnostic_n),
        seed=int(args.diagnostic_seed),
        device=args.device,
    )
    _write_json(
        output / "score_diagnostic.json",
        {"round1": diagnostic1, "round2": diagnostic2},
    )

    repeated = None
    if int(args.round1_replicates) > 0:
        repeated = _repeated_round1(
            round1,
            config,
            replicates=int(args.round1_replicates),
            seed=int(args.repeated_seed),
            root_maxiter=int(args.root_maxiter),
            device=args.device,
        )
        _write_json(output / "round1_repeated.json", repeated)

    exact = exact_mle(observed, config)
    summary = {
        "experiment": asdict(config),
        "exact_mle_observed": exact,
        "round1": {
            "p_hat": inference1["root"]["p_hat"],
            "absolute_error_to_true": abs(inference1["root"]["p_hat"] - config.p_true),
            "absolute_error_to_exact_mle": abs(inference1["root"]["p_hat"] - exact),
            "root": inference1["root"],
            "normal_intervals": inference1["normal_intervals"],
            "mean_score_std_mse": float(np.mean([row["std_mse"] for row in diagnostic1])),
            "mean_score_corr": float(np.mean([row["corr"] for row in diagnostic1])),
        },
        "round2": {
            "proposal": asdict(proposal),
            "p_hat": inference2["root"]["p_hat"],
            "absolute_error_to_true": abs(inference2["root"]["p_hat"] - config.p_true),
            "absolute_error_to_exact_mle": abs(inference2["root"]["p_hat"] - exact),
            "root": inference2["root"],
            "normal_intervals": inference2["normal_intervals"],
            "mean_score_std_mse": float(np.mean([row["std_mse"] for row in diagnostic2])),
            "mean_score_corr": float(np.mean([row["corr"] for row in diagnostic2])),
        },
        "round1_repeated": None if repeated is None else repeated["summary"],
        "interpretation": {
            "single_dataset_round2": "official data-dependent two-round result",
            "round1_repeated": "valid repeated-data evaluation of one observation-independent global score model",
            "round2_coverage": "requires retraining round 2 for every observed dataset and is not estimated here",
        },
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(_json_value(summary), indent=2))
    print("saved to", output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("model1", "model2"), required=True)
    parser.add_argument("--profile", choices=("smoke", "pilot", "paper"), default="pilot")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-seed", type=int, default=20260731)
    parser.add_argument("--observed-seed", type=int, default=20260732)
    parser.add_argument("--diagnostic-seed", type=int, default=20260733)
    parser.add_argument("--repeated-seed", type=int, default=20260734)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--p-min", type=float, default=0.05)
    parser.add_argument("--p-max", type=float, default=0.70)
    parser.add_argument("--p-true", type=float, default=0.30)
    parser.add_argument("--n-observations", type=int)
    parser.add_argument("--num-bootstrap", type=int, default=0)
    parser.add_argument("--round1-replicates", type=int, default=100)
    parser.add_argument("--diagnostic-n", type=int, default=1_000)
    parser.add_argument("--p-values", default="0.10,0.30,0.50,0.65")
    parser.add_argument("--root-maxiter", type=int, default=3_000)
    parser.add_argument("--training-size-override", type=int)
    parser.add_argument("--batch-size-override", type=int)
    parser.add_argument("--coord-epochs-override", type=int)
    parser.add_argument("--joint-epochs-override", type=int)
    parser.add_argument("--fisher-epochs-override", type=int)
    parser.add_argument("--extra-sample-size-override", type=int)
    parser.add_argument("--extra-obs-size-override", type=int)
    parser.add_argument("--debias-epochs-override", type=int)
    parser.add_argument("--debias-fisher-epochs-override", type=int)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
