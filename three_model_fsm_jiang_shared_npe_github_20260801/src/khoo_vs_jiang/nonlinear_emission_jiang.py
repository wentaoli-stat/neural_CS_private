"""Jiang two-round workflow on the nonlinear-emission HMM.

Only the simulator is changed in the default ``official_mlp`` arm.  The
controlled ``raw_sequence_gru`` arm changes only the single-observation score
architecture.  Both arms retain direct score matching, the Fisher penalty,
conditional debiasing, proposal rounds, root solver, and confidence-set
formulas.  Jiang never receives an NPE in this runner.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from .dgp import HMMConfig
from .jiang_official import (
    GaussianProposal,
    JiangTrainingConfig,
    debiased_scores,
    fit_jiang_round,
    infer_dataset,
    save_fitted,
)
from .nonlinear_emission_screen import (
    SparseScaleMixtureEmission,
    exact_score_u,
    simulate_trajectories,
)
from .upstreams import UpstreamModules, load_upstreams


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


def _configs(
    profile: str,
) -> tuple[JiangTrainingConfig, JiangTrainingConfig]:
    if profile == "smoke":
        first = JiangTrainingConfig.smoke()
        return first, replace(first, lambda_fisher=1e-3)
    if profile == "pilot":
        first = JiangTrainingConfig(
            training_size=3_000,
            coord_epochs=30,
            joint_epochs=30,
            fisher_epochs=15,
            extra_sample_size=200,
            extra_obs_size=100,
            debias_epochs=100,
            debias_fisher_epochs=15,
        )
        second = replace(
            first,
            fisher_epochs=10,
            fisher_lr=1e-5,
            lambda_fisher=1e-3,
        )
        return first, second
    if profile != "paper":
        raise ValueError(profile)
    first = JiangTrainingConfig()
    second = replace(
        first,
        fisher_epochs=20,
        fisher_lr=1e-5,
        lambda_fisher=1e-3,
    )
    return first, second


def _proposal(
    round1: dict[str, Any], n_observations: int, hmm: HMMConfig
) -> GaussianProposal:
    covariance = float(round1["covariance"]["Cov_sand"])
    if not np.isfinite(covariance) or covariance <= 0.0:
        raise RuntimeError("round-one sandwich covariance is not positive and finite")
    return GaussianProposal(
        mean=float(np.clip(round1["root"]["p_hat"], hmm.p_min, hmm.p_max)),
        sd=max(6.0 * np.sqrt(covariance / int(n_observations)), 1e-6),
    )


def score_diagnostic(
    fitted,
    upstreams: UpstreamModules,
    hmm: HMMConfig,
    emission: SparseScaleMixtureEmission,
    p_values: list[float],
    *,
    n_per_value: int,
    seed: int,
    device: str,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float]] = []
    for p in p_values:
        y = simulate_trajectories(rng, p, hmm, emission, n=n_per_value)
        prediction = debiased_scores(
            fitted, p, y, device=device, physical_coordinate=True
        )
        truth = exact_score_u(y, p, hmm, emission, upstreams) / (p * (1.0 - p))
        truth_sd = max(float(truth.std()), 1e-12)
        error = prediction - truth
        rows.append(
            {
                "p": float(p),
                "n": int(n_per_value),
                "mse": float(np.mean(np.square(error))),
                "std_mse": float(np.mean(np.square(error / truth_sd))),
                "corr": float(np.corrcoef(prediction[:, 0], truth[:, 0])[0, 1]),
                "prediction_mean": float(prediction.mean()),
                "truth_mean": float(truth.mean()),
                "truth_sd": truth_sd,
            }
        )
    return rows


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
    first_config, second_config = _configs(args.profile)
    if args.batch_size_override is not None:
        first_config = replace(first_config, batch_size=int(args.batch_size_override))
        second_config = replace(second_config, batch_size=int(args.batch_size_override))
    shared_overrides = {
        field: int(value)
        for field, value in (
            ("extra_sample_size", args.extra_sample_size_override),
            ("extra_obs_size", args.extra_obs_size_override),
            ("debias_epochs", args.debias_epochs_override),
            ("debias_fisher_epochs", args.debias_fisher_epochs_override),
        )
        if value is not None
    }
    if shared_overrides:
        first_config = replace(first_config, **shared_overrides)
        second_config = replace(second_config, **shared_overrides)
    if args.fisher_epochs_override is not None:
        first_config = replace(
            first_config, fisher_epochs=int(args.fisher_epochs_override)
        )
        second_config = replace(
            second_config, fisher_epochs=int(args.fisher_epochs_override)
        )
    architecture = str(args.score_architecture)
    method_description = (
        "official MLP"
        if architecture == "official_mlp"
        else "architecture-adapted raw-sequence GRU"
    )

    def simulator(rng, p, config, ignored_upstreams):
        del ignored_upstreams
        return simulate_trajectories(rng, p, config, emission)

    observed = simulate_trajectories(
        np.random.default_rng(int(args.observed_seed)),
        hmm.p_true,
        hmm,
        emission,
        n=int(args.n_observations),
    )
    np.save(output / "observed_iid_trajectories.npy", observed)
    _write_json(
        output / "config.json",
        {
            "profile": args.profile,
            "training_seed": int(args.training_seed),
            "observed_seed": int(args.observed_seed),
            "n_observations": int(args.n_observations),
            "hmm": hmm.to_dict(),
            "emission": asdict(emission),
            "round1": asdict(first_config),
            "round2": asdict(second_config),
            "jiang_method": (
                f"{method_description} + direct SM + Fisher penalty + conditional debias + "
                "two-round proposal + official roots/confidence sets"
            ),
            "score_architecture": architecture,
            "architecture_change_only": architecture == "raw_sequence_gru",
            "batch_size_override": args.batch_size_override,
            "published_jiang_batch_size": args.batch_size_override is None,
            "budget_overrides": {
                "fisher_epochs": args.fisher_epochs_override,
                "extra_sample_size": args.extra_sample_size_override,
                "extra_obs_size": args.extra_obs_size_override,
                "debias_epochs": args.debias_epochs_override,
                "debias_fisher_epochs": args.debias_fisher_epochs_override,
            },
            "published_jiang_budget": all(
                value is None
                for value in (
                    args.batch_size_override,
                    args.fisher_epochs_override,
                    args.extra_sample_size_override,
                    args.extra_obs_size_override,
                    args.debias_epochs_override,
                    args.debias_fisher_epochs_override,
                )
            ),
            "npe_for_jiang": False,
        },
    )

    print(f"=== Jiang {architecture} round 1 ===")
    round1 = fit_jiang_round(
        upstreams=upstreams,
        hmm=hmm,
        config=first_config,
        proposal=None,
        round_id=1,
        seed=int(args.training_seed),
        device=args.device,
        score_architecture=architecture,
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

    proposal = _proposal(inference1, int(args.n_observations), hmm)
    _write_json(output / "inference/round2_proposal.json", asdict(proposal))
    print(f"=== Jiang {architecture} round 2 ===")
    round2 = fit_jiang_round(
        upstreams=upstreams,
        hmm=hmm,
        config=second_config,
        proposal=proposal,
        round_id=2,
        seed=int(args.training_seed) + 10_000,
        device=args.device,
        score_architecture=architecture,
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
    diagnostic1 = score_diagnostic(
        round1,
        upstreams,
        hmm,
        emission,
        p_values,
        n_per_value=int(args.diagnostic_n),
        seed=int(args.diagnostic_seed),
        device=args.device,
    )
    diagnostic2 = score_diagnostic(
        round2,
        upstreams,
        hmm,
        emission,
        p_values,
        n_per_value=int(args.diagnostic_n),
        seed=int(args.diagnostic_seed),
        device=args.device,
    )
    _write_json(
        output / "score_diagnostic.json",
        {"round1": diagnostic1, "round2": diagnostic2},
    )
    summary = {
        "p_true": hmm.p_true,
        "round1": {
            "p_hat": inference1["root"]["p_hat"],
            "absolute_error": abs(inference1["root"]["p_hat"] - hmm.p_true),
            "root": inference1["root"],
            "normal_intervals": inference1["normal_intervals"],
            "mean_score_std_mse": float(
                np.mean([row["std_mse"] for row in diagnostic1])
            ),
            "mean_score_corr": float(np.mean([row["corr"] for row in diagnostic1])),
        },
        "round2": {
            "p_hat": inference2["root"]["p_hat"],
            "absolute_error": abs(inference2["root"]["p_hat"] - hmm.p_true),
            "root": inference2["root"],
            "normal_intervals": inference2["normal_intervals"],
            "mean_score_std_mse": float(
                np.mean([row["std_mse"] for row in diagnostic2])
            ),
            "mean_score_corr": float(np.mean([row["corr"] for row in diagnostic2])),
        },
        "warning": (
            "one observed dataset does not estimate frequentist coverage; "
            + (
                "the MLP arm is the published Jiang architecture"
                if architecture == "official_mlp"
                else "the GRU arm is architecture-adapted Jiang, not the published architecture"
            )
        ),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(_json_value(summary), indent=2))
    print("saved to", output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "pilot", "paper"), default="pilot")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-seed", type=int, default=20260726)
    parser.add_argument("--observed-seed", type=int, default=20260724)
    parser.add_argument("--diagnostic-seed", type=int, default=20260727)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--fixed-p01", type=float, default=0.06)
    parser.add_argument("--init-prob", type=float, default=0.5)
    parser.add_argument("--p-min", type=float, default=0.80)
    parser.add_argument("--p-max", type=float, default=0.99)
    parser.add_argument("--p-true", type=float, default=0.94)
    parser.add_argument("--rho", type=float, default=0.20)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--n-observations", type=int, default=100)
    parser.add_argument("--num-bootstrap", type=int, default=0)
    parser.add_argument("--diagnostic-n", type=int, default=1_000)
    parser.add_argument("--p-values", default="0.84,0.90,0.94,0.97")
    parser.add_argument("--root-maxiter", type=int, default=3_000)
    parser.add_argument("--batch-size-override", type=int)
    parser.add_argument("--fisher-epochs-override", type=int)
    parser.add_argument("--extra-sample-size-override", type=int)
    parser.add_argument("--extra-obs-size-override", type=int)
    parser.add_argument("--debias-epochs-override", type=int)
    parser.add_argument("--debias-fisher-epochs-override", type=int)
    parser.add_argument(
        "--score-architecture",
        choices=("official_mlp", "raw_sequence_gru"),
        default="official_mlp",
    )
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
