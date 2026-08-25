"""Matched Stage-2 NPE ablation for Jiang and the structured Khoo arm.

All learned-score arms receive the same simulated parameters, trajectories,
data-only pilot, NPE architecture, optimizer settings, and random seed.  Jiang
features use observation-independent Round-1 checkpoints.  A Jiang physical
parameter score is converted to the logit-coordinate score used by the Khoo
models before it enters the shared two-dimensional NPE context.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .jiang_official import (
    JiangFitted,
    debiased_scores_tensor,
    load_fitted as load_jiang,
    resolve_device,
)
from .nonlinear_emission_screen import load_fitted as load_ours, simulate_trajectories
from .nonlinear_emission_stage2 import (
    EXACT_LABEL,
    _stage2_upstream,
    data_only_pilot,
    exact_posterior,
    frozen_features,
)
from .upstreams import UpstreamModules, load_upstreams


JIANG_MLP_NPE = "Jiang official MLP score + shared NPE"
JIANG_GRU_NPE = "Jiang raw-sequence GRU score + shared NPE"
OURS_GATE_NPE = "local composite + nonlinear gate + GRU + shared NPE"
METHODS = (JIANG_MLP_NPE, JIANG_GRU_NPE, OURS_GATE_NPE)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_compatible(ours, jiang: JiangFitted, label: str) -> None:
    fields = ("length", "block_size", "fixed_p01", "init_prob", "p_min", "p_max")
    mismatches = [
        field
        for field in fields
        if not np.isclose(getattr(ours.hmm, field), getattr(jiang.hmm, field))
    ]
    if mismatches:
        raise ValueError(f"{label} HMM mismatch in {mismatches}")
    if int(jiang.round_id) != 1:
        raise ValueError(f"{label} must be an observation-independent Round-1 checkpoint")


def jiang_score_u_at_pilot(
    fitted: JiangFitted,
    y: np.ndarray,
    pilot_u: np.ndarray,
    *,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Return Jiang's debiased single-trajectory score in u=logit(p)."""
    device = resolve_device(device)
    y = np.asarray(y, dtype=np.float32)
    pilot_u = np.asarray(pilot_u, dtype=np.float64).reshape(-1)
    if y.ndim != 3 or y.shape[0] != pilot_u.size:
        raise ValueError("y and pilot_u batch sizes do not agree")
    pilot_p = 1.0 / (1.0 + np.exp(-pilot_u))
    flat = np.ascontiguousarray(y.reshape(y.shape[0], -1))
    output: list[np.ndarray] = []
    fitted.score_model.eval()
    fitted.debias_model.eval()
    for start in range(0, flat.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), flat.shape[0])
        x_tensor = torch.as_tensor(flat[start:stop], dtype=torch.float32, device=device)
        p_tensor = torch.as_tensor(
            pilot_p[start:stop, None], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            score_v = debiased_scores_tensor(fitted, p_tensor, x_tensor).reshape(-1)
            score_p = score_v * float(fitted.scale_theta)
            score_u = score_p * p_tensor.reshape(-1) * (1.0 - p_tensor.reshape(-1))
        output.append(score_u.cpu().numpy())
    result = np.concatenate(output).astype(np.float32)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("non-finite Jiang score feature")
    return result


def shared_features(
    ours,
    jiang_mlp: JiangFitted,
    jiang_gru: JiangFitted,
    upstreams: UpstreamModules,
    y: np.ndarray,
    pilot_u: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> dict[str, np.ndarray]:
    ours_feature = frozen_features(ours, upstreams, y, pilot_u, device=device)
    mlp = jiang_score_u_at_pilot(
        jiang_mlp, y, pilot_u, device=device, batch_size=batch_size
    )
    gru = jiang_score_u_at_pilot(
        jiang_gru, y, pilot_u, device=device, batch_size=batch_size
    )
    return {
        JIANG_MLP_NPE: np.column_stack([pilot_u, mlp]).astype(np.float32),
        JIANG_GRU_NPE: np.column_stack([pilot_u, gru]).astype(np.float32),
        OURS_GATE_NPE: np.asarray(ours_feature[
            "positive nonlinear local-gate GRU FSM"
        ], dtype=np.float32),
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in (EXACT_LABEL, *METHODS):
        selected = [row for row in rows if row["method"] == method]
        squared = np.asarray(
            [row["posterior_mean_sq_error_to_exact"] for row in selected],
            dtype=np.float64,
        )
        output.append(
            {
                "method": method,
                "n_test": len(selected),
                "posterior_mean_rmse_to_exact": float(np.sqrt(squared.mean())),
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
    ours = load_ours(Path(args.ours_checkpoint), upstreams, device=device)
    jiang_mlp = load_jiang(Path(args.jiang_mlp_round1), upstreams, device=device)
    jiang_gru = load_jiang(Path(args.jiang_gru_round1), upstreams, device=device)
    _assert_compatible(ours, jiang_mlp, "Jiang MLP")
    _assert_compatible(ours, jiang_gru, "Jiang GRU")
    if jiang_mlp.score_architecture != "official_mlp":
        raise ValueError("--jiang-mlp-round1 is not an official MLP checkpoint")
    if jiang_gru.score_architecture != "raw_sequence_gru":
        raise ValueError("--jiang-gru-round1 is not a raw-sequence GRU checkpoint")

    stage2 = _stage2_upstream()
    NPE, posterior_nn, BoxUniform = stage2.import_sbi()
    rng = np.random.default_rng(int(args.sbi_train_seed))
    strata = (
        np.arange(int(args.n_sbi_train), dtype=np.float64)
        + rng.uniform(size=int(args.n_sbi_train))
    ) / int(args.n_sbi_train)
    rng.shuffle(strata)
    p_train = ours.hmm.p_min + (ours.hmm.p_max - ours.hmm.p_min) * strata
    y_train = simulate_trajectories(rng, p_train, ours.hmm, ours.emission)
    pilot_u, pilot_status, _, _ = data_only_pilot(
        y_train,
        fitted=ours,
        grid_size=int(args.pilot_grid_size),
        batch_size=int(args.pilot_batch_size),
    )
    features = shared_features(
        ours,
        jiang_mlp,
        jiang_gru,
        upstreams,
        y_train,
        pilot_u,
        device=device,
        batch_size=int(args.score_batch_size),
    )
    pilot_p = 1.0 / (1.0 + np.exp(-pilot_u))
    pilot_corr = float(np.corrcoef(np.log(p_train / (1.0 - p_train)), pilot_u)[0, 1])
    print(
        "pilot corr/RMSE:",
        pilot_corr,
        float(np.sqrt(np.mean(np.square(pilot_p - p_train)))),
        flush=True,
    )

    args.p_prior_min = ours.hmm.p_min
    args.p_prior_max = ours.hmm.p_max
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
        for method in METHODS
    ]

    rows: list[dict[str, Any]] = []
    p_values = [float(value) for value in args.test_p_values.split(",")]
    observation_seeds = [int(value) for value in args.test_seeds.split(",")]
    for p_true in p_values:
        for observation_seed in observation_seeds:
            y_obs = simulate_trajectories(
                np.random.default_rng(observation_seed),
                p_true,
                ours.hmm,
                ours.emission,
                n=1,
            )
            obs_pilot_u, _, _, _ = data_only_pilot(
                y_obs,
                fitted=ours,
                grid_size=int(args.pilot_grid_size),
                batch_size=int(args.pilot_batch_size),
            )
            obs_features = shared_features(
                ours,
                jiang_mlp,
                jiang_gru,
                upstreams,
                y_obs,
                obs_pilot_u,
                device=device,
                batch_size=int(args.score_batch_size),
            )
            axis, cdf, exact = exact_posterior(y_obs[0], ours, int(args.grid_size))
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
            evaluation_seed = int(observation_seed) + 91_000
            for result in npe_results:
                method = str(result["method"])
                samples = stage2.posterior_samples(
                    result,
                    obs_features[method][0],
                    int(args.posterior_n),
                    evaluation_seed,
                    args,
                )
                mean = float(samples.mean())
                sd = float(samples.std())
                q05, q50, q95 = np.quantile(samples, [0.05, 0.50, 0.95])
                rows.append(
                    {
                        "method": method,
                        "p_true": p_true,
                        "observation_seed": observation_seed,
                        "posterior_mean": mean,
                        "posterior_sd": sd,
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
                        "abs_sd_error_to_exact": abs(sd - exact["posterior_sd"]),
                        "interval_endpoint_mae_to_exact": 0.5
                        * (
                            abs(float(q05) - exact["q05"])
                            + abs(float(q95) - exact["q95"])
                        ),
                        "abs_width_error_to_exact": abs(
                            float(q95 - q05) - (exact["q95"] - exact["q05"])
                        ),
                        "w1_to_exact": float(stage2.exact_w1(samples, axis, cdf)),
                    }
                )

    summary = _summarize(rows)
    _write_csv(output / "posterior_by_seed.csv", rows)
    _write_csv(output / "posterior_summary.csv", summary)
    config = {
        **vars(args),
        "ours_checkpoint": str(Path(args.ours_checkpoint).resolve()),
        "jiang_mlp_round1": str(Path(args.jiang_mlp_round1).resolve()),
        "jiang_gru_round1": str(Path(args.jiang_gru_round1).resolve()),
        "jiang_checkpoints_observation_independent": True,
        "stage1_frozen": True,
        "stage2_context": "(shared data-only pilot u_hat, frozen u-score at u_hat)",
        "training_simulations_shared": True,
        "npe_architecture_optimizer_and_seed_shared": True,
        "posterior_sampling_seed_shared_within_observation": True,
        "pilot_corr_true_u": pilot_corr,
        "pilot_status_fractions": {
            str(status): float(np.mean(pilot_status == status))
            for status in np.unique(pilot_status)
        },
        "exact_score_used_for_training": False,
        "exact_posterior_used_for_training": False,
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("saved to", output, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--jiang-mlp-round1", type=Path, required=True)
    parser.add_argument("--jiang-gru-round1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-sbi-train", type=int, default=50_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260725)
    parser.add_argument("--pilot-grid-size", type=int, default=201)
    parser.add_argument("--pilot-batch-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=512)
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
