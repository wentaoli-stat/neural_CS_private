#!/usr/bin/env python3
"""Evaluate trained score-only diffusion posteriors with a non-oracle pilot.

Stage 2 was trained with contexts ``S(Y, logit(pi_true))``.  This script leaves
the trained posterior untouched and, for each test dataset, replaces the true
evaluation location by the root of an equal-channel marginal/pairwise composite
score.  The pilot is used only to evaluate the frozen score and is not appended
to the diffusion context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import model2_amortized_score_runtime as score_runtime
import run_blockwise_common_factor_two_stage_torch_diffusion_experiment as diffusion
import run_model2_score_only_sbi_npe as common
import run_model2_mode_a_mle as mle


def load_diffusion_posterior(
    checkpoint_path: Path,
    device: str,
) -> tuple[diffusion.TorchDiffusionPosterior, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    model = diffusion.ConditionalEpsilonNet(
        x_dim=1,
        hidden=int(config["diffusion_hidden"]),
        depth=int(config["diffusion_depth"]),
        t_dim=int(config["diffusion_t_embed_dim"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    posterior = diffusion.TorchDiffusionPosterior(
        model,
        x_mean=payload["x_mean"],
        x_sd=payload["x_sd"],
        z_mean=payload["z_mean"],
        z_sd=payload["z_sd"],
        betas=payload["betas"],
        device=device,
        pi_min=float(config["pi_prior_min"]),
        pi_max=float(config["pi_prior_max"]),
    )
    return posterior, payload


def pooled_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    frame = frame[frame["method"] != "exact likelihood grid"]
    out: list[dict[str, object]] = []
    for method, group in frame.groupby("method"):
        out.append(
            {
                "method": method,
                "n": len(group),
                "pi_avg_mse": float(group["pi_sq_err"].mean()),
                "rmse_mean_to_exact": float(np.sqrt(group["mean_sq_err_to_exact"].mean())),
                "w1_to_exact": float(group["w1_to_exact"].mean()),
                "avg_post_sd": float(group["pi_post_sd"].mean()),
                "coverage90": float(group["coverage90"].mean()),
                "pilot_rmse": float(np.sqrt(group["pilot_sq_err"].mean())),
                "pilot_boundary_rate": float(group["pilot_at_boundary"].mean()),
            }
        )
    return sorted(out, key=lambda row: float(row["w1_to_exact"]))


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "posterior_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    device = diffusion.resolve_device(str(args.device))
    methods = common.parse_list(args.methods, str)

    runtimes = {
        method: score_runtime.AmortizedScoreRuntime(args.stage1_run_dir, method, device=device)
        for method in methods
    }
    metadata = common.validate_matching_runtimes(runtimes)
    reference = next(iter(runtimes.values()))
    posterior_payloads = {
        method: load_diffusion_posterior(
            Path(args.diffusion_run_dir) / f"diffusion_{method}.pt",
            device,
        )
        for method in methods
    }

    print("device:", device)
    print("Stage-1 run:", args.stage1_run_dir)
    print("Diffusion run:", args.diffusion_run_dir)
    print("Pilot: equal-channel marginal+pairwise composite-score global root")
    print("Pilot is not concatenated to the diffusion context")

    records: list[dict[str, object]] = []
    data: list[np.ndarray] = []
    for pi_true in common.parse_list(args.test_pi_values, float):
        for seed in common.parse_list(args.test_seeds, int):
            rng = np.random.default_rng(int(seed))
            data.append(common.simulate(rng, np.asarray([pi_true]), metadata))
            records.append({"pi_true": pi_true, "seed": seed})
    y_all = np.concatenate(data, axis=0)

    u_grid = np.linspace(reference.u_min, reference.u_max, int(args.pilot_grid_size))
    pilot_score = mle.composite_pilot_score_grid(y_all, u_grid, float(metadata["tau"]))
    pilot_u, pilot_status, n_stationary, pilot_residual = mle.constrained_modes_from_score_grid(
        pilot_score,
        u_grid,
    )
    pilot_pi = 1.0 / (1.0 + np.exp(-pilot_u))
    contexts = common.score_contexts(
        runtimes,
        y_all,
        pilot_pi,
        batch_size=int(args.score_batch_size),
    )

    pilot_rows: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        pi_true = float(record["pi_true"])
        seed = int(record["seed"])
        at_boundary = pilot_status[index] in {"left_boundary", "right_boundary"}
        pilot_rows.append(
            {
                "pi_true": pi_true,
                "seed": seed,
                "pilot_pi": float(pilot_pi[index]),
                "pilot_u": float(pilot_u[index]),
                "pilot_error": float(pilot_pi[index] - pi_true),
                "pilot_sq_err": float((pilot_pi[index] - pi_true) ** 2),
                "pilot_status": str(pilot_status[index]),
                "pilot_at_boundary": float(at_boundary),
                "pilot_n_stationary": int(n_stationary[index]),
                "pilot_score_residual": float(pilot_residual[index]),
                **{
                    f"{method}_score_at_pilot": float(contexts[method][index, 0])
                    for method in methods
                },
            }
        )

        exact_grid, _, exact_cdf, exact = common.exact_posterior(y_all[index], args)
        rows.append(
            {
                "pi_true": pi_true,
                "seed": seed,
                "method": "exact likelihood grid",
                "pilot_pi": float(pilot_pi[index]),
                "pilot_sq_err": float((pilot_pi[index] - pi_true) ** 2),
                "pilot_at_boundary": float(at_boundary),
                "score_context": float("nan"),
                "pi_post_mean": exact["mean"],
                "pi_post_sd": exact["sd"],
                "q05": exact["q05"],
                "q50": exact["q50"],
                "q95": exact["q95"],
                "pi_sq_err": float((exact["mean"] - pi_true) ** 2),
                "mean_sq_err_to_exact": 0.0,
                "w1_to_exact": 0.0,
                "coverage90": float(exact["q05"] <= pi_true <= exact["q95"]),
            }
        )

        for method_index, method in enumerate(methods):
            posterior, _ = posterior_payloads[method]
            label = common.METHOD_LABELS[method].replace("NPE", "diffusion") + " (pilot anchor)"
            samples = posterior.sample(
                contexts[method][index],
                int(args.posterior_n),
                int(args.posterior_seed) + 100_003 * method_index + seed,
            )
            np.save(samples_dir / f"{method}_pi{pi_true:g}_seed{seed}.npy", samples)
            mean = float(np.mean(samples))
            sd = float(np.std(samples))
            q05, q50, q95 = [float(x) for x in np.quantile(samples, [0.05, 0.50, 0.95])]
            rows.append(
                {
                    "pi_true": pi_true,
                    "seed": seed,
                    "method": label,
                    "pilot_pi": float(pilot_pi[index]),
                    "pilot_sq_err": float((pilot_pi[index] - pi_true) ** 2),
                    "pilot_at_boundary": float(at_boundary),
                    "score_context": float(contexts[method][index, 0]),
                    "pi_post_mean": mean,
                    "pi_post_sd": sd,
                    "q05": q05,
                    "q50": q50,
                    "q95": q95,
                    "pi_sq_err": float((mean - pi_true) ** 2),
                    "mean_sq_err_to_exact": float((mean - exact["mean"]) ** 2),
                    "w1_to_exact": diffusion.exact_sample_w1(samples, exact_grid, exact_cdf),
                    "coverage90": float(q05 <= pi_true <= q95),
                }
            )

    summary = common.summarize(rows)
    pooled = pooled_summary(rows)
    common.write_csv(out_dir / "pilot_diagnostics.csv", pilot_rows)
    common.write_csv(out_dir / "posterior_by_seed.csv", rows)
    common.write_csv(out_dir / "posterior_summary.csv", summary)
    common.write_csv(out_dir / "posterior_summary_pooled.csv", pooled)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stage1_metadata": metadata,
        "pilot_definition": "equal-channel marginal+pairwise composite-score global root",
        "context_definition": "S(Y, logit(pi_pilot)) only",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n==== Pilot diagnostics ====")
    pilot_frame = pd.DataFrame(pilot_rows)
    print(
        pilot_frame.groupby("pi_true").agg(
            pilot_rmse=("pilot_sq_err", lambda x: float(np.sqrt(np.mean(x)))),
            boundary_rate=("pilot_at_boundary", "mean"),
        ).to_string()
    )
    print("\n==== Pilot-anchor score-only diffusion ====")
    print(pd.DataFrame(pooled).to_string(index=False))
    print("\nSaved:", out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-run-dir", type=Path, required=True)
    parser.add_argument("--diffusion-run-dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="linear,radial")
    parser.add_argument("--pi-prior-min", type=float, default=0.05)
    parser.add_argument("--pi-prior-max", type=float, default=0.70)
    parser.add_argument("--test-pi-values", type=str, default="0.10,0.30,0.50,0.65")
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--pilot-grid-size", type=int, default=401)
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--posterior-seed", type=int, default=117000)
    parser.add_argument("--grid-size", type=int, default=5_000)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/model2_score_only_diffusion_pilot_eval"),
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
