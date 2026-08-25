#!/usr/bin/env python3
"""Score-only conditional diffusion using a frozen amortized Model 2 FSM field.

Training contexts contain exactly

    S_frozen(Y, logit(pi_true)),

with no parameter concatenated to the context.  Test simulations use the same
true-parameter score evaluation, so this is the diffusion counterpart of
``run_model2_score_only_sbi_npe.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import model2_amortized_score_runtime as score_runtime
import run_blockwise_common_factor_two_stage_torch_diffusion_experiment as diffusion
import run_model2_score_only_sbi_npe as common


def save_diffusion_checkpoint(
    path: Path,
    method: str,
    result: dict[str, object],
    args: argparse.Namespace,
) -> None:
    posterior = result["posterior"]
    torch.save(
        {
            "method": method,
            "label": common.METHOD_LABELS[method],
            "model_state_dict": posterior.model.state_dict(),
            "x_mean": posterior.x_mean,
            "x_sd": posterior.x_sd,
            "z_mean": posterior.z_mean,
            "z_sd": posterior.z_sd,
            "betas": posterior.betas.detach().cpu(),
            "stage1_run_dir": str(args.stage1_run_dir),
            "context": "S(Y, logit(pi_true)) only",
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "posterior_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    methods = common.parse_list(args.methods, str)
    bad = sorted(set(methods) - set(common.METHOD_LABELS))
    if bad or not methods:
        raise ValueError(f"--methods must use {sorted(common.METHOD_LABELS)}; got {methods}")
    device = diffusion.resolve_device(str(args.device))

    runtimes = {
        method: score_runtime.AmortizedScoreRuntime(args.stage1_run_dir, method, device=device)
        for method in methods
    }
    metadata = common.validate_matching_runtimes(runtimes)
    args.tau = float(metadata["tau"])
    if not (
        float(metadata["pi_min"]) <= float(args.pi_prior_min)
        < float(args.pi_prior_max) <= float(metadata["pi_max"])
    ):
        raise ValueError(
            "The diffusion prior must lie inside the trained Stage-1 anchor support "
            f"[{metadata['pi_min']}, {metadata['pi_max']}]"
        )

    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "device:", device)
    print("Stage-1 run:", args.stage1_run_dir)
    print("Context: scalar S(Y, logit(pi_true)); pi_true is not concatenated")
    print("Methods:", methods)
    print("Model metadata:", metadata)

    sanity_rows: list[dict[str, object]] = []
    for method, runtime in runtimes.items():
        checks = runtime.run_sanity_checks(seed=int(args.sanity_seed))
        sanity_rows.append(checks)
        if not bool(checks["passed"]):
            raise AssertionError(f"Frozen Stage-1 runtime sanity failed for {method}: {checks}")
    common.write_csv(out_dir / "runtime_sanity_checks.csv", sanity_rows)

    rng = np.random.default_rng(int(args.diffusion_train_seed))
    pi_train = rng.uniform(
        float(args.pi_prior_min),
        float(args.pi_prior_max),
        size=int(args.n_diffusion_train),
    )
    y_train = common.simulate(rng, pi_train, metadata)
    train_contexts = common.score_contexts(
        runtimes,
        y_train,
        pi_train,
        batch_size=int(args.score_batch_size),
    )
    del y_train

    context_rows: list[dict[str, object]] = []
    for method in methods:
        z = train_contexts[method].reshape(-1).astype(np.float64)
        context_rows.append(
            {
                "method": common.METHOD_LABELS[method],
                "n": z.size,
                "context_dim": 1,
                "score_mean": float(z.mean()),
                "score_sd": float(z.std()),
                "score_min": float(z.min()),
                "score_max": float(z.max()),
                "corr_score_pi": float(np.corrcoef(z, pi_train)[0, 1]),
            }
        )
    common.write_csv(out_dir / "training_context_summary.csv", context_rows)
    if bool(args.save_training_context):
        np.savez_compressed(
            out_dir / "training_contexts.npz",
            pi_train=pi_train.astype(np.float32),
            **{f"score_{method}": train_contexts[method] for method in methods},
        )

    results: dict[str, dict[str, object]] = {}
    trace_rows: list[dict[str, object]] = []
    for idx, method in enumerate(methods):
        label = common.METHOD_LABELS[method].replace("NPE", "diffusion")
        result = diffusion.train_diffusion_method(
            label,
            train_contexts[method],
            pi_train,
            int(args.diffusion_seed) + 173 * idx,
            args,
            device,
        )
        result["method"] = label
        results[method] = result
        trace_rows.extend(result["trace"])
        save_diffusion_checkpoint(out_dir / f"diffusion_{method}.pt", method, result, args)

    rows: list[dict[str, object]] = []
    test_pi_values = common.parse_list(args.test_pi_values, float)
    test_seeds = common.parse_list(args.test_seeds, int)
    for pi_true in test_pi_values:
        if not (float(args.pi_prior_min) <= pi_true <= float(args.pi_prior_max)):
            raise ValueError(f"test pi {pi_true} lies outside the diffusion prior")
        for obs_seed in test_seeds:
            obs_rng = np.random.default_rng(int(obs_seed))
            y_obs = common.simulate(obs_rng, np.asarray([pi_true]), metadata)
            obs_contexts = common.score_contexts(
                runtimes,
                y_obs,
                np.asarray([pi_true]),
                batch_size=1,
            )
            exact_grid, _, exact_cdf, exact = common.exact_posterior(y_obs[0], args)
            rows.append(
                {
                    "pi_true": pi_true,
                    "seed": obs_seed,
                    "method": "exact likelihood grid",
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

            for idx, method in enumerate(methods):
                label = common.METHOD_LABELS[method].replace("NPE", "diffusion")
                samples = results[method]["posterior"].sample(
                    obs_contexts[method][0],
                    int(args.posterior_n),
                    int(args.posterior_seed) + 100_003 * idx + obs_seed,
                )
                np.save(samples_dir / f"{method}_pi{pi_true:g}_seed{obs_seed}.npy", samples)
                mean = float(np.mean(samples))
                sd = float(np.std(samples))
                q05, q50, q95 = [float(x) for x in np.quantile(samples, [0.05, 0.50, 0.95])]
                rows.append(
                    {
                        "pi_true": pi_true,
                        "seed": obs_seed,
                        "method": label,
                        "score_context": float(obs_contexts[method][0, 0]),
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
    common.write_csv(out_dir / "posterior_by_seed.csv", rows)
    common.write_csv(out_dir / "posterior_summary.csv", summary)
    common.write_csv(out_dir / "diffusion_training_trace.csv", trace_rows)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stage1_metadata": metadata,
        "context_definition": "S(Y, logit(pi_true)) only",
        "context_dim": 1,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n==== Score-only diffusion posterior summary ====")
    print(f"{'pi':>6}  {'method':<40}{'MSE':>12}{'RMSE-exact':>14}{'W1-exact':>12}{'post SD':>11}{'cov90':>9}")
    print("-" * 106)
    for row in summary:
        print(
            f"{float(row['pi_true']):>6.2f}  {str(row['method']):<40}"
            f"{float(row['pi_avg_mse']):>12.4g}"
            f"{float(row['rmse_mean_to_exact']):>14.4g}"
            f"{float(row['w1_to_exact']):>12.4g}"
            f"{float(row['avg_post_sd']):>11.4g}"
            f"{float(row['coverage90']):>9.3g}"
        )
    print("\nSaved:")
    for name in (
        "config.json",
        "runtime_sanity_checks.csv",
        "training_context_summary.csv",
        "diffusion_training_trace.csv",
        "posterior_by_seed.csv",
        "posterior_summary.csv",
    ):
        print(out_dir / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-run-dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="linear,radial")
    parser.add_argument("--pi-prior-min", type=float, default=0.05)
    parser.add_argument("--pi-prior-max", type=float, default=0.70)
    parser.add_argument("--n-diffusion-train", type=int, default=20_000)
    parser.add_argument("--diffusion-train-seed", type=int, default=20260710)
    parser.add_argument("--diffusion-seed", type=int, default=20260711)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--beta-min", type=float, default=1e-4)
    parser.add_argument("--beta-max", type=float, default=0.02)
    parser.add_argument("--train-steps", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--diffusion-hidden", type=int, default=128)
    parser.add_argument("--diffusion-depth", type=int, default=3)
    parser.add_argument("--diffusion-t-embed-dim", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--test-pi-values", type=str, default="0.10,0.30,0.50,0.65")
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--posterior-seed", type=int, default=97000)
    parser.add_argument("--grid-size", type=int, default=5_000)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--sanity-seed", type=int, default=20260710)
    parser.add_argument("--save-training-context", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/model2_score_only_torch_diffusion"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.pi_prior_min < args.pi_prior_max < 1.0):
        raise ValueError("Require 0 < pi_prior_min < pi_prior_max < 1")
    if args.n_diffusion_train < 100:
        raise ValueError("--n-diffusion-train must be at least 100")
    run(args)


if __name__ == "__main__":
    main()
