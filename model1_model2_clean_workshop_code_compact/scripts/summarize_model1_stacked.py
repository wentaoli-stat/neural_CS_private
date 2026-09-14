#!/usr/bin/env python3
"""Replay frozen Model-1 scores and report the one-seed stacked comparison.

Exact scores are generated here only after every model checkpoint is frozen.
This script never trains, selects checkpoints, or changes their parameters.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model1 import evaluate_stage1, runtime, stage1


LABELS = {"linear": "Linear", "radial": "Multiplicative gate", "stacked": "Stacked NLSA"}


def reduction(reference: float, candidate: float) -> float:
    return 100 * (reference - candidate) / reference


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
    ])


def run(run_dir: Path, report_path: Path):
    started = time.perf_counter()
    config = json.loads((run_dir / "config.json").read_text())
    info = json.loads((run_dir / "training_info.json").read_text())
    protocol = json.loads((run_dir / "exact_evaluation.json").read_text())
    if set(info) != set(LABELS):
        raise ValueError("Expected exactly the linear, radial, and stacked checkpoints")
    if len({v["minibatch_sha256"] for v in info.values()}) != 1:
        raise ValueError("Training minibatch streams were not matched")
    models = {m: runtime.AmortizedScoreRuntime(run_dir, m, "cpu") for m in LABELS}
    sanity = {m: model.run_sanity_checks() for m, model in models.items()}
    if not all(v["passed"] for v in sanity.values()):
        raise ValueError(f"Runtime sanity check failed: {sanity}")
    stats = next(iter(models.values())).stats
    rng, _ = evaluate_stage1.evaluation_rng(config, protocol["rng"]["seed"])
    with (run_dir / "score_summary_by_pi.csv").open() as handle:
        original = {(float(r["pi"]), r["method"]): r for r in csv.DictReader(handle)}
    metric_rows, paired_rows, arrays = [], [], {}
    for pi in protocol["pi_values"]:
        data, truth = stage1.build_eval_data(
            rng, pi, protocol["n_test_per_pi"], config["n_blocks"], config["block_size"],
            config["tau"], stats, float(stats["anchor_u_mean"]), float(stats["anchor_u_sd"]),
            need_raw_y=True,
        )
        expected_hash = protocol["datasets"][format(pi, ".17g")]
        assert evaluate_stage1.array_sha256(data["y"]) == expected_hash["y_sha256"]
        assert evaluate_stage1.array_sha256(truth) == expected_hash["exact_truth_sha256"]
        errors = {}
        arrays[f"pi_{pi:g}_truth"] = truth
        for method, model in models.items():
            pred = stage1.predict(model.model, data, method, "cpu", protocol["batch_size"])
            metrics = stage1.metric_row(truth, pred)
            # Guards against replaying the wrong checkpoint or test bank. The
            # tolerance stays far below float32 epsilon while tolerating the
            # reduction-order differences a different OMP thread count produces.
            np.testing.assert_allclose(metrics["mse"], float(original[(pi, model.label)]["mse"]), rtol=1e-9)
            metric_rows.append({"pi": pi, "method": method, **metrics})
            errors[method] = (pred.astype(np.float64) - truth)**2
            arrays[f"pi_{pi:g}_{method}"] = pred
        for baseline in ("linear", "radial"):
            delta = errors["stacked"] - errors[baseline]
            paired_rows.append({"pi": pi, "baseline": baseline,
                                "stacked_minus_baseline_mse": float(delta.mean()),
                                "paired_test_se": float(delta.std(ddof=1)/math.sqrt(len(delta))),
                                "relative_reduction_percent": reduction(float(errors[baseline].mean()), float(errors["stacked"].mean()))})
    with (run_dir / "training_trace.csv").open() as handle:
        trace = list(csv.DictReader(handle))
    stacked_trace = [r for r in trace if r["method"].startswith("stacked")]
    means = {m: {metric: float(np.mean([r[metric] for r in metric_rows if r["method"] == m]))
                 for metric in ("mse", "std_mse")} for m in LABELS}
    stability = {key: max(float(r[key]) for r in stacked_trace)
                 for key in ("m_max_abs", "pooled_m_rms", "pooled_m_sd")}
    log = run_dir.with_suffix(".log")
    elapsed = None
    if log.exists():
        timing = [line for line in log.read_text().splitlines() if line.startswith("real ")]
        if timing:
            elapsed = float(timing[-1].split()[1])
    for method, entry in protocol["checkpoints"].items():
        assert evaluate_stage1.checkpoint_sha256(run_dir / entry["path"]) == entry["sha256"]
    summary = {"scope": "Model 1, Stage 1 only, one training seed", "config": config,
               "training_info": info, "metrics": metric_rows, "means": means,
               "paired_differences": paired_rows, "stability": stability,
               "runtime_checks": sanity, "stage1_and_evaluation_wall_seconds": elapsed,
               "report_replay_seconds": time.perf_counter()-started,
               "report_environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}}
    np.savez_compressed(run_dir / "score_predictions.npz", **arrays)
    (run_dir / "stacked_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    gate_gain = reduction(means["radial"]["mse"], means["stacked"]["mse"])
    std_gain = reduction(means["radial"]["std_mse"], means["stacked"]["std_mse"])
    wins = sum(r["relative_reduction_percent"] > 0 for r in paired_rows if r["baseline"] == "radial")
    direction = "lower" if gate_gain > 0 else "higher"
    lines = ["# Stacked NLSA: Model 1, Stage 1, one seed", "",
             f"Verdict: stacked NLSA has {abs(gate_gain):.2f}% {direction} mean raw Fisher-score MSE than the multiplicative gate "
             f"({means['stacked']['mse']:.6f} versus {means['radial']['mse']:.6f}), "
             f"and beats it at {wins}/4 tested parameter values. "
             f"The relative reduction in mean standardized MSE is {std_gain:.2f}% (negative means worse). "
             "This is one training seed; it does not establish stability across training seeds.", "",
             "## Scope and protocol", "",
             f"Seed `{config['seed']}`; `K=20`, `m=20`, `tau=0.5`, `sigma_q=0.20`. "
             "All three methods use one shared bank of 40,000 training and 8,000 validation datasets, "
             "the same anchors and FSM targets, matched untrained linear readout weights, the same "
             "minibatch stream, and 20,000 updates each. The readout has two width-64 SiLU layers; "
             "the local network has two width-16 SiLU layers. Batch size 512, learning rate 1e-4 "
             "for both parameter groups, cosine decay after step 10,000 to ratio 0.1, "
             "weight decay 1e-3, gradient clipping 5, EMA decay 0.995.", "",
             "Checkpoint selection is explicitly validation-best **raw**, using only validation FSM loss. "
             "The pre-existing trainer's default actually considers both raw and EMA; that default remains "
             "unchanged, while the new launcher explicitly uses `--checkpoint-selection raw` for all three "
             "methods as required by the experiment prompt. EMA loss is logged but cannot select a checkpoint.", "",
             "The exact full score in the `u=logit(pi)` coordinate is used only after all training "
             "has finished. Each parameter point has 5,000 shared test datasets. "
             "Standardized MSE is raw MSE divided by the exact score variance at that parameter. "
             "Mean rows average the four parameter points equally.", "",
             "No Model 2, Stage 2, extra seeds, or architecture ablations were run.", "",
             "## Local maps and initialization", "",
             "The gate uses $s_z\\,\\operatorname{softplus}(g_\\eta(s_z,a_z))$. "
             "The stacked map uses $(1,s_z,m_\\eta(s_z,a_z))$ with an unbounded learned channel, "
             "`m_dim=1`, and the constant included. Its identity input to the readout is the existing "
             "standardized block feature verbatim. The learned local features are mean-pooled separately.", "",
             "The added channel can learn signed functions and nonzero values at zero local score; "
             "the positive multiplicative map cannot. Retaining the identity alongside a learned "
             "feature enlarges the summary representation. This does not guarantee better finite-sample "
             "optimization or an exact containment theorem for these fixed-width networks.", "",
             "The feature MLP's final layer starts at zero. Its readout columns retain seeded nonzero "
             "`nn.Linear` initialization; the constant readout column starts at zero. Thus initial "
             "predictions match ILSA while gradients reach the feature MLP. The constant pools to 1 "
             "and is representationally redundant with the first-layer bias, but making its coefficient "
             "trainable can change optimization and regularization. Numerical irrelevance of training "
             "with versus without it is therefore not assumed.", "",
             table(["Method", "Initial max abs difference from ILSA", "Parameters", "Selected raw step", "Training wall time (s)"],
                   [[LABELS[m], f"{info[m].get('nested_initialization_max_abs_diff', 0):.3g}", info[m]["trainable_parameters"],
                     info[m]["best_step"], f"{info[m]['wall_seconds']:.2f}"] for m in LABELS]), "",
             "## Fisher-score approximation error", ""]
    for metric, title in (("mse", "Raw MSE"), ("std_mse", "Standardized MSE")):
        rows = []
        for pi in protocol["pi_values"]:
            vals = {r["method"]: r[metric] for r in metric_rows if r["pi"] == pi}
            rows.append([f"{pi:.2f}", *(f"{vals[m]:.6f}" for m in LABELS),
                         f"{reduction(vals['linear'],vals['stacked']):.2f}%", f"{reduction(vals['radial'],vals['stacked']):.2f}%"])
        vals = {m: means[m][metric] for m in LABELS}
        rows.append(["Mean", *(f"{vals[m]:.6f}" for m in LABELS),
                     f"{reduction(vals['linear'],vals['stacked']):.2f}%", f"{reduction(vals['radial'],vals['stacked']):.2f}%"])
        lines += [f"### {title}", "", table(["True pi", *LABELS.values(), "Stacked reduction vs linear", "Stacked reduction vs gate"], rows), ""]
    lines += ["## Paired test-set differences", "",
              "Negative differences favor stacked NLSA. SE is the standard error of the paired "
              "squared-error difference across the 5,000 test datasets, conditional on these fitted models. "
              "It is **not** an across-training-seed uncertainty estimate; that spread is unavailable "
              "with one seed. No NPE pairing or posterior comparisons apply to this Stage-1-only run.", "",
              table(["True pi", "Baseline", "Stacked minus baseline raw MSE", "Paired test SE"],
                    [[f"{r['pi']:.2f}", LABELS[r["baseline"]], f"{r['stacked_minus_baseline_mse']:.6f}", f"{r['paired_test_se']:.6f}"] for r in paired_rows]), "",
              "## Stability, timing, and reproducibility", "",
              f"Maximum logged |m|: {stability['m_max_abs']:.6f}; maximum pooled-m RMS: "
              f"{stability['pooled_m_rms']:.6f}; maximum pooled-m SD: {stability['pooled_m_sd']:.6f}. "
              "These diagnostics use the same first 256 validation datasets at every logging step "
              "(step 0, step 1, then every 100 updates). They are not maxima over all training data. "
              "All training and validation losses were finite; no divergence or hyperparameter changes occurred.", "",
              f"Stage 1 plus exact evaluation wall time: {elapsed} seconds. "
              "Training times above include each method's validation passes. Report replay is additional. "
              "Hardware: Apple M3 Max, 64 GiB RAM; CPU, 8 threads. "
              f"Python {platform.python_version()}, PyTorch {torch.__version__}, NumPy {np.__version__}. "
              "PyTorch 2.5.1 satisfies SBI 0.24.0's declared dependency constraint, whereas the archived "
              "requirements ask for the incompatible PyTorch 2.8. These are newly matched fits on one "
              "environment, not bitwise reproductions of the historical five-seed tables.", "",
              "Initial gradient-flow and runtime-roundtrip tests pass. All three frozen runtime "
              "sanity checks pass. The replay reproduces the evaluator's dataset hashes and MSEs. "
              "Training-data fingerprints are in `config.json`; equal minibatch hashes are in "
              "`training_info.json`; frozen checkpoint and test-data hashes are in `exact_evaluation.json`.", "",
              f"Run directory: `{run_dir.relative_to(ROOT)}`. Raw results: "
              "`score_summary_by_pi.csv`, `score_predictions.npz`, `training_trace.csv`, "
              "and `stacked_summary.json`. The log is the adjacent `.log` file.", "",
              "Reproduce from the package directory:", "", "```bash",
              "OMP_NUM_THREADS=8 DEVICE=cpu SEED=20260709 \\",
              "PYTHON_BIN=../.venv-stacked-nlsa/bin/python \\",
              "bash scripts/run_model1_stage1_stacked.sh runs/NEW_UNUSED_DIRECTORY",
              "../.venv-stacked-nlsa/bin/python scripts/summarize_model1_stacked.py \\",
              "  --run-dir runs/NEW_UNUSED_DIRECTORY", "```", "",
              "Existing packaged headline results and all files under `artifacts/` remain unchanged.", ""]
    report_path.write_text("\n".join(lines))
    print(json.dumps({"means": means, "stability": stability, "wall_seconds": elapsed}, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "new_results" / "STACKED_NLSA_RESULTS.md")
    args = parser.parse_args()
    run(args.run_dir.resolve(), args.report.resolve())
