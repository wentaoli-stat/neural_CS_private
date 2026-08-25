#!/usr/bin/env python3
"""Evaluate frozen Model-2 Stage-1 checkpoints against the exact score.

This command is deliberately separate from :mod:`model2.stage1`: training and
checkpoint selection finish before any held-out exact-score data are generated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from model2 import runtime, stage1


def evaluation_rng(
    config: dict[str, Any], explicit_seed: int | None
) -> tuple[np.random.Generator, dict[str, Any]]:
    if explicit_seed is not None:
        return np.random.default_rng(explicit_seed), {
            "source": "explicit_evaluation_seed",
            "seed": int(explicit_seed),
        }
    state = config.get("heldout_evaluation_rng_state")
    if not isinstance(state, dict):
        raise ValueError(
            "Stage-1 config predates separated evaluation and has no stored "
            "heldout RNG state; pass --evaluation-seed explicitly"
        )
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    return rng, {"source": "stored_post_train_validation_rng_state", "seed": None}


def parse_pi_values(text: str, pi_min: float, pi_max: float) -> list[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("--pi-values must contain at least one value")
    if any(value < pi_min or value > pi_max for value in values):
        raise ValueError("all --pi-values must lie inside the trained anchor support")
    return values


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.json"
    stats_path = run_dir / "feature_stats.npz"
    for path in (config_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_csv = run_dir / "score_summary_by_pi.csv"
    milestone_csv = run_dir / "milestone_score_summary_by_pi.csv"
    protocol_path = run_dir / "exact_evaluation.json"
    if output_csv.exists() or milestone_csv.exists() or protocol_path.exists():
        raise FileExistsError(
            "exact evaluation output already exists; use a fresh Stage-1 run directory"
        )
    if args.n_test <= 0 or args.batch_size <= 0:
        raise ValueError("--n-test and --batch-size must be positive")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    methods = runtime.parse_method_list(str(config["methods"]))
    pi_min = float(config["anchor_pi_min"])
    pi_max = float(config["anchor_pi_max"])
    pi_values = parse_pi_values(args.pi_values, pi_min, pi_max)
    stats = {key: np.asarray(value) for key, value in np.load(stats_path).items()}
    anchor_mean = float(np.asarray(stats["anchor_u_mean"]))
    anchor_sd = float(np.asarray(stats["anchor_u_sd"]))
    rng, rng_metadata = evaluation_rng(config, args.evaluation_seed)

    best_runtimes = {
        method: runtime.AmortizedScoreRuntime(run_dir, method, device=args.device)
        for method in methods
    }
    milestone_runtimes: list[tuple[str, int, str, runtime.AmortizedScoreRuntime]] = []
    for method in methods:
        for step in config.get("milestone_steps_resolved", []):
            for source in ("raw", "ema"):
                checkpoint = (
                    run_dir
                    / "milestones"
                    / f"model_{method}_step{int(step)}_{source}.pt"
                )
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                milestone_runtimes.append(
                    (
                        method,
                        int(step),
                        source,
                        runtime.AmortizedScoreRuntime(
                            run_dir,
                            method,
                            device=args.device,
                            checkpoint_path=checkpoint,
                        ),
                    )
                )

    rows: list[dict[str, Any]] = []
    milestone_rows: list[dict[str, Any]] = []
    dataset_fingerprints: dict[str, Any] = {}
    for pi in pi_values:
        eval_data, truth = stage1.build_eval_data(
            rng,
            pi,
            args.n_test,
            int(config["n_blocks"]),
            int(config["block_size"]),
            float(config["tau"]),
            stats,
            anchor_mean,
            anchor_sd,
        )
        dataset_fingerprints[format(pi, ".17g")] = {
            "y_sha256": stage1.array_sha256(eval_data["y"]),
            "exact_truth_sha256": stage1.array_sha256(truth),
        }
        for method, score_runtime in best_runtimes.items():
            prediction = stage1.predict(
                score_runtime.model,
                eval_data,
                method,
                score_runtime.device,
                args.batch_size,
            )
            row = stage1.metric_row(truth, prediction)
            row.update(
                {
                    "method": score_runtime.label,
                    "pi": pi,
                    "n": args.n_test,
                }
            )
            rows.append(row)
        for method, step, source, score_runtime in milestone_runtimes:
            prediction = stage1.predict(
                score_runtime.model,
                eval_data,
                method,
                score_runtime.device,
                args.batch_size,
            )
            row = stage1.metric_row(truth, prediction)
            row.update(
                {
                    "method": score_runtime.label,
                    "checkpoint_step": step,
                    "checkpoint_source": source,
                    "pi": pi,
                    "n": args.n_test,
                }
            )
            milestone_rows.append(row)

    stage1.write_csv(output_csv, rows)
    if milestone_rows:
        stage1.write_csv(milestone_csv, milestone_rows)
    protocol = {
        "model": "model2",
        "separated_from_training": True,
        "run_dir": str(run_dir),
        "n_test_per_pi": int(args.n_test),
        "pi_values": pi_values,
        "batch_size": int(args.batch_size),
        "rng": rng_metadata,
        "datasets": dataset_fingerprints,
        "validation_best_checkpoints": {
            method: best_runtimes[method].metadata() for method in methods
        },
        "milestone_checkpoints": [
            {
                "method": method,
                "step": step,
                "source": source,
                **score_runtime.metadata(),
            }
            for method, step, source, score_runtime in milestone_runtimes
        ],
    }
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("\n==== Frozen exact-score evaluation ====")
    for row in rows:
        print(
            f"pi={row['pi']:.3g} {row['method']:<45} "
            f"std_mse={row['std_mse']:.4g} corr={row['corr']:.4g} "
            f"cosine={row['cosine']:.4g}"
        )
    print(f"Saved: {output_csv}")
    if milestone_rows:
        print(f"Saved: {milestone_csv}")
    print(f"Saved: {protocol_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--pi-values", default="0.07,0.10,0.30,0.50,0.65,0.68")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
