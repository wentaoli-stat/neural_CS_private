#!/usr/bin/env python3
"""Aggregate paired fixed-sigma Stage-1 FSM runs across training seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def short_method(label: str) -> str:
    if label.startswith("linear "):
        return "linear"
    if label.startswith("radial "):
        return "radial"
    raise ValueError(f"Unknown method label: {label}")


def seed_from_dir(path: Path) -> int:
    match = re.search(r"seed(\d+)$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse seed from {path}")
    return int(match.group(1))


def format_mean_sd(mu: float, sd: float) -> str:
    return f"{mu:.4f} +/- {sd:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", type=str, default="formal40k_fixed_sigma_seed")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("runs/formal40k_fixed_sigma_summary"),
    )
    args = parser.parse_args()

    run_dirs = sorted(path for path in args.runs_dir.glob(f"{args.prefix}*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matching {args.runs_dir / (args.prefix + '*')}")

    score_rows: list[dict[str, Any]] = []
    fsm_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    configs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        seed = seed_from_dir(run_dir)
        configs.append(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        for row in read_csv(run_dir / "score_summary_by_pi.csv"):
            score_rows.append(
                {
                    "seed": seed,
                    "pi": float(row["pi"]),
                    "method": short_method(row["method"]),
                    "mse": float(row["mse"]),
                    "std_mse": float(row["std_mse"]),
                    "corr": float(row["corr"]),
                    "cosine": float(row["cosine"]),
                }
            )
        for row in read_csv(run_dir / "fixed_grid_fsm_loss.csv"):
            fsm_rows.append(
                {
                    "seed": seed,
                    "pi": float(row["pi"]),
                    "method": short_method(row["method"]),
                    "fsm_relative_mse": float(row["fsm_relative_mse"]),
                }
            )
        info = json.loads((run_dir / "training_info.json").read_text(encoding="utf-8"))
        for method in ("linear", "radial"):
            training_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "best_step": int(info[method]["best_step"]),
                    "best_val_loss": float(info[method]["best_val_loss"]),
                    "best_source": info[method]["best_source"],
                }
            )

    paired_score: dict[tuple[int, float], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in score_rows:
        paired_score[(row["seed"], row["pi"])][row["method"]] = row

    summary_by_pi: list[dict[str, Any]] = []
    all_improvements: list[float] = []
    all_wins = 0
    for pi in sorted({row["pi"] for row in score_rows}):
        pairs = [pair for (seed, pair_pi), pair in paired_score.items() if pair_pi == pi]
        if any(set(pair) != {"linear", "radial"} for pair in pairs):
            raise RuntimeError(f"Incomplete method pair at pi={pi}")
        linear_values = [pair["linear"]["std_mse"] for pair in pairs]
        radial_values = [pair["radial"]["std_mse"] for pair in pairs]
        improvements = [100.0 * (linear - radial) / linear for linear, radial in zip(linear_values, radial_values)]
        wins = sum(radial < linear for linear, radial in zip(linear_values, radial_values))
        all_improvements.extend(improvements)
        all_wins += wins
        summary_by_pi.append(
            {
                "pi": pi,
                "n_seeds": len(pairs),
                "linear_std_mse_mean": mean(linear_values),
                "linear_std_mse_sd": sample_sd(linear_values),
                "radial_std_mse_mean": mean(radial_values),
                "radial_std_mse_sd": sample_sd(radial_values),
                "radial_improvement_pct_mean": mean(improvements),
                "radial_improvement_pct_sd": sample_sd(improvements),
                "radial_wins": wins,
            }
        )

    seed_summary: list[dict[str, Any]] = []
    for seed in sorted({row["seed"] for row in score_rows}):
        pairs = [pair for (pair_seed, pi), pair in paired_score.items() if pair_seed == seed]
        linear_mean = mean(pair["linear"]["std_mse"] for pair in pairs)
        radial_mean = mean(pair["radial"]["std_mse"] for pair in pairs)
        seed_summary.append(
            {
                "seed": seed,
                "n_pi": len(pairs),
                "linear_mean_std_mse": linear_mean,
                "radial_mean_std_mse": radial_mean,
                "radial_improvement_pct": 100.0 * (linear_mean - radial_mean) / linear_mean,
                "radial_wins": int(radial_mean < linear_mean),
            }
        )

    paired_t = float("nan")
    paired_p = float("nan")
    try:
        from scipy.stats import ttest_rel

        test = ttest_rel(
            [row["linear_mean_std_mse"] for row in seed_summary],
            [row["radial_mean_std_mse"] for row in seed_summary],
        )
        paired_t = float(test.statistic)
        paired_p = float(test.pvalue)
    except ImportError:
        pass

    paired_fsm: dict[tuple[int, float], dict[str, float]] = defaultdict(dict)
    for row in fsm_rows:
        paired_fsm[(row["seed"], row["pi"])][row["method"]] = row["fsm_relative_mse"]
    fsm_summary: list[dict[str, Any]] = []
    for pi in sorted({row["pi"] for row in fsm_rows}):
        pairs = [pair for (seed, pair_pi), pair in paired_fsm.items() if pair_pi == pi]
        linear_values = [pair["linear"] for pair in pairs]
        radial_values = [pair["radial"] for pair in pairs]
        improvements = [100.0 * (linear - radial) / linear for linear, radial in zip(linear_values, radial_values)]
        fsm_summary.append(
            {
                "pi": pi,
                "n_seeds": len(pairs),
                "linear_fsm_relative_mse_mean": mean(linear_values),
                "linear_fsm_relative_mse_sd": sample_sd(linear_values),
                "radial_fsm_relative_mse_mean": mean(radial_values),
                "radial_fsm_relative_mse_sd": sample_sd(radial_values),
                "radial_improvement_pct_mean": mean(improvements),
                "radial_wins": sum(radial < linear for linear, radial in zip(linear_values, radial_values)),
            }
        )

    output_prefix = args.output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_prefix.with_name(output_prefix.name + "_by_pi.csv"), summary_by_pi)
    write_csv(output_prefix.with_name(output_prefix.name + "_by_seed.csv"), seed_summary)
    write_csv(output_prefix.with_name(output_prefix.name + "_fsm_grid.csv"), fsm_summary)
    write_csv(output_prefix.with_name(output_prefix.name + "_training.csv"), training_rows)

    config = configs[0]
    total_pairs = len(paired_score)
    report_lines = [
        "# Model 2 amortized FSM: fixed-sigma formal results",
        "",
        "## Setting",
        "",
        f"- Training seeds: `{', '.join(str(seed_from_dir(path)) for path in run_dirs)}`",
        f"- `n_train={config['n_train']}`, `n_val={config['n_val']}`, `n_test={config['n_test']}` per diagnostic anchor",
        f"- `K={config['n_blocks']}`, `m={config['block_size']}`, `tau={config['tau']}`",
        f"- Continuous stratified anchors: `pi in [{config['anchor_pi_min']}, {config['anchor_pi_max']}]`",
        f"- Fixed FSM proposal width: `sigma_q={config['sigma_q']}` in logit coordinate",
        f"- Architecture: `{config['rho_activation']}` rho head, positive signed anchor-conditioned radial gates",
        "- Exact full score is used only for the following post-training diagnostics.",
        "",
        "## Exact-score standardized MSE",
        "",
        "| pi | linear mean +/- sd | radial mean +/- sd | radial improvement | wins |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary_by_pi:
        report_lines.append(
            "| "
            f"{row['pi']:.2f} | "
            f"{format_mean_sd(row['linear_std_mse_mean'], row['linear_std_mse_sd'])} | "
            f"{format_mean_sd(row['radial_std_mse_mean'], row['radial_std_mse_sd'])} | "
            f"{row['radial_improvement_pct_mean']:.1f}% | "
            f"{row['radial_wins']}/{row['n_seeds']} |"
        )
    report_lines += [
        "",
        "## Overall paired result",
        "",
        f"- Radial wins: **{all_wins}/{total_pairs}** seed-anchor comparisons.",
        f"- Mean paired relative reduction in exact-score std-MSE: **{mean(all_improvements):.1f}%** "
        f"(SD across pairs {sample_sd(all_improvements):.1f} percentage points).",
        f"- After averaging the six anchors within each independent training seed, radial wins **{sum(row['radial_wins'] for row in seed_summary)}/{len(seed_summary)}** seeds.",
        (
            f"- Exploratory paired seed-level test: `t({len(seed_summary) - 1})={paired_t:.2f}`, `p={paired_p:.4f}`. "
            "The seed, not the seed-anchor row, is treated as the independent unit."
            if math.isfinite(paired_t)
            else "- SciPy was unavailable, so no paired seed-level p-value was computed."
        ),
        "- Five seeds satisfy the planned formal replication count, although wider architecture claims should still be checked in another model setting.",
        "",
        "## Fixed-grid FSM objective",
        "",
        "| pi | linear relative MSE | radial relative MSE | radial improvement | wins |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in fsm_summary:
        report_lines.append(
            "| "
            f"{row['pi']:.2f} | "
            f"{format_mean_sd(row['linear_fsm_relative_mse_mean'], row['linear_fsm_relative_mse_sd'])} | "
            f"{format_mean_sd(row['radial_fsm_relative_mse_mean'], row['radial_fsm_relative_mse_sd'])} | "
            f"{row['radial_improvement_pct_mean']:.2f}% | "
            f"{row['radial_wins']}/{row['n_seeds']} |"
        )
    report_lines += [
        "",
        "The FSM regression target has variance `1/sigma_q^2`, so differences in its relative MSE are expected to be much smaller than differences in the noiseless exact-score diagnostic.",
        "",
    ]
    report_path = output_prefix.with_suffix(".md")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
