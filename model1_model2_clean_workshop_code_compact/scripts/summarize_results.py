#!/usr/bin/env python3
"""Recompute the compact headline tables from the packaged CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATORY = "artifacts/confirmatory_20260820"


def has_confirmatory() -> bool:
    """Return whether the optional full multi-seed result tree is present."""
    return (ROOT / CONFIRMATORY).is_dir()


def rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct_reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def paired_stage1_means(
    table: list[dict[str, str]], nonlinear_prefix: str
) -> tuple[float, float]:
    linear = [float(row["std_mse"]) for row in table if row["method"].startswith("linear")]
    nonlinear = [
        float(row["std_mse"])
        for row in table
        if row["method"].startswith(nonlinear_prefix)
    ]
    return mean(linear), mean(nonlinear)


def pooled_w1(path: str, linear_prefix: str, nonlinear_prefix: str) -> tuple[float, float]:
    table = rows(path)
    linear = [row for row in table if row["method"].startswith(linear_prefix)]
    nonlinear = [row for row in table if row["method"].startswith(nonlinear_prefix)]
    column = "mean_w1_to_exact" if "mean_w1_to_exact" in linear[0] else "w1_to_exact"
    return (
        mean(float(row[column]) for row in linear),
        mean(float(row[column]) for row in nonlinear),
    )


def stage2_metrics(path: str, method: str) -> dict[str, float]:
    table = [row for row in rows(path) if row["method"] == method]
    return {
        "truth_mse": mean(float(row["pi_avg_mse"]) for row in table),
        "rmse_exact": math.sqrt(
            mean(float(row["mean_mse_to_exact"]) for row in table)
        ),
        "w1": mean(float(row["w1_to_exact"]) for row in table),
        "sd": mean(float(row["avg_post_sd"]) for row in table),
        "coverage": mean(float(row["coverage90"]) for row in table),
    }


def model1(*, include_baselines: bool = False) -> None:
    linear = mean(
        float(row["std_mse"])
        for row in rows("artifacts/model1/stage1/linear_validation_best/score_summary_by_pi.csv")
    )
    radial = mean(
        float(row["std_mse"])
        for row in rows("artifacts/model1/stage1/radial_matched_validation_best/score_summary_by_pi.csv")
    )
    pooled = rows("artifacts/model1/stage2/npe_60/posterior_summary_pooled.csv")
    w1 = {row["method"]: float(row["mean_w1_to_exact"]) for row in pooled}
    print("Model 1 (20x20)")
    print(
        f"  Stage 1 mean stdMSE: Linear={linear:.6f}, Nonlinear gate={radial:.6f}, "
        f"reduction={pct_reduction(linear, radial):.2f}%"
    )
    print(
        "  Stage 2 mean W1: "
        f"Linear={w1['linear FSM pilot+score NPE']:.6f}, "
        f"Nonlinear gate={w1['radial FSM pilot+score NPE']:.6f}, "
        f"reduction={pct_reduction(w1['linear FSM pilot+score NPE'], w1['radial FSM pilot+score NPE']):.2f}%"
    )
    if has_confirmatory():
        stage1_pairs = [(linear, radial)]
        stage2_pairs = [
            (
                w1["linear FSM pilot+score NPE"],
                w1["radial FSM pilot+score NPE"],
            )
        ]
        for seed in range(20260710, 20260714):
            stage1_pairs.append(
                paired_stage1_means(
                    rows(
                        f"{CONFIRMATORY}/multiseed/model1/stage1/seed_{seed}/"
                        "score_summary_by_pi.csv"
                    ),
                    "radial",
                )
            )
            stage2_pairs.append(
                pooled_w1(
                    f"{CONFIRMATORY}/multiseed/model1/stage2/"
                    f"stage1_seed_{seed}_npe_seed_54000/posterior_summary_pooled.csv",
                    "linear FSM",
                    "radial FSM",
                )
            )
        stage1_linear, stage1_radial = map(mean, zip(*stage1_pairs, strict=True))
        stage2_linear, stage2_radial = map(mean, zip(*stage2_pairs, strict=True))
        print(
            f"  Five-seed Stage 1 stdMSE: {stage1_linear:.6f}->{stage1_radial:.6f} "
            f"({pct_reduction(stage1_linear, stage1_radial):.2f}% reduction)"
        )
        print(
            f"  Five-seed Stage 2 W1: {stage2_linear:.6f}->{stage2_radial:.6f} "
            f"({pct_reduction(stage2_linear, stage2_radial):.2f}% reduction)"
        )
    else:
        print("  Five-seed summary: skipped (confirmatory artifacts not packaged)")
    if include_baselines:
        raw = stage2_metrics(
            "artifacts/model1/stage2/raw_data_npe_50k/posterior_summary.csv",
            "raw data NPE",
        )
        print(
            f"  Optional raw-data NPE: truth MSE={raw['truth_mse']:.6f}, "
            f"RMSE-exact={raw['rmse_exact']:.6f}, W1={raw['w1']:.6f}, "
            f"coverage={raw['coverage']:.3f}"
        )
        direct_stage1 = mean(
            float(row["std_mse"])
            for row in rows(
                "artifacts/model1/direct_raw_fsm/stage1/score_summary_by_pi.csv"
            )
        )
        direct = stage2_metrics(
            "artifacts/model1/direct_raw_fsm/stage2/posterior_summary.csv",
            "direct amortized raw-FSM pilot+score NPE",
        )
        print(
            f"  Optional direct raw-FSM: Stage1 stdMSE={direct_stage1:.6f}; "
            f"truth MSE={direct['truth_mse']:.6f}, "
            f"RMSE-exact={direct['rmse_exact']:.6f}, W1={direct['w1']:.6f}, "
            f"coverage={direct['coverage']:.3f}"
        )


def stage1_policy(seed: str, source: str) -> tuple[float, float, int]:
    if source == "validation_best":
        table = rows(f"artifacts/model2/k40_m40/stage1/{seed}/score_summary_by_pi.csv")
    else:
        table = [
            row
            for row in rows(
                f"artifacts/model2/k40_m40/stage1/{seed}/milestone_score_summary_by_pi.csv"
            )
            if row["checkpoint_step"] == "20000" and row["checkpoint_source"] == "ema"
        ]
    linear = [float(row["std_mse"]) for row in table if row["method"].startswith("linear CS")]
    shared = [float(row["std_mse"]) for row in table if row["method"].startswith("shared-raw-gate")]
    wins = sum(candidate < reference for reference, candidate in zip(linear, shared, strict=True))
    return mean(linear), mean(shared), wins


def stage2_policy(directory: str) -> tuple[dict[str, float], dict[str, float]]:
    table = rows(f"artifacts/model2/k40_m40/stage2/{directory}/posterior_summary.csv")
    output: dict[str, dict[str, float]] = {}
    for label, prefix in (("linear", "linear FSM"), ("shared", "shared-raw-gate")):
        selected = [row for row in table if row["method"].startswith(prefix)]
        output[label] = {
            "w1": mean(float(row["w1_to_exact"]) for row in selected),
            "rmse": math.sqrt(mean(float(row["rmse_mean_to_exact"]) ** 2 for row in selected)),
            "mse": mean(float(row["pi_avg_mse"]) for row in selected),
            "coverage": mean(float(row["coverage90"]) for row in selected),
        }
    return output["linear"], output["shared"]


def model2(*, include_baselines: bool = False) -> None:
    print("Model 2 (40x40)")
    for seed in ("seed_20260709", "seed_20260710"):
        for policy in ("fixed20k_ema", "validation_best"):
            linear, shared, wins = stage1_policy(seed, policy)
            print(
                f"  Stage 1 {seed.removeprefix('seed_')} {policy}: "
                f"Linear={linear:.6f}, Nonlinear gate={shared:.6f}, "
                f"reduction={pct_reduction(linear, shared):.2f}%, pi wins={wins}/6"
            )
    if has_confirmatory():
        stage1_pairs = []
        stage2_pairs = []
        for seed in range(20260709, 20260714):
            if seed <= 20260710:
                stage1_path = (
                    f"artifacts/model2/k40_m40/stage1/seed_{seed}/score_summary_by_pi.csv"
                )
            else:
                stage1_path = (
                    f"{CONFIRMATORY}/multiseed/model2/stage1/seed_{seed}/"
                    "score_summary_by_pi.csv"
                )
            stage1_pairs.append(
                paired_stage1_means(rows(stage1_path), "shared-raw-gate")
            )
            if seed == 20260709:
                stage2_path = (
                    "artifacts/model2/k40_m40/stage2/"
                    "npe_validation_best_seed_20260709/posterior_summary.csv"
                )
            else:
                stage2_path = (
                    f"{CONFIRMATORY}/multiseed/model2/stage2/"
                    f"stage1_seed_{seed}_npe_seed_54000/posterior_summary.csv"
                )
            stage2_pairs.append(
                pooled_w1(stage2_path, "linear FSM", "shared-raw-gate FSM")
            )
        stage1_linear, stage1_shared = map(mean, zip(*stage1_pairs, strict=True))
        stage2_linear, stage2_shared = map(mean, zip(*stage2_pairs, strict=True))
        print(
            f"  Five-seed Stage 1 stdMSE: {stage1_linear:.6f}->{stage1_shared:.6f} "
            f"({pct_reduction(stage1_linear, stage1_shared):.2f}% reduction)"
        )
        print(
            f"  Five-seed Stage 2 W1: {stage2_linear:.6f}->{stage2_shared:.6f} "
            f"({pct_reduction(stage2_linear, stage2_shared):.2f}% reduction)"
        )
    else:
        print("  Five-seed summary: skipped (confirmatory artifacts not packaged)")
    for label, directory in (
        ("fixed20k_ema", "npe_fixed20k_ema_seed_20260709"),
        ("validation_best", "npe_validation_best_seed_20260709"),
    ):
        linear, shared = stage2_policy(directory)
        print(
            f"  Stage 2 {label}: W1 {linear['w1']:.6f}->{shared['w1']:.6f} "
            f"({pct_reduction(linear['w1'], shared['w1']):.2f}% reduction); "
            f"RMSE-exact {linear['rmse']:.6f}->{shared['rmse']:.6f}; "
            f"MSE {linear['mse']:.6f}->{shared['mse']:.6f}; "
            f"coverage {linear['coverage']:.3f}->{shared['coverage']:.3f}"
        )
    if include_baselines:
        raw = stage2_metrics(
            "artifacts/model2/k40_m40/stage2/raw_data_npe_50k/posterior_summary.csv",
            "raw data NPE",
        )
        print(
            f"  Optional raw-data NPE: truth MSE={raw['truth_mse']:.6f}, "
            f"RMSE-exact={raw['rmse_exact']:.6f}, W1={raw['w1']:.6f}, "
            f"coverage={raw['coverage']:.3f}"
        )
        direct_stage1 = mean(
            float(row["std_mse"])
            for row in rows(
                "artifacts/model2/k40_m40/direct_raw_fsm/stage1/score_summary_by_pi.csv"
            )
        )
        direct = stage2_metrics(
            "artifacts/model2/k40_m40/direct_raw_fsm/stage2/posterior_summary.csv",
            "direct amortized raw-FSM pilot+score NPE",
        )
        print(
            f"  Optional direct raw-FSM: Stage1 stdMSE={direct_stage1:.6f}; "
            f"truth MSE={direct['truth_mse']:.6f}, "
            f"RMSE-exact={direct['rmse_exact']:.6f}, W1={direct['w1']:.6f}, "
            f"coverage={direct['coverage']:.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="also print the optional naive/direct raw-data diagnostics",
    )
    args = parser.parse_args()
    model1(include_baselines=args.include_baselines)
    model2(include_baselines=args.include_baselines)
