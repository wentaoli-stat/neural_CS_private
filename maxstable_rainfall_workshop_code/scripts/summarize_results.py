#!/usr/bin/env python3
"""Recompute the retained tables from the archived per-dataset rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _json_vector(value: str) -> np.ndarray:
    return np.asarray(json.loads(value), dtype=np.float64)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prior_wide(path: Path) -> dict[str, Any]:
    rows = _rows(path)
    error = np.stack([_json_vector(row["posterior_mean_error"]) for row in rows])
    coordinate_rmse = np.sqrt(np.mean(np.square(error), axis=0))
    return {
        "n": len(rows),
        "rmse_by_coordinate": coordinate_rmse.tolist(),
        "joint_l2_rmse": float(np.sqrt(np.mean(np.sum(np.square(error), axis=1)))),
    }


def _fixed_truth(path: Path) -> dict[str, Any]:
    rows = _rows(path)
    output: dict[str, Any] = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        truth = np.stack([_json_vector(row["truth"]) for row in selected])
        estimate = np.stack([_json_vector(row["estimate"]) for row in selected])
        mse = np.mean(np.square(estimate - truth), axis=0)
        summary: dict[str, Any] = {
            "n": len(selected),
            "mse_by_parameter": mse.tolist(),
            "sum_mse": float(np.sum(mse)),
        }
        if selected[0]["sd"]:
            summary["mean_sd_or_asymptotic_se"] = np.mean(
                [_json_vector(row["sd"]) for row in selected], axis=0
            ).tolist()
        display_method = (
            "Nonlinear-gate NPE" if method == "Positive(s,w_a) NPE" else method
        )
        output[display_method] = summary
    linear = output["rough Linear NPE"]["sum_mse"]
    nonlinear = output["Nonlinear-gate NPE"]["sum_mse"]
    output["nonlinear_gate_sum_mse_reduction_percent"] = 100.0 * (
        linear - nonlinear
    ) / linear
    return output


def _extremal(path: Path) -> dict[str, Any]:
    rows = _rows(path)
    output: dict[str, Any] = {}
    for truth in ("low", "center", "high"):
        selected = [row for row in rows if row["truth_label"] == truth]
        linear = float(
            np.mean(
                [
                    float(row["delta_integrated_squared_error"])
                    for row in selected
                    if row["method"] == "Linear"
                ]
            )
        )
        nonlinear = float(
            np.mean(
                [
                    float(row["delta_integrated_squared_error"])
                    for row in selected
                    if row["method"] == "Positive(s,w_a)"
                ]
            )
        )
        mean_error = {
            "Linear": linear,
            "Nonlinear gate": nonlinear,
            "nonlinear_gate_reduction_percent": 100.0
            * (linear - nonlinear)
            / linear,
        }
        output[truth] = mean_error
    return output


def summarize(root: Path = ROOT) -> dict[str, Any]:
    results = root / "results"
    return {
        "prior_wide": {
            "Linear": _prior_wide(results / "stage2_linear" / "posterior_by_dataset.csv"),
            "Nonlinear gate": _prior_wide(
                results / "stage2_positive_anchor" / "posterior_by_dataset.csv"
            ),
        },
        "fixed_truth": {
            label: _fixed_truth(results / "fixed_truth" / label / "by_dataset.csv")
            for label in ("low", "center", "high")
        },
        "extremal_coefficient": _extremal(
            results / "extremal" / "extremal_coefficient_by_dataset.csv"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = summarize()
    rendered = json.dumps(value, indent=2)
    if args.output is not None:
        args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
