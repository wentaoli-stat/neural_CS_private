#!/usr/bin/env python3
"""Cheap integrity and provenance checks for the selected snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_one(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 1:
        raise SystemExit(f"expected one selected Stage-2 row in {path}")
    return value[0]


def main() -> None:
    required = (
        ROOT / "README.md",
        ROOT / "MANIFEST.sha256",
        ROOT / "config.json",
        ROOT / "code" / "src" / "maxstable_rainfall79" / "core.py",
        ROOT / "code" / "runners" / "train_stage1.py",
        ROOT / "code" / "runners" / "run_stage2.py",
        ROOT / "scripts" / "summarize_results.py",
        ROOT / "results" / "stage2_linear" / "npe.pt",
        ROOT / "results" / "stage2_positive_anchor" / "npe.pt",
        ROOT / "results" / "extremal" / "summary.json",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing selected files:\n" + "\n".join(missing))

    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected_digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected_digest:
            raise SystemExit(f"manifest mismatch: {relative}")

    linear = load_one(ROOT / "results" / "stage2_linear" / "summary.json")
    positive = load_one(
        ROOT / "results" / "stage2_positive_anchor" / "summary.json"
    )
    expected = (0.07576637564662042, 0.0715136193696381)
    actual = (
        float(linear["posterior_mean_l2_rmse_to_generating_parameter"]),
        float(positive["posterior_mean_l2_rmse_to_generating_parameter"]),
    )
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-14):
        raise SystemExit(f"unexpected selected Stage-2 metrics: {actual}")

    reductions = []
    for label in ("low", "center", "high"):
        summary = json.loads(
            (ROOT / "results" / "fixed_truth" / label / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        methods = {row["method"]: row for row in summary["methods"]}
        expected_methods = {
            "full all-pair MPLE",
            "rough pilot-only NPE",
            "rough Linear NPE",
            "Positive(s,w_a) NPE",
        }
        if set(methods) != expected_methods:
            raise SystemExit(f"unexpected methods in fixed truth {label}: {set(methods)}")
        linear_mse = sum(methods["rough Linear NPE"]["mse_by_parameter"])
        positive_mse = sum(methods["Positive(s,w_a) NPE"]["mse_by_parameter"])
        reductions.append(100.0 * (linear_mse - positive_mse) / linear_mse)
    if not np.allclose(
        reductions,
        (31.145559298496654, 9.966033612220457, 18.614424902766356),
        rtol=0.0,
        atol=1e-12,
    ):
        raise SystemExit(f"unexpected fixed-truth reductions: {reductions}")

    print("Selected Max-stable snapshot passed")
    print(
        "prior-wide Identity/Score Gate joint RMSE: "
        f"{actual[0]:.8f}/{actual[1]:.8f}"
    )
    print("fixed-truth summed-MSE reductions:", ", ".join(f"{x:.2f}%" for x in reductions))
    for path in required[-3:]:
        print(f"{path.relative_to(ROOT)} sha256={sha256(path)}")


if __name__ == "__main__":
    main()
