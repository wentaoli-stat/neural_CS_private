#!/usr/bin/env python3
"""Fast integrity checks that do not require checkpoints or the Jiang checkout."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_TEXT = ("/root/autodl-tmp", "connect.west", "seetacloud.com")
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".npy", ".npz", ".pkl", ".pickle")


def fail(message: str) -> None:
    raise SystemExit(f"package check failed: {message}")


def check_json() -> int:
    count = 0
    for path in ROOT.rglob("*.json"):
        with path.open(encoding="utf-8") as handle:
            json.load(handle)
        count += 1
    return count


def check_csv() -> int:
    count = 0
    for path in ROOT.rglob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            fail(f"empty CSV: {path.relative_to(ROOT)}")
        count += 1
    return count


def check_portability() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            fail(f"large/generated binary was included: {path.relative_to(ROOT)}")
        if path.suffix.lower() not in {".py", ".md", ".json", ".toml", ".txt", ".csv", ""}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in FORBIDDEN_TEXT:
            if needle in text:
                fail(f"non-portable text {needle!r} in {path.relative_to(ROOT)}")


def check_headlines() -> None:
    for name in ("headline_score_stdmse.csv", "headline_posterior_mean_rmse.csv"):
        path = ROOT / "results" / name
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if [row["model"] for row in rows] != ["model1", "model2", "model3"]:
            fail(f"unexpected model rows in {path.relative_to(ROOT)}")


def main() -> None:
    json_count = check_json()
    csv_count = check_csv()
    check_portability()
    check_headlines()
    print(f"OK: {json_count} JSON files and {csv_count} CSV files validated")
    print("OK: headline tables contain all three models")
    print("OK: no checkpoint binaries or machine-specific remote paths found")


if __name__ == "__main__":
    main()
