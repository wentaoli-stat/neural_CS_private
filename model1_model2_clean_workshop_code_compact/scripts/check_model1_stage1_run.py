#!/usr/bin/env python3
"""Gate a matched-readout Model 1 Stage-1 directory before Stage 2.

Checks that lr, gate_lr and joint_rho_lr are all equal to --expected-lr, that
linear/radial/stacked were trained, share one minibatch SHA-256, and that the
nested-initialization error is at most --max-nesting. Exits non-zero otherwise.

    python scripts/check_model1_stage1_run.py runs/s1_sq0.20_20260710 [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check(run_dir: Path, expected_lr: float, max_nesting: float) -> bool:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    info = json.loads((run_dir / "training_info.json").read_text(encoding="utf-8"))
    lrs = (float(config["lr"]), float(config["gate_lr"]), float(config["joint_rho_lr"]))
    shas = {item["minibatch_sha256"] for item in info.values()}
    nesting = {m: item.get("nested_initialization_max_abs_diff") for m, item in info.items()}
    good = (
        all(abs(lr - expected_lr) < 1e-12 for lr in lrs)
        and set(info) == {"linear", "radial", "stacked"}
        and len(shas) == 1
        and all(value is None or float(value) <= max_nesting for value in nesting.values())
        and (run_dir / "score_summary_by_pi.csv").exists()
    )
    print(run_dir, "lrs", lrs, "n_sha", len(shas), "nesting", nesting,
          "best_step", {m: item["best_step"] for m, item in info.items()},
          "OK" if good else "FAILED", flush=True)
    return good


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--expected-lr", type=float, default=1e-3)
    parser.add_argument("--max-nesting", type=float, default=1e-5)
    args = parser.parse_args()
    results = [check(d, args.expected_lr, args.max_nesting) for d in args.run_dirs]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
