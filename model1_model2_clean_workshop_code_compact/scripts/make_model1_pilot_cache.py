#!/usr/bin/env python3
"""Precompute the Model 1 Stage-2 marginal-pilot cache for one NPE training bank.

Uses the same simulator, seeds, grid and pilot code that ``model1.stage2_npe``
runs on first use, and the same array names, but writes to a temporary file and
renames it into place, so several Stage-2 runs can start without racing to
create the cache or reading a half-written one.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model1 import stage2_npe as s2  # noqa: E402  (package root must be importable)

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--sbi-train-seed", type=int, default=20260723)
ap.add_argument("--n-sbi-train", type=int, default=50_000)
ap.add_argument("--pi-prior-min", type=float, default=0.05)
ap.add_argument("--pi-prior-max", type=float, default=0.70)
ap.add_argument("--pilot-grid-size", type=int, default=201)
ap.add_argument("--pilot-batch-size", type=int, default=512)
ap.add_argument("--pilot-grid-chunk-size", type=int, default=16)
ap.add_argument("--pilot-progress-every", type=int, default=10)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
if a.out.exists():
    raise SystemExit(f"refusing to overwrite existing cache: {a.out}")
metadata = {"n_blocks": 20, "block_size": 20, "tau": 0.5}
rng = np.random.default_rng(a.sbi_train_seed)
pi_train = rng.uniform(a.pi_prior_min, a.pi_prior_max, size=a.n_sbi_train)
y_train = s2.simulate(rng, pi_train, metadata)
pilot_u, status = s2.data_only_pilot(y_train, metadata, a)
a.out.parent.mkdir(parents=True, exist_ok=True)
tmp = a.out.with_name(a.out.stem + ".tmp.npz")
np.savez_compressed(
    tmp, pi_train=pi_train.astype(np.float32), pilot_u=pilot_u.astype(np.float64),
    pilot_status=status, pilot_grid_size=np.asarray(a.pilot_grid_size), tau=np.asarray(0.5),
    sbi_train_seed=np.asarray(a.sbi_train_seed), pilot_mode=np.asarray(s2.SELECTED_PILOT_MODE),
    pilot_space=np.asarray("uniform_pi"),
)
os.replace(tmp, a.out)
print(f"wrote {a.out}: n={pilot_u.size}, boundary rate "
      f"{np.mean(np.isin(status, ['left_boundary', 'right_boundary'])):.3f}")
