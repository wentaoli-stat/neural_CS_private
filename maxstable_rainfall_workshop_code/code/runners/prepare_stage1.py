#!/usr/bin/env python3
"""Simulate the locked 47-year by 79-site Direct-FSM rows."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from maxstable_rainfall79.core import (
    FormalConfig,
    jsonable,
    make_pair_design,
    sample_stage1_rows,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=8_000)
    parser.add_argument("--sigma-q", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args(argv)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = FormalConfig(
        n_years=47,
        n_sites=79,
        n_angle_groups=1,
        n_distance_groups=20,
        pairs_per_group=100_000,
        pair_seed=20260731,
        active_indices=(0, 1, 2),
        fixed_u=(0.5, 0.5, 0.5),
    )
    config.validate()
    rng = np.random.default_rng(int(args.seed))
    design = make_pair_design(config)

    print("Stage-1 train rows", flush=True)
    sigma_q = float(args.sigma_q)
    train_anchor, train_target, train_parameter, train_y = sample_stage1_rows(
        rng,
        int(args.n_train),
        config,
        sigma_q,
        int(args.seed) + 10_000,
    )
    print("Stage-1 validation rows", flush=True)
    val_anchor, val_target, val_parameter, val_y = sample_stage1_rows(
        rng,
        int(args.n_val),
        config,
        sigma_q,
        int(args.seed) + 100_000,
    )

    # Pair features are streamed separately so this file stays a compact source bank.
    train_feature = np.empty((0,), dtype=np.float32)
    val_feature = np.empty((0,), dtype=np.float32)

    np.savez_compressed(
        output / "stage1_arrays.npz",
        train_anchor_w=train_anchor,
        train_target=train_target,
        train_parameter_u=train_parameter,
        train_y=train_y,
        train_feature=train_feature,
        val_anchor_w=val_anchor,
        val_target=val_target,
        val_parameter_u=val_parameter,
        val_y=val_y,
        val_feature=val_feature,
        design_pairs=design.pairs,
        design_group_ids=design.group_ids,
        design_group_geometry=design.group_geometry,
        design_population_counts=design.population_counts,
    )
    metadata = {
        "seed": int(args.seed),
        "n_train_complete_datasets": int(args.n_train),
        "n_val_complete_datasets": int(args.n_val),
        "sigma_q": sigma_q,
        "config": asdict(config),
        "one_stage1_row": f"one complete ({config.n_years},{config.n_sites}) dataset",
        "outer_blocks": f"{config.n_years} iid annual fields",
        "grouped_features_deferred_to_all_pair_pass": True,
        "inner_tokens": (
            f"{config.n_groups} direction-distance composite-score groups, "
            f"{config.pairs_per_group} selected spatial pairs per group"
        ),
        "true_full_likelihood_or_score_used": False,
        "target": "diag(sigma_q^-2) @ (sampled_w-anchor_w)",
        "arrays": "stage1_arrays.npz",
    }
    (output / "metadata.json").write_text(
        json.dumps(jsonable(metadata), indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
