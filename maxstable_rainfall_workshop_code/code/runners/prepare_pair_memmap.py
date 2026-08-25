#!/usr/bin/env python3
"""Stream all-pair Max-stable subscores into separate memory-mapped arrays.

This avoids materializing the all-pair three-parameter table in memory.  The
confirmatory protocol stores individual-pair scores as float32 so Stage 1 and
the dynamically recomputed Stage-2 features use the same numerical precision.
Float16 remains available only as an explicitly labelled diagnostic option.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from maxstable_rainfall79.core import (
    FormalConfig,
    jsonable,
    make_annual_pair_feature_function,
    make_pair_design,
)


def stream_split(
    *,
    label: str,
    anchor: np.ndarray,
    datasets: np.ndarray,
    output_path: Path,
    grouped_path: Path,
    batch_function,
    group_indices: list[np.ndarray],
    theta_dim: int,
    chunk_size: int,
    accumulate_statistics: bool,
    storage_dtype: np.dtype,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, float]]:
    n = len(anchor)
    years = int(datasets.shape[1])
    pairs = int(sum(len(index) for index in group_indices))
    groups = len(group_indices)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=storage_dtype,
        shape=(n, years, pairs, int(theta_dim)),
    )
    grouped = np.lib.format.open_memmap(
        grouped_path,
        mode="w+",
        dtype=np.float32,
        shape=(n, years, groups, int(theta_dim)),
    )
    total = np.zeros((groups, theta_dim), dtype=np.float64)
    total_square = np.zeros_like(total)
    count = np.zeros(groups, dtype=np.int64)
    quantization_abs_sum = 0.0
    quantization_square_sum = 0.0
    quantization_count = 0
    quantization_max_abs = 0.0
    grouped_replay_max_abs = 0.0
    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        value = np.asarray(
            jax.device_get(
                batch_function(
                    jnp.asarray(anchor[start:stop], dtype=jnp.float64),
                    jnp.asarray(datasets[start:stop], dtype=jnp.float64),
                )
            ),
            dtype=np.float32,
        )
        raw[start:stop] = value.astype(storage_dtype)
        stored = np.asarray(raw[start:stop], dtype=np.float32)
        difference = stored.astype(np.float64) - value.astype(np.float64)
        quantization_abs_sum += float(np.abs(difference).sum())
        quantization_square_sum += float(np.square(difference).sum())
        quantization_count += int(difference.size)
        quantization_max_abs = max(
            quantization_max_abs, float(np.max(np.abs(difference)))
        )
        for group, index in enumerate(group_indices):
            selected = value[:, :, index, :]
            grouped[start:stop, :, group, :] = selected.mean(axis=2)
            replay = stored[:, :, index, :].mean(axis=2)
            grouped_replay_max_abs = max(
                grouped_replay_max_abs,
                float(np.max(np.abs(replay - selected.mean(axis=2)))),
            )
            if accumulate_statistics:
                total[group] += selected.sum(axis=(0, 1, 2), dtype=np.float64)
                total_square[group] += np.square(
                    selected, dtype=np.float64
                ).sum(axis=(0, 1, 2), dtype=np.float64)
                count[group] += selected.shape[0] * selected.shape[1] * selected.shape[2]
        raw.flush()
        grouped.flush()
        print(f"{label}: {stop}/{n}", flush=True)
    del raw, grouped
    audit = {
        "storage_max_abs_error_to_float32": quantization_max_abs,
        "storage_mean_abs_error_to_float32": (
            quantization_abs_sum / max(quantization_count, 1)
        ),
        "storage_rmse_to_float32": math.sqrt(
            quantization_square_sum / max(quantization_count, 1)
        ),
        "stored_pair_to_float32_group_mean_max_abs": grouped_replay_max_abs,
    }
    if not accumulate_statistics:
        return None, None, audit
    mean = total / count[:, None]
    variance = np.maximum(total_square / count[:, None] - np.square(mean), 1e-12)
    return (
        mean.reshape(1, 1, groups, theta_dim),
        np.sqrt(variance).reshape(1, 1, groups, theta_dim),
        audit,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-pair-path", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=8_000)
    parser.add_argument("--n-distance-groups", type=int, default=20)
    parser.add_argument("--feature-chunk-size", type=int, default=16)
    args = parser.parse_args(argv)

    source = args.source_prepared_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    source_arrays = np.load(source / metadata["arrays"], allow_pickle=False)
    original = FormalConfig(**metadata["config"])
    config = replace(
        original,
        n_angle_groups=1,
        n_distance_groups=int(args.n_distance_groups),
        pairs_per_group=100_000,
    )
    design = make_pair_design(config)
    if len(design.pairs) != config.n_sites * (config.n_sites - 1) // 2:
        raise RuntimeError("the requested design did not retain every spatial pair")
    group_indices = [
        np.flatnonzero(design.group_ids == group) for group in range(config.n_groups)
    ]
    if any(len(index) == 0 for index in group_indices):
        raise RuntimeError("every distance group must contain at least one pair")
    _, batch_function = make_annual_pair_feature_function(config, design)
    n_train = min(int(args.n_train), len(source_arrays["train_anchor_w"]))
    n_val = min(int(args.n_val), len(source_arrays["val_anchor_w"]))
    train_pair_path = args.train_pair_path.resolve()
    val_pair_path = output / "val_pair.npy"
    train_grouped_path = output / "train_grouped.npy"
    val_grouped_path = output / "val_grouped.npy"

    storage_dtype = np.dtype("float32")
    pair_mean, pair_sd, train_storage_audit = stream_split(
        label="train all-pair features",
        anchor=source_arrays["train_anchor_w"][:n_train],
        datasets=source_arrays["train_y"][:n_train],
        output_path=train_pair_path,
        grouped_path=train_grouped_path,
        batch_function=batch_function,
        group_indices=group_indices,
        theta_dim=config.theta_dim,
        chunk_size=int(args.feature_chunk_size),
        accumulate_statistics=True,
        storage_dtype=storage_dtype,
    )
    _, _, val_storage_audit = stream_split(
        label="validation all-pair features",
        anchor=source_arrays["val_anchor_w"][:n_val],
        datasets=source_arrays["val_y"][:n_val],
        output_path=val_pair_path,
        grouped_path=val_grouped_path,
        batch_function=batch_function,
        group_indices=group_indices,
        theta_dim=config.theta_dim,
        chunk_size=int(args.feature_chunk_size),
        accumulate_statistics=False,
        storage_dtype=storage_dtype,
    )
    np.savez(
        output / "compact_arrays.npz",
        train_anchor_w=source_arrays["train_anchor_w"][:n_train],
        train_target=source_arrays["train_target"][:n_train],
        val_anchor_w=source_arrays["val_anchor_w"][:n_val],
        val_target=source_arrays["val_target"][:n_val],
        pair_feature_mean=pair_mean,
        pair_feature_sd=pair_sd,
        design_pairs=design.pairs,
        design_group_ids=design.group_ids,
        design_group_geometry=design.group_geometry,
        design_population_counts=design.population_counts,
    )
    output_metadata = {
        "source_prepared_dir": str(source),
        "n_train_complete_datasets": n_train,
        "n_val_complete_datasets": n_val,
        "sigma_q": metadata["sigma_q"],
        "config": asdict(config),
        "array_format": "separate_npy_memmap",
        "arrays": "compact_arrays.npz",
        "train_feature": str(train_pair_path),
        "val_feature": str(val_pair_path),
        "train_grouped_feature": str(train_grouped_path),
        "val_grouped_feature": str(val_grouped_path),
        "individual_pair_storage_dtype": str(storage_dtype),
        "grouped_storage_dtype": "float32",
        "pair_statistics_accumulation_dtype": "float64_from_float32_chunks",
        "feature_level": (
            f"all {config.n_sites * (config.n_sites - 1) // 2} pair subscores "
            f"before {config.n_groups} distance-bin means"
        ),
        "one_stage1_row": f"one complete ({config.n_years},{config.n_sites}) dataset",
        "true_full_likelihood_or_score_used": False,
        "train_storage_audit": train_storage_audit,
        "validation_storage_audit": val_storage_audit,
    }
    (output / "metadata.json").write_text(
        json.dumps(jsonable(output_metadata), indent=2), encoding="utf-8"
    )
    print(json.dumps(jsonable(output_metadata), indent=2), flush=True)


if __name__ == "__main__":
    main()
