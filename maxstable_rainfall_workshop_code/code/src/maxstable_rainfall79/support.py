"""Small, self-contained Smith-model support used by the selected experiment.

This module replaces the historical Model-3/Jiang import chain.  It contains
only the rainfall79 covariance map, pair grouping, and frozen simulator loader
needed by the 47 x 79 Direct-FSM experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # Training from an existing feature cache only needs torch.
    jax = None
    jnp = None


RAINFALL_BASE_THETA = np.array(
    [
        332.15273678,
        70.39821371,
        184.62656960,
        20.65249877,
        0.06445933,
        -0.15529697,
        3.54050776,
        0.02308246,
        -0.03934086,
        0.19174323,
    ],
    dtype=np.float64,
)


def _require_maxstable_upstream():
    if jax is None or jnp is None:
        raise RuntimeError("JAX is required for simulation and feature preparation")
    package_root = Path(__file__).resolve().parents[3]
    upstream = Path(
        os.environ.get(
            "MAXSTABLE_UPSTREAM",
            package_root / "code" / "upstream" / "maxstable",
        )
    ).expanduser().resolve()
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    try:
        import dataset as dataset_module
        from maxstable_jax_likelihood import (
            max_stable_pairwise_loglik_multi_obs_jax as pair_one,
            max_stable_pairwise_loglik_pairs_jax as pair_many,
        )
    except ImportError as exc:
        raise RuntimeError(
            "R/rpy2/SpatialExtremes and the bundled Smith support are required"
        ) from exc
    return dataset_module, pair_one, pair_many


@dataclass(frozen=True)
class Model3Config:
    u_min: float = 0.15
    u_max: float = 0.85
    u_true: tuple[float, float, float] = (0.5, 0.5, 0.5)
    log_sd_span: float = 0.35
    fisher_z_span: float = 0.70
    n_years: int = 47
    n_sites: int = 79
    n_angle_groups: int = 4
    n_distance_groups: int = 5

    @property
    def theta_dim(self) -> int:
        return 3

    @property
    def n_groups(self) -> int:
        return self.n_angle_groups * self.n_distance_groups


@dataclass
class DirectionalPairDesign:
    pairs: np.ndarray
    group_ids: np.ndarray
    group_geometry: np.ndarray
    group_counts: np.ndarray


def theta10_from_u_jax(u: "jnp.ndarray", config: Model3Config) -> "jnp.ndarray":
    base = jnp.asarray(RAINFALL_BASE_THETA, dtype=jnp.float64)
    sd_x0 = jnp.sqrt(base[0])
    sd_y0 = jnp.sqrt(base[2])
    rho0 = base[1] / (sd_x0 * sd_y0)
    centered = 2.0 * (jnp.asarray(u, dtype=jnp.float64) - 0.5)
    sd_x = sd_x0 * jnp.exp(float(config.log_sd_span) * centered[0])
    sd_y = sd_y0 * jnp.exp(float(config.log_sd_span) * centered[1])
    rho = jnp.tanh(jnp.arctanh(rho0) + float(config.fisher_z_span) * centered[2])
    return base.at[:3].set(
        jnp.asarray((sd_x * sd_x, rho * sd_x * sd_y, sd_y * sd_y))
    )


def theta10_from_u_np(u: np.ndarray, config: Model3Config) -> np.ndarray:
    value = np.asarray(u, dtype=np.float64)
    base = RAINFALL_BASE_THETA
    sd_x0, sd_y0 = math.sqrt(float(base[0])), math.sqrt(float(base[2]))
    rho0 = float(base[1] / (sd_x0 * sd_y0))
    centered = 2.0 * (value - 0.5)
    sd_x = sd_x0 * np.exp(float(config.log_sd_span) * centered[..., 0])
    sd_y = sd_y0 * np.exp(float(config.log_sd_span) * centered[..., 1])
    rho = np.tanh(
        np.arctanh(rho0) + float(config.fisher_z_span) * centered[..., 2]
    )
    output = np.broadcast_to(base, value.shape[:-1] + (10,)).copy()
    output[..., 0] = sd_x * sd_x
    output[..., 1] = rho * sd_x * sd_y
    output[..., 2] = sd_y * sd_y
    return output


def make_directional_pair_design(
    config: Model3Config,
    coords: np.ndarray | None = None,
) -> DirectionalPairDesign:
    """Use all pairs and form direction-by-equal-count-distance groups."""

    if coords is None:
        dataset_module, _, _ = _require_maxstable_upstream()
        coords = np.asarray(dataset_module.create_2d_grid(), dtype=np.float64)
    else:
        coords = np.asarray(coords, dtype=np.float64)
    if coords.shape != (config.n_sites, 2):
        raise ValueError(f"expected coordinates ({config.n_sites},2), got {coords.shape}")
    pairs = np.asarray(
        [(i, j) for i in range(config.n_sites) for j in range(i + 1, config.n_sites)],
        dtype=np.int32,
    )
    delta = coords[pairs[:, 1]] - coords[pairs[:, 0]]
    distance = np.linalg.norm(delta, axis=1)
    angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), np.pi)
    angle_id = np.minimum(
        (angle / np.pi * config.n_angle_groups).astype(np.int32),
        config.n_angle_groups - 1,
    )
    group_ids = np.empty(len(pairs), dtype=np.int32)
    for angle_group in range(config.n_angle_groups):
        index = np.flatnonzero(angle_id == angle_group)
        ordered = index[np.argsort(distance[index], kind="stable")]
        for radial, part in enumerate(
            np.array_split(ordered, config.n_distance_groups)
        ):
            group_ids[part] = angle_group * config.n_distance_groups + radial

    raw_geometry = np.column_stack(
        (np.log(np.maximum(distance, 1e-12)), np.cos(2.0 * angle), np.sin(2.0 * angle))
    )
    normalized = (raw_geometry - raw_geometry.mean(axis=0)) / np.maximum(
        raw_geometry.std(axis=0), 1e-12
    )
    group_geometry = np.empty((config.n_groups, 3), dtype=np.float64)
    group_counts = np.empty(config.n_groups, dtype=np.int64)
    for group in range(config.n_groups):
        mask = group_ids == group
        if not np.any(mask):
            raise RuntimeError(f"empty directional pair group {group}")
        group_geometry[group] = normalized[mask].mean(axis=0)
        group_counts[group] = int(mask.sum())
    return DirectionalPairDesign(pairs, group_ids, group_geometry, group_counts)
