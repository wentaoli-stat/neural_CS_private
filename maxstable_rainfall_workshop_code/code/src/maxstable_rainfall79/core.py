"""Full-dataset Direct-FSM utilities for the Smith Max-stable experiment.

One Direct-FSM row is one complete 47-year by 79-site dataset.  The 47 annual
fields are the iid outer blocks, exactly as the iid blocks in Model 2.  Within
an annual field, all 3,081 spatial pairs are grouped into 20 distance-bin
composite-score tokens.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from scipy.special import ndtr
import torch
from torch import nn
from torch.nn import functional as F

from .support import (
    Model3Config,
    make_directional_pair_design,
    theta10_from_u_np,
    theta10_from_u_jax,
    _require_maxstable_upstream,
)

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # Model training from frozen features only needs PyTorch.
    jax = None
    jnp = None


PARAMETER_NAMES = ("log_sd_x", "log_sd_y", "fisher_z_rho")
SELECTED_PROTOCOL_VERSION = "rainfall79_10k_linear_positive_anchor_v1"


def covariance_matrix_from_parameters(parameters: np.ndarray) -> np.ndarray:
    """Convert ``(Sigma_11, Sigma_12, Sigma_22)`` rows to 2x2 matrices."""

    value = np.asarray(parameters, dtype=np.float64)
    if value.shape[-1:] != (3,):
        raise ValueError("covariance parameters must have trailing dimension 3")
    output = np.empty(value.shape[:-1] + (2, 2), dtype=np.float64)
    output[..., 0, 0] = value[..., 0]
    output[..., 0, 1] = value[..., 1]
    output[..., 1, 0] = value[..., 1]
    output[..., 1, 1] = value[..., 2]
    determinant = value[..., 0] * value[..., 2] - value[..., 1] ** 2
    if np.any(value[..., 0] <= 0.0) or np.any(determinant <= 0.0):
        raise ValueError("covariance parameters must define positive-definite matrices")
    return output


def smith_extremal_coefficient(
    covariance_parameters: np.ndarray, displacements: np.ndarray
) -> np.ndarray:
    """Evaluate Smith's pairwise extremal coefficient on displacement vectors.

    For covariance matrix ``Sigma`` and spatial displacement ``h``, the Smith
    extremal coefficient is

        delta(h) = 2 Phi(sqrt(h' Sigma^{-1} h) / 2).

    The return shape is ``covariance_parameters.shape[:-1] + (n_h,)``.
    """

    covariance = np.asarray(covariance_parameters, dtype=np.float64)
    h = np.asarray(displacements, dtype=np.float64)
    if covariance.shape[-1:] != (3,):
        raise ValueError("covariance parameters must have trailing dimension 3")
    if h.ndim != 2 or h.shape[1] != 2:
        raise ValueError("displacements must have shape (n_h, 2)")
    s11, s12, s22 = covariance[..., 0], covariance[..., 1], covariance[..., 2]
    determinant = s11 * s22 - s12**2
    if np.any(s11 <= 0.0) or np.any(s22 <= 0.0) or np.any(determinant <= 0.0):
        raise ValueError("covariance parameters must define positive-definite matrices")
    dx, dy = h[:, 0], h[:, 1]
    quadratic = (
        s22[..., None] * dx**2
        - 2.0 * s12[..., None] * dx * dy
        + s11[..., None] * dy**2
    ) / determinant[..., None]
    return 2.0 * ndtr(0.5 * np.sqrt(np.maximum(quadratic, 0.0)))


def covariance_anisotropy(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return major-axis orientation in [0, pi) and major/minor SD ratio."""

    matrix = covariance_matrix_from_parameters(parameters)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    orientation = np.mod(
        np.arctan2(eigenvectors[..., 1, 1], eigenvectors[..., 0, 1]), np.pi
    )
    axis_ratio = np.sqrt(eigenvalues[..., 1] / eigenvalues[..., 0])
    return orientation, axis_ratio


def axial_angle_difference(angle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Signed shortest angle difference for unoriented axes, in [-pi/2, pi/2)."""

    difference = np.asarray(angle, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return np.mod(difference + 0.5 * np.pi, np.pi) - 0.5 * np.pi


def extremal_evaluation_grid(
    coordinates: np.ndarray,
    n_radii: int = 12,
    n_angles: int = 8,
    lower_distance_quantile: float = 0.05,
    upper_distance_quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a direction-balanced displacement grid over observed distance scales."""

    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 2:
        raise ValueError("coordinates must have shape (n_sites, 2), n_sites >= 2")
    if int(n_radii) < 1 or int(n_angles) < 1:
        raise ValueError("n_radii and n_angles must be positive")
    if not 0.0 <= lower_distance_quantile < upper_distance_quantile <= 1.0:
        raise ValueError("distance quantiles must satisfy 0 <= lower < upper <= 1")
    first, second = np.triu_indices(len(coords), k=1)
    distances = np.linalg.norm(coords[second] - coords[first], axis=1)
    probabilities = np.linspace(
        float(lower_distance_quantile),
        float(upper_distance_quantile),
        int(n_radii),
    )
    radii = np.quantile(distances, probabilities)
    angles = np.arange(int(n_angles), dtype=np.float64) * np.pi / int(n_angles)
    radius_grid, angle_grid = np.meshgrid(radii, angles, indexing="ij")
    displacement = np.column_stack(
        (
            (radius_grid * np.cos(angle_grid)).reshape(-1),
            (radius_grid * np.sin(angle_grid)).reshape(-1),
        )
    )
    return displacement, radius_grid.reshape(-1), angle_grid.reshape(-1)


@dataclass(frozen=True)
class FormalConfig:
    """Locked 47-year, 79-site, three-parameter rainfall experiment."""

    u_min: float = 0.15
    u_max: float = 0.85
    n_years: int = 47
    n_sites: int = 79
    n_angle_groups: int = 1
    n_distance_groups: int = 20
    pairs_per_group: int = 100_000
    pair_seed: int = 20260731
    log_sd_span: float = 0.35
    fisher_z_span: float = 0.70
    active_indices: tuple[int, ...] = (0, 1, 2)
    fixed_u: tuple[float, float, float] = (0.5, 0.5, 0.5)

    @property
    def theta_dim(self) -> int:
        return 3

    @property
    def n_groups(self) -> int:
        return self.n_angle_groups * self.n_distance_groups

    @property
    def range(self) -> float:
        return self.u_max - self.u_min

    def validate(self) -> None:
        if not 0.0 < self.u_min < self.u_max < 1.0:
            raise ValueError("u bounds must lie inside (0,1)")
        if (self.n_years, self.n_sites) != (47, 79):
            raise ValueError("the selected experiment requires exactly (47,79)")
        if self.n_angle_groups != 1 or self.n_distance_groups != 20:
            raise ValueError("the selected experiment requires 20 distance bins")
        if self.pairs_per_group < 3_081:
            raise ValueError("the selected experiment must retain all 3,081 pairs")
        if tuple(self.active_indices) != (0, 1, 2):
            raise ValueError("the selected experiment infers all three dependence coordinates")
        if tuple(self.fixed_u) != (0.5, 0.5, 0.5):
            raise ValueError("fixed_u is retained only as a compatibility constant")

    def model3(self) -> Model3Config:
        return Model3Config(
            u_min=self.u_min,
            u_max=self.u_max,
            log_sd_span=self.log_sd_span,
            fisher_z_span=self.fisher_z_span,
            n_years=self.n_years,
            n_sites=self.n_sites,
            n_angle_groups=self.n_angle_groups,
            n_distance_groups=self.n_distance_groups,
        )


def formal_coordinates(config: FormalConfig) -> np.ndarray:
    """Return the locked coordinates of the 79 Swiss rainfall sites."""

    dataset_module, _, _ = _require_maxstable_upstream()
    value = np.asarray(dataset_module.create_2d_grid(), dtype=np.float64)
    if value.shape != (config.n_sites, 2):
        raise ValueError(
            f"coordinate design has shape {value.shape}, expected ({config.n_sites},2)"
        )
    return value


@dataclass(frozen=True)
class TrainConfig:
    sigma_q: float | tuple[float, ...]
    hidden: int
    depth: int
    gate_hidden: int
    iters: int
    batch_size: int
    lr: float
    gate_lr: float
    joint_lr: float
    gate_only_steps: int
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = 20
    print_every: int = 100
    ema_decay: float = 0.995


@dataclass
class PairDesign:
    pairs: np.ndarray
    group_ids: np.ndarray
    group_geometry: np.ndarray
    population_counts: np.ndarray


def stratified_pair_subdesign(
    design: PairDesign,
    pairs_per_group: int,
    seed: int,
) -> PairDesign:
    """Return a reproducible distance/angle-stratified pilot pair subset.

    ``population_counts`` deliberately remains the count in the full design.
    The composite-pilot code therefore assigns each retained pair the usual
    population-count / selected-count weight.  Passing zero retains the full
    design and is the confirmatory default.
    """

    requested = int(pairs_per_group)
    if requested < 0:
        raise ValueError("pairs_per_group must be nonnegative")
    if requested == 0:
        return design
    pairs = np.asarray(design.pairs)
    group_ids = np.asarray(design.group_ids)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or group_ids.shape != (len(pairs),):
        raise ValueError("invalid pair design arrays")
    n_groups = len(np.asarray(design.population_counts))
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    for group in range(n_groups):
        candidates = np.flatnonzero(group_ids == group)
        if len(candidates) == 0:
            raise ValueError(f"pilot pair design has an empty group {group}")
        if len(candidates) <= requested:
            chosen = candidates
        else:
            chosen = np.sort(rng.choice(candidates, size=requested, replace=False))
        selected.append(chosen)
    index = np.concatenate(selected)
    return PairDesign(
        pairs=pairs[index].copy(),
        group_ids=group_ids[index].copy(),
        group_geometry=np.asarray(design.group_geometry).copy(),
        population_counts=np.asarray(design.population_counts).copy(),
    )


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sigmoid_np(value: np.ndarray | float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def physical_to_latent(u: np.ndarray, config: FormalConfig) -> np.ndarray:
    unit = (np.asarray(u, dtype=np.float64) - config.u_min) / config.range
    unit = np.clip(unit, 1e-8, 1.0 - 1e-8)
    return np.log(unit) - np.log1p(-unit)


def latent_to_physical(value: np.ndarray, config: FormalConfig) -> np.ndarray:
    return config.u_min + config.range * sigmoid_np(value)


def physical_to_unit(u: np.ndarray, config: FormalConfig) -> np.ndarray:
    return (np.asarray(u, dtype=np.float64) - config.u_min) / config.range


def unit_to_physical(unit: np.ndarray, config: FormalConfig) -> np.ndarray:
    return config.u_min + config.range * np.asarray(unit, dtype=np.float64)


def du_dlatent(value: np.ndarray, config: FormalConfig) -> np.ndarray:
    unit = sigmoid_np(value)
    return config.range * unit * (1.0 - unit)


def latent_to_physical_jax(value: "jnp.ndarray", config: FormalConfig) -> "jnp.ndarray":
    return float(config.u_min) + float(config.range) * jax.nn.sigmoid(value)


def active_to_full_physical(
    value: np.ndarray, config: FormalConfig
) -> np.ndarray:
    """Insert the inferred coordinates into the fixed three-parameter vector."""

    active = np.asarray(value, dtype=np.float64)
    if active.shape[-1] != config.theta_dim:
        raise ValueError(
            f"expected {config.theta_dim} active coordinates, got {active.shape}"
        )
    output = np.broadcast_to(
        np.asarray(config.fixed_u, dtype=np.float64), active.shape[:-1] + (3,)
    ).copy()
    output[..., np.asarray(config.active_indices, dtype=np.int64)] = active
    return output


def active_to_full_physical_jax(
    value: "jnp.ndarray", config: FormalConfig
) -> "jnp.ndarray":
    active = jnp.asarray(value, dtype=jnp.float64)
    output = jnp.broadcast_to(
        jnp.asarray(config.fixed_u, dtype=jnp.float64), active.shape[:-1] + (3,)
    )
    return output.at[..., jnp.asarray(config.active_indices, dtype=jnp.int32)].set(
        active
    )


def normalized_to_theta10_np(value: np.ndarray, config: FormalConfig) -> np.ndarray:
    """Map the three normalized dependence coordinates to the Smith/GEV vector."""

    return theta10_from_u_np(active_to_full_physical(value, config), config.model3())


def normalized_to_theta10_jax(
    value: "jnp.ndarray", config: FormalConfig
) -> "jnp.ndarray":
    return theta10_from_u_jax(
        active_to_full_physical_jax(value, config), config.model3()
    )


def latin_hypercube(
    rng: np.random.Generator,
    n: int,
    dimension: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    output = np.empty((int(n), int(dimension)), dtype=np.float64)
    for coordinate in range(int(dimension)):
        strata = (np.arange(int(n)) + rng.uniform(size=int(n))) / int(n)
        rng.shuffle(strata)
        output[:, coordinate] = lower + (upper - lower) * strata
    return output


def make_pair_design(config: FormalConfig, coords: np.ndarray | None = None) -> PairDesign:
    """Retain all pairs and assign them to the 20 locked distance bins."""

    coords = formal_coordinates(config) if coords is None else np.asarray(coords)
    full = make_directional_pair_design(config.model3(), coords=coords)
    rng = np.random.default_rng(int(config.pair_seed))
    selected_pairs: list[np.ndarray] = []
    selected_groups: list[np.ndarray] = []
    for group in range(config.n_groups):
        candidates = np.flatnonzero(full.group_ids == group)
        take = min(int(config.pairs_per_group), candidates.size)
        index = np.sort(rng.choice(candidates, size=take, replace=False))
        selected_pairs.append(full.pairs[index])
        selected_groups.append(np.full(take, group, dtype=np.int32))
    return PairDesign(
        pairs=np.concatenate(selected_pairs).astype(np.int32),
        group_ids=np.concatenate(selected_groups).astype(np.int32),
        group_geometry=np.asarray(full.group_geometry, dtype=np.float64),
        population_counts=np.asarray(full.group_counts, dtype=np.int64),
    )


def _require_jax() -> None:
    if jax is None or jnp is None:
        raise RuntimeError("JAX is required for Max-stable feature construction")


def make_annual_feature_function(config: FormalConfig, design: PairDesign):
    """Return batched (dataset,year,group,coordinate) composite subscores."""

    _require_jax()
    dataset_module, pair_one, _ = _require_maxstable_upstream()
    coords = jnp.asarray(formal_coordinates(config), dtype=jnp.float64)
    pairs = jnp.asarray(design.pairs, dtype=jnp.int32)
    group_ids = jnp.asarray(design.group_ids, dtype=jnp.int32)
    selected_counts = np.bincount(
        design.group_ids, minlength=config.n_groups
    ).astype(np.float64)
    counts = jnp.asarray(selected_counts, dtype=jnp.float64).reshape(-1, 1)
    def pair_score(anchor, annual, pair):
        i, j = pair[0], pair[1]

        def objective(latent):
            physical = latent_to_physical_jax(latent, config)
            return pair_one(
                normalized_to_theta10_jax(physical, config),
                coords[i],
                coords[j],
                annual[i].reshape(1),
                annual[j].reshape(1),
                jacobian_mode="corrected",
            )

        return jax.grad(objective)(anchor)

    def one_year(anchor, annual):
        scores = jax.vmap(lambda pair: pair_score(anchor, annual, pair))(pairs)
        total = jax.ops.segment_sum(
            scores, group_ids, num_segments=config.n_groups
        )
        return total / counts

    def one_dataset(anchor, dataset):
        return jax.vmap(lambda annual: one_year(anchor, annual))(dataset)

    return jax.jit(one_dataset), jax.jit(
        jax.vmap(one_dataset, in_axes=(0, 0))
    )


def make_annual_pair_feature_function(config: FormalConfig, design: PairDesign):
    """Return individual selected-pair subscores before group aggregation."""

    _require_jax()
    dataset_module, pair_one, _ = _require_maxstable_upstream()
    coords = jnp.asarray(formal_coordinates(config), dtype=jnp.float64)
    pairs = jnp.asarray(design.pairs, dtype=jnp.int32)
    def pair_score(anchor, annual, pair):
        i, j = pair[0], pair[1]

        def objective(latent):
            physical = latent_to_physical_jax(latent, config)
            return pair_one(
                normalized_to_theta10_jax(physical, config),
                coords[i],
                coords[j],
                annual[i].reshape(1),
                annual[j].reshape(1),
                jacobian_mode="corrected",
            )

        return jax.grad(objective)(anchor)

    def one_year(anchor, annual):
        return jax.vmap(lambda pair: pair_score(anchor, annual, pair))(pairs)

    def one_dataset(anchor, dataset):
        return jax.vmap(lambda annual: one_year(anchor, annual))(dataset)

    return jax.jit(one_dataset), jax.jit(
        jax.vmap(one_dataset, in_axes=(0, 0))
    )


def compute_annual_features(
    anchor: np.ndarray,
    datasets: np.ndarray,
    config: FormalConfig,
    design: PairDesign,
    *,
    chunk_size: int = 8,
    batch_function: Any | None = None,
    label: str = "features",
) -> np.ndarray:
    if batch_function is None:
        _, batch_function = make_annual_feature_function(config, design)
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1, config.theta_dim)
    datasets = np.asarray(datasets, dtype=np.float64).reshape(
        -1, config.n_years, config.n_sites
    )
    output: list[np.ndarray] = []
    for start in range(0, len(anchor), int(chunk_size)):
        stop = min(start + int(chunk_size), len(anchor))
        value = batch_function(
            jnp.asarray(anchor[start:stop]), jnp.asarray(datasets[start:stop])
        )
        output.append(np.asarray(jax.device_get(value), dtype=np.float32))
        print(f"{label}: {stop}/{len(anchor)}", flush=True)
    return np.concatenate(output, axis=0)


def simulate_complete_dataset(
    physical: np.ndarray, config: FormalConfig, seed: int
) -> np.ndarray:
    config.validate()
    dataset_module, _, _ = _require_maxstable_upstream()
    theta = normalized_to_theta10_np(np.asarray(physical), config)
    value = np.asarray(dataset_module.your_simulator(theta, int(seed)), dtype=np.float64)
    if value.shape != (config.n_years, config.n_sites):
        raise RuntimeError(f"bad simulator shape {value.shape}")
    if not np.all(np.isfinite(value)):
        raise RuntimeError(f"non-finite simulator output at seed {seed}")
    return value


def sample_stage1_rows(
    rng: np.random.Generator,
    n: int,
    config: FormalConfig,
    sigma_q: float | tuple[float, ...] | np.ndarray,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_u = latin_hypercube(
        rng, n, config.theta_dim, config.u_min, config.u_max
    )
    anchor_w = physical_to_latent(anchor_u, config)
    sigma = np.asarray(sigma_q, dtype=np.float64)
    if sigma.ndim == 0:
        sigma = np.repeat(sigma.reshape(1), config.theta_dim)
    sigma = sigma.reshape(config.theta_dim)
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("all proposal standard deviations must be positive")
    sampled_w = anchor_w + sigma.reshape(1, -1) * rng.normal(
        size=(int(n), config.theta_dim)
    )
    sampled_u = latent_to_physical(sampled_w, config)
    target = (sampled_w - anchor_w) / sigma.reshape(1, -1) ** 2
    datasets = []
    for index, physical in enumerate(sampled_u):
        datasets.append(
            simulate_complete_dataset(physical, config, int(seed_offset) + index)
        )
        print(f"simulated complete datasets: {index + 1}/{n}", flush=True)
    return (
        anchor_w,
        target.astype(np.float32),
        sampled_u,
        np.stack(datasets).astype(np.float32),
    )


def softplus_inverse(value: float) -> float:
    return math.log(math.expm1(float(value)))


def make_mlp(in_dim: int, out_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(in_dim)
    for _ in range(int(depth)):
        layers.extend((nn.Linear(current, int(hidden)), nn.SiLU()))
        current = int(hidden)
    layers.append(nn.Linear(current, int(out_dim)))
    return nn.Sequential(*layers)


class DistanceBinMLPFullDatasetScore(nn.Module):
    """Toy-style non-recurrent readout over fixed distance-bin summaries.

    A year is represented by the ordered vector of distance-bin composite
    scores.  A shared MLP maps that vector (and the local parameter anchor) to
    one annual score, and the 47 iid annual scores are summed.  There is no
    recurrent layer in this architecture.
    """

    architecture = "distance_bin_mlp"

    def __init__(
        self,
        theta_dim: int,
        hidden: int,
        depth: int,
        geometry: np.ndarray,
        population_counts: np.ndarray,
    ) -> None:
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.n_groups = int(len(geometry))
        self.year_readout = make_mlp(
            self.n_groups * self.theta_dim + self.theta_dim,
            self.theta_dim,
            int(hidden),
            int(depth),
        )
        self.register_buffer(
            "geometry",
            torch.as_tensor(geometry, dtype=torch.float32).reshape(
                1, 1, self.n_groups, 3
            ),
        )
        counts = np.asarray(population_counts, dtype=np.float32)
        self.register_buffer(
            "population_counts",
            torch.as_tensor(counts).reshape(1, 1, self.n_groups, 1),
        )

    def forward_from_tokens(
        self, score: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        batch, years, groups, coordinates = score.shape
        if groups != self.n_groups or coordinates != self.theta_dim:
            raise ValueError("unexpected distance-bin score shape")
        annual = score.reshape(batch, years, groups * coordinates)
        anchor = anchor_z[:, None, :].expand(batch, years, self.theta_dim)
        per_year = self.year_readout(
            torch.cat((annual, anchor), dim=-1)
        )
        return per_year.sum(dim=1)

    def forward(self, score: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        return self.forward_from_tokens(score, anchor_z)


class DistanceBinPairMLPFullDatasetScore(nn.Module):
    """Pair-local adapter for :class:`DistanceBinMLPFullDatasetScore`."""

    architecture = "distance_bin_pair_mlp"

    def __init__(
        self,
        theta_dim: int,
        hidden: int,
        depth: int,
        geometry: np.ndarray,
        population_counts: np.ndarray,
        group_ids: np.ndarray,
        block_mean: np.ndarray | None = None,
        block_sd: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.core = DistanceBinMLPFullDatasetScore(
            theta_dim,
            hidden,
            depth,
            geometry,
            population_counts,
        )
        ids = np.asarray(group_ids, dtype=np.int64)
        counts = np.bincount(ids, minlength=len(geometry)).astype(np.float32)
        if np.any(counts == 0):
            raise ValueError("every distance group needs at least one pair")
        self.register_buffer(
            "pair_group_ids", torch.as_tensor(ids, dtype=torch.long)
        )
        self.register_buffer(
            "selected_group_counts",
            torch.as_tensor(counts, dtype=torch.float32).reshape(1, 1, -1, 1),
        )
        if block_mean is None:
            block_mean = np.zeros(
                (1, 1, len(geometry), self.theta_dim), dtype=np.float32
            )
        if block_sd is None:
            block_sd = np.ones(
                (1, 1, len(geometry), self.theta_dim), dtype=np.float32
            )
        self.register_buffer(
            "block_mean", torch.as_tensor(block_mean, dtype=torch.float32)
        )
        self.register_buffer(
            "block_sd", torch.as_tensor(block_sd, dtype=torch.float32)
        )

    @property
    def geometry(self) -> torch.Tensor:
        return self.core.geometry

    def aggregate_pairs(self, score: torch.Tensor) -> torch.Tensor:
        batch, years, pairs, coordinates = score.shape
        if pairs != len(self.pair_group_ids):
            raise ValueError("unexpected number of spatial pairs")
        groups = int(self.selected_group_counts.shape[2])
        output = score.new_zeros(batch, years, groups, coordinates)
        index = self.pair_group_ids.reshape(1, 1, pairs, 1).expand(
            batch, years, pairs, coordinates
        )
        output.scatter_add_(2, index, score)
        return output / self.selected_group_counts

    def forward_from_pairs(
        self, score: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        grouped = self.aggregate_pairs(score)
        grouped = (grouped - self.block_mean) / self.block_sd
        return self.core.forward_from_tokens(
            grouped, anchor_z
        )

    def forward(self, score: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        return self.forward_from_pairs(score, anchor_z)


class _RawPairLocalGateBase(nn.Module):
    """Raw-score adapter for the selected nested Positive(s, anchor) arm.

    It reconstructs each raw pair score, applies the gate in raw score units,
    and adds only the resulting correction to the Linear group summary. A zero
    correction is therefore exactly the Linear model.
    """

    def __init__(
        self,
        linear: DistanceBinPairMLPFullDatasetScore,
        gate_hidden: int,
        pair_feature_mean: np.ndarray,
        pair_feature_sd: np.ndarray,
        group_feature_sd: np.ndarray,
        gate_score_mean: np.ndarray,
        gate_score_sd: np.ndarray,
    ) -> None:
        super().__init__()
        self.core = copy.deepcopy(linear)
        self.theta_dim = int(self.core.theta_dim)
        ids = self.core.pair_group_ids.detach().cpu().numpy()
        pair_mean = np.asarray(pair_feature_mean, dtype=np.float32)
        pair_sd = np.asarray(pair_feature_sd, dtype=np.float32)
        group_sd = np.asarray(group_feature_sd, dtype=np.float32)
        if pair_mean.shape[-2:] != (len(self.core.geometry[0, 0]), self.theta_dim):
            raise ValueError("bad groupwise pair-score mean shape")
        if pair_sd.shape != pair_mean.shape or group_sd.shape != pair_mean.shape:
            raise ValueError("pair/group score normalizers must have matching shapes")
        if np.any(pair_sd <= 0.0) or np.any(group_sd <= 0.0):
            raise ValueError("pair/group score standard deviations must be positive")
        self.register_buffer(
            "pair_feature_mean",
            torch.as_tensor(pair_mean[:, :, ids, :], dtype=torch.float32),
        )
        self.register_buffer(
            "pair_feature_sd",
            torch.as_tensor(pair_sd[:, :, ids, :], dtype=torch.float32),
        )
        self.register_buffer(
            "group_feature_sd", torch.as_tensor(group_sd, dtype=torch.float32)
        )
        gate_mean = np.asarray(gate_score_mean, dtype=np.float32).reshape(
            1, 1, 1, self.theta_dim
        )
        gate_sd = np.asarray(gate_score_sd, dtype=np.float32).reshape(
            1, 1, 1, self.theta_dim
        )
        if np.any(gate_sd <= 0.0):
            raise ValueError("global gate-score standard deviations must be positive")
        self.register_buffer("gate_score_mean", torch.as_tensor(gate_mean))
        self.register_buffer("gate_score_sd", torch.as_tensor(gate_sd))
        self.gate = make_mlp(2 * self.theta_dim, self.theta_dim, gate_hidden, 2)

    def raw_score(self, standardized_score: torch.Tensor) -> torch.Tensor:
        return standardized_score * self.pair_feature_sd + self.pair_feature_mean

    def gate_inputs(
        self, raw_score: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        batch, years, pairs, _ = raw_score.shape
        normalized_raw = (raw_score - self.gate_score_mean) / self.gate_score_sd
        anchor = anchor_z[:, None, None, :].expand(batch, years, pairs, -1)
        return torch.cat((normalized_raw, anchor), dim=-1)

    def raw_correction(
        self, raw_score: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, score: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        raw = self.raw_score(score)
        correction = self.raw_correction(raw, anchor_z)
        identity_grouped = self.core.aggregate_pairs(score)
        identity_grouped = (
            identity_grouped - self.core.block_mean
        ) / self.core.block_sd
        correction_grouped = self.core.aggregate_pairs(correction)
        correction_grouped = correction_grouped / self.group_feature_sd
        return self.core.core.forward_from_tokens(
            identity_grouped + correction_grouped, anchor_z
        )


class PositiveScoreAnchorPairGate(_RawPairLocalGateBase):
    """Positive multiplier whose only inputs are raw score and anchor."""

    architecture = "raw_positive_distance_bin_pair_mlp"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        final = self.gate[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("gate final layer must be linear")
        identity_logit = softplus_inverse(1.0)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, identity_logit)
        self.register_buffer(
            "identity_logit", torch.tensor(identity_logit, dtype=torch.float32)
        )

    def gate_values(
        self, raw_score: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        value = F.softplus(self.gate(self.gate_inputs(raw_score, anchor_z)))
        # Division by the exact same initialized scalar makes step-zero nesting
        # bitwise, while retaining a strictly positive multiplier thereafter.
        return value / F.softplus(self.identity_logit)

    def raw_correction(
        self, raw_score: torch.Tensor, anchor_z: torch.Tensor
    ) -> torch.Tensor:
        return raw_score * (self.gate_values(raw_score, anchor_z) - 1.0)


def feature_statistics(
    features: np.ndarray, anchor_w: np.ndarray
) -> dict[str, np.ndarray]:
    feature = np.asarray(features, dtype=np.float64)
    mean = feature.mean(axis=(0, 1), keepdims=True)
    sd = feature.std(axis=(0, 1), keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    anchor_mean = np.asarray(anchor_w, dtype=np.float64).mean(axis=0)
    anchor_sd = np.asarray(anchor_w, dtype=np.float64).std(axis=0)
    anchor_sd = np.where(anchor_sd < 1e-8, 1.0, anchor_sd)
    return {
        "feature_mean": mean,
        "feature_sd": sd,
        "anchor_mean": anchor_mean,
        "anchor_sd": anchor_sd,
    }


def transform_inputs(
    features: np.ndarray, anchor_w: np.ndarray, stats: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    score = (
        (np.asarray(features) - stats["feature_mean"]) / stats["feature_sd"]
    ).astype(np.float32)
    anchor = (
        (np.asarray(anchor_w) - stats["anchor_mean"]) / stats["anchor_sd"]
    ).astype(np.float32)
    return score, anchor


def transform_pair_inputs(
    features: np.ndarray,
    anchor_w: np.ndarray,
    stats: dict[str, np.ndarray],
    group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize pair scores with their parent group's frozen statistics."""

    ids = np.asarray(group_ids, dtype=np.int64)
    mean_key = (
        "pair_feature_mean" if "pair_feature_mean" in stats else "feature_mean"
    )
    sd_key = "pair_feature_sd" if "pair_feature_sd" in stats else "feature_sd"
    mean = np.asarray(stats[mean_key])[:, :, ids, :]
    sd = np.asarray(stats[sd_key])[:, :, ids, :]
    score = ((np.asarray(features) - mean) / sd).astype(np.float32)
    anchor = (
        (np.asarray(anchor_w) - stats["anchor_mean"]) / stats["anchor_sd"]
    ).astype(np.float32)
    return score, anchor


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_stage1_checkpoint(path: Path, device: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("protocol_version") == SELECTED_PROTOCOL_VERSION:
        return load_selected_stage1_checkpoint_from_payload(checkpoint, device)
    raise ValueError(
        "checkpoint does not implement the selected Linear/Nonlinear-gate protocol: "
        f"expected {SELECTED_PROTOCOL_VERSION!r}, "
        f"got {checkpoint.get('protocol_version')!r}"
    )


def build_selected_models(
    config: FormalConfig,
    training: TrainConfig,
    design: PairDesign,
    stats: dict[str, np.ndarray],
    device: str,
) -> dict[str, nn.Module]:
    """Construct the selected nested Linear and Positive(s, anchor) arms."""

    required = (
        "pair_feature_mean",
        "pair_feature_sd",
        "feature_sd",
        "pair_block_mean",
        "pair_block_sd",
        "gate_score_mean",
        "gate_score_sd",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        raise ValueError(f"selected checkpoint is missing statistics: {missing}")
    linear = DistanceBinPairMLPFullDatasetScore(
        config.theta_dim,
        training.hidden,
        training.depth,
        design.group_geometry,
        design.population_counts,
        design.group_ids,
        stats["pair_block_mean"],
        stats["pair_block_sd"],
    ).to(device)
    positive_anchor = PositiveScoreAnchorPairGate(
        linear,
        training.gate_hidden,
        stats["pair_feature_mean"],
        stats["pair_feature_sd"],
        stats["feature_sd"],
        stats["gate_score_mean"],
        stats["gate_score_sd"],
    ).to(device)
    return {"linear": linear, "positive_anchor": positive_anchor}


def save_selected_stage1_checkpoint(
    path: Path,
    states: dict[str, dict[str, torch.Tensor]],
    stats: dict[str, np.ndarray],
    config: FormalConfig,
    training: TrainConfig,
    design: PairDesign,
    diagnostics: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    """Save the selected two-arm fair Stage-1 checkpoint."""

    if set(states) != {"linear", "positive_anchor"}:
        raise ValueError("selected checkpoint must contain Linear and Positive(s, anchor)")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": SELECTED_PROTOCOL_VERSION,
            "models": states,
            "stats": stats,
            "config": asdict(config),
            "training": asdict(training),
            "design": asdict(design),
            "diagnostics": diagnostics,
            "selection": selection,
        },
        path,
    )


def load_selected_stage1_checkpoint_from_payload(
    checkpoint: dict[str, Any], device: str
) -> dict[str, Any]:
    config = FormalConfig(**checkpoint["config"])
    training = TrainConfig(**checkpoint["training"])
    design = PairDesign(
        **{
            key: np.asarray(value)
            for key, value in checkpoint["design"].items()
        }
    )
    stats = {
        key: np.asarray(value) for key, value in checkpoint["stats"].items()
    }
    models = build_selected_models(config, training, design, stats, device)
    models = {
        method: model
        for method, model in models.items()
        if method in checkpoint["models"]
    }
    for method, model in models.items():
        model.load_state_dict(checkpoint["models"][method])
        model.eval()
    return {
        "models": models,
        **models,
        "nonlinear": models["positive_anchor"],
        "stats": stats,
        "config": config,
        "training": training,
        "design": design,
        "diagnostics": checkpoint["diagnostics"],
        "selection": checkpoint.get("selection", {}),
        "protocol_version": checkpoint["protocol_version"],
    }


def make_full_composite_score_function(
    config: FormalConfig, design: PairDesign
):
    """Composite score in latent coordinates for the shared data-only pilot."""

    _require_jax()
    dataset_module, pair_one, _ = _require_maxstable_upstream()
    coords = jnp.asarray(formal_coordinates(config), dtype=jnp.float64)
    pairs = jnp.asarray(design.pairs, dtype=jnp.int32)
    selected_counts = np.bincount(
        design.group_ids, minlength=config.n_groups
    ).astype(np.float64)
    pair_weight = (
        design.population_counts[design.group_ids]
        / selected_counts[design.group_ids]
    )
    pair_weight = jnp.asarray(pair_weight, dtype=jnp.float64)
    def objective(latent, dataset):
        physical = latent_to_physical_jax(latent, config)
        theta = normalized_to_theta10_jax(physical, config)

        def one(pair):
            i, j = pair[0], pair[1]
            return pair_one(
                theta,
                coords[i],
                coords[j],
                dataset[:, i],
                dataset[:, j],
                jacobian_mode="corrected",
            )

        values = jax.vmap(one)(pairs)
        return jnp.sum(values * pair_weight)

    one_score = jax.grad(objective)
    return jax.jit(one_score), jax.jit(
        jax.vmap(one_score, in_axes=(0, 0))
    )


def composite_sandwich_covariance_at_roots(
    datasets: np.ndarray,
    roots: np.ndarray,
    config: FormalConfig,
    design: PairDesign,
    *,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """Estimate the MPLE Godambe covariance from iid annual score blocks.

    The pairwise composite likelihood is summed over the 47 independent annual
    fields.  At each supplied latent-coordinate MPLE, this routine forms the
    annual composite scores ``s_t``, the observed sensitivity
    ``H=-mean_t Hessian(ell_t)``, and the variability
    ``J=mean_t[(s_t-sbar)(s_t-sbar)']``.  The returned latent covariance is
    ``H^-1 J H^-T / T``; physical-coordinate covariance uses the delta method.

    Conventional sandwich/Wald inference is regular only for interior roots.
    Boundary status is deliberately left to the caller because the optimizer,
    rather than the covariance calculation, defines the active constraints.
    """

    _require_jax()
    dataset_module, pair_one, _ = _require_maxstable_upstream()
    coords = jnp.asarray(formal_coordinates(config), dtype=jnp.float64)
    pairs = jnp.asarray(design.pairs, dtype=jnp.int32)
    selected_counts = np.bincount(
        design.group_ids, minlength=config.n_groups
    ).astype(np.float64)
    pair_weight = jnp.asarray(
        design.population_counts[design.group_ids]
        / selected_counts[design.group_ids],
        dtype=jnp.float64,
    )
    def annual_objective(latent, field):
        physical = latent_to_physical_jax(latent, config)
        theta = normalized_to_theta10_jax(physical, config)

        def one(pair):
            i, j = pair[0], pair[1]
            return pair_one(
                theta,
                coords[i],
                coords[j],
                field[None, i],
                field[None, j],
                jacobian_mode="corrected",
            )

        return jnp.sum(jax.vmap(one)(pairs) * pair_weight)

    def complete_objective(latent, dataset):
        return jnp.sum(
            jax.vmap(lambda field: annual_objective(latent, field))(dataset)
        )

    annual_score = jax.grad(annual_objective)
    complete_curvature = jax.hessian(complete_objective)
    n_years = float(config.n_years)

    def evaluate_one(latent, dataset):
        scores = jax.vmap(lambda field: annual_score(latent, field))(dataset)
        centered = scores - jnp.mean(scores, axis=0, keepdims=True)
        variability = centered.T @ centered / n_years
        sensitivity = -complete_curvature(latent, dataset) / n_years
        sensitivity = 0.5 * (sensitivity + sensitivity.T)
        eigenvalues, eigenvectors = jnp.linalg.eigh(sensitivity)
        tolerance = jnp.maximum(jnp.max(jnp.abs(eigenvalues)) * 1e-10, 1e-12)
        inverse_eigenvalues = jnp.where(
            jnp.abs(eigenvalues) > tolerance, 1.0 / eigenvalues, 0.0
        )
        inverse_sensitivity = (
            eigenvectors * inverse_eigenvalues[None, :]
        ) @ eigenvectors.T
        covariance_w = (
            inverse_sensitivity @ variability @ inverse_sensitivity.T / n_years
        )
        covariance_w = 0.5 * (covariance_w + covariance_w.T)
        unit = jax.nn.sigmoid(latent)
        derivative = float(config.range) * unit * (1.0 - unit)
        covariance_u = (
            derivative[:, None] * covariance_w * derivative[None, :]
        )
        covariance_u = 0.5 * (covariance_u + covariance_u.T)
        condition = jnp.max(jnp.abs(eigenvalues)) / jnp.maximum(
            jnp.min(jnp.abs(eigenvalues)), tolerance
        )
        return (
            covariance_w,
            covariance_u,
            jnp.sqrt(jnp.maximum(jnp.diag(covariance_u), 0.0)),
            eigenvalues,
            condition,
            jnp.sum(jnp.abs(eigenvalues) > tolerance),
            jnp.linalg.norm(jnp.mean(scores, axis=0)),
        )

    batch_evaluate = jax.jit(jax.vmap(evaluate_one))
    datasets = np.asarray(datasets, dtype=np.float64)
    roots = np.asarray(roots, dtype=np.float64)
    if datasets.ndim != 3 or datasets.shape[1:] != (
        config.n_years,
        config.n_sites,
    ):
        raise ValueError(
            "datasets must have shape "
            f"(n,{config.n_years},{config.n_sites}), got {datasets.shape}"
        )
    if roots.shape != (len(datasets), config.theta_dim):
        raise ValueError(
            f"roots must have shape {(len(datasets), config.theta_dim)}, "
            f"got {roots.shape}"
        )
    names = (
        "covariance_w",
        "covariance_u",
        "se_u",
        "sensitivity_eigenvalues",
        "sensitivity_condition",
        "sensitivity_rank",
        "annual_score_mean_norm",
    )
    output: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for start in range(0, len(datasets), int(batch_size)):
        stop = min(start + int(batch_size), len(datasets))
        values = batch_evaluate(
            jnp.asarray(roots[start:stop]), jnp.asarray(datasets[start:stop])
        )
        for name, value in zip(names, values):
            output[name].append(np.asarray(jax.device_get(value)))
        print(f"MPLE sandwich covariance: {stop}/{len(datasets)}", flush=True)
    return {name: np.concatenate(values) for name, values in output.items()}


def composite_newton_pilot(
    datasets: np.ndarray,
    config: FormalConfig,
    design: PairDesign,
    *,
    iterations: int = 8,
    batch_size: int = 32,
    max_step: float = 1.0,
    ridge: float = 1e-3,
    n_starts: int = 15,
    n_refine_starts: int = 3,
    start_radius: float = 2.0,
    backtracking_steps: int = 6,
    latent_limit: float = 6.0,
    convergence_tolerance: float = 1e-3,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, dict[str, np.ndarray]]:
    """Safeguarded deterministic multistart composite-likelihood pilot.

    Every Newton proposal is compared with a fixed backtracking ladder and the
    current point, and may only move to the finite candidate with the largest
    composite objective.  The final result is the best objective across a
    deterministic set of starts.  Both ordinary score residual and a
    constrained projected-score residual are returned for audit: a genuine
    boundary optimum need not have zero ordinary score.
    """

    _require_jax()
    dataset_module, pair_one, _ = _require_maxstable_upstream()
    coords = jnp.asarray(formal_coordinates(config), dtype=jnp.float64)
    pairs = jnp.asarray(design.pairs, dtype=jnp.int32)
    selected_counts = np.bincount(
        design.group_ids, minlength=config.n_groups
    ).astype(np.float64)
    pair_weight = jnp.asarray(
        design.population_counts[design.group_ids]
        / selected_counts[design.group_ids],
        dtype=jnp.float64,
    )
    def objective(latent, dataset):
        physical = latent_to_physical_jax(latent, config)
        theta = normalized_to_theta10_jax(physical, config)

        def one(pair):
            i, j = pair[0], pair[1]
            return pair_one(
                theta,
                coords[i],
                coords[j],
                dataset[:, i],
                dataset[:, j],
                jacobian_mode="corrected",
            )

        return jnp.sum(jax.vmap(one)(pairs) * pair_weight)

    score = jax.grad(objective)
    curvature = jax.jacfwd(score)

    dim = int(config.theta_dim)
    starts = [np.zeros(dim, dtype=np.float64)]
    for coordinate in range(dim):
        for sign in (-1.0, 1.0):
            value = np.zeros(dim, dtype=np.float64)
            value[coordinate] = sign * float(start_radius)
            starts.append(value)
    corner_radius = float(start_radius) / math.sqrt(float(dim))
    for code in range(2**dim):
        starts.append(
            np.asarray(
                [
                    corner_radius if (code >> coordinate) & 1 else -corner_radius
                    for coordinate in range(dim)
                ],
                dtype=np.float64,
            )
        )
    ordered_starts: list[np.ndarray] = []
    for candidate in starts:
        if not any(np.array_equal(candidate, prior) for prior in ordered_starts):
            ordered_starts.append(candidate)
    start_table = np.stack(ordered_starts)
    if int(n_starts) < 1:
        raise ValueError("n_starts must be positive")
    start_table = start_table[: min(int(n_starts), len(start_table))]
    refine_count = min(max(1, int(n_refine_starts)), len(start_table))
    starts_jax = jnp.asarray(start_table, dtype=jnp.float64)
    scales = jnp.asarray(
        [0.5**index for index in range(max(1, int(backtracking_steps)))] + [0.0],
        dtype=jnp.float64,
    )
    identity = jnp.eye(dim, dtype=jnp.float64)
    lower = -float(latent_limit)
    upper = float(latent_limit)

    def solve_start(initial, dataset):
        def update(_, latent):
            value = score(latent, dataset)
            jac = curvature(latent, dataset)
            regularized = jac - float(ridge) * identity
            step = jnp.linalg.solve(regularized, value)
            step = jnp.where(jnp.all(jnp.isfinite(step)), step, jnp.zeros_like(step))
            step_norm = jnp.linalg.norm(step)
            step = step * jnp.minimum(
                1.0, float(max_step) / jnp.maximum(step_norm, 1e-12)
            )
            candidates = jnp.clip(
                latent[None, :] - scales[:, None] * step[None, :], lower, upper
            )
            candidate_objective = jax.vmap(lambda point: objective(point, dataset))(
                candidates
            )
            candidate_objective = jnp.where(
                jnp.isfinite(candidate_objective), candidate_objective, -jnp.inf
            )
            return candidates[jnp.argmax(candidate_objective)]

        latent = jax.lax.fori_loop(0, int(iterations), update, initial)
        value = score(latent, dataset)
        raw_residual = jnp.linalg.norm(value)
        projected = latent - jnp.clip(latent + value, lower, upper)
        projected_residual = jnp.linalg.norm(projected)
        objective_value = objective(latent, dataset)
        boundary = jnp.any(jnp.abs(latent) >= upper - 1e-6)
        return latent, objective_value, raw_residual, projected_residual, boundary

    def solve_one(dataset):
        initial_objective = jax.vmap(lambda point: objective(point, dataset))(
            starts_jax
        )
        safe_initial_objective = jnp.where(
            jnp.isfinite(initial_objective), initial_objective, -jnp.inf
        )
        refine_indices = jnp.argsort(safe_initial_objective)[-refine_count:]
        refine_starts = starts_jax[refine_indices]
        latent, objective_value, raw_residual, projected_residual, boundary = jax.vmap(
            lambda initial: solve_start(initial, dataset)
        )(refine_starts)
        safe_objective = jnp.where(
            jnp.isfinite(objective_value), objective_value, -jnp.inf
        )
        selected = jnp.argmax(safe_objective)
        fallback = ~jnp.any(jnp.isfinite(initial_objective)) | ~jnp.any(
            jnp.isfinite(objective_value)
        )
        selected = jnp.where(fallback, 0, selected)
        return (
            latent[selected],
            raw_residual[selected],
            projected_residual[selected],
            objective_value[selected],
            boundary[selected],
            refine_indices[selected],
            fallback,
        )

    batch_solve = jax.jit(jax.vmap(solve_one))
    datasets = np.asarray(datasets, dtype=np.float64)
    roots: list[np.ndarray] = []
    diagnostics: dict[str, list[np.ndarray]] = {
        "raw_residual": [],
        "projected_residual": [],
        "objective": [],
        "boundary": [],
        "selected_start": [],
        "fallback": [],
    }
    for start in range(0, len(datasets), int(batch_size)):
        stop = min(start + int(batch_size), len(datasets))
        result = batch_solve(jnp.asarray(datasets[start:stop]))
        converted = [np.asarray(jax.device_get(value)) for value in result]
        roots.append(converted[0])
        for key, value in zip(diagnostics, converted[1:]):
            diagnostics[key].append(value)
        print(f"shared composite pilot: {stop}/{len(datasets)}", flush=True)
    root_array = np.concatenate(roots)
    diagnostic_arrays = {
        key: np.concatenate(value) for key, value in diagnostics.items()
    }
    diagnostic_arrays["converged"] = (
        diagnostic_arrays["projected_residual"] <= float(convergence_tolerance)
    )
    diagnostic_arrays["start_table"] = start_table
    if return_diagnostics:
        return root_array, diagnostic_arrays
    return root_array, diagnostic_arrays["projected_residual"]
