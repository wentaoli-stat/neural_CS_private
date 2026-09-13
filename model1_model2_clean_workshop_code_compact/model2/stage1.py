#!/usr/bin/env python3
"""Amortized conditional FSM for the blockwise common-factor mixture.

This is the cleaned current implementation of the amortized score field

    S(Y, beta_anchor) ~= d/dbeta log p_beta(Y)

for the Model 2 common-factor variance mixture. We work in the unconstrained
coordinate u=logit(pi), sample anchors u_a over a pi range, sample local
perturbations u | u_a ~ N(u_a, sigma_q^2), simulate Y ~ p_u, and regress

    S(Y, u_a) -> (u - u_a) / sigma_q^2.

The exact score is used only for diagnostics after training. By default, the
local composite-score features are evaluated at each sample's own anchor
pi_a, so the network really sees a beta-conditional approximation problem.
Only the deployable anchor-conditioned Linear and shared-raw-gate paths are exposed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F

from model1.stage1 import load_matched_linear_rho


def sigmoid_np(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit_np(p: np.ndarray | float) -> np.ndarray:
    p_arr = np.asarray(p, dtype=np.float64)
    return np.log(p_arr) - np.log1p(-p_arr)


def softplus_inverse(x: float) -> float:
    return math.log(math.expm1(x))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def score_u_from_log_ratio(log_ratio: np.ndarray, pi: float | np.ndarray) -> np.ndarray:
    """Mixture score wrt u=logit(pi) from a likelihood ratio log f1/f0.

    For a local two-component mixture, posterior active probability is
    sigmoid(logit(pi) + log_ratio), hence the u-score is posterior_prob - pi.
    """
    log_ratio = np.asarray(log_ratio, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)
    if pi_arr.ndim == 0:
        logit_pi = logit_np(float(pi_arr))
        w = sigmoid_np(logit_pi + log_ratio)
        return w - float(pi_arr)
    shape = (pi_arr.shape[0],) + (1,) * (log_ratio.ndim - 1)
    pi_b = pi_arr.reshape(shape)
    w = sigmoid_np(logit_np(pi_arr).reshape(shape) + log_ratio)
    return w - pi_b


def simulate_common_factor(
    rng: np.random.Generator,
    u: np.ndarray,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> np.ndarray:
    """Simulate Y_{ki} = eps_{ki} + B_k * tau * Z_k."""
    u = np.asarray(u, dtype=np.float64)
    pi = sigmoid_np(u)
    active = rng.binomial(1, pi[:, None], size=(u.shape[0], n_blocks)).astype(np.float64)
    eps = rng.normal(size=(u.shape[0], n_blocks, block_size))
    z = rng.normal(size=(u.shape[0], n_blocks, 1))
    return eps + active[:, :, None] * tau * z


def full_block_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    """Block likelihood ratio log f1(Y_k) / f0(Y_k)."""
    m = y.shape[-1]
    block_sum = y.sum(axis=-1)
    return -0.5 * math.log1p(m * tau**2) + (
        tau**2 / (2.0 * (1.0 + m * tau**2))
    ) * block_sum**2


def marginal_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    return -0.5 * math.log1p(tau**2) + (tau**2 / (2.0 * (1.0 + tau**2))) * y**2


def pairwise_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    m = y.shape[-1]
    i_idx, j_idx = np.triu_indices(m, k=1)
    pair_sum = y[:, :, i_idx] + y[:, :, j_idx]
    return -0.5 * math.log1p(2.0 * tau**2) + (
        tau**2 / (2.0 * (1.0 + 2.0 * tau**2))
    ) * pair_sum**2


def full_score_u(y: np.ndarray, pi: float | np.ndarray, tau: float) -> np.ndarray:
    return score_u_from_log_ratio(full_block_log_ratio(y, tau), pi).sum(axis=1)


def sample_from_anchor_pi(
    rng: np.random.Generator,
    anchor_pi: np.ndarray,
    sigma_q: float,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> dict[str, np.ndarray]:
    """Simulate FSM pairs for a supplied vector of anchor probabilities."""
    anchor_pi = np.asarray(anchor_pi, dtype=np.float64)
    anchor_u = logit_np(anchor_pi)
    eps = rng.normal(size=anchor_pi.shape[0])
    u = anchor_u + sigma_q * eps
    pi = sigmoid_np(u)
    y = simulate_common_factor(rng, u, n_blocks, block_size, tau)
    target = (u - anchor_u) / (sigma_q**2)
    return {
        "y": y,
        "anchor_pi": anchor_pi,
        "anchor_u": anchor_u,
        "u": u,
        "pi": pi,
        "target": target,
    }


def sample_anchor_batch(
    rng: np.random.Generator,
    n: int,
    pi_min: float,
    pi_max: float,
    sigma_q: float,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> dict[str, np.ndarray]:
    """Sample continuous stratified anchors and their local FSM pairs."""
    grid = (np.arange(n, dtype=np.float64) + rng.uniform(size=n)) / max(n, 1)
    rng.shuffle(grid)
    anchor_pi = pi_min + (pi_max - pi_min) * grid
    return sample_from_anchor_pi(rng, anchor_pi, sigma_q, n_blocks, block_size, tau)


# Datasets per featurization chunk. Pairwise float64 scores for 40k x 40 x 780
# take ~10 GB per temporary array, so full-bank featurization exceeds host memory
# on smaller machines. Chunking along the dataset axis leaves every feature value
# unchanged; see make_feature_stats for the standardization constants.
FEATURE_CHUNK_ROWS = 2_000


def _rows_of_pi(feature_pi: np.ndarray | float, start: int, stop: int) -> np.ndarray | float:
    pi_arr = np.asarray(feature_pi, dtype=np.float64)
    return feature_pi if pi_arr.ndim == 0 else pi_arr[start:stop]


def make_feature_stats(
    y: np.ndarray,
    tau: float,
    feature_pi: np.ndarray | float,
    chunk_rows: int = FEATURE_CHUNK_ROWS,
) -> dict[str, np.ndarray]:
    if y.shape[0] > int(chunk_rows):
        return _make_feature_stats_chunked(y, tau, feature_pi, int(chunk_rows))
    s1 = score_u_from_log_ratio(marginal_log_ratio(y, tau), feature_pi)
    s2 = score_u_from_log_ratio(pairwise_log_ratio(y, tau), feature_pi)
    stats = {
        "s1_mean": np.array(float(s1.mean()), dtype=np.float64),
        "s1_sd": np.array(float(s1.std()), dtype=np.float64),
        "s2_mean": np.array(float(s2.mean()), dtype=np.float64),
        "s2_sd": np.array(float(s2.std()), dtype=np.float64),
    }
    for key in ("s1_sd", "s2_sd"):
        stats[key] = np.where(stats[key] < 1e-8, 1.0, stats[key])
    s1_z = (s1 - stats["s1_mean"]) / stats["s1_sd"]
    s2_z = (s2 - stats["s2_mean"]) / stats["s2_sd"]
    block_raw = np.stack([s1_z.mean(axis=2), s2_z.mean(axis=2)], axis=-1)
    stats["block_mean"] = block_raw.mean(axis=(0, 1), keepdims=True)
    stats["block_sd"] = block_raw.std(axis=(0, 1), keepdims=True)
    stats["block_sd"] = np.where(stats["block_sd"] < 1e-8, 1.0, stats["block_sd"])
    return stats


def _make_feature_stats_chunked(
    y: np.ndarray,
    tau: float,
    feature_pi: np.ndarray | float,
    chunk_rows: int,
) -> dict[str, np.ndarray]:
    """Same population means/SDs as the one-shot path, from float64 running sums.

    Agrees with the one-shot numpy mean/std up to floating-point rounding
    (relative ~1e-15); it never holds pairwise float64 scores for all rows.
    """
    n = y.shape[0]
    count = {"s1": 0, "s2": 0}
    total = {"s1": 0.0, "s2": 0.0}
    square = {"s1": 0.0, "s2": 0.0}
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        pi = _rows_of_pi(feature_pi, start, stop)
        for key, log_ratio in (("s1", marginal_log_ratio), ("s2", pairwise_log_ratio)):
            s = score_u_from_log_ratio(log_ratio(y[start:stop], tau), pi)
            count[key] += s.size
            total[key] += float(s.sum())
            square[key] += float(np.square(s).sum())
    stats: dict[str, np.ndarray] = {}
    for key in ("s1", "s2"):
        mean = total[key] / count[key]
        stats[f"{key}_mean"] = np.array(mean, dtype=np.float64)
        stats[f"{key}_sd"] = np.array(math.sqrt(max(square[key] / count[key] - mean**2, 0.0)), dtype=np.float64)
    for key in ("s1_sd", "s2_sd"):
        stats[key] = np.where(stats[key] < 1e-8, 1.0, stats[key])

    block_count = 0
    block_total = np.zeros(2, dtype=np.float64)
    block_square = np.zeros(2, dtype=np.float64)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        pi = _rows_of_pi(feature_pi, start, stop)
        s1 = score_u_from_log_ratio(marginal_log_ratio(y[start:stop], tau), pi)
        s2 = score_u_from_log_ratio(pairwise_log_ratio(y[start:stop], tau), pi)
        s1_z = (s1 - stats["s1_mean"]) / stats["s1_sd"]
        s2_z = (s2 - stats["s2_mean"]) / stats["s2_sd"]
        block_raw = np.stack([s1_z.mean(axis=2), s2_z.mean(axis=2)], axis=-1)
        block_count += block_raw.shape[0] * block_raw.shape[1]
        block_total += block_raw.sum(axis=(0, 1))
        block_square += np.square(block_raw).sum(axis=(0, 1))
    block_mean = block_total / block_count
    block_sd = np.sqrt(np.maximum(block_square / block_count - block_mean**2, 0.0))
    stats["block_mean"] = block_mean.reshape(1, 1, 2)
    stats["block_sd"] = np.where(block_sd < 1e-8, 1.0, block_sd).reshape(1, 1, 2)
    return stats


def featurize(
    y: np.ndarray,
    tau: float,
    feature_pi: np.ndarray | float,
    stats: dict[str, np.ndarray],
    chunk_rows: int = FEATURE_CHUNK_ROWS,
) -> dict[str, np.ndarray]:
    if y.shape[0] > int(chunk_rows):
        # Elementwise per dataset, so chunked outputs are bit-identical.
        out: dict[str, np.ndarray] = {}
        for start in range(0, y.shape[0], int(chunk_rows)):
            stop = min(start + int(chunk_rows), y.shape[0])
            part = featurize(y[start:stop], tau, _rows_of_pi(feature_pi, start, stop), stats, chunk_rows)
            if not out:
                out = {k: np.empty((y.shape[0],) + v.shape[1:], dtype=v.dtype) for k, v in part.items()}
            for key, value in part.items():
                out[key][start:stop] = value
        return out
    s1 = score_u_from_log_ratio(marginal_log_ratio(y, tau), feature_pi)
    s2 = score_u_from_log_ratio(pairwise_log_ratio(y, tau), feature_pi)
    s1_z = ((s1 - stats["s1_mean"]) / stats["s1_sd"]).astype(np.float32)
    s2_z = ((s2 - stats["s2_mean"]) / stats["s2_sd"]).astype(np.float32)
    block_raw = np.stack([s1_z.mean(axis=2), s2_z.mean(axis=2)], axis=-1)
    block = ((block_raw - stats["block_mean"]) / stats["block_sd"]).astype(np.float32)
    return {"s1": s1_z, "s2": s2_z, "block": block}


def standardize_anchor(
    anchor_u_train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = float(anchor_u_train.mean())
    sd = float(anchor_u_train.std())
    if sd < 1e-8:
        sd = 1.0
    return tuple(((x - mean) / sd).astype(np.float32) for x in (anchor_u_train, *others))


def concatenate_prepared_splits(
    splits: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate independently simulated training-cache replicas.

    Each replica is featurized separately before concatenation.  This keeps
    peak host memory close to one raw replica plus the final float32 cache,
    instead of materializing pairwise float64 features for all replicas at
    once.
    """
    if not splits:
        raise ValueError("at least one prepared split is required")
    expected_keys = set(splits[0])
    if any(set(split) != expected_keys for split in splits[1:]):
        raise ValueError("prepared cache replicas have inconsistent fields")
    if len(splits) == 1:
        return splits[0]
    return {
        key: np.concatenate([split[key] for split in splits], axis=0)
        for key in splits[0]
    }


def array_sha256(value: np.ndarray) -> str:
    """Stable fingerprint including array dtype, shape, and raw contents."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def mapping_sha256(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(key.encode("utf-8"))
        digest.update(array_sha256(values[key]).encode("ascii"))
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class PositiveMLPMultiplier(nn.Module):
    """Positive scalar multiplier, initialized exactly at m=1.

    With ``input_dim=1`` this is m(s). With ``input_dim=2`` this is the
    beta-conditioned gate m(s, u_anchor), which is the default radial map in
    the amortized setting.
    """

    def __init__(self, hidden: int, input_dim: int = 1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, softplus_inverse(1.0))

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if len(features) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} gate features, got {len(features)}")
        x = torch.stack(features, dim=-1)
        return F.softplus(self.net(x).squeeze(-1))


def make_rho(in_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.SiLU())
        d = hidden
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class LinearCSBetaDeepSets(nn.Module):
    """DeepSets readout over block-level linear CS, conditioned on anchor beta."""

    def __init__(self, hidden: int, depth: int):
        super().__init__()
        self.rho = make_rho(3, hidden, depth)

    def forward(self, block: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        anchor = anchor_z.reshape(-1, 1, 1).expand(-1, block.shape[1], 1)
        x = torch.cat([block, anchor], dim=-1)
        return self.rho(x).squeeze(-1).sum(dim=1)


class SharedRawRadialCSBetaDeepSets(nn.Module):
    """One raw-score gate shared by the marginal and pairwise channels.

    ``s1`` and ``s2`` arrive in their channel-specific standardized
    coordinates.  The shared multiplier sees each channel's raw local score;
    its output is then mapped back to that channel's standardized coordinate
    before the unchanged block aggregation and rho readout.  Thus both
    channels remain present while the nonlinear calibration map is shared.
    """

    def __init__(
        self,
        hidden: int,
        depth: int,
        gate_hidden: int,
        gate_condition_on_anchor: bool,
        s1_mean: np.ndarray,
        s1_sd: np.ndarray,
        s2_mean: np.ndarray,
        s2_sd: np.ndarray,
        block_mean: np.ndarray,
        block_sd: np.ndarray,
    ):
        super().__init__()
        self.gate_condition_on_anchor = bool(gate_condition_on_anchor)
        gate_input_dim = 2 if self.gate_condition_on_anchor else 1
        self.shared_gate = PositiveMLPMultiplier(gate_hidden, input_dim=gate_input_dim)
        self.rho = make_rho(3, hidden, depth)
        self.register_buffer("s1_mean", torch.as_tensor(s1_mean, dtype=torch.float32))
        self.register_buffer("s1_sd", torch.as_tensor(s1_sd, dtype=torch.float32))
        self.register_buffer("s2_mean", torch.as_tensor(s2_mean, dtype=torch.float32))
        self.register_buffer("s2_sd", torch.as_tensor(s2_sd, dtype=torch.float32))
        self.register_buffer("block_mean", torch.as_tensor(block_mean, dtype=torch.float32))
        self.register_buffer("block_sd", torch.as_tensor(block_sd, dtype=torch.float32))

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        raw_s1 = s1 * self.s1_sd + self.s1_mean
        raw_s2 = s2 * self.s2_sd + self.s2_mean
        if self.gate_condition_on_anchor:
            anchor_s1 = anchor_z.reshape(-1, 1, 1).expand_as(raw_s1)
            anchor_s2 = anchor_z.reshape(-1, 1, 1).expand_as(raw_s2)
            multiplier_s1 = self.shared_gate(raw_s1, anchor_s1)
            multiplier_s2 = self.shared_gate(raw_s2, anchor_s2)
        else:
            multiplier_s1 = self.shared_gate(raw_s1)
            multiplier_s2 = self.shared_gate(raw_s2)

        # These delta forms equal (raw_s * multiplier - channel_mean) /
        # channel_sd, but make the m=1 nesting numerically exact whenever the
        # initialized multiplier is represented as exactly one.
        gated_s1 = s1 + raw_s1 * (multiplier_s1 - 1.0) / self.s1_sd
        gated_s2 = s2 + raw_s2 * (multiplier_s2 - 1.0) / self.s2_sd
        block_raw = torch.stack([gated_s1.mean(dim=2), gated_s2.mean(dim=2)], dim=-1)
        block = (block_raw - self.block_mean) / self.block_sd
        anchor = anchor_z.reshape(-1, 1, 1).expand(-1, block.shape[1], 1)
        x = torch.cat([block, anchor], dim=-1)
        return self.rho(x).squeeze(-1).sum(dim=1)


class LocalFeatureMLP(nn.Module):
    """Unbounded learned channel m(s, a) in R^m_dim, initially identically zero.

    Consumes the raw local score, matching the gate's input convention, so the
    only difference between the two nonlinear maps is stacking versus
    elementwise multiplication.
    """

    def __init__(self, hidden: int, m_dim: int = 1):
        super().__init__()
        if m_dim < 1:
            raise ValueError("m_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, m_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return self.net(torch.stack((s, anchor), dim=-1))


class StackedCSBetaDeepSets(nn.Module):
    """phi(s, a) = (1, s, m(s, a)) for the marginal and pairwise channels.

    ``share_local_features`` selects the Model-2-specific question: one learned
    map shared by both channel types, as the multiplicative gate does, or one
    per type. The identity channels reuse the existing standardized ``block``
    feature verbatim, so ILSA nesting is exact rather than approximate.
    """

    def __init__(self, hidden: int, depth: int, gate_hidden: int,
                 s1_mean, s1_sd, s2_mean, s2_sd,
                 m_dim: int = 1, include_constant_channel: bool = True,
                 share_local_features: bool = True):
        super().__init__()
        self.m_dim = int(m_dim)
        self.include_constant_channel = bool(include_constant_channel)
        self.share_local_features = bool(share_local_features)
        self.local_features = LocalFeatureMLP(gate_hidden, self.m_dim)
        self.local_features_pairwise = (
            None if self.share_local_features else LocalFeatureMLP(gate_hidden, self.m_dim)
        )
        width = int(self.include_constant_channel) + 2 + 2 * self.m_dim + 1
        self.rho = make_rho(width, hidden, depth)
        for key, value in (("s1_mean", s1_mean), ("s1_sd", s1_sd),
                           ("s2_mean", s2_mean), ("s2_sd", s2_sd)):
            self.register_buffer(key, torch.as_tensor(value, dtype=torch.float32))

    @property
    def linear_column_map(self) -> tuple[int, int, int]:
        off = int(self.include_constant_channel)
        return off, off + 1, off + 2 + 2 * self.m_dim

    def _pairwise_module(self) -> nn.Module:
        return self.local_features if self.share_local_features else self.local_features_pairwise

    def readout_inputs(self, block, s1, s2, anchor_z):
        raw1 = s1 * self.s1_sd + self.s1_mean
        raw2 = s2 * self.s2_sd + self.s2_mean
        a1 = anchor_z.reshape(-1, 1, 1).expand_as(raw1)
        a2 = anchor_z.reshape(-1, 1, 1).expand_as(raw2)
        pooled1 = self.local_features(raw1, a1).mean(dim=2)
        pooled2 = self._pairwise_module()(raw2, a2).mean(dim=2)
        anchor = anchor_z.reshape(-1, 1, 1).expand(-1, block.shape[1], 1)
        parts = [torch.ones_like(block[..., :1])] if self.include_constant_channel else []
        return torch.cat([*parts, block, pooled1, pooled2, anchor], dim=-1)

    def forward(self, block, s1, s2, anchor_z):
        return self.rho(self.readout_inputs(block, s1, s2, anchor_z)).squeeze(-1).sum(dim=1)


GATED_METHODS = {"shared_radial"}
STACKED_METHODS = {"stacked_shared", "stacked_split"}
NONLINEAR_METHODS = GATED_METHODS | STACKED_METHODS
SELECTED_GATE_CONDITION_ON_ANCHOR = True
SELECTED_RADIAL_INIT = "matched_random"
SELECTED_GATE_ONLY_STEPS = 0


def tensors_for_method(data: dict[str, np.ndarray], method: str, device: str) -> dict[str, torch.Tensor]:
    if method == "linear":
        keys = ("block", "anchor_z")
    elif method in GATED_METHODS:
        keys = ("s1", "s2", "anchor_z")
    elif method in STACKED_METHODS:
        keys = ("block", "s1", "s2", "anchor_z")
    else:
        raise ValueError(f"Unknown method: {method}")
    return {key: torch.as_tensor(data[key], dtype=torch.float32, device=device) for key in keys}


def forward_method(model: nn.Module, batch: dict[str, torch.Tensor], method: str) -> torch.Tensor:
    if method == "linear":
        return model(batch["block"], batch["anchor_z"])
    if method in GATED_METHODS:
        return model(batch["s1"], batch["s2"], batch["anchor_z"])
    if method in STACKED_METHODS:
        return model(batch["block"], batch["s1"], batch["s2"], batch["anchor_z"])
    raise ValueError(f"Unknown method: {method}")


def mse_over_tensors(
    model: nn.Module,
    x: dict[str, torch.Tensor],
    y: torch.Tensor,
    method: str,
    batch_size: int,
) -> float:
    model.eval()
    sqerr = 0.0
    n_total = 0
    with torch.no_grad():
        n = int(y.shape[0])
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            xb = {k: v[sl] for k, v in x.items()}
            pred = forward_method(model, xb, method)
            err = pred - y[sl]
            sqerr += float(torch.sum(err**2).item())
            n_total += int(err.numel())
    model.train()
    return sqerr / max(n_total, 1)


def predict(
    model: nn.Module,
    data: dict[str, np.ndarray],
    method: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    x = tensors_for_method(data, method, device)
    n = int(next(iter(x.values())).shape[0])
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            xb = {k: v[sl] for k, v in x.items()}
            out.append(forward_method(model, xb, method).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def clone_state_dict(model: nn.Module, *, cpu: bool) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        cloned = value.detach().clone()
        state[key] = cloned.cpu() if cpu else cloned
    return state


@torch.no_grad()
def update_ema_state(
    ema_state: dict[str, torch.Tensor],
    model: nn.Module,
    decay: float,
) -> None:
    for key, value in model.state_dict().items():
        if torch.is_floating_point(value):
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[key].copy_(value.detach())


def learning_rate_multiplier(
    step: int,
    iters: int,
    schedule: str,
    decay_start_step: int,
    min_ratio: float,
) -> float:
    """Return a deterministic per-step multiplier for every optimizer group."""
    if schedule == "constant" or step <= decay_start_step:
        return 1.0
    if schedule != "cosine_tail":
        raise ValueError(f"Unknown learning-rate schedule: {schedule}")
    progress = (step - decay_start_step) / (iters - decay_start_step)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def train_model(
    model: nn.Module,
    method: str,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    *,
    iters: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    print_every: int,
    seed: int,
    grad_clip: float,
    ema_decay: float,
    gate_lr: float,
    joint_rho_lr: float,
    lr_schedule: str,
    lr_decay_start_step: int,
    lr_min_ratio: float,
    milestone_steps: tuple[int, ...],
    device: str,
    name: str,
) -> tuple[
    nn.Module,
    list[dict[str, Any]],
    dict[str, Any],
    dict[int, dict[str, dict[str, torch.Tensor]]],
]:
    if lr_schedule == "cosine_tail" and not (0 <= lr_decay_start_step < iters):
        raise ValueError(
            "cosine_tail requires 0 <= lr_decay_start_step < the method iteration budget"
        )
    model = model.to(device)
    x_train = tensors_for_method(train, method, device)
    x_val = tensors_for_method(val, method, device)
    y_train = torch.as_tensor(train["target"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val["target"], dtype=torch.float32, device=device)
    if method in NONLINEAR_METHODS:
        gate_params = list(
            model.shared_gate.parameters() if method in GATED_METHODS
            else (m for module in (model.local_features, model.local_features_pairwise)
                  if module is not None for m in module.parameters())
        )
        rho_params = list(model.rho.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": gate_params, "lr": gate_lr, "name": "gate"},
                {
                    "params": rho_params,
                    "lr": joint_rho_lr,
                    "name": "rho",
                },
            ],
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_lrs = {
        "model": lr,
        "gate": gate_lr,
        "rho": joint_rho_lr,
    }

    def set_step_learning_rates(step: int) -> float:
        multiplier = learning_rate_multiplier(
            step,
            iters,
            lr_schedule,
            lr_decay_start_step,
            lr_min_ratio,
        )
        if method in NONLINEAR_METHODS:
            opt.param_groups[0]["lr"] = base_lrs["gate"] * multiplier
            opt.param_groups[1]["lr"] = base_lrs["rho"] * multiplier
        else:
            opt.param_groups[0]["lr"] = base_lrs["model"] * multiplier
        return multiplier

    def trace_learning_rates() -> dict[str, float | str]:
        if method in NONLINEAR_METHODS:
            return {
                "lr_model": "",
                "lr_gate": float(opt.param_groups[0]["lr"]),
                "lr_rho": float(opt.param_groups[1]["lr"]),
            }
        return {
            "lr_model": float(opt.param_groups[0]["lr"]),
            "lr_gate": "",
            "lr_rho": "",
        }

    set_step_learning_rates(0)
    rng = np.random.default_rng(seed)
    minibatch_digest = hashlib.sha256()
    best_state = clone_state_dict(model, cpu=True)
    ema_state = clone_state_dict(model, cpu=False)
    best_val = mse_over_tensors(model, x_val, y_val, method, batch_size)
    if not np.isfinite(best_val):
        raise RuntimeError(f"{name} produced non-finite initial validation loss")
    best_step = 0
    best_source = "raw"
    milestone_step_set = set(milestone_steps)
    milestone_states: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    trace: list[dict[str, Any]] = [
        {
            "method": name,
            "step": 0,
            "stage": "joint",
            "train_loss": best_val,
            "val_loss_raw": best_val,
            "val_loss_ema": best_val,
            "best_val_loss": best_val,
            "best_source": best_source,
            **trace_learning_rates(),
        }
    ]
    bad = 0
    n_train = int(y_train.shape[0])
    print(f"{name} step 0/{iters}: val={best_val:.6g}")

    for step in range(1, iters + 1):
        lr_multiplier = set_step_learning_rates(step)
        idx_numpy = np.ascontiguousarray(
            rng.integers(0, n_train, size=min(batch_size, n_train)),
            dtype=np.int64,
        )
        minibatch_digest.update(idx_numpy.tobytes(order="C"))
        idx = torch.as_tensor(
            idx_numpy,
            dtype=torch.long,
            device=device,
        )
        xb = {k: v.index_select(0, idx) for k, v in x_train.items()}
        yb = y_train.index_select(0, idx)
        opt.zero_grad(set_to_none=True)
        pred = forward_method(model, xb, method)
        loss = torch.mean((pred - yb) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced non-finite training loss at step {step}")
        loss.backward()
        for param in model.parameters():
            if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                raise RuntimeError(f"{name} produced non-finite gradients at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"{name} produced non-finite gradient norm at step {step}")
        opt.step()
        update_ema_state(ema_state, model, ema_decay)
        if step in milestone_step_set:
            milestone_states[step] = {
                "raw": clone_state_dict(model, cpu=True),
                "ema": {
                    key: value.detach().cpu().clone()
                    for key, value in ema_state.items()
                },
            }
        scheduled_validation = (
            step == 1
            or step % print_every == 0
            or step == iters
        )
        if scheduled_validation or step in milestone_step_set:
            val_loss_raw = mse_over_tensors(model, x_val, y_val, method, batch_size)
            raw_state = clone_state_dict(model, cpu=False)
            model.load_state_dict(ema_state)
            val_loss_ema = mse_over_tensors(model, x_val, y_val, method, batch_size)
            model.load_state_dict(raw_state)
            if not np.isfinite(val_loss_raw) or not np.isfinite(val_loss_ema):
                raise RuntimeError(f"{name} produced non-finite validation loss at step {step}")
            if val_loss_ema < val_loss_raw:
                selected_val = val_loss_ema
                selected_source = "ema"
                selected_state = {key: value.detach().cpu().clone() for key, value in ema_state.items()}
            else:
                selected_val = val_loss_raw
                selected_source = "raw"
                selected_state = clone_state_dict(model, cpu=True)
            reported_best_val = (
                min(best_val, selected_val) if scheduled_validation else best_val
            )
            reported_best_source = (
                selected_source
                if scheduled_validation and selected_val < best_val
                else best_source
            )
            trace.append(
                {
                    "method": name,
                    "step": step,
                    "stage": "joint",
                    "train_loss": float(loss.detach().cpu().item()),
                    "val_loss_raw": float(val_loss_raw),
                    "val_loss_ema": float(val_loss_ema),
                    "best_val_loss": float(reported_best_val),
                    "best_source": reported_best_source,
                    "grad_norm": float(grad_norm.detach().cpu().item()),
                    **trace_learning_rates(),
                }
            )
            lr_values = "/".join(f"{group['lr']:.3g}" for group in opt.param_groups)
            print(
                f"{name} step {step}/{iters} [joint]: train={loss.item():.6g}, "
                f"val_raw={val_loss_raw:.6g}, val_ema={val_loss_ema:.6g}, "
                f"best={reported_best_val:.6g}, lr={lr_values}, "
                f"lr_mult={lr_multiplier:.4f}"
            )
            if scheduled_validation:
                if selected_val < best_val - 1e-6:
                    best_val = selected_val
                    best_state = selected_state
                    best_step = step
                    best_source = selected_source
                    bad = 0
                else:
                    bad += 1
                if patience > 0 and bad >= patience:
                    print(f"{name}: early stopping at step {step}; best val={best_val:.6g}")
                    break
    missing_milestones = sorted(milestone_step_set - set(milestone_states))
    if missing_milestones:
        raise RuntimeError(
            f"{name} stopped before requested milestone steps: {missing_milestones}"
        )
    model.load_state_dict(best_state)
    info = {
        "best_step": best_step,
        "best_val_loss": float(best_val),
        "best_source": best_source,
        "stopped_step": int(trace[-1]["step"]),
        "lr_schedule": lr_schedule,
        "lr_decay_start_step": lr_decay_start_step,
        "lr_min_ratio": lr_min_ratio,
        "final_learning_rates": trace_learning_rates(),
        "milestone_steps_saved": sorted(milestone_states),
        "minibatch_seed": int(seed),
        "minibatch_indices_sha256": minibatch_digest.hexdigest(),
        "minibatch_rows_drawn": int(
            int(trace[-1]["step"]) * min(batch_size, n_train)
        ),
    }
    return model, trace, info, milestone_states


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    var = float(np.var(y_true) + 1e-12)
    pred_sd = float(np.std(y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if pred_sd > 1e-12 else float("nan")
    denom = float(np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    cosine = float(np.dot(y_true, y_pred) / denom) if denom > 1e-12 else float("nan")
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "std_mse": mse / var,
        "std_rmse": math.sqrt(mse / var),
        "corr": corr,
        "cosine": cosine,
        "target_mean": float(np.mean(y_true)),
        "target_sd": float(np.std(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_sd": pred_sd,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(
    raw: dict[str, np.ndarray],
    tau: float,
    stats: dict[str, np.ndarray],
    anchor_z: np.ndarray,
) -> dict[str, np.ndarray]:
    feat = featurize(raw["y"], tau, raw["anchor_pi"], stats)
    feat["anchor_z"] = anchor_z.astype(np.float32)
    feat["target"] = raw["target"].astype(np.float32)
    feat["anchor_pi"] = raw["anchor_pi"].astype(np.float64)
    return feat


def build_eval_data(
    rng: np.random.Generator,
    pi: float,
    n: int,
    n_blocks: int,
    block_size: int,
    tau: float,
    stats: dict[str, np.ndarray],
    anchor_mean: float,
    anchor_sd: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    u = np.full(n, float(logit_np(pi)), dtype=np.float64)
    y = simulate_common_factor(rng, u, n_blocks, block_size, tau)
    feature_pi: np.ndarray | float = np.full(n, pi, dtype=np.float64)
    feat = featurize(y, tau, feature_pi, stats)
    feat["y"] = y.astype(np.float32)
    feat["anchor_z"] = ((u - anchor_mean) / anchor_sd).astype(np.float32)
    feat["anchor_pi"] = np.full(n, pi, dtype=np.float64)
    truth = full_score_u(y, pi, tau)
    return feat, truth


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (0.0 < args.anchor_pi_min < args.anchor_pi_max < 1.0):
        raise ValueError("anchor pi range must lie inside (0, 1)")
    if args.sigma_q <= 0:
        raise ValueError("--sigma-q must be positive")
    if args.train_cache_replicates < 1:
        raise ValueError("--train-cache-replicates must be at least 1")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")
    if not (0.0 <= args.ema_decay < 1.0):
        raise ValueError("--ema-decay must lie in [0, 1)")
    if args.lr_decay_start_step < 0:
        raise ValueError("--lr-decay-start-step must be nonnegative")
    if not (0.0 < args.lr_min_ratio <= 1.0):
        raise ValueError("--lr-min-ratio must lie in (0, 1]")
    milestone_steps = tuple(
        sorted(
            {
                int(value)
                for value in args.milestone_steps.split(",")
                if value.strip()
            }
        )
    )
    if any(step < 1 or step > args.iters for step in milestone_steps):
        raise ValueError("--milestone-steps must lie between 1 and --iters")
    if args.save_final_ema_validation and args.iters not in milestone_steps:
        raise ValueError(
            "--save-final-ema-validation requires the final --iters value in "
            "--milestone-steps"
        )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"torch device: {device}")
    print(f"Output directory: {out_dir}")
    print(
        "setting:",
        {
            "n_blocks": args.n_blocks,
            "block_size": args.block_size,
            "tau": args.tau,
            "anchor_pi_range": [args.anchor_pi_min, args.anchor_pi_max],
            "sigma_q": args.sigma_q,
            "gate_condition_on_anchor": SELECTED_GATE_CONDITION_ON_ANCHOR,
            "radial_init": SELECTED_RADIAL_INIT,
            "methods": args.methods,
        },
    )

    rng = np.random.default_rng(args.seed)
    print("\nGenerating amortized FSM data...")
    # Keep the legacy base-train -> validation RNG path exactly unchanged.
    # Extra cache replicas are generated only after validation, from separate
    # SeedSequence streams, so enabling them cannot change the old base cache,
    # validation cache, or the later exact-score diagnostic RNG state.
    train_raw_base = sample_anchor_batch(
        rng,
        args.n_train,
        args.anchor_pi_min,
        args.anchor_pi_max,
        args.sigma_q,
        args.n_blocks,
        args.block_size,
        args.tau,
    )
    val_raw = sample_anchor_batch(
        rng,
        args.n_val,
        args.anchor_pi_min,
        args.anchor_pi_max,
        args.sigma_q,
        args.n_blocks,
        args.block_size,
        args.tau,
    )
    print(
        "base train y:",
        train_raw_base["y"].shape,
        "cache replicas:",
        args.train_cache_replicates,
        "val y:",
        val_raw["y"].shape,
    )

    print("Computing anchor/beta-conditioned local CS features...")
    # Always fit preprocessing on the legacy base cache.  Consequently a
    # cache-size ablation changes training-row diversity, not feature scaling.
    stats = make_feature_stats(train_raw_base["y"], args.tau, train_raw_base["anchor_pi"])
    anchor_z_train, anchor_z_val = standardize_anchor(
        train_raw_base["anchor_u"], val_raw["anchor_u"]
    )
    anchor_mean = float(train_raw_base["anchor_u"].mean())
    anchor_sd = float(train_raw_base["anchor_u"].std())
    if anchor_sd < 1e-8:
        anchor_sd = 1.0
    train_splits = [
        prepare_split(
            train_raw_base,
            args.tau,
            stats,
            anchor_z_train,
        )
    ]
    cache_fingerprints: dict[str, Any] = {
        "base_train": {
            "y_sha256": array_sha256(train_raw_base["y"]),
            "anchor_u_sha256": array_sha256(train_raw_base["anchor_u"]),
            "target_sha256": array_sha256(train_raw_base["target"]),
        },
        "validation": {
            "y_sha256": array_sha256(val_raw["y"]),
            "anchor_u_sha256": array_sha256(val_raw["anchor_u"]),
            "target_sha256": array_sha256(val_raw["target"]),
        },
        "extra_train_replicas": [],
    }
    for replica_index in range(1, args.train_cache_replicates):
        seed_words = [int(args.seed), 410_003, replica_index]
        replica_rng = np.random.default_rng(np.random.SeedSequence(seed_words))
        replica_raw = sample_anchor_batch(
            replica_rng,
            args.n_train,
            args.anchor_pi_min,
            args.anchor_pi_max,
            args.sigma_q,
            args.n_blocks,
            args.block_size,
            args.tau,
        )
        replica_anchor_z = (
            (replica_raw["anchor_u"] - anchor_mean) / anchor_sd
        ).astype(np.float32)
        train_splits.append(
            prepare_split(
                replica_raw,
                args.tau,
                stats,
                replica_anchor_z,
            )
        )
        cache_fingerprints["extra_train_replicas"].append(
            {
                "replica_index": replica_index,
                "seed_sequence_words": seed_words,
                "y_sha256": array_sha256(replica_raw["y"]),
                "anchor_u_sha256": array_sha256(replica_raw["anchor_u"]),
                "target_sha256": array_sha256(replica_raw["target"]),
            }
        )
        del replica_raw
    train = concatenate_prepared_splits(train_splits)
    val = prepare_split(val_raw, args.tau, stats, anchor_z_val)
    effective_n_train = int(train["target"].shape[0])
    print("effective train rows:", effective_n_train)
    print("linear block features:", train["block"].shape)
    print("local marginal/pairwise:", train["s1"].shape, train["s2"].shape)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = sorted(set(methods) - ({"linear"} | NONLINEAR_METHODS))
    if bad:
        raise ValueError(f"Unknown methods: {bad}")
    gated_methods_requested = set(methods) & NONLINEAR_METHODS
    config = vars(args).copy()
    config["gate_condition_on_anchor"] = SELECTED_GATE_CONDITION_ON_ANCHOR
    config["radial_init"] = SELECTED_RADIAL_INIT
    config["gate_only_steps"] = SELECTED_GATE_ONLY_STEPS
    config["device_resolved"] = device
    config["anchor_sampler"] = "continuous_stratified"
    config["sigma_q_mode"] = "fixed"
    config["block_aggregation"] = "mean"
    config["rho_activation"] = "silu"
    config["anchor_u_mean"] = anchor_mean
    config["anchor_u_sd"] = anchor_sd
    config["heldout_evaluation_rng_state"] = rng.bit_generator.state
    config["exact_evaluation_separated"] = True
    config["n_train_base"] = int(args.n_train)
    config["n_train_effective"] = effective_n_train
    sampled_row_presentations = int(
        args.iters * min(args.batch_size, effective_n_train)
    )
    config["sampled_row_presentations"] = sampled_row_presentations
    config["expected_presentations_per_train_row"] = (
        sampled_row_presentations / effective_n_train
    )
    config["preprocessing_fit_scope"] = "base_train_cache"
    config["extra_train_replica_seed_scheme"] = (
        "numpy.SeedSequence([seed, 410003, replica_index])"
    )
    config["milestone_steps_resolved"] = list(milestone_steps)
    config["checkpoint_selection"] = "validation_best_raw_or_ema"
    config["exact_score_available_to_checkpoint_selection"] = False
    config["lr_screen_selection_metric"] = (
        "final_ema_fsm_validation_mse"
        if args.save_final_ema_validation
        else None
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "feature_stats.npz",
        **stats,
        anchor_u_mean=np.array(anchor_mean),
        anchor_u_sd=np.array(anchor_sd),
    )
    reproducibility_fingerprints: dict[str, Any] = {
        "feature_stats_sha256": mapping_sha256(stats),
        "data_cache": cache_fingerprints,
        "method_initializations": {},
    }
    (out_dir / "reproducibility_fingerprints.json").write_text(
        json.dumps(reproducibility_fingerprints, indent=2), encoding="utf-8"
    )
    del train_raw_base, val_raw, train_splits

    models: dict[str, tuple[str, nn.Module]] = {}
    all_trace: list[dict[str, Any]] = []
    training_info: dict[str, Any] = {}
    milestone_states_by_method: dict[
        str, dict[int, dict[str, dict[str, torch.Tensor]]]
    ] = {}
    linear_initial_rho_state: dict[str, torch.Tensor] | None = None
    linear_initial_check: np.ndarray | None = None
    check_n = min(256, int(train["target"].shape[0]))
    check_data = {key: value[:check_n] for key, value in train.items()}
    if gated_methods_requested and "linear" not in methods:
        set_global_seed(args.seed + 11)
        initial_linear_reference = LinearCSBetaDeepSets(args.hidden, args.depth)
        linear_initial_rho_state = clone_state_dict(initial_linear_reference.rho, cpu=True)
        initial_linear_reference = initial_linear_reference.to(device)
        linear_initial_check = predict(
            initial_linear_reference,
            check_data,
            "linear",
            device,
            args.batch_size,
        )
    ordered = ("linear", "shared_radial", "stacked_shared", "stacked_split")
    for method in (candidate for candidate in ordered if candidate in methods):
        if method == "linear":
            label = "linear CS beta-conditioned DeepSets FSM"
            set_global_seed(args.seed + 11)
            model = LinearCSBetaDeepSets(args.hidden, args.depth)
            if gated_methods_requested:
                linear_initial_rho_state = clone_state_dict(model.rho, cpu=True)
                model = model.to(device)
                linear_initial_check = predict(model, check_data, "linear", device, args.batch_size)
        elif method in STACKED_METHODS:
            shared = method == "stacked_shared"
            label = ("stacked NLSA local map (shared)" if shared
                     else "stacked NLSA local map (per-channel)")
            set_global_seed(args.seed + (33 if shared else 44))
            model = StackedCSBetaDeepSets(
                args.hidden, args.depth, args.gate_hidden,
                s1_mean=stats["s1_mean"], s1_sd=stats["s1_sd"],
                s2_mean=stats["s2_mean"], s2_sd=stats["s2_sd"],
                m_dim=args.m_dim,
                include_constant_channel=bool(args.include_constant_channel),
                share_local_features=shared,
            )
            nesting_error = None
            if linear_initial_rho_state is None or linear_initial_check is None:
                raise RuntimeError("matched_random requires the saved initial linear rho")
            load_matched_linear_rho(
                model.rho, linear_initial_rho_state, model.linear_column_map,
                constant_column=0 if model.include_constant_channel else None,
            )
            reference_check = linear_initial_check
            label = f"{label} [matched_random]"
            print(f"{method} initialization: matched untrained linear rho, m == 0")
            model = model.to(device)
            gated_check = predict(model, check_data, method, device, args.batch_size)
            nesting_error = float(np.max(np.abs(reference_check - gated_check)))
        else:
            label = "shared-raw-gate CS beta-conditioned gate+DeepSets FSM"
            set_global_seed(args.seed + 22)
            model = SharedRawRadialCSBetaDeepSets(
                args.hidden,
                args.depth,
                args.gate_hidden,
                gate_condition_on_anchor=SELECTED_GATE_CONDITION_ON_ANCHOR,
                s1_mean=stats["s1_mean"],
                s1_sd=stats["s1_sd"],
                s2_mean=stats["s2_mean"],
                s2_sd=stats["s2_sd"],
                block_mean=stats["block_mean"],
                block_sd=stats["block_sd"],
            )
            nesting_error: float | None = None
            if linear_initial_rho_state is None or linear_initial_check is None:
                raise RuntimeError("matched_random requires the saved initial linear rho")
            model.rho.load_state_dict(linear_initial_rho_state)
            reference_check = linear_initial_check
            label = f"{label} [matched_random]"
            init_description = "matched untrained linear rho"
            print(f"{method} initialization: {init_description}")
            model = model.to(device)
            if reference_check is not None:
                gated_check = predict(model, check_data, method, device, args.batch_size)
                nesting_error = float(np.max(np.abs(reference_check - gated_check)))
                print(f"nested initialization max abs diff: {nesting_error:.3e}")
                if nesting_error > 1e-5:
                    raise RuntimeError(
                        f"{method} initialization does not reproduce its linear reference: "
                        f"{nesting_error:.3e}"
                    )
        initial_state_fingerprint = state_dict_sha256(model.state_dict())
        reproducibility_fingerprints["method_initializations"][method] = {
            "state_dict_sha256": initial_state_fingerprint,
            "rho_state_dict_sha256": state_dict_sha256(model.rho.state_dict()),
        }
        print(f"\nTraining {label}...")
        model, trace, info, method_milestone_states = train_model(
            model,
            method,
            train,
            val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            seed=args.seed + 30_001,
            grad_clip=args.grad_clip,
            ema_decay=args.ema_decay,
            gate_lr=args.gate_lr,
            joint_rho_lr=args.joint_rho_lr,
            lr_schedule=args.lr_schedule,
            lr_decay_start_step=args.lr_decay_start_step,
            lr_min_ratio=args.lr_min_ratio,
            milestone_steps=milestone_steps,
            device=device,
            name=label,
        )
        info["method"] = method
        info["label"] = label
        info["initial_state_sha256"] = initial_state_fingerprint
        if method in NONLINEAR_METHODS:
            info["initialization"] = SELECTED_RADIAL_INIT
            info["local_map"] = ("multiplicative_gate" if method in GATED_METHODS
                                 else "stacked_(1,s,m)")
            info["gate_sharing"] = "shared" if method != "stacked_split" else "per_channel"
            info["gate_input_scale"] = "raw_local_score"
            if nesting_error is not None:
                info["nested_initialization_max_abs_diff"] = nesting_error
        training_info[method] = info
        milestone_states_by_method[method] = method_milestone_states
        torch.save(
            {
                "state_dict": model.state_dict(),
                "method": method,
                "label": label,
                "config": config,
                "training_info": info,
            },
            out_dir / f"model_{method}.pt",
        )
        milestone_dir = out_dir / "milestones"
        for milestone_step, source_states in method_milestone_states.items():
            for source, state_dict in source_states.items():
                milestone_path = (
                    milestone_dir
                    / f"model_{method}_step{milestone_step}_{source}.pt"
                )
                milestone_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": state_dict,
                        "method": method,
                        "label": label,
                        "checkpoint_step": milestone_step,
                        "checkpoint_source": source,
                        "selection": "fixed_milestone",
                        "config": config,
                        "training_info": info,
                    },
                    milestone_path,
                )
        models[label] = (method, model)
        all_trace.extend(trace)
    write_csv(out_dir / "training_trace.csv", all_trace)
    (out_dir / "training_info.json").write_text(json.dumps(training_info, indent=2), encoding="utf-8")

    final_ema_validation_rows: list[dict[str, Any]] = []
    if args.save_final_ema_validation:
        print("\nSaving predeclared final-EMA paired FSM validation losses...")
        val_target = np.asarray(val["target"], dtype=np.float64)
        for label, (method, model) in models.items():
            source_states = milestone_states_by_method.get(method, {}).get(args.iters)
            if source_states is None or "ema" not in source_states:
                raise RuntimeError(
                    f"Missing final EMA state for {method} at step {args.iters}"
                )
            selected_state = clone_state_dict(model, cpu=True)
            model.load_state_dict(source_states["ema"])
            prediction = predict(model, val, method, device, args.batch_size).astype(
                np.float64
            )
            model.load_state_dict(selected_state)
            squared_fsm_error = (prediction - val_target) ** 2
            artifact_path = out_dir / f"lr_selection_final_ema_{method}.npz"
            np.savez_compressed(
                artifact_path,
                row_index=np.arange(val_target.size, dtype=np.int64),
                squared_fsm_error=squared_fsm_error,
            )
            n_val_rows = int(squared_fsm_error.size)
            loss_sd = float(np.std(squared_fsm_error, ddof=1)) if n_val_rows > 1 else 0.0
            final_ema_validation_rows.append(
                {
                    "method_id": method,
                    "method": label,
                    "selection_metric": "final_ema_fsm_validation_mse",
                    "checkpoint_step": args.iters,
                    "checkpoint_source": "ema",
                    "n_validation": n_val_rows,
                    "fsm_val_mse": float(np.mean(squared_fsm_error)),
                    "fsm_val_mse_se": loss_sd / math.sqrt(max(n_val_rows, 1)),
                    "lr": args.lr if method == "linear" else args.joint_rho_lr,
                    "gate_lr": "" if method == "linear" else args.gate_lr,
                    "validation_y_sha256": cache_fingerprints["validation"]["y_sha256"],
                    "validation_target_sha256": cache_fingerprints["validation"][
                        "target_sha256"
                    ],
                    "minibatch_indices_sha256": training_info[method][
                        "minibatch_indices_sha256"
                    ],
                    "exact_score_used": False,
                    "per_row_artifact": artifact_path.name,
                }
            )
        write_csv(
            out_dir / "lr_selection_final_ema_summary.csv",
            final_ema_validation_rows,
        )

    (out_dir / "reproducibility_fingerprints.json").write_text(
        json.dumps(reproducibility_fingerprints, indent=2), encoding="utf-8"
    )

    print("\nSaved:")
    print(out_dir / "config.json")
    print(out_dir / "feature_stats.npz")
    print(out_dir / "reproducibility_fingerprints.json")
    print(out_dir / "training_trace.csv")
    print(out_dir / "training_info.json")
    if args.save_final_ema_validation:
        print(out_dir / "lr_selection_final_ema_summary.csv")
        for method in methods:
            print(out_dir / f"lr_selection_final_ema_{method}.npz")
    if milestone_steps:
        print(out_dir / "milestones")
    for method in methods:
        print(out_dir / f"model_{method}.pt")
    print("Exact-score evaluation is a separate command: python -m model2.evaluate_stage1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=40_000)
    parser.add_argument(
        "--train-cache-replicates",
        type=int,
        default=1,
        help=(
            "Number of independent n_train-row training-cache replicas. Replica 1 "
            "is the unchanged legacy cache; extras use separate RNG streams and "
            "the legacy base-cache preprocessing statistics. Default 1 preserves "
            "all prior behavior."
        ),
    )
    parser.add_argument("--n-val", type=int, default=8_000)
    parser.add_argument("--n-blocks", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--anchor-pi-min", type=float, default=0.05)
    parser.add_argument("--anchor-pi-max", type=float, default=0.7)
    parser.add_argument("--sigma-q", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument(
        "--methods",
        type=str,
        default="linear,shared_radial",
        help="Comma-separated subset of linear and shared_radial.",
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--m-dim", type=int, default=1,
                        help="Learned-channel width for the stacked local map.")
    parser.add_argument("--include-constant-channel", type=int, choices=(0, 1), default=1)
    parser.add_argument("--iters", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4, help="Linear-model learning rate.")
    parser.add_argument("--gate-lr", type=float, default=1e-4)
    parser.add_argument("--joint-rho-lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=("constant", "cosine_tail"),
        help="constant, or cosine decay after --lr-decay-start-step.",
    )
    parser.add_argument("--lr-decay-start-step", type=int, default=10_000)
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=0.1,
        help="Final/base LR ratio for cosine_tail.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument(
        "--milestone-steps",
        type=str,
        default="10000,15000,20000",
        help=(
            "Comma-separated optimizer steps at which both raw and EMA "
            "checkpoints are saved and evaluated; empty preserves legacy behavior."
        ),
    )
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--save-final-ema-validation",
        action="store_true",
        help=(
            "Save ordered per-row squared FSM losses on the shared validation cache "
            "for the final EMA checkpoint. Requires --iters in --milestone-steps."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/blockwise_common_factor_amortized_fsm",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
