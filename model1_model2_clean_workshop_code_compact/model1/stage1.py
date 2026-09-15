#!/usr/bin/env python3
"""Amortized conditional FSM for the blockwise mean-shift mixture.

This is the cleaned current implementation of the amortized score field

    S(Y, beta_anchor) ~= d/dbeta log p_beta(Y)

for the Model 1 blockwise mean-shift mixture. We work in the unconstrained
coordinate u=logit(pi), sample anchors u_a over a pi range, sample local
perturbations u | u_a ~ N(u_a, sigma_q^2), simulate Y ~ p_u, and regress

    S(Y, u_a) -> (u - u_a) / sigma_q^2.

The exact score is used only for diagnostics after training. By default, the
local composite-score features are evaluated at each sample's own anchor
pi_a, so the network really sees a beta-conditional approximation problem.
Only the deployable anchor-conditioned feature path is exposed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ArchSpec:
    input_keys: tuple[str, ...]
    needs_subscores: bool
    needs_raw_y: bool
    within_block_perm_invariant: bool
    parameter_input: str
    seed_offset: int
    label: str
    warm_start_from: str | None = None


ARCHITECTURES: dict[str, ArchSpec] = {
    "linear": ArchSpec(
        input_keys=("block", "anchor_z"),
        needs_subscores=True,
        needs_raw_y=False,
        within_block_perm_invariant=True,
        parameter_input="anchor_z",
        seed_offset=11,
        label="linear CS beta-conditioned DeepSets FSM",
    ),
    "radial": ArchSpec(
        input_keys=("s", "anchor_z"),
        needs_subscores=True,
        needs_raw_y=False,
        within_block_perm_invariant=True,
        parameter_input="anchor_z",
        seed_offset=22,
        label="radial CS beta-conditioned gate+DeepSets FSM",
        warm_start_from="linear",
    ),
    "stacked": ArchSpec(
        input_keys=("block", "s", "anchor_z"),
        needs_subscores=True,
        needs_raw_y=False,
        within_block_perm_invariant=True,
        parameter_input="anchor_z",
        seed_offset=33,
        label="stacked NLSA local map",
        warm_start_from="linear",
    ),
}

M_POOL_SCALES = ("none", "block_sd")
M_PARAMETRIZATIONS = ("free", "score_scaled")

SELECTED_GATE_CONDITION_ON_ANCHOR = True
SELECTED_RADIAL_INIT = "matched_random"
SELECTED_GATE_ONLY_STEPS = 0


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


def simulate_mean_shift(
    rng: np.random.Generator,
    u: np.ndarray,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> np.ndarray:
    """Simulate Y_{ki} = eps_{ki} + B_k * tau."""
    u = np.asarray(u, dtype=np.float64)
    pi = sigmoid_np(u)
    active = rng.binomial(1, pi[:, None], size=(u.shape[0], n_blocks)).astype(np.float64)
    eps = rng.normal(size=(u.shape[0], n_blocks, block_size))
    return eps + active[:, :, None] * tau


def full_block_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    """Block likelihood ratio log f1(Y_k) / f0(Y_k)."""
    m = y.shape[-1]
    return tau * y.sum(axis=-1) - 0.5 * m * tau**2


def marginal_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    return tau * y - 0.5 * tau**2


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
    y = simulate_mean_shift(rng, u, n_blocks, block_size, tau)
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


def make_feature_stats(
    y: np.ndarray,
    tau: float,
    feature_pi: np.ndarray | float,
) -> dict[str, np.ndarray]:
    s = score_u_from_log_ratio(marginal_log_ratio(y, tau), feature_pi)
    stats = {
        "s_mean": np.array(float(s.mean()), dtype=np.float64),
        "s_sd": np.array(float(s.std()), dtype=np.float64),
    }
    stats["s_sd"] = np.where(stats["s_sd"] < 1e-8, 1.0, stats["s_sd"])
    s_z = (s - stats["s_mean"]) / stats["s_sd"]
    block_raw = s_z.mean(axis=2, keepdims=True)
    stats["block_mean"] = block_raw.mean(axis=(0, 1), keepdims=True)
    stats["block_sd"] = block_raw.std(axis=(0, 1), keepdims=True)
    stats["block_sd"] = np.where(stats["block_sd"] < 1e-8, 1.0, stats["block_sd"])
    return stats


def featurize(
    y: np.ndarray,
    tau: float,
    feature_pi: np.ndarray | float,
    stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    s = score_u_from_log_ratio(marginal_log_ratio(y, tau), feature_pi)
    s_z = ((s - stats["s_mean"]) / stats["s_sd"]).astype(np.float32)
    block_raw = s_z.mean(axis=2, keepdims=True)
    block = ((block_raw - stats["block_mean"]) / stats["block_sd"]).astype(np.float32)
    return {"s": s_z, "block": block}


def standardize_anchor(
    anchor_u_train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = float(anchor_u_train.mean())
    sd = float(anchor_u_train.std())
    if sd < 1e-8:
        sd = 1.0
    return tuple(((x - mean) / sd).astype(np.float32) for x in (anchor_u_train, *others))


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


class LocalFeatureMLP(nn.Module):
    """Unbounded anchor-conditioned features, initially identically zero.

    ``parametrization`` selects how the learned channel is written:

    ``free``
        ``m(s, a) = MLP(s, a)``, the form in Appendix I.1.
    ``score_scaled``
        ``m(s, a) = s * MLP(s, a)``.

    Both start at ``m == 0`` and so nest ILSA exactly, but they have very
    different gradients there. With mean pooling, ``d(pooled m)/d m_j`` is the
    constant ``1/m`` for every local score in a block, so under ``free`` the
    whole initial learning signal reaching the MLP output layer is one scalar
    per block: the fastest-learned component is an ``s``-independent constant,
    which the readout's biases can already represent. Writing the factor ``s``
    explicitly makes ``d m_j / d MLP_j = s_j``, restoring the ``s``-dependent
    gradient that the multiplicative gate gets for free. It forces
    ``m(0, a) = 0``, which matches the local inverse link ``h_a^{-1}(0) = 0``.

    ``init_std`` > 0 perturbs the output layer instead, trading exact ILSA
    nesting for immediate ``s``-dependence; the realized deviation is measured
    and reported by the caller rather than assumed negligible.
    """

    def __init__(self, hidden: int, m_dim: int = 1, parametrization: str = "free",
                 init_std: float = 0.0):
        super().__init__()
        if m_dim < 1:
            raise ValueError("m_dim must be positive")
        if parametrization not in M_PARAMETRIZATIONS:
            raise ValueError(f"Unknown m parametrization: {parametrization}")
        if init_std < 0:
            raise ValueError("init_std must be non-negative")
        self.parametrization = str(parametrization)
        self.init_std = float(init_std)
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, m_dim),
        )
        if self.init_std > 0:
            nn.init.normal_(self.net[-1].weight, std=self.init_std)
        else:
            nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.stack((s, anchor), dim=-1))
        if self.parametrization == "score_scaled":
            out = out * s.unsqueeze(-1)
        return out


def load_matched_linear_rho(
    stacked_rho: nn.Sequential,
    linear_state: dict[str, torch.Tensor],
    column_map: tuple[int, ...],
    constant_column: int | None = None,
) -> None:
    """Embed a saved linear rho while retaining seeded nonzero m columns.

    ``column_map[j]`` is the destination column of linear input j. Since
    m=0, its ordinary random readout columns preserve ILSA predictions and
    transmit gradients into m. Zeroing BOTH would disable that branch.
    """
    state = stacked_rho.state_dict()
    source = linear_state["0.weight"]
    first = state["0.weight"].clone()
    if (len(column_map) != source.shape[1]
            or len(set(column_map)) != len(column_map)
            or any(i < 0 or i >= first.shape[1] for i in column_map)):
        raise ValueError("column_map must map every linear input to a distinct valid column")
    if constant_column is not None:
        if constant_column in column_map or not 0 <= constant_column < first.shape[1]:
            raise ValueError("constant column must be distinct from mapped columns")
        first[:, constant_column] = 0
    first[:, list(column_map)] = source.to(first)
    for key in state:
        state[key] = first if key == "0.weight" else linear_state[key].clone()
    stacked_rho.load_state_dict(state, strict=True)


def stacked_linear_column_map(
    stacked: "StackedCSBetaDeepSets", linear_constant_channel: bool,
) -> tuple[tuple[int, ...], int | None]:
    """Column map and zeroed constant column for embedding ILSA in the stacked readout.

    ILSA with a constant reads (1, block, anchor) and the stacked map reads
    (1, block, m, anchor), so the constant's weights carry over to column 0.
    Without it, the stacked constant column starts at zero instead.
    """
    block_col, anchor_col = stacked.linear_column_map
    if linear_constant_channel:
        if not stacked.include_constant_channel:
            raise ValueError("ILSA with a constant channel nests only in a stacked map that has one")
        return (0, block_col, anchor_col), None
    return (block_col, anchor_col), (0 if stacked.include_constant_channel else None)


class StackedCSBetaDeepSets(nn.Module):
    """Mean-pool (1, s, m(s,a)), then sum shared block readouts."""

    def __init__(self, hidden: int, depth: int, gate_hidden: int,
                 m_dim: int = 1, include_constant_channel: bool = True,
                 m_pool_scale: str = "none", block_sd: np.ndarray | None = None,
                 m_parametrization: str = "free", m_init_std: float = 0.0):
        super().__init__()
        if m_pool_scale not in M_POOL_SCALES:
            raise ValueError(f"Unknown m_pool_scale: {m_pool_scale}")
        self.include_constant_channel = bool(include_constant_channel)
        self.m_dim = int(m_dim)
        self.m_pool_scale = str(m_pool_scale)
        self.local_features = LocalFeatureMLP(
            gate_hidden, self.m_dim, m_parametrization, m_init_std,
        )
        self.rho = make_rho(int(self.include_constant_channel) + 2 + self.m_dim, hidden, depth)
        # Registered only when used, so that default-configuration checkpoints
        # keep the state_dict they had before this option existed.
        if self.m_pool_scale == "block_sd":
            if block_sd is None:
                raise ValueError("m_pool_scale='block_sd' requires the block_sd feature statistic")
            self.register_buffer(
                "m_pool_divisor", torch.as_tensor(block_sd, dtype=torch.float32)
            )

    @property
    def linear_column_map(self) -> tuple[int, int]:
        offset = int(self.include_constant_channel)
        return offset, offset + 1 + self.m_dim

    def learned_features(self, s: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        anchor = anchor_z.reshape(-1, 1, 1).expand_as(s)
        return self.local_features(s, anchor)

    def readout_inputs(self, block: torch.Tensor, s: torch.Tensor,
                       anchor_z: torch.Tensor) -> torch.Tensor:
        pooled = self.learned_features(s, anchor_z).mean(dim=2)
        if self.m_pool_scale == "block_sd":
            # Put the learned channel on the same footing as the identity
            # channel, which featurize() already divided by this constant.
            # m == 0 at initialization, so this cannot disturb ILSA nesting.
            pooled = pooled / self.m_pool_divisor.reshape(1, 1, -1)
        anchor = anchor_z.reshape(-1, 1, 1).expand(-1, block.shape[1], 1)
        # The identity channel is the exact existing linear input, not a
        # recomputation from local scores. The constant is a pooled mean of 1.
        parts = [torch.ones_like(block)] if self.include_constant_channel else []
        return torch.cat([*parts, block, pooled, anchor], dim=-1)

    def forward(self, block: torch.Tensor, s: torch.Tensor,
                anchor_z: torch.Tensor) -> torch.Tensor:
        return self.rho(self.readout_inputs(block, s, anchor_z)).squeeze(-1).sum(dim=1)


class LinearCSBetaDeepSets(nn.Module):
    """DeepSets readout over block-level linear CS, conditioned on anchor beta.

    With ``include_constant_channel`` the local map is phi(s) = (1, s), the same
    identity-plus-constant channels the stacked NLSA map starts from, so the
    stacked map differs from ILSA only by its learned channel m(s, a).
    """

    def __init__(self, hidden: int, depth: int, include_constant_channel: bool = False):
        super().__init__()
        self.include_constant_channel = bool(include_constant_channel)
        self.rho = make_rho(int(self.include_constant_channel) + 2, hidden, depth)

    def forward(self, block: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        anchor = anchor_z.reshape(-1, 1, 1).expand(-1, block.shape[1], 1)
        parts = [torch.ones_like(block)] if self.include_constant_channel else []
        x = torch.cat([*parts, block, anchor], dim=-1)
        return self.rho(x).squeeze(-1).sum(dim=1)


class RadialCSBetaDeepSets(nn.Module):
    """Signed positive radial-style gates followed by beta-conditioned DeepSets."""

    def __init__(
        self,
        hidden: int,
        depth: int,
        gate_hidden: int,
        gate_condition_on_anchor: bool,
        block_mean: np.ndarray,
        block_sd: np.ndarray,
        include_constant_channel: bool = False,
    ):
        super().__init__()
        self.gate_condition_on_anchor = bool(gate_condition_on_anchor)
        self.include_constant_channel = bool(include_constant_channel)
        gate_input_dim = 2 if self.gate_condition_on_anchor else 1
        self.gate = PositiveMLPMultiplier(gate_hidden, input_dim=gate_input_dim)
        self.rho = make_rho(int(self.include_constant_channel) + 2, hidden, depth)
        self.register_buffer("block_mean", torch.as_tensor(block_mean, dtype=torch.float32))
        self.register_buffer("block_sd", torch.as_tensor(block_sd, dtype=torch.float32))

    def forward(self, s: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        if self.gate_condition_on_anchor:
            anchor_s = anchor_z.reshape(-1, 1, 1).expand_as(s)
            gated_s = s * self.gate(s, anchor_s)
        else:
            gated_s = s * self.gate(s)
        block_raw = gated_s.mean(dim=2, keepdim=True)
        block = (block_raw - self.block_mean) / self.block_sd
        anchor = anchor_z.reshape(-1, 1, 1).expand(-1, block.shape[1], 1)
        parts = [torch.ones_like(block)] if self.include_constant_channel else []
        x = torch.cat([*parts, block, anchor], dim=-1)
        return self.rho(x).squeeze(-1).sum(dim=1)


def build_model(
    method: str,
    config: dict[str, Any],
    stats: dict[str, np.ndarray],
) -> nn.Module:
    # Configs written before the flag existed trained ILSA on s alone.
    linear_constant = bool(config.get("linear_constant_channel", 0))
    if method == "linear":
        return LinearCSBetaDeepSets(
            int(config["hidden"]), int(config["depth"]), include_constant_channel=linear_constant,
        )
    if method == "radial":
        return RadialCSBetaDeepSets(
            int(config["hidden"]),
            int(config["depth"]),
            int(config["gate_hidden"]),
            gate_condition_on_anchor=bool(config.get("gate_condition_on_anchor", 0)),
            block_mean=stats["block_mean"],
            block_sd=stats["block_sd"],
            include_constant_channel=linear_constant,
        )
    if method == "stacked":
        return StackedCSBetaDeepSets(
            int(config["hidden"]), int(config["depth"]), int(config["gate_hidden"]),
            m_dim=int(config.get("m_dim", 1)),
            include_constant_channel=bool(config.get("include_constant_channel", 1)),
            m_pool_scale=str(config.get("m_pool_scale", "none")),
            block_sd=stats["block_sd"],
            m_parametrization=str(config.get("m_parametrization", "free")),
            m_init_std=float(config.get("m_init_std", 0.0)),
        )
    raise ValueError(f"Unknown method: {method}")


def tensors_for_method(data: dict[str, np.ndarray], method: str, device: str) -> dict[str, torch.Tensor]:
    if method not in ARCHITECTURES:
        raise ValueError(f"Unknown method: {method}")
    keys = ARCHITECTURES[method].input_keys
    return {key: torch.as_tensor(data[key], dtype=torch.float32, device=device) for key in keys}


def forward_method(model: nn.Module, batch: dict[str, torch.Tensor], method: str) -> torch.Tensor:
    if method not in ARCHITECTURES:
        raise ValueError(f"Unknown method: {method}")
    return model(*(batch[key] for key in ARCHITECTURES[method].input_keys))


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
    device: str,
    name: str,
    checkpoint_selection: str = "raw_or_ema",
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    if checkpoint_selection not in {"raw", "raw_or_ema"}:
        raise ValueError("unknown checkpoint selection policy")
    if lr_schedule == "cosine_tail" and not (0 <= lr_decay_start_step < iters):
        raise ValueError(
            "cosine_tail requires 0 <= lr_decay_start_step < the method iteration budget"
        )
    model = model.to(device)
    x_train = tensors_for_method(train, method, device)
    x_val = tensors_for_method(val, method, device)
    y_train = torch.as_tensor(train["target"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val["target"], dtype=torch.float32, device=device)
    if method in {"radial", "stacked"}:
        gate_params = list((model.local_features if method == "stacked" else model.gate).parameters())
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
        if method in {"radial", "stacked"}:
            opt.param_groups[0]["lr"] = base_lrs["gate"] * multiplier
            opt.param_groups[1]["lr"] = base_lrs["rho"] * multiplier
        else:
            opt.param_groups[0]["lr"] = base_lrs["model"] * multiplier
        return multiplier

    def trace_learning_rates() -> dict[str, float | str]:
        if method in {"radial", "stacked"}:
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

    @torch.no_grad()
    def feature_diagnostics() -> dict[str, float]:
        if method != "stacked":
            return {}
        # Fixed, target-free validation subset; diagnostics never select weights.
        m = model.learned_features(x_val["s"][:256], x_val["anchor_z"][:256])
        pooled = m.mean(dim=2)
        return {"m_max_abs": float(m.abs().max()),
                "pooled_m_rms": float(pooled.square().mean().sqrt()),
                "pooled_m_sd": float(pooled.std(unbiased=False))}

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
            **feature_diagnostics(),
        }
    ]
    bad = 0
    n_train = int(y_train.shape[0])
    print(f"{name} step 0/{iters}: val={best_val:.6g}")

    for step in range(1, iters + 1):
        lr_multiplier = set_step_learning_rates(step)
        indices = rng.integers(0, n_train, size=min(batch_size, n_train))
        minibatch_digest.update(indices.tobytes())
        idx = torch.as_tensor(
            indices,
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
        if step == 1 or step % print_every == 0 or step == iters:
            val_loss_raw = mse_over_tensors(model, x_val, y_val, method, batch_size)
            raw_state = clone_state_dict(model, cpu=False)
            model.load_state_dict(ema_state)
            val_loss_ema = mse_over_tensors(model, x_val, y_val, method, batch_size)
            model.load_state_dict(raw_state)
            if not np.isfinite(val_loss_raw) or not np.isfinite(val_loss_ema):
                raise RuntimeError(f"{name} produced non-finite validation loss at step {step}")
            if checkpoint_selection == "raw_or_ema" and val_loss_ema < val_loss_raw:
                selected_val = val_loss_ema
                selected_source = "ema"
                selected_state = {key: value.detach().cpu().clone() for key, value in ema_state.items()}
            else:
                selected_val = val_loss_raw
                selected_source = "raw"
                selected_state = clone_state_dict(model, cpu=True)
            trace.append(
                {
                    "method": name,
                    "step": step,
                    "stage": "joint",
                    "train_loss": float(loss.detach().cpu().item()),
                    "val_loss_raw": float(val_loss_raw),
                    "val_loss_ema": float(val_loss_ema),
                    "best_val_loss": float(min(best_val, selected_val)),
                    "best_source": selected_source if selected_val < best_val else best_source,
                    "grad_norm": float(grad_norm.detach().cpu().item()),
                    **trace_learning_rates(),
                    **feature_diagnostics(),
                }
            )
            lr_values = "/".join(f"{group['lr']:.3g}" for group in opt.param_groups)
            print(
                f"{name} step {step}/{iters} [joint]: train={loss.item():.6g}, "
                f"val_raw={val_loss_raw:.6g}, val_ema={val_loss_ema:.6g}, "
                f"best={min(best_val, selected_val):.6g}, lr={lr_values}, "
                f"lr_mult={lr_multiplier:.4f}"
            )
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
        "checkpoint_selection": checkpoint_selection,
        "minibatch_sha256": minibatch_digest.hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, trace, info


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
    *,
    need_subscores: bool = True,
    need_raw_y: bool = False,
) -> dict[str, np.ndarray]:
    feat: dict[str, np.ndarray] = {}
    if need_subscores:
        feat.update(featurize(raw["y"], tau, raw["anchor_pi"], stats))
    if need_raw_y:
        feat["y"] = raw["y"].astype(np.float32)
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
    *,
    need_subscores: bool = True,
    need_raw_y: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    u = np.full(n, float(logit_np(pi)), dtype=np.float64)
    y = simulate_mean_shift(rng, u, n_blocks, block_size, tau)
    feat: dict[str, np.ndarray] = {}
    if need_subscores:
        feature_pi: np.ndarray | float = np.full(n, pi, dtype=np.float64)
        feat.update(featurize(y, tau, feature_pi, stats))
    if need_raw_y:
        feat["y"] = y.astype(np.float32)
    feat["anchor_z"] = ((u - anchor_mean) / anchor_sd).astype(np.float32)
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
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.m_dim < 1:
        raise ValueError("--m-dim must be positive")
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = sorted(set(methods) - set(ARCHITECTURES))
    if bad:
        raise ValueError(f"Unknown methods: {bad}")
    need_subscores = any(ARCHITECTURES[method].needs_subscores for method in methods)
    need_raw_y = any(ARCHITECTURES[method].needs_raw_y for method in methods)
    if args.linear_constant_channel and "stacked" in methods and not args.include_constant_channel:
        raise ValueError("--linear-constant-channel 1 needs --include-constant-channel 1 for stacked")

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
            "iters": args.iters,
            "methods": args.methods,
        },
    )

    rng = np.random.default_rng(args.seed)
    print("\nGenerating amortized FSM data...")
    train_raw = sample_anchor_batch(
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
    print("train y:", train_raw["y"].shape, "val y:", val_raw["y"].shape)

    print("Computing anchor/beta-conditioned local CS features...")
    stats = make_feature_stats(train_raw["y"], args.tau, train_raw["anchor_pi"])
    anchor_z_train, anchor_z_val = standardize_anchor(train_raw["anchor_u"], val_raw["anchor_u"])
    anchor_mean = float(train_raw["anchor_u"].mean())
    anchor_sd = float(train_raw["anchor_u"].std())
    if anchor_sd < 1e-8:
        anchor_sd = 1.0
    train = prepare_split(
        train_raw,
        args.tau,
        stats,
        anchor_z_train,
        need_subscores=need_subscores,
        need_raw_y=need_raw_y,
    )
    val = prepare_split(
        val_raw,
        args.tau,
        stats,
        anchor_z_val,
        need_subscores=need_subscores,
        need_raw_y=need_raw_y,
    )
    if need_subscores:
        print("linear block features:", train["block"].shape)
        print("local marginal subscores:", train["s"].shape)
    if need_raw_y:
        print("raw observations:", train["y"].shape)

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
    config["exact_score_available_to_checkpoint_selection"] = False
    config["software"] = {"torch": str(torch.__version__), "numpy": np.__version__}
    config["shared_data_sha256"] = {
        f"{split}_{key}": hashlib.sha256(np.ascontiguousarray(raw[key]).tobytes()).hexdigest()
        for split, raw in (("train", train_raw), ("validation", val_raw))
        for key in ("y", "anchor_u", "target")
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "feature_stats.npz",
        **stats,
        anchor_u_mean=np.array(anchor_mean),
        anchor_u_sd=np.array(anchor_sd),
    )

    all_trace: list[dict[str, Any]] = []
    training_info: dict[str, Any] = {}
    linear_initial_rho_state: dict[str, torch.Tensor] | None = None
    linear_initial_check: np.ndarray | None = None
    check_n = min(256, int(train["target"].shape[0]))
    check_data = {key: value[:check_n] for key, value in train.items()}
    needs_linear_init = any(ARCHITECTURES[m].warm_start_from == "linear" for m in methods)
    if needs_linear_init and "linear" not in methods:
        set_global_seed(args.seed + ARCHITECTURES["linear"].seed_offset)
        initial_linear_reference = build_model("linear", config, stats)
        linear_initial_rho_state = clone_state_dict(initial_linear_reference.rho, cpu=True)
        initial_linear_reference = initial_linear_reference.to(device)
        linear_initial_check = predict(
            initial_linear_reference,
            check_data,
            "linear",
            device,
            args.batch_size,
        )
    for method in (candidate for candidate in ARCHITECTURES if candidate in methods):
        spec = ARCHITECTURES[method]
        if method == "linear":
            label = spec.label
            set_global_seed(args.seed + spec.seed_offset)
            model = build_model(method, config, stats)
            if needs_linear_init:
                linear_initial_rho_state = clone_state_dict(model.rho, cpu=True)
                model = model.to(device)
                linear_initial_check = predict(
                    model,
                    check_data,
                    "linear",
                    device,
                    args.batch_size,
                )
        elif method in {"radial", "stacked"}:
            label = spec.label
            set_global_seed(args.seed + spec.seed_offset)
            model = build_model(method, config, stats)
            nesting_error: float | None = None
            if linear_initial_rho_state is None or linear_initial_check is None:
                raise RuntimeError("matched_random requires the saved initial linear rho")
            if method == "stacked":
                column_map, constant_column = stacked_linear_column_map(
                    model, bool(args.linear_constant_channel),
                )
                load_matched_linear_rho(
                    model.rho, linear_initial_rho_state, column_map,
                    constant_column=constant_column,
                )
            else:
                model.rho.load_state_dict(linear_initial_rho_state)
            model = model.to(device)
            radial_check = predict(model, check_data, method, device, args.batch_size)
            nesting_error = float(np.max(np.abs(linear_initial_check - radial_check)))
            print(f"matched random initialization max abs diff: {nesting_error:.3e}")
            # --m-init-std deliberately perturbs the learned channel, so exact
            # nesting is not expected there; the deviation is reported instead.
            approximate_nesting = method == "stacked" and args.m_init_std > 0
            if nesting_error > 1e-5 and not approximate_nesting:
                raise RuntimeError(
                    f"matched_random {method} initialization does not reproduce the "
                    f"untrained linear reference: {nesting_error:.3e}"
                )
            label = f"{label} [matched_random]"
            print(f"{method} initialization: matched untrained linear predictions")
        else:
            label = spec.label
            set_global_seed(args.seed + spec.seed_offset)
            model = build_model(method, config, stats)
        print(f"\nTraining {label}...")
        model, trace, info = train_model(
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
            device=device,
            name=label,
            checkpoint_selection=args.checkpoint_selection,
        )
        info["method"] = method
        info["label"] = label
        if method in {"radial", "stacked"}:
            info["initialization"] = SELECTED_RADIAL_INIT
            if nesting_error is not None:
                info["nested_initialization_max_abs_diff"] = nesting_error
        if method == "stacked":
            info["learned_readout_initialization"] = "nonzero_seeded_linear_default"
            info["m_parametrization"] = args.m_parametrization
            info["m_init_std"] = float(args.m_init_std)
            info["m_pool_scale"] = args.m_pool_scale
            info["ilsa_nesting"] = "approximate" if args.m_init_std > 0 else "exact"
        training_info[method] = info
        torch.save(
            {
                "state_dict": model.state_dict(),
                "method": method,
                "label": label,
                "config": config,
                "training_info": info,
                "input_mode": "raw" if spec.needs_raw_y else "subscore",
            },
            out_dir / f"model_{method}.pt",
        )
        all_trace.extend(trace)
        # Preserve finished methods and traces even if a later method fails.
        write_csv(out_dir / "training_trace.csv", all_trace)
        (out_dir / "training_info.json").write_text(json.dumps(training_info, indent=2), encoding="utf-8")
    write_csv(out_dir / "training_trace.csv", all_trace)
    (out_dir / "training_info.json").write_text(json.dumps(training_info, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(out_dir / "config.json")
    print(out_dir / "feature_stats.npz")
    print(out_dir / "training_trace.csv")
    print(out_dir / "training_info.json")
    for method in methods:
        print(out_dir / f"model_{method}.pt")
    print("Exact-score evaluation is a separate command: python -m model1.evaluate_stage1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=40_000)
    parser.add_argument("--n-val", type=int, default=8_000)
    parser.add_argument("--n-blocks", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--anchor-pi-min", type=float, default=0.05)
    parser.add_argument("--anchor-pi-max", type=float, default=0.7)
    parser.add_argument("--sigma-q", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--methods", type=str, default="linear,radial")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--m-dim", type=int, default=1)
    parser.add_argument("--include-constant-channel", type=int, choices=(0, 1), default=1)
    parser.add_argument("--linear-constant-channel", type=int, choices=(0, 1), default=1,
                        help="ILSA local map (1, s) when 1, s alone when 0. The gate uses the "
                             "same readout inputs so it still starts exactly at ILSA. Use 0 to "
                             "reproduce runs made before this option existed.")
    parser.add_argument("--m-parametrization", choices=M_PARAMETRIZATIONS, default="free",
                        help="'free' is m=MLP(s,a); 'score_scaled' is m=s*MLP(s,a), which "
                             "keeps exact ILSA nesting but restores s-dependent gradients.")
    parser.add_argument("--m-init-std", type=float, default=0.0,
                        help="Output-layer init SD for the learned channel. Above zero this "
                             "gives up exact ILSA nesting; the deviation is reported.")
    parser.add_argument("--m-pool-scale", choices=M_POOL_SCALES, default="none",
                        help="Divisor for the pooled learned channel; 'block_sd' reuses "
                             "the identity channel's standardizing constant.")
    parser.add_argument("--checkpoint-selection", choices=("raw_or_ema", "raw"),
                        default="raw_or_ema",
                        help="Legacy raw/EMA selection, or explicitly validation-best raw.")
    parser.add_argument("--iters", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4, help="Linear-model learning rate.")
    parser.add_argument("--gate-lr", type=float, default=1e-4)
    parser.add_argument("--joint-rho-lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine_tail",
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
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/blockwise_mean_shift_amortized_fsm",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
