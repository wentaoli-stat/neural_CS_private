#!/usr/bin/env python3
"""Likelihood-free FSM training for the blockwise common-factor variance mixture.

Model:
    Z_k iid Bernoulli(pi)
    Y_.k | Z_k=0 ~ N(0, I_n)
    Y_.k | Z_k=1 ~ N(0, I_n + tau^2 11^T)

Only pi is unknown. Training is non-oracle Direct FSM:
    u ~ N(u0, sigma_q^2)
    Y ~ p_{pi(u)}
    S(feature(Y; u0)) is regressed to (u-u0)/sigma_q^2.

The exact full score is used only for post-training diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def score_u_from_log_ratio(log_ratio: np.ndarray, pi: float) -> np.ndarray:
    w = sigmoid_np(logit(pi) + log_ratio)
    score_pi = (w - pi) / (pi * (1.0 - pi))
    return score_pi * pi * (1.0 - pi)


def log_ratio_from_score_u(score_u: np.ndarray, pi: float, eps: float = 1e-6) -> np.ndarray:
    """Invert a local iid-mixture u-score into local log-likelihood-ratio evidence."""
    posterior_prob = np.clip(pi + np.asarray(score_u, dtype=np.float64), eps, 1.0 - eps)
    return np.log(posterior_prob) - np.log1p(-posterior_prob) - logit(pi)


def pi_from_u(u: np.ndarray) -> np.ndarray:
    return sigmoid_np(u)


def simulate_from_u(
    rng: np.random.Generator,
    u: np.ndarray,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> np.ndarray:
    pi = pi_from_u(u)
    z = rng.binomial(1, pi[:, None], size=(u.shape[0], n_blocks)).astype(np.float64)
    eps = rng.normal(size=(u.shape[0], n_blocks, block_size))
    factor = rng.normal(scale=tau, size=(u.shape[0], n_blocks, 1))
    return eps + z[:, :, None] * factor


def make_fsm_data(
    rng: np.random.Generator,
    n: int,
    u0: float,
    sigma_q: float,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.normal(loc=u0, scale=sigma_q, size=n)
    y = simulate_from_u(rng, u, n_blocks, block_size, tau)
    target = (u - u0) / (sigma_q**2)
    return y, target


def simulate_center(
    rng: np.random.Generator,
    n: int,
    pi0: float,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> np.ndarray:
    return simulate_from_u(rng, np.full(n, logit(pi0), dtype=np.float64), n_blocks, block_size, tau)


def full_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    n = y.shape[2]
    block_sum = y.sum(axis=2)
    return -0.5 * math.log1p(n * tau**2) + (tau**2 / (2.0 * (1.0 + n * tau**2))) * block_sum**2


def marginal_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    return -0.5 * math.log1p(tau**2) + (tau**2 / (2.0 * (1.0 + tau**2))) * y**2


def pairwise_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    n = y.shape[2]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_sum = y[:, :, i] + y[:, :, j]
            pairs.append(
                -0.5 * math.log1p(2.0 * tau**2)
                + (tau**2 / (2.0 * (1.0 + 2.0 * tau**2))) * pair_sum**2
            )
    return np.stack(pairs, axis=2)


def recover_block_log_ratio_from_marginal_pairwise_evidence(
    marginal_lr_sum: np.ndarray,
    pairwise_lr_sum: np.ndarray,
    tau: float,
    block_size: int,
) -> np.ndarray:
    """Recover common-factor block evidence from marginal and pairwise evidences.

    For a block of size m, the pairwise evidence contains
    sum_{i<j}(y_i+y_j)^2 = (m-2) sum_i y_i^2 + (sum_i y_i)^2.
    The marginal evidence recovers sum_i y_i^2, so the two together recover the
    full block log-likelihood ratio under the common-factor covariance model.
    """
    if tau <= 0:
        raise ValueError("tau must be positive to recover block evidence from local evidences")
    m = int(block_size)
    n_pairs = m * (m - 1) // 2
    a1 = -0.5 * math.log1p(tau**2)
    b1 = tau**2 / (2.0 * (1.0 + tau**2))
    a2 = -0.5 * math.log1p(2.0 * tau**2)
    b2 = tau**2 / (2.0 * (1.0 + 2.0 * tau**2))
    afull = -0.5 * math.log1p(m * tau**2)
    bfull = tau**2 / (2.0 * (1.0 + m * tau**2))

    sum_y2 = (marginal_lr_sum - m * a1) / b1
    pair_quad = (pairwise_lr_sum - n_pairs * a2) / b2
    block_sum_sq = pair_quad - (m - 2) * sum_y2
    return afull + bfull * block_sum_sq


def standardize_train_eval(
    x_train: np.ndarray,
    *others: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, ...]:
    mean = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return tuple(((x - mean) / sd).astype(np.float32) for x in (x_train, *others))


def standardize_scalar(
    x_train: np.ndarray,
    *others: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, ...]:
    mean = float(x_train.mean())
    sd = float(x_train.std())
    if sd < eps:
        sd = 1.0
    return tuple(((x - mean) / sd).astype(np.float32) for x in (x_train, *others))


def fit_linear_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
    y = np.asarray(y, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("fit_linear_ridge received non-finite x")
    if not np.all(np.isfinite(y)):
        raise ValueError("fit_linear_ridge received non-finite y")
    design = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    gram = np.einsum("ni,nj->ij", design, design)
    gram[1:, 1:] += ridge * np.eye(gram.shape[0] - 1)
    rhs = np.einsum("ni,n->i", design, y)
    return np.linalg.solve(gram, rhs)


def predict_linear(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    coef = np.asarray(coef, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64).reshape(x.shape[0], -1)
    if not np.all(np.isfinite(coef)):
        raise ValueError("predict_linear received non-finite coef")
    if not np.all(np.isfinite(x)):
        raise ValueError("predict_linear received non-finite x")
    design = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return design @ coef


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PositiveMonotoneMultiplier(nn.Module):
    """Positive monotone scalar map m(r), initialized near m(r)=1."""

    def __init__(self, hidden: int):
        super().__init__()
        self.raw_w1 = nn.Parameter(torch.zeros(hidden))
        self.b1 = nn.Parameter(torch.linspace(-1.0, 1.0, hidden))
        self.raw_w2 = nn.Parameter(torch.full((hidden,), -10.0))
        self.bias = nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float32))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        w1 = F.softplus(self.raw_w1)
        w2 = F.softplus(self.raw_w2)
        h = F.softplus(r * w1 + self.b1)
        z = self.bias + torch.sum(h * w2, dim=-1)
        return F.softplus(z)


class PositiveMLPMultiplier(nn.Module):
    """Positive scalar map m(r) without monotonicity constraints, initialized near m(r)=1."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, float(math.log(math.expm1(1.0))))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(r).squeeze(-1))


class PositiveLogPolynomialMultiplier(nn.Module):
    """Positive polynomial gate, initialized at m(r)=1.

    With nonnegative radial inputs this uses log(1+r). With signed inputs it
    uses asinh(r), which is defined on the whole real line and avoids silent
    NaNs from log1p(r) when r <= -1.
    """

    def __init__(self, degree: int, signed_basis: bool = False):
        super().__init__()
        if degree < 0:
            raise ValueError("degree must be non-negative")
        self.signed_basis = bool(signed_basis)
        coeff = torch.zeros(degree + 1, dtype=torch.float32)
        coeff[0] = float(math.log(math.expm1(1.0)))
        self.coeff = nn.Parameter(coeff)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r.squeeze(-1)
        z = torch.asinh(x) if self.signed_basis else torch.log1p(x)
        powers = [torch.ones_like(z)]
        for _ in range(1, self.coeff.numel()):
            powers.append(powers[-1] * z)
        poly = torch.stack(powers, dim=-1) @ self.coeff
        return F.softplus(poly)


def make_multiplier(kind: str, hidden: int, degree: int, gate_input: str = "abs") -> nn.Module:
    if kind == "monotone_mlp":
        return PositiveMonotoneMultiplier(hidden)
    if kind == "positive_mlp":
        return PositiveMLPMultiplier(hidden)
    if kind == "log_polynomial":
        return PositiveLogPolynomialMultiplier(degree, signed_basis=(gate_input == "signed"))
    raise ValueError(f"Unknown gate kind: {kind}")


def gate_input_tensor(x: torch.Tensor, gate_input: str) -> torch.Tensor:
    if gate_input == "abs":
        return torch.abs(x)
    if gate_input == "signed":
        return x
    raise ValueError(f"Unknown gate input: {gate_input}")


def gate_label_for(kind: str) -> str:
    if kind == "monotone_mlp":
        return "MLP"
    if kind == "positive_mlp":
        return "positive-MLP"
    if kind == "log_polynomial":
        return "polynomial"
    return str(kind)


class RadialGateMarginalPairwiseCS(nn.Module):
    """Structure-preserving s -> s*m(g(s)) map with a linear CS head."""

    def __init__(
        self,
        n_blocks: int,
        block_size: int,
        hidden: int,
        linear_coef: np.ndarray,
        gate_kind: str,
        poly_degree: int,
        gate_input: str = "abs",
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_pairs = block_size * (block_size - 1) // 2
        coef = np.asarray(linear_coef, dtype=np.float32)
        if coef.shape != (1 + 2 * n_blocks,):
            raise ValueError(f"linear_coef must have shape {(1 + 2 * n_blocks,)}, got {coef.shape}")
        self.intercept = nn.Parameter(torch.tensor(coef[0], dtype=torch.float32))
        self.weights = nn.Parameter(torch.tensor(coef[1:], dtype=torch.float32))
        self.marginal_gate = make_multiplier(gate_kind, hidden, poly_degree, gate_input)
        self.pairwise_gate = make_multiplier(gate_kind, hidden, poly_degree, gate_input)
        self.gate_input = gate_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        gated_s1 = s1 * self.marginal_gate(gate_input_tensor(s1, self.gate_input).unsqueeze(-1))
        gated_s2 = s2 * self.pairwise_gate(gate_input_tensor(s2, self.gate_input).unsqueeze(-1))
        block_marg = gated_s1.sum(dim=2)
        block_pair = gated_s2.sum(dim=2)
        features = torch.cat([block_marg, block_pair], dim=1)
        return self.intercept + features @ self.weights


class RadialGateMarginalPairwiseCSMLPHead(nn.Module):
    """Structure-preserving marginal/pairwise gates followed by the same MLP head."""

    def __init__(
        self,
        n_blocks: int,
        block_size: int,
        hidden: int,
        head_hidden: int,
        head_depth: int,
        feature_mean: np.ndarray,
        feature_sd: np.ndarray,
        gate_kind: str,
        poly_degree: int,
        gate_input: str = "abs",
    ):
        super().__init__()
        self.block_size = block_size
        self.n_pairs = block_size * (block_size - 1) // 2
        self.marginal_gate = make_multiplier(gate_kind, hidden, poly_degree, gate_input)
        self.pairwise_gate = make_multiplier(gate_kind, hidden, poly_degree, gate_input)
        self.head = MLP(2 * n_blocks, head_hidden, head_depth)
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_sd", torch.as_tensor(feature_sd, dtype=torch.float32))
        self.gate_input = gate_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        gated_s1 = s1 * self.marginal_gate(gate_input_tensor(s1, self.gate_input).unsqueeze(-1))
        gated_s2 = s2 * self.pairwise_gate(gate_input_tensor(s2, self.gate_input).unsqueeze(-1))
        block_marg = gated_s1.sum(dim=2)
        block_pair = gated_s2.sum(dim=2)
        features = torch.cat([block_marg, block_pair], dim=1)
        features = (features - self.feature_mean) / self.feature_sd
        return self.head(features)


class RadialGateMarginalPairwiseDeepSets(nn.Module):
    """Structure-preserving marginal/pairwise radial gates followed by shared block rho."""

    def __init__(
        self,
        block_size: int,
        hidden: int,
        gate_hidden: int,
        gate_kind: str,
        poly_degree: int,
        gate_input: str = "abs",
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.n_pairs = self.block_size * (self.block_size - 1) // 2
        self.marginal_gate = make_multiplier(gate_kind, gate_hidden, poly_degree, gate_input)
        self.pairwise_gate = make_multiplier(gate_kind, gate_hidden, poly_degree, gate_input)
        self.gate_input = gate_input
        self.rho = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        gated_s1 = s1 * self.marginal_gate(gate_input_tensor(s1, self.gate_input).unsqueeze(-1))
        gated_s2 = s2 * self.pairwise_gate(gate_input_tensor(s2, self.gate_input).unsqueeze(-1))
        block_features = torch.stack([gated_s1.sum(dim=2), gated_s2.sum(dim=2)], dim=-1)
        return self.rho(block_features).squeeze(-1).sum(dim=1)


class LinearCSMarginalOnlyDeepSets(nn.Module):
    """DeepSets readout applied only to marginal block CS features."""

    def __init__(self, block_size: int, hidden: int):
        super().__init__()
        self.block_size = int(block_size)
        self.rho = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs); ignore pairwise scores.
        s1 = x[:, :, : self.block_size]
        block_scores = s1.sum(dim=2, keepdim=True)
        return self.rho(block_scores).squeeze(-1).sum(dim=1)


class RadialGateMarginalOnlyDeepSets(nn.Module):
    """Marginal-only radial gate followed by shared block rho."""

    def __init__(
        self,
        block_size: int,
        hidden: int,
        gate_hidden: int,
        gate_kind: str,
        poly_degree: int,
        gate_input: str = "abs",
    ):
        super().__init__()
        self.block_size = int(block_size)
        self.marginal_gate = make_multiplier(gate_kind, gate_hidden, poly_degree, gate_input)
        self.gate_input = gate_input
        self.rho = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs); ignore pairwise scores.
        s1 = x[:, :, : self.block_size]
        gated_s1 = s1 * self.marginal_gate(gate_input_tensor(s1, self.gate_input).unsqueeze(-1))
        block_features = gated_s1.sum(dim=2, keepdim=True)
        return self.rho(block_features).squeeze(-1).sum(dim=1)


class LearnedPhiVectorMarginalPairwiseCS(nn.Module):
    """Block vector E_k=(sum_i phi1(s1_ki), sum_{i<j} phi2(s2_kij)) with linear head.

    Both phi networks are residual maps initialized at identity, so the model
    starts exactly at the linear marginal+pairwise block-CS ridge baseline.
    """

    def __init__(self, n_blocks: int, block_size: int, hidden: int, linear_coef: np.ndarray):
        super().__init__()
        self.block_size = int(block_size)
        self.n_pairs = self.block_size * (self.block_size - 1) // 2
        coef = np.asarray(linear_coef, dtype=np.float32)
        if coef.shape != (1 + 2 * n_blocks,):
            raise ValueError(f"linear_coef must have shape {(1 + 2 * n_blocks,)}, got {coef.shape}")
        self.intercept = nn.Parameter(torch.tensor(coef[0], dtype=torch.float32))
        self.weights = nn.Parameter(torch.tensor(coef[1:], dtype=torch.float32))
        self.phi1_residual = self._make_residual(hidden)
        self.phi2_residual = self._make_residual(hidden)

    @staticmethod
    def _make_residual(hidden: int) -> nn.Sequential:
        net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        return net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        phi1 = s1 + self.phi1_residual(s1.unsqueeze(-1)).squeeze(-1)
        phi2 = s2 + self.phi2_residual(s2.unsqueeze(-1)).squeeze(-1)
        block_marg = phi1.sum(dim=2)
        block_pair = phi2.sum(dim=2)
        features = torch.cat([block_marg, block_pair], dim=1)
        return self.intercept + features @ self.weights


class OddMonotoneResidualMap(nn.Module):
    """Odd, sign-preserving monotone scalar map initialized at identity."""

    def __init__(self, n_basis: int):
        super().__init__()
        if n_basis <= 0:
            raise ValueError("n_basis must be positive")
        self.raw_slope = nn.Parameter(torch.tensor(math.log(math.expm1(1.0)), dtype=torch.float32))
        self.raw_coeff = nn.Parameter(torch.full((n_basis,), -12.0))
        self.raw_rate = nn.Parameter(torch.zeros(n_basis))
        self.bias = nn.Parameter(torch.linspace(-2.0, 2.0, n_basis))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        r = torch.abs(s)
        slope = F.softplus(self.raw_slope) + 1e-6
        coeff = F.softplus(self.raw_coeff)
        rate = F.softplus(self.raw_rate) + 1e-6
        basis = F.softplus(r.unsqueeze(-1) * rate + self.bias) - F.softplus(self.bias)
        h = slope * r + basis @ coeff
        return torch.sign(s) * h


class OddMonotoneVectorMarginalPairwiseCS(nn.Module):
    """Block vector with odd monotone marginal and pairwise subscore maps."""

    def __init__(self, n_blocks: int, block_size: int, n_basis: int, linear_coef: np.ndarray):
        super().__init__()
        self.block_size = int(block_size)
        self.n_pairs = self.block_size * (self.block_size - 1) // 2
        coef = np.asarray(linear_coef, dtype=np.float32)
        if coef.shape != (1 + 2 * n_blocks,):
            raise ValueError(f"linear_coef must have shape {(1 + 2 * n_blocks,)}, got {coef.shape}")
        self.intercept = nn.Parameter(torch.tensor(coef[0], dtype=torch.float32))
        self.weights = nn.Parameter(torch.tensor(coef[1:], dtype=torch.float32))
        self.phi1 = OddMonotoneResidualMap(n_basis)
        self.phi2 = OddMonotoneResidualMap(n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        block_marg = self.phi1(s1).sum(dim=2)
        block_pair = self.phi2(s2).sum(dim=2)
        features = torch.cat([block_marg, block_pair], dim=1)
        return self.intercept + features @ self.weights


class PhiEmbeddingVectorMarginalPairwiseCS(nn.Module):
    """Generic neural phi maps over marginal/pairwise subscores with a linear head."""

    def __init__(self, n_blocks: int, block_size: int, hidden: int, phi_dim: int):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.block_size = int(block_size)
        self.n_pairs = self.block_size * (self.block_size - 1) // 2
        self.phi_dim = int(phi_dim)
        self.phi1 = self._make_phi(hidden, self.phi_dim)
        self.phi2 = self._make_phi(hidden, self.phi_dim)
        self.head = nn.Linear(2 * self.n_blocks * self.phi_dim, 1)

    @staticmethod
    def _make_phi(hidden: int, phi_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, phi_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        e1 = self.phi1(s1.unsqueeze(-1)).sum(dim=2)
        e2 = self.phi2(s2.unsqueeze(-1)).sum(dim=2)
        features = torch.cat(
            [
                e1.reshape(x.shape[0], self.n_blocks * self.phi_dim),
                e2.reshape(x.shape[0], self.n_blocks * self.phi_dim),
            ],
            dim=1,
        )
        return self.head(features).squeeze(-1)


class StructuredMarginalPairwiseDeepSets(nn.Module):
    """Blockwise DeepSets over marginal and same-block pairwise subscores."""

    def __init__(self, block_size: int, hidden: int):
        super().__init__()
        self.block_size = block_size
        self.n_pairs = block_size * (block_size - 1) // 2
        self.phi1 = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.phi2 = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        e1 = self.phi1(s1.unsqueeze(-1)).sum(dim=2)
        e2 = self.phi2(s2.unsqueeze(-1)).sum(dim=2)
        block_score = self.rho(torch.cat([e1, e2], dim=-1)).squeeze(-1)
        return block_score.sum(dim=1)


class LinearCSMarginalPairwiseDeepSets(nn.Module):
    """DeepSets readout applied to linear marginal/pairwise block CS features."""

    def __init__(self, block_size: int, hidden: int):
        super().__init__()
        self.block_size = int(block_size)
        self.rho = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        block_scores = torch.stack([s1.sum(dim=2), s2.sum(dim=2)], dim=-1)
        return self.rho(block_scores).squeeze(-1).sum(dim=1)


class NestedStructuredMarginalPairwiseDeepSets(nn.Module):
    """Residual DeepSets that starts from the linear marginal+pairwise CS baseline."""

    def __init__(
        self,
        n_blocks: int,
        block_size: int,
        hidden: int,
        linear_coef: np.ndarray,
        residual_scale_init: float = 0.0,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_pairs = block_size * (block_size - 1) // 2
        coef = np.asarray(linear_coef, dtype=np.float32)
        if coef.shape != (1 + 2 * n_blocks,):
            raise ValueError(f"linear_coef must have shape {(1 + 2 * n_blocks,)}, got {coef.shape}")
        self.register_buffer("linear_intercept", torch.tensor(coef[0], dtype=torch.float32))
        self.register_buffer("linear_weights", torch.tensor(coef[1:], dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))
        self.phi1 = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.phi2 = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        block_marg = s1.sum(dim=2)
        block_pair = s2.sum(dim=2)
        linear_x = torch.cat([block_marg, block_pair], dim=1)
        linear_score = self.linear_intercept + linear_x @ self.linear_weights

        # Mean aggregation keeps the nonlinear residual scale stable as block_size changes.
        e1 = self.phi1(s1.unsqueeze(-1)).mean(dim=2)
        e2 = self.phi2(s2.unsqueeze(-1)).mean(dim=2)
        residual_blocks = self.rho(torch.cat([e1, e2], dim=-1)).squeeze(-1)
        residual_score = residual_blocks.sum(dim=1)
        return linear_score + self.residual_scale * residual_score


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    iters: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    print_every: int,
    device: str,
    name: str,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state = None
    bad = 0
    trace: list[dict[str, float]] = []
    step = 0

    def validation_loss() -> float:
        sqerr = 0.0
        n_total = 0
        model.eval()
        with torch.no_grad():
            for start in range(0, x_val_t.shape[0], batch_size):
                xb = x_val_t[start : start + batch_size].to(device)
                yb = y_val_t[start : start + batch_size].to(device)
                err = model(xb) - yb
                sqerr += float(torch.sum(err**2).item())
                n_total += int(yb.numel())
        model.train()
        return sqerr / max(n_total, 1)

    initial_val_loss = validation_loss()
    if not np.isfinite(initial_val_loss):
        raise RuntimeError(f"{name} produced non-finite initial validation loss: {initial_val_loss}")
    best_loss = initial_val_loss
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    trace.append({"step": 0.0, "train_loss": initial_val_loss, "val_loss": initial_val_loss})
    print(f"{name} step 0/{iters}: train={initial_val_loss:.4g}, val={initial_val_loss:.4g}")

    while step < iters:
        for xb, yb in loader:
            step += 1
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} produced non-finite training loss at step {step}: {loss.item()}")
            loss.backward()
            opt.step()
            if step == 1 or step % print_every == 0 or step == iters:
                val_loss = validation_loss()
                if not np.isfinite(val_loss):
                    raise RuntimeError(f"{name} produced non-finite validation loss at step {step}: {val_loss}")
                trace.append({"step": float(step), "train_loss": float(loss.item()), "val_loss": val_loss})
                print(f"{name} step {step}/{iters}: train={loss.item():.4g}, val={val_loss:.4g}")
                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if patience > 0 and bad >= patience:
                    step = iters
                    break
            if step >= iters:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, trace


def predict_torch(model: nn.Module, x: np.ndarray, device: str, batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    var = float(np.var(y_true) + 1e-12)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 1e-12 else float("nan")
    denom = np.linalg.norm(y_true) * np.linalg.norm(y_pred)
    cosine = float(np.dot(y_true, y_pred) / denom) if denom > 1e-12 else float("nan")
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "std_mse": mse / var,
        "std_rmse": math.sqrt(mse / var),
        "corr": corr,
        "cosine": cosine,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    u0 = logit(args.pi)

    print(f"device: {device}")
    print(f"pi0={args.pi}, u0={u0:.4f}, tau={args.tau}, K={args.n_blocks}, block_size={args.block_size}")
    print(f"sigma_q={args.sigma_q}")

    y_train, target_train = make_fsm_data(
        rng, args.n_train, u0, args.sigma_q, args.n_blocks, args.block_size, args.tau
    )
    y_val, target_val = make_fsm_data(
        rng, args.n_val, u0, args.sigma_q, args.n_blocks, args.block_size, args.tau
    )
    y_test = simulate_center(rng, args.n_test, args.pi, args.n_blocks, args.block_size, args.tau)

    full_test = score_u_from_log_ratio(full_log_ratio(y_test, args.tau), args.pi).sum(axis=1)

    s1_train = score_u_from_log_ratio(marginal_log_ratio(y_train, args.tau), args.pi)
    s1_val = score_u_from_log_ratio(marginal_log_ratio(y_val, args.tau), args.pi)
    s1_test = score_u_from_log_ratio(marginal_log_ratio(y_test, args.tau), args.pi)
    s2_train = score_u_from_log_ratio(pairwise_log_ratio(y_train, args.tau), args.pi)
    s2_val = score_u_from_log_ratio(pairwise_log_ratio(y_val, args.tau), args.pi)
    s2_test = score_u_from_log_ratio(pairwise_log_ratio(y_test, args.tau), args.pi)

    evidence1_train = log_ratio_from_score_u(s1_train, args.pi).sum(axis=2)
    evidence1_val = log_ratio_from_score_u(s1_val, args.pi).sum(axis=2)
    evidence1_test = log_ratio_from_score_u(s1_test, args.pi).sum(axis=2)
    evidence2_train = log_ratio_from_score_u(s2_train, args.pi).sum(axis=2)
    evidence2_val = log_ratio_from_score_u(s2_val, args.pi).sum(axis=2)
    evidence2_test = log_ratio_from_score_u(s2_test, args.pi).sum(axis=2)
    evidence_both_train = np.concatenate([evidence1_train, evidence2_train], axis=1)
    evidence_both_val = np.concatenate([evidence1_val, evidence2_val], axis=1)
    evidence_both_test = np.concatenate([evidence1_test, evidence2_test], axis=1)

    recovered_log_ratio_train = recover_block_log_ratio_from_marginal_pairwise_evidence(
        evidence1_train, evidence2_train, args.tau, args.block_size
    )
    recovered_log_ratio_val = recover_block_log_ratio_from_marginal_pairwise_evidence(
        evidence1_val, evidence2_val, args.tau, args.block_size
    )
    recovered_log_ratio_test = recover_block_log_ratio_from_marginal_pairwise_evidence(
        evidence1_test, evidence2_test, args.tau, args.block_size
    )
    recovered_block_score_train = score_u_from_log_ratio(recovered_log_ratio_train, args.pi)
    recovered_block_score_val = score_u_from_log_ratio(recovered_log_ratio_val, args.pi)
    recovered_block_score_test = score_u_from_log_ratio(recovered_log_ratio_test, args.pi)

    raw_train, raw_val, raw_test = standardize_train_eval(
        y_train.reshape(args.n_train, -1),
        y_val.reshape(args.n_val, -1),
        y_test.reshape(args.n_test, -1),
    )

    s1_train_std, s1_val_std, s1_test_std = standardize_scalar(s1_train, s1_val, s1_test)
    s2_train_std, s2_val_std, s2_test_std = standardize_scalar(s2_train, s2_val, s2_test)

    block_marg_train = s1_train_std.sum(axis=2)
    block_marg_val = s1_val_std.sum(axis=2)
    block_marg_test = s1_test_std.sum(axis=2)
    block_pair_train = s2_train_std.sum(axis=2)
    block_pair_val = s2_val_std.sum(axis=2)
    block_pair_test = s2_test_std.sum(axis=2)
    block_both_train = np.concatenate([block_marg_train, block_pair_train], axis=1)
    block_both_val = np.concatenate([block_marg_val, block_pair_val], axis=1)
    block_both_test = np.concatenate([block_marg_test, block_pair_test], axis=1)
    block_marg_train_mlp, block_marg_val_mlp, block_marg_test_mlp = standardize_train_eval(
        block_marg_train, block_marg_val, block_marg_test
    )
    block_both_train_mlp, block_both_val_mlp, block_both_test_mlp = standardize_train_eval(
        block_both_train, block_both_val, block_both_test
    )
    evidence_both_train_std, evidence_both_val_std, evidence_both_test_std = standardize_train_eval(
        evidence_both_train, evidence_both_val, evidence_both_test
    )
    recovered_block_score_train_std, recovered_block_score_val_std, recovered_block_score_test_std = (
        standardize_train_eval(
            recovered_block_score_train,
            recovered_block_score_val,
            recovered_block_score_test,
        )
    )
    block_both_mean = block_both_train.mean(axis=0, keepdims=True)
    block_both_sd = block_both_train.std(axis=0, keepdims=True)
    block_both_sd = np.where(block_both_sd < 1e-8, 1.0, block_both_sd)

    struct_train = np.concatenate([s1_train_std, s2_train_std], axis=2)
    struct_val = np.concatenate([s1_val_std, s2_val_std], axis=2)
    struct_test = np.concatenate([s1_test_std, s2_test_std], axis=2)

    predictions: dict[str, np.ndarray] = {}
    traces: list[dict[str, object]] = []

    raw_linear_coef = fit_linear_ridge(raw_train, target_train, args.ridge)
    marg_linear_coef = fit_linear_ridge(block_marg_train, target_train, args.ridge)
    both_linear_coef = fit_linear_ridge(block_both_train, target_train, args.ridge)

    # Default comparison: all default methods use a linear final head.
    predictions["raw linear ridge FSM"] = predict_linear(raw_linear_coef, raw_test)
    predictions["linear marginal+pairwise block CS FSM"] = predict_linear(
        both_linear_coef, block_both_test
    )

    if args.include_clean_five:
        gate_label = gate_label_for(args.clean_gate_kind)

        predictions["linear marginal block CS FSM"] = predict_linear(
            marg_linear_coef, block_marg_test
        )

        marginal_linear_ds_model, trace = train_torch_model(
            LinearCSMarginalOnlyDeepSets(args.block_size, args.hidden),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear marginal CS DeepSets FSM",
        )
        predictions["linear marginal CS DeepSets FSM"] = predict_torch(
            marginal_linear_ds_model, struct_test, device
        )
        traces += [
            {"method": "linear marginal CS DeepSets FSM", **row}
            for row in trace
        ]

        marginal_radial_ds_model, trace = train_torch_model(
            RadialGateMarginalOnlyDeepSets(
                args.block_size,
                args.hidden,
                args.gate_hidden,
                args.clean_gate_kind,
                args.poly_degree,
                args.gate_input,
            ),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name=f"{gate_label} radial-gate marginal-only CS DeepSets FSM",
        )
        predictions[f"{gate_label} radial-gate marginal-only CS DeepSets FSM"] = predict_torch(
            marginal_radial_ds_model, struct_test, device
        )
        traces += [
            {"method": f"{gate_label} radial-gate marginal-only CS DeepSets FSM", **row}
            for row in trace
        ]

        linear_ds_model, trace = train_torch_model(
            LinearCSMarginalPairwiseDeepSets(args.block_size, args.hidden),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear CS marginal+pairwise DeepSets FSM",
        )
        predictions["linear CS marginal+pairwise DeepSets FSM"] = predict_torch(
            linear_ds_model, struct_test, device
        )
        traces += [
            {"method": "linear CS marginal+pairwise DeepSets FSM", **row}
            for row in trace
        ]

        radial_model, trace = train_torch_model(
            RadialGateMarginalPairwiseCS(
                args.n_blocks,
                args.block_size,
                args.gate_hidden,
                both_linear_coef,
                args.clean_gate_kind,
                args.poly_degree,
                args.gate_input,
            ),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name=f"{gate_label} radial-gate marginal+pairwise CS linear FSM",
        )
        predictions[f"{gate_label} radial-gate marginal+pairwise CS linear FSM"] = predict_torch(
            radial_model, struct_test, device
        )
        traces += [
            {"method": f"{gate_label} radial-gate marginal+pairwise CS linear FSM", **row}
            for row in trace
        ]

        radial_ds_model, trace = train_torch_model(
            RadialGateMarginalPairwiseDeepSets(
                args.block_size,
                args.hidden,
                args.gate_hidden,
                args.clean_gate_kind,
                args.poly_degree,
                args.gate_input,
            ),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name=f"{gate_label} radial-gate marginal+pairwise CS DeepSets FSM",
        )
        predictions[f"{gate_label} radial-gate marginal+pairwise CS DeepSets FSM"] = predict_torch(
            radial_ds_model, struct_test, device
        )
        traces += [
            {"method": f"{gate_label} radial-gate marginal+pairwise CS DeepSets FSM", **row}
            for row in trace
        ]

    if args.include_extra_linear_ablations:
        predictions["linear marginal block CS FSM"] = predict_linear(marg_linear_coef, block_marg_test)

    if args.include_nonlinear_vector_ablations:
        evidence_coef = fit_linear_ridge(evidence_both_train_std, target_train, args.ridge)
        predictions["nonlinear evidence marginal+pairwise vector FSM"] = predict_linear(
            evidence_coef, evidence_both_test_std
        )
        recovered_coef = fit_linear_ridge(recovered_block_score_train_std, target_train, args.ridge)
        predictions["nonlinear recovered block-score vector FSM"] = predict_linear(
            recovered_coef, recovered_block_score_test_std
        )

    if args.include_learned_phi_vector:
        phi_vector_model, trace = train_torch_model(
            LearnedPhiVectorMarginalPairwiseCS(
                args.n_blocks,
                args.block_size,
                args.hidden,
                both_linear_coef,
            ),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="learned phi-vector marginal+pairwise CS linear-head FSM",
        )
        predictions["learned phi-vector marginal+pairwise CS linear-head FSM"] = predict_torch(
            phi_vector_model, struct_test, device
        )
        traces += [
            {"method": "learned phi-vector marginal+pairwise CS linear-head FSM", **row}
            for row in trace
        ]

    if args.include_odd_monotone_vector:
        odd_model, trace = train_torch_model(
            OddMonotoneVectorMarginalPairwiseCS(
                args.n_blocks,
                args.block_size,
                args.hidden,
                both_linear_coef,
            ),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="odd-monotone phi-vector marginal+pairwise CS linear-head FSM",
        )
        predictions["odd-monotone phi-vector marginal+pairwise CS linear-head FSM"] = predict_torch(
            odd_model, struct_test, device
        )
        traces += [
            {"method": "odd-monotone phi-vector marginal+pairwise CS linear-head FSM", **row}
            for row in trace
        ]

    if args.include_phi_embedding_vector:
        phi_embedding_model, trace = train_torch_model(
            PhiEmbeddingVectorMarginalPairwiseCS(
                args.n_blocks,
                args.block_size,
                args.hidden,
                args.phi_dim,
            ),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="phi-embedding vector marginal+pairwise CS linear-head FSM",
        )
        predictions["phi-embedding vector marginal+pairwise CS linear-head FSM"] = predict_torch(
            phi_embedding_model, struct_test, device
        )
        traces += [
            {"method": "phi-embedding vector marginal+pairwise CS linear-head FSM", **row}
            for row in trace
        ]

    if args.include_linear_cs_deepset and not args.include_clean_five:
        linear_ds_model, trace = train_torch_model(
            LinearCSMarginalPairwiseDeepSets(args.block_size, args.hidden),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear CS marginal+pairwise DeepSets FSM",
        )
        predictions["linear CS marginal+pairwise DeepSets FSM"] = predict_torch(
            linear_ds_model, struct_test, device
        )
        traces += [
            {"method": "linear CS marginal+pairwise DeepSets FSM", **row}
            for row in trace
        ]

    if args.include_structured_deepset:
        struct_model, trace = train_torch_model(
            StructuredMarginalPairwiseDeepSets(args.block_size, args.hidden),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="structured phi+rho marginal+pairwise CS DeepSets FSM",
        )
        predictions["structured phi+rho marginal+pairwise CS DeepSets FSM"] = predict_torch(
            struct_model, struct_test, device
        )
        traces += [
            {"method": "structured phi+rho marginal+pairwise CS DeepSets FSM", **row}
            for row in trace
        ]

    if args.include_nonlinear_head_methods:
        both_mlp_model, trace = train_torch_model(
            MLP(block_both_train_mlp.shape[1], args.hidden, args.depth),
            block_both_train_mlp,
            target_train,
            block_both_val_mlp,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear marginal+pairwise block CS MLP head FSM",
        )
        predictions["linear marginal+pairwise block CS MLP head FSM"] = predict_torch(
            both_mlp_model, block_both_test_mlp, device
        )
        traces += [{"method": "linear marginal+pairwise block CS MLP head FSM", **row} for row in trace]

        for gate_kind, method_name in [
            ("monotone_mlp", "MLP radial-gate marginal+pairwise CS MLP head FSM"),
            ("log_polynomial", "polynomial radial-gate marginal+pairwise CS MLP head FSM"),
        ]:
            radial_mlp_model, trace = train_torch_model(
                RadialGateMarginalPairwiseCSMLPHead(
                    args.n_blocks,
                    args.block_size,
                    args.gate_hidden,
                    args.hidden,
                    args.depth,
                    block_both_mean,
                    block_both_sd,
                    gate_kind,
                    args.poly_degree,
                    args.gate_input,
                ),
                struct_train,
                target_train,
                struct_val,
                target_val,
                iters=args.iters,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                print_every=args.print_every,
                device=device,
                name=method_name,
            )
            predictions[method_name] = predict_torch(radial_mlp_model, struct_test, device)
            traces += [{"method": method_name, **row} for row in trace]

    if args.include_mlp_head_ablations:
        marg_mlp_model, trace = train_torch_model(
            MLP(block_marg_train_mlp.shape[1], args.hidden, args.depth),
            block_marg_train_mlp,
            target_train,
            block_marg_val_mlp,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear marginal block CS MLP FSM",
        )
        predictions["linear marginal block CS MLP FSM"] = predict_torch(
            marg_mlp_model, block_marg_test_mlp, device
        )
        traces += [{"method": "linear marginal block CS MLP FSM", **row} for row in trace]

        both_mlp_model, trace = train_torch_model(
            MLP(block_both_train_mlp.shape[1], args.hidden, args.depth),
            block_both_train_mlp,
            target_train,
            block_both_val_mlp,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear marginal+pairwise block CS MLP FSM",
        )
        predictions["linear marginal+pairwise block CS MLP FSM"] = predict_torch(
            both_mlp_model, block_both_test_mlp, device
        )
        traces += [{"method": "linear marginal+pairwise block CS MLP FSM", **row} for row in trace]

        raw_model, trace = train_torch_model(
            MLP(raw_train.shape[1], args.hidden, args.depth),
            raw_train,
            target_train,
            raw_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="raw MLP FSM",
        )
        predictions["raw MLP FSM"] = predict_torch(raw_model, raw_test, device)
        traces += [{"method": "raw MLP FSM", **row} for row in trace]

        struct_model, trace = train_torch_model(
            StructuredMarginalPairwiseDeepSets(args.block_size, args.hidden),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="structured marginal+pairwise CS FSM",
        )
        predictions["structured marginal+pairwise CS FSM"] = predict_torch(struct_model, struct_test, device)
        traces += [{"method": "structured marginal+pairwise CS FSM", **row} for row in trace]

    if not args.skip_radial_gates and not args.include_clean_five:
        for gate_kind, method_name in [
            ("monotone_mlp", "MLP radial-gate marginal+pairwise CS FSM"),
            ("log_polynomial", "polynomial radial-gate marginal+pairwise CS FSM"),
        ]:
            radial_model, trace = train_torch_model(
                RadialGateMarginalPairwiseCS(
                    args.n_blocks,
                    args.block_size,
                    args.gate_hidden,
                    both_linear_coef,
                    gate_kind,
                    args.poly_degree,
                    args.gate_input,
                ),
                struct_train,
                target_train,
                struct_val,
                target_val,
                iters=args.iters,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                print_every=args.print_every,
                device=device,
                name=method_name,
            )
            predictions[method_name] = predict_torch(radial_model, struct_test, device)
            traces += [{"method": method_name, **row} for row in trace]

    if args.include_mlp_head_ablations or args.include_nested_structured:
        nested_model, trace = train_torch_model(
            NestedStructuredMarginalPairwiseDeepSets(
                args.n_blocks,
                args.block_size,
                args.hidden,
                both_linear_coef,
                residual_scale_init=args.nested_residual_scale_init,
            ),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="nested structured marginal+pairwise CS FSM",
        )
        predictions["nested structured marginal+pairwise CS FSM"] = predict_torch(
            nested_model, struct_test, device
        )
        traces += [{"method": "nested structured marginal+pairwise CS FSM", **row} for row in trace]

    rows = []
    for method, pred in predictions.items():
        row: dict[str, object] = {
            "method": method,
            "n_test": args.n_test,
            "pi": args.pi,
            "tau": args.tau,
            "n_blocks": args.n_blocks,
            "block_size": args.block_size,
            "sigma_q": args.sigma_q,
        }
        row.update(metrics(full_test, pred))
        rows.append(row)
    rows = sorted(rows, key=lambda x: float(x["std_mse"]))
    write_csv(out_dir / "score_summary_direct.csv", rows)
    write_csv(out_dir / "training_trace.csv", traces)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n==== Direct score error (true score used only for diagnostics) ====")
    print(f"{'method':<46}{'std_mse':>10}{'mse':>12}{'corr':>10}{'cosine':>10}")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['method']:<46}"
            f"{float(row['std_mse']):>10.4g}"
            f"{float(row['mse']):>12.4g}"
            f"{float(row['corr']):>10.4g}"
            f"{float(row['cosine']):>10.4g}"
        )
    print("\nSaved:")
    print(out_dir / "score_summary_direct.csv")
    print(out_dir / "training_trace.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--n-val", type=int, default=4_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--n-blocks", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--pi", type=float, default=0.3)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--sigma-q", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--phi-dim", type=int, default=8)
    parser.add_argument("--gate-hidden", type=int, default=8)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--include-nonlinear-head-methods", action="store_true")
    parser.add_argument("--include-extra-linear-ablations", action="store_true")
    parser.add_argument("--include-nonlinear-vector-ablations", action="store_true")
    parser.add_argument("--include-learned-phi-vector", action="store_true")
    parser.add_argument("--include-odd-monotone-vector", action="store_true")
    parser.add_argument("--include-phi-embedding-vector", action="store_true")
    parser.add_argument("--include-linear-cs-deepset", action="store_true")
    parser.add_argument("--include-structured-deepset", action="store_true")
    parser.add_argument("--include-mlp-head-ablations", action="store_true")
    parser.add_argument("--include-nested-structured", action="store_true")
    parser.add_argument("--include-clean-five", action="store_true")
    parser.add_argument(
        "--clean-gate-kind",
        choices=["monotone_mlp", "positive_mlp", "log_polynomial"],
        default="positive_mlp",
    )
    parser.add_argument(
        "--gate-input",
        choices=["abs", "signed"],
        default="signed",
        help="Input to radial multiplier: abs keeps the old odd map; signed allows asymmetric positive multipliers.",
    )
    parser.add_argument("--skip-radial-gates", action="store_true")
    parser.add_argument("--nested-residual-scale-init", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="runs/blockwise_common_factor_fsm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.pi < 1.0):
        raise ValueError("--pi must be in (0, 1)")
    if args.sigma_q <= 0:
        raise ValueError("--sigma-q must be positive")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    run(args)


if __name__ == "__main__":
    main()
