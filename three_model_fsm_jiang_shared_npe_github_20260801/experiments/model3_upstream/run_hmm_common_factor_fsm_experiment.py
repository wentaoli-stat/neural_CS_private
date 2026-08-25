#!/usr/bin/env python3
"""Likelihood-free FSM for a hidden-Markov common-factor variance mixture.

Model:
    B_t in {0, 1}
    P(B_t=1 | B_{t-1}=0) = p01
    P(B_t=1 | B_{t-1}=1) = p11

    Y_t | B_t=0 ~ N(0, I_n)
    Y_t | B_t=1 ~ N(0, I_n + tau^2 11')

The first-stage FSM training is likelihood-free:
    u ~ N(u0, diag(sigma_q^2))
    Y ~ p_{theta(u)}
    S_psi(summary(Y; theta0)) is regressed to (u-u0)/sigma_q^2.

The exact HMM Fisher score is used only after training as a diagnostic.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PARAM_NAMES = ("p01", "p11")


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


def u_to_params(u: np.ndarray) -> np.ndarray:
    return sigmoid_np(u)


def fixed_stationary_prob(p01: float, p11: float) -> float:
    return p01 / max(p01 + 1.0 - p11, 1e-12)


def logsumexp2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def simulate_hmm_common_factor(
    rng: np.random.Generator,
    u: np.ndarray,
    length: int,
    block_size: int,
    tau: float,
    init_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Y and hidden states for a batch of unconstrained parameters."""
    theta = u_to_params(u)
    p01 = theta[:, 0]
    p11 = theta[:, 1]
    n = u.shape[0]
    states = np.zeros((n, length), dtype=np.int64)
    states[:, 0] = rng.binomial(1, init_prob, size=n)
    for t in range(1, length):
        prob_one = np.where(states[:, t - 1] == 1, p11, p01)
        states[:, t] = rng.binomial(1, prob_one)

    eps = rng.normal(size=(n, length, block_size))
    factor = rng.normal(scale=tau, size=(n, length, 1))
    y = eps + states[:, :, None] * factor
    return y.astype(np.float64), states


def make_fsm_data(
    rng: np.random.Generator,
    n: int,
    u0: np.ndarray,
    sigma_q: np.ndarray,
    length: int,
    block_size: int,
    tau: float,
    init_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = rng.normal(loc=u0[None, :], scale=sigma_q[None, :], size=(n, 2))
    y, _ = simulate_hmm_common_factor(rng, u, length, block_size, tau, init_prob)
    target = (u - u0[None, :]) / (sigma_q[None, :] ** 2)
    return y, target.astype(np.float64)


def simulate_center(
    rng: np.random.Generator,
    n: int,
    u0: np.ndarray,
    length: int,
    block_size: int,
    tau: float,
    init_prob: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.repeat(u0[None, :], n, axis=0)
    return simulate_hmm_common_factor(rng, u, length, block_size, tau, init_prob)


def full_emission_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    n = y.shape[2]
    block_sum = y.sum(axis=2)
    return -0.5 * math.log1p(n * tau**2) + (
        tau**2 / (2.0 * (1.0 + n * tau**2))
    ) * block_sum**2


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


def local_score_u_from_log_ratio(log_ratio: np.ndarray, pi_ref: float) -> np.ndarray:
    """Reference iid-mixture u-score used only as an emission evidence map."""
    w = sigmoid_np(logit(pi_ref) + log_ratio)
    score_pi = (w - pi_ref) / (pi_ref * (1.0 - pi_ref))
    return score_pi * pi_ref * (1.0 - pi_ref)


def hmm_transition_score_u(
    emission_lr: np.ndarray,
    p01: float,
    p11: float,
    init_prob: float,
) -> np.ndarray:
    """Exact observed HMM score in u=(logit p01, logit p11) coordinates."""
    n, length = emission_lr.shape
    log_p00 = math.log1p(-p01)
    log_p01 = math.log(p01)
    log_p10 = math.log1p(-p11)
    log_p11 = math.log(p11)
    log_init0 = math.log1p(-init_prob)
    log_init1 = math.log(init_prob)

    alpha = np.empty((n, length, 2), dtype=np.float64)
    alpha[:, 0, 0] = log_init0
    alpha[:, 0, 1] = log_init1 + emission_lr[:, 0]
    for t in range(1, length):
        alpha[:, t, 0] = logsumexp2(alpha[:, t - 1, 0] + log_p00, alpha[:, t - 1, 1] + log_p10)
        alpha[:, t, 1] = emission_lr[:, t] + logsumexp2(
            alpha[:, t - 1, 0] + log_p01,
            alpha[:, t - 1, 1] + log_p11,
        )
    loglik = logsumexp2(alpha[:, -1, 0], alpha[:, -1, 1])

    beta = np.zeros((n, length, 2), dtype=np.float64)
    for t in range(length - 1, 0, -1):
        beta[:, t - 1, 0] = logsumexp2(
            log_p00 + beta[:, t, 0],
            log_p01 + emission_lr[:, t] + beta[:, t, 1],
        )
        beta[:, t - 1, 1] = logsumexp2(
            log_p10 + beta[:, t, 0],
            log_p11 + emission_lr[:, t] + beta[:, t, 1],
        )

    xi00 = np.zeros(n, dtype=np.float64)
    xi01 = np.zeros(n, dtype=np.float64)
    xi10 = np.zeros(n, dtype=np.float64)
    xi11 = np.zeros(n, dtype=np.float64)
    for t in range(1, length):
        xi00 += np.exp(alpha[:, t - 1, 0] + log_p00 + beta[:, t, 0] - loglik)
        xi01 += np.exp(alpha[:, t - 1, 0] + log_p01 + emission_lr[:, t] + beta[:, t, 1] - loglik)
        xi10 += np.exp(alpha[:, t - 1, 1] + log_p10 + beta[:, t, 0] - loglik)
        xi11 += np.exp(alpha[:, t - 1, 1] + log_p11 + emission_lr[:, t] + beta[:, t, 1] - loglik)

    score_p01 = xi01 / p01 - xi00 / (1.0 - p01)
    score_p11 = xi11 / p11 - xi10 / (1.0 - p11)
    score_u01 = score_p01 * p01 * (1.0 - p01)
    score_u11 = score_p11 * p11 * (1.0 - p11)
    return np.stack([score_u01, score_u11], axis=1)


def standardize_train_eval(
    x_train: np.ndarray,
    *others: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, ...]:
    mean = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return tuple(((x - mean) / sd).astype(np.float32) for x in (x_train, *others))


def standardize_scalar_subscores(
    s_train: np.ndarray,
    *others: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, ...]:
    mean = float(s_train.mean())
    sd = float(s_train.std())
    if sd < eps:
        sd = 1.0
    return tuple(((x - mean) / sd).astype(np.float32) for x in (s_train, *others))


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
    rhs = design.T @ y
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
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.SiLU())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequenceCSNet(nn.Module):
    """Nonlinear head on per-time aggregated marginal/pairwise CS features."""

    def __init__(self, length: int, hidden: int, depth: int):
        super().__init__()
        self.head = MLP(2 * length, 2, hidden, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.reshape(x.shape[0], -1))


class LinearCSGRU(nn.Module):
    """GRU head on the linear per-time composite score sequence."""

    def __init__(self, in_dim: int, gru_hidden: int, out_dim: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=gru_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(gru_hidden, gru_hidden), nn.SiLU(), nn.Linear(gru_hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1])


class RawSequenceGRU(nn.Module):
    """GRU head on the raw per-time observations."""

    def __init__(self, block_size: int, gru_hidden: int, out_dim: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size=block_size, hidden_size=gru_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(gru_hidden, gru_hidden), nn.SiLU(), nn.Linear(gru_hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1])


class TimeDeepSets(nn.Module):
    """Exchangeable DeepSets readout over a sequence of per-time features."""

    def __init__(self, in_dim: int, hidden: int, depth: int, out_dim: int = 2):
        super().__init__()
        self.rho = MLP(in_dim, out_dim, hidden, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rho(x).sum(dim=1)


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
    """Positive scalar map m(r) without monotonicity constraints, initialized at m(r)=1."""

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

    Nonnegative radial inputs use log(1+r). Signed inputs use asinh(r), which is
    defined on the whole real line and avoids silent NaNs from log1p(r) when
    standardized subscores are <= -1.
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


class RadialGateEmissionLinearHead(nn.Module):
    """Structure-preserving emission gates with a time-indexed linear CS head."""

    def __init__(
        self,
        length: int,
        block_size: int,
        hidden: int,
        linear_coef: np.ndarray,
        gate_kind: str,
        poly_degree: int,
        gate_input: str = "abs",
    ):
        super().__init__()
        self.length = int(length)
        self.block_size = int(block_size)
        self.n_pairs = self.block_size * (self.block_size - 1) // 2
        coef = np.asarray(linear_coef, dtype=np.float32)
        if coef.shape != (1 + 2 * self.length, 2):
            raise ValueError(f"linear_coef must have shape {(1 + 2 * self.length, 2)}, got {coef.shape}")
        self.intercept = nn.Parameter(torch.as_tensor(coef[0], dtype=torch.float32))
        self.weights = nn.Parameter(torch.as_tensor(coef[1:], dtype=torch.float32))
        self.marginal_gate = make_multiplier(gate_kind, hidden, poly_degree, gate_input)
        self.pairwise_gate = make_multiplier(gate_kind, hidden, poly_degree, gate_input)
        self.gate_input = gate_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, T, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        gated_s1 = s1 * self.marginal_gate(gate_input_tensor(s1, self.gate_input).unsqueeze(-1))
        gated_s2 = s2 * self.pairwise_gate(gate_input_tensor(s2, self.gate_input).unsqueeze(-1))
        time_features = torch.stack([gated_s1.sum(dim=2), gated_s2.sum(dim=2)], dim=-1)
        features = time_features.reshape(x.shape[0], -1)
        return self.intercept + features @ self.weights


class RadialGateEmissionDeepSets(nn.Module):
    """Structure-preserving emission gates followed by a shared time DeepSets rho."""

    def __init__(
        self,
        block_size: int,
        hidden: int,
        gate_hidden: int,
        gate_kind: str,
        poly_degree: int,
        gate_input: str = "abs",
        out_dim: int = 2,
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
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, T, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        gated_s1 = s1 * self.marginal_gate(gate_input_tensor(s1, self.gate_input).unsqueeze(-1))
        gated_s2 = s2 * self.pairwise_gate(gate_input_tensor(s2, self.gate_input).unsqueeze(-1))
        time_features = torch.stack([gated_s1.sum(dim=2), gated_s2.sum(dim=2)], dim=-1)
        return self.rho(time_features).sum(dim=1)


class StructuredEmissionGRU(nn.Module):
    """DeepSets over within-time subscores, then GRU over time."""

    def __init__(self, block_size: int, embed_dim: int, gru_hidden: int, out_dim: int = 2):
        super().__init__()
        self.block_size = block_size
        self.n_pairs = block_size * (block_size - 1) // 2
        self.phi1 = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU())
        self.phi2 = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU())
        self.gru = nn.GRU(input_size=2 * embed_dim, hidden_size=gru_hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(gru_hidden, gru_hidden), nn.SiLU(), nn.Linear(gru_hidden, out_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        e1 = self.phi1(s1.unsqueeze(-1)).mean(dim=2)
        e2 = self.phi2(s2.unsqueeze(-1)).mean(dim=2)
        return torch.cat([e1, e2], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encode(x)
        _, h = self.gru(seq)
        return self.head(h[-1])


class StructuredEmissionDeepSets(nn.Module):
    """DeepSets over within-time subscores, then exchangeable sum over time.

    This tests the Model-1-style readout
        E_t = (mean_i phi1(s_ti), mean_ij phi2(s_tij)),
        S_hat(Y) = sum_t rho(E_t),
    against the order-aware GRU aggregation.
    """

    def __init__(self, block_size: int, embed_dim: int, hidden: int, out_dim: int = 2):
        super().__init__()
        self.block_size = block_size
        self.n_pairs = block_size * (block_size - 1) // 2
        self.phi1 = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU())
        self.phi2 = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim), nn.SiLU())
        self.rho = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, T, block_size + n_pairs)
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        e1 = self.phi1(s1.unsqueeze(-1)).mean(dim=2)
        e2 = self.phi2(s2.unsqueeze(-1)).mean(dim=2)
        emb = torch.cat([e1, e2], dim=-1)
        return self.rho(emb).sum(dim=1)


class NestedStructuredEmissionGRU(StructuredEmissionGRU):
    """Residual version initialized exactly at the linear time-CS ridge baseline."""

    def __init__(
        self,
        length: int,
        block_size: int,
        embed_dim: int,
        gru_hidden: int,
        linear_coef: np.ndarray,
        residual_scale_init: float = 0.0,
    ):
        super().__init__(block_size=block_size, embed_dim=embed_dim, gru_hidden=gru_hidden, out_dim=2)
        coef = np.asarray(linear_coef, dtype=np.float32)
        self.register_buffer("linear_intercept", torch.as_tensor(coef[0], dtype=torch.float32))
        self.register_buffer("linear_weights", torch.as_tensor(coef[1:], dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))
        self.length = length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = x[:, :, : self.block_size]
        s2 = x[:, :, self.block_size :]
        time_cs = torch.stack([s1.sum(dim=2), s2.sum(dim=2)], dim=2).reshape(x.shape[0], -1)
        linear_score = self.linear_intercept + time_cs @ self.linear_weights
        residual_score = super().forward(x)
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

    def validation_loss() -> float:
        model.eval()
        sqerr = 0.0
        n_total = 0
        with torch.no_grad():
            for start in range(0, x_val_t.shape[0], batch_size):
                xb = x_val_t[start : start + batch_size].to(device)
                yb = y_val_t[start : start + batch_size].to(device)
                err = model(xb) - yb
                sqerr += float(torch.sum(err**2).item())
                n_total += int(err.numel())
        model.train()
        return sqerr / max(n_total, 1)

    val_loss = validation_loss()
    if not np.isfinite(val_loss):
        raise RuntimeError(f"{name} produced non-finite initial validation loss: {val_loss}")
    best_loss = val_loss
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    trace.append({"step": 0.0, "train_loss": float("nan"), "val_loss": val_loss})
    print(f"{name} step 0/{iters}: train=nan, val={val_loss:.4g}")

    step = 0
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


def predict_torch(model: nn.Module, x: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    scale = np.std(y_true, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    std_mse = float(np.mean(((err) / scale[None, :]) ** 2))
    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)
    corr = float(np.corrcoef(flat_true, flat_pred)[0, 1]) if np.std(flat_pred) > 1e-12 else float("nan")
    denom = np.linalg.norm(flat_true) * np.linalg.norm(flat_pred)
    cosine = float(np.dot(flat_true, flat_pred) / denom) if denom > 1e-12 else float("nan")
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "std_mse": std_mse,
        "std_rmse": math.sqrt(std_mse),
        "corr": corr,
        "cosine": cosine,
    }


def by_param_metrics(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> list[dict[str, object]]:
    rows = []
    for j, name in enumerate(PARAM_NAMES):
        err = y_pred[:, j] - y_true[:, j]
        mse = float(np.mean(err**2))
        scale = float(np.std(y_true[:, j]))
        if scale < 1e-8:
            scale = 1.0
        corr = float(np.corrcoef(y_true[:, j], y_pred[:, j])[0, 1]) if np.std(y_pred[:, j]) > 1e-12 else float("nan")
        rows.append(
            {
                "method": method,
                "param": name,
                "mse": mse,
                "rmse": math.sqrt(mse),
                "std_mse": mse / (scale**2),
                "std_rmse": math.sqrt(mse / (scale**2)),
                "corr": corr,
            }
        )
    return rows


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
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    u0 = np.array([logit(args.p01), logit(args.p11)], dtype=np.float64)
    if args.init_prob < 0:
        init_prob = fixed_stationary_prob(args.p01, args.p11)
    else:
        init_prob = args.init_prob
    if args.sigma_q_values:
        sigma_q = np.array([float(x) for x in args.sigma_q_values.split(",")], dtype=np.float64)
    else:
        sigma_q = np.array([args.sigma_q, args.sigma_q], dtype=np.float64)
    if sigma_q.shape != (2,):
        raise ValueError("--sigma-q-values must contain exactly two comma-separated values")

    print(f"device: {device}")
    print(f"p01={args.p01}, p11={args.p11}, u0={u0.tolist()}, tau={args.tau}, pi_ref={args.pi_ref}")
    print(f"length={args.length}, block_size={args.block_size}, init_prob={init_prob:.4f}")
    print(f"sigma_q={sigma_q.tolist()}")

    y_train, target_train = make_fsm_data(
        rng, args.n_train, u0, sigma_q, args.length, args.block_size, args.tau, init_prob
    )
    y_val, target_val = make_fsm_data(
        rng, args.n_val, u0, sigma_q, args.length, args.block_size, args.tau, init_prob
    )
    y_test, _ = simulate_center(rng, args.n_test, u0, args.length, args.block_size, args.tau, init_prob)

    emission_test = full_emission_log_ratio(y_test, args.tau)
    true_score = hmm_transition_score_u(emission_test, args.p01, args.p11, init_prob)

    s1_train = local_score_u_from_log_ratio(marginal_log_ratio(y_train, args.tau), args.pi_ref)
    s1_val = local_score_u_from_log_ratio(marginal_log_ratio(y_val, args.tau), args.pi_ref)
    s1_test = local_score_u_from_log_ratio(marginal_log_ratio(y_test, args.tau), args.pi_ref)
    s2_train = local_score_u_from_log_ratio(pairwise_log_ratio(y_train, args.tau), args.pi_ref)
    s2_val = local_score_u_from_log_ratio(pairwise_log_ratio(y_val, args.tau), args.pi_ref)
    s2_test = local_score_u_from_log_ratio(pairwise_log_ratio(y_test, args.tau), args.pi_ref)

    s1_train, s1_val, s1_test = standardize_scalar_subscores(s1_train, s1_val, s1_test)
    s2_train, s2_val, s2_test = standardize_scalar_subscores(s2_train, s2_val, s2_test)

    raw_train, raw_val, raw_test = standardize_train_eval(
        y_train.reshape(args.n_train, -1),
        y_val.reshape(args.n_val, -1),
        y_test.reshape(args.n_test, -1),
    )
    raw_seq_train = raw_train.reshape(args.n_train, args.length, args.block_size)
    raw_seq_val = raw_val.reshape(args.n_val, args.length, args.block_size)
    raw_seq_test = raw_test.reshape(args.n_test, args.length, args.block_size)
    time_cs_train = np.stack([s1_train.sum(axis=2), s2_train.sum(axis=2)], axis=2)
    time_cs_val = np.stack([s1_val.sum(axis=2), s2_val.sum(axis=2)], axis=2)
    time_cs_test = np.stack([s1_test.sum(axis=2), s2_test.sum(axis=2)], axis=2)
    global_cs_train = time_cs_train.sum(axis=1)
    global_cs_val = time_cs_val.sum(axis=1)
    global_cs_test = time_cs_test.sum(axis=1)
    time_cs_mlp_train, time_cs_mlp_val, time_cs_mlp_test = standardize_train_eval(
        time_cs_train.reshape(args.n_train, -1),
        time_cs_val.reshape(args.n_val, -1),
        time_cs_test.reshape(args.n_test, -1),
    )
    time_cs_gru_train = time_cs_mlp_train.reshape(args.n_train, args.length, 2)
    time_cs_gru_val = time_cs_mlp_val.reshape(args.n_val, args.length, 2)
    time_cs_gru_test = time_cs_mlp_test.reshape(args.n_test, args.length, 2)

    struct_train = np.concatenate([s1_train, s2_train], axis=2)
    struct_val = np.concatenate([s1_val, s2_val], axis=2)
    struct_test = np.concatenate([s1_test, s2_test], axis=2)

    predictions: dict[str, np.ndarray] = {}
    traces: list[dict[str, object]] = []

    raw_coef = fit_linear_ridge(raw_train, target_train, args.ridge)
    global_coef = fit_linear_ridge(global_cs_train, target_train, args.ridge)
    time_coef = fit_linear_ridge(time_cs_train, target_train, args.ridge)
    predictions["raw linear ridge FSM"] = predict_linear(raw_coef, raw_test)
    if args.include_clean_five:
        predictions["linear time-indexed marginal+pairwise CS ridge FSM"] = predict_linear(time_coef, time_cs_test)

        linear_deepsets, trace = train_torch_model(
            TimeDeepSets(2, args.hidden, args.depth),
            time_cs_gru_train,
            target_train,
            time_cs_gru_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear CS time-deepsets FSM",
        )
        predictions["linear CS time-deepsets FSM"] = predict_torch(
            linear_deepsets, time_cs_gru_test, device, args.batch_size
        )
        traces += [{"method": "linear CS time-deepsets FSM", **row} for row in trace]

        gate_label = gate_label_for(args.clean_gate_kind)
        radial_linear, trace = train_torch_model(
            RadialGateEmissionLinearHead(
                args.length,
                args.block_size,
                args.gate_hidden,
                time_coef,
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
            name=f"{gate_label} radial-gate CS linear FSM",
        )
        predictions[f"{gate_label} radial-gate CS linear FSM"] = predict_torch(
            radial_linear, struct_test, device, args.batch_size
        )
        traces += [{"method": f"{gate_label} radial-gate CS linear FSM", **row} for row in trace]

        radial_deepsets, trace = train_torch_model(
            RadialGateEmissionDeepSets(
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
            name=f"{gate_label} radial-gate CS time-deepsets FSM",
        )
        predictions[f"{gate_label} radial-gate CS time-deepsets FSM"] = predict_torch(
            radial_deepsets, struct_test, device, args.batch_size
        )
        traces += [{"method": f"{gate_label} radial-gate CS time-deepsets FSM", **row} for row in trace]
    else:
        predictions["global-sum linear CS ridge FSM"] = predict_linear(global_coef, global_cs_test)
        predictions["time-indexed linear CS ridge FSM"] = predict_linear(time_coef, time_cs_test)

    if args.include_raw_gru and not args.include_clean_five:
        raw_gru, trace = train_torch_model(
            RawSequenceGRU(args.block_size, args.gru_hidden),
            raw_seq_train,
            target_train,
            raw_seq_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="raw GRU FSM",
        )
        predictions["raw GRU FSM"] = predict_torch(raw_gru, raw_seq_test, device, args.batch_size)
        traces += [{"method": "raw GRU FSM", **row} for row in trace]

    if args.include_time_deepsets_raw and not args.include_clean_five:
        raw_deepsets, trace = train_torch_model(
            TimeDeepSets(args.block_size, args.hidden, args.depth),
            raw_seq_train,
            target_train,
            raw_seq_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="raw time-deepsets FSM",
        )
        predictions["raw time-deepsets FSM"] = predict_torch(raw_deepsets, raw_seq_test, device, args.batch_size)
        traces += [{"method": "raw time-deepsets FSM", **row} for row in trace]

    if args.include_time_deepsets_linear and not args.include_clean_five:
        linear_deepsets, trace = train_torch_model(
            TimeDeepSets(2, args.hidden, args.depth),
            time_cs_gru_train,
            target_train,
            time_cs_gru_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear CS time-deepsets FSM",
        )
        predictions["linear CS time-deepsets FSM"] = predict_torch(
            linear_deepsets, time_cs_gru_test, device, args.batch_size
        )
        traces += [{"method": "linear CS time-deepsets FSM", **row} for row in trace]

    if not args.skip_time_indexed_mlp and not args.include_clean_five:
        time_mlp, trace = train_torch_model(
            SequenceCSNet(args.length, args.hidden, args.depth),
            time_cs_mlp_train,
            target_train,
            time_cs_mlp_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="time-indexed CS MLP FSM",
        )
        predictions["time-indexed CS MLP FSM"] = predict_torch(
            time_mlp, time_cs_mlp_test, device, args.batch_size
        )
        traces += [{"method": "time-indexed CS MLP FSM", **row} for row in trace]

    if args.include_linear_gru and not args.include_clean_five:
        linear_gru, trace = train_torch_model(
            LinearCSGRU(2, args.gru_hidden),
            time_cs_gru_train,
            target_train,
            time_cs_gru_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear CS GRU FSM",
        )
        predictions["linear CS GRU FSM"] = predict_torch(linear_gru, time_cs_gru_test, device, args.batch_size)
        traces += [{"method": "linear CS GRU FSM", **row} for row in trace]

    if not args.skip_nested_structured and not args.include_clean_five:
        nested, trace = train_torch_model(
            NestedStructuredEmissionGRU(
                args.length,
                args.block_size,
                args.embed_dim,
                args.gru_hidden,
                time_coef,
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
            name="nested structured sequential CS GRU FSM",
        )
        predictions["nested structured sequential CS GRU FSM"] = predict_torch(
            nested, struct_test, device, args.batch_size
        )
        traces += [{"method": "nested structured sequential CS GRU FSM", **row} for row in trace]

    if args.include_non_nested_structured and not args.include_clean_five:
        structured, trace = train_torch_model(
            StructuredEmissionGRU(args.block_size, args.embed_dim, args.gru_hidden),
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
            name="structured sequential CS GRU FSM",
        )
        predictions["structured sequential CS GRU FSM"] = predict_torch(
            structured, struct_test, device, args.batch_size
        )
        traces += [{"method": "structured sequential CS GRU FSM", **row} for row in trace]

    if args.include_time_deepsets_structured and not args.include_clean_five:
        time_deepsets, trace = train_torch_model(
            StructuredEmissionDeepSets(args.block_size, args.embed_dim, args.gru_hidden),
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
            name="structured time-deepsets CS FSM",
        )
        predictions["structured time-deepsets CS FSM"] = predict_torch(
            time_deepsets, struct_test, device, args.batch_size
        )
        traces += [{"method": "structured time-deepsets CS FSM", **row} for row in trace]

    rows = []
    by_param_rows = []
    for method, pred in predictions.items():
        row: dict[str, object] = {
            "method": method,
            "n_test": args.n_test,
            "p01": args.p01,
            "p11": args.p11,
            "tau": args.tau,
            "length": args.length,
            "block_size": args.block_size,
            "sigma_q_01": sigma_q[0],
            "sigma_q_11": sigma_q[1],
        }
        row.update(summary_metrics(true_score, pred))
        rows.append(row)
        by_param_rows.extend(by_param_metrics(true_score, pred, method))
    rows = sorted(rows, key=lambda x: float(x["std_mse"]))
    write_csv(out_dir / "score_summary_direct.csv", rows)
    write_csv(out_dir / "score_by_param_direct.csv", by_param_rows)
    write_csv(out_dir / "training_trace.csv", traces)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n==== Direct score error (true HMM score used only for diagnostics) ====")
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
    print(out_dir / "score_by_param_direct.csv")
    print(out_dir / "training_trace.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--n-test", type=int, default=3_000)
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--p01", type=float, default=0.08)
    parser.add_argument("--p11", type=float, default=0.90)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--pi-ref", type=float, default=0.3)
    parser.add_argument("--init-prob", type=float, default=-1.0, help="Negative means fixed stationary prob at center.")
    parser.add_argument("--sigma-q", type=float, default=0.4)
    parser.add_argument("--sigma-q-values", type=str, default="")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--iters", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--nested-residual-scale-init", type=float, default=0.0)
    parser.add_argument("--include-raw-gru", action="store_true")
    parser.add_argument("--include-time-deepsets-raw", action="store_true")
    parser.add_argument("--include-time-deepsets-linear", action="store_true")
    parser.add_argument("--include-linear-gru", action="store_true")
    parser.add_argument("--include-non-nested-structured", action="store_true")
    parser.add_argument("--include-time-deepsets-structured", action="store_true")
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
    parser.add_argument("--skip-time-indexed-mlp", action="store_true")
    parser.add_argument("--skip-nested-structured", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="runs/hmm_common_factor_fsm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("p01", "p11", "pi_ref"):
        value = getattr(args, name)
        if not (0.0 < value < 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1)")
    if args.sigma_q <= 0:
        raise ValueError("--sigma-q must be positive")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    run(args)


if __name__ == "__main__":
    main()
