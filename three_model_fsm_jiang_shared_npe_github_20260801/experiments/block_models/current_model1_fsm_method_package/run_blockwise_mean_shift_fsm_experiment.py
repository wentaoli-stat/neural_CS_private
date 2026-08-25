#!/usr/bin/env python3
"""Likelihood-free FSM training for the blockwise shared mean-shift mixture.

This script intentionally does *not* use the analytic nonlinear recovery map as
a method. Training uses only:

  1. a local Gaussian proposal q(u | u0),
  2. simulations Y ~ p_{pi(u)},
  3. the proposal-score target tau=(u-u0)/sigma_q^2.

The exact full score is used only after training as a diagnostic metric.
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


def sigmoid_scalar(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def pi_from_u(u: np.ndarray | float) -> np.ndarray | float:
    if isinstance(u, float):
        return sigmoid_scalar(u)
    return sigmoid_np(u)


def score_pi_from_log_ratio(log_ratio: np.ndarray, pi: float) -> np.ndarray:
    """d/d pi log[(1-pi) f0 + pi f1] from log(f1/f0)."""
    w = sigmoid_np(logit(pi) + log_ratio)
    return (w - pi) / (pi * (1.0 - pi))


def score_u_from_log_ratio(log_ratio: np.ndarray, pi: float) -> np.ndarray:
    return score_pi_from_log_ratio(log_ratio, pi) * pi * (1.0 - pi)


def log_ratio_from_score_u(score_u: np.ndarray, pi: float, eps: float = 1e-6) -> np.ndarray:
    """Invert the pilot marginal u-score into local log-likelihood ratio evidence.

    For this toy model, score_u(y; pi) = P(Z=1 | y, pi) - pi. Hence
    log f1(y)/f0(y) = logit(pi + score_u) - logit(pi).
    This is used only as a nonlinear vector-summary design, not as a full-score oracle.
    """
    posterior_prob = np.clip(pi + np.asarray(score_u, dtype=np.float64), eps, 1.0 - eps)
    return np.log(posterior_prob) - np.log1p(-posterior_prob) - logit(pi)


def simulate_from_u(
    rng: np.random.Generator,
    u: np.ndarray,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    pi = pi_from_u(u)
    z = rng.binomial(1, pi[:, None], size=(u.shape[0], n_blocks)).astype(np.float64)
    y = rng.normal(size=(u.shape[0], n_blocks, block_size)) + tau * z[:, :, None]
    log_r = tau * y - 0.5 * tau**2
    return y, log_r


def simulate_center(
    rng: np.random.Generator,
    n: int,
    pi0: float,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    u0 = logit(pi0)
    u = np.full(n, u0, dtype=np.float64)
    return simulate_from_u(rng, u, n_blocks, block_size, tau)


def make_fsm_data(
    rng: np.random.Generator,
    n: int,
    u0: float,
    sigma_q: float,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = rng.normal(loc=u0, scale=sigma_q, size=n)
    y, log_r = simulate_from_u(rng, u, n_blocks, block_size, tau)
    target = (u - u0) / (sigma_q**2)
    return y, log_r, target


def standardize_train_eval(
    x_train: np.ndarray,
    *others: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, ...]:
    mean = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return tuple((x - mean) / sd for x in (x_train, *others))


def standardize_scalar_subscores(
    s_train: np.ndarray,
    *others: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, ...]:
    mean = float(s_train.mean())
    sd = float(s_train.std())
    if sd < eps:
        sd = 1.0
    return tuple((x - mean) / sd for x in (s_train, *others))


def fit_linear_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("fit_linear_ridge received non-finite x or y")
    design = np.concatenate([np.ones((x.shape[0], 1)), x.reshape(x.shape[0], -1)], axis=1)
    gram = np.einsum("ni,nj->ij", design, design)
    gram[1:, 1:] += ridge * np.eye(gram.shape[0] - 1)
    rhs = np.einsum("ni,n->i", design, y)
    return np.linalg.solve(gram, rhs)


def predict_linear(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    coef = np.asarray(coef, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(coef).all() or not np.isfinite(x).all():
        raise ValueError("predict_linear received non-finite coef or x")
    design = np.concatenate([np.ones((x.shape[0], 1)), x.reshape(x.shape[0], -1)], axis=1)
    return np.einsum("ni,i->n", design, coef)


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

    With nonnegative inputs this uses log(1+r), matching the original radial
    version. With signed inputs it uses asinh(r), which is defined on the
    whole real line and avoids the silent NaNs caused by log1p(r) for r <= -1.
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


class RadialGateMarginalCS(nn.Module):
    """Structure-preserving marginal subscore gate with a block-level linear CS head."""

    def __init__(
        self,
        n_blocks: int,
        hidden: int,
        linear_coef: np.ndarray,
        gate_kind: str,
        poly_degree: int,
        gate_input: str = "abs",
    ):
        super().__init__()
        coef = np.asarray(linear_coef, dtype=np.float32)
        if coef.shape != (1 + n_blocks,):
            raise ValueError(f"linear_coef must have shape {(1 + n_blocks,)}, got {coef.shape}")
        self.intercept = nn.Parameter(torch.tensor(coef[0], dtype=torch.float32))
        self.weights = nn.Parameter(torch.tensor(coef[1:], dtype=torch.float32))
        self.gate = make_multiplier(gate_kind, hidden, poly_degree, gate_input)
        self.gate_input = gate_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size)
        gated_x = x * self.gate(gate_input_tensor(x, self.gate_input).unsqueeze(-1))
        block_scores = gated_x.sum(dim=2)
        return self.intercept + block_scores @ self.weights


class RadialGateMarginalCSMLPHead(nn.Module):
    """Structure-preserving marginal subscore gate followed by the same MLP head."""

    def __init__(
        self,
        n_blocks: int,
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
        self.gate = make_multiplier(gate_kind, hidden, poly_degree, gate_input)
        self.head = MLP(n_blocks, head_hidden, head_depth)
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_sd", torch.as_tensor(feature_sd, dtype=torch.float32))
        self.gate_input = gate_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size)
        gated_x = x * self.gate(gate_input_tensor(x, self.gate_input).unsqueeze(-1))
        block_scores = gated_x.sum(dim=2)
        block_scores = (block_scores - self.feature_mean) / self.feature_sd
        return self.head(block_scores)


class RadialGateMarginalDeepSets(nn.Module):
    """Structure-preserving radial gate followed by a shared block DeepSets rho."""

    def __init__(
        self,
        hidden: int,
        gate_hidden: int,
        gate_kind: str,
        poly_degree: int,
        gate_input: str = "abs",
    ):
        super().__init__()
        self.gate = make_multiplier(gate_kind, gate_hidden, poly_degree, gate_input)
        self.gate_input = gate_input
        self.rho = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, block_size)
        gated_x = x * self.gate(gate_input_tensor(x, self.gate_input).unsqueeze(-1))
        block_scores = gated_x.sum(dim=2, keepdim=True)
        return self.rho(block_scores).squeeze(-1).sum(dim=1)


class StructuredMarginalDeepSets(nn.Module):
    """Sum_k rho(sum_i phi(s_ik)) for marginal subscores."""

    def __init__(self, hidden: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, n)
        emb = self.phi(x.unsqueeze(-1)).sum(dim=2)
        block_score = self.rho(emb).squeeze(-1)
        return block_score.sum(dim=1)


class BlockVectorStructuredMLP(nn.Module):
    """Keep per-block nonlinear embeddings, then use one global MLP head.

    This ablation tests
        E_k = sum_i phi(s_ki),  (E_1, ..., E_K) -> MLP -> S_hat,
    instead of the exchangeable Deep Sets readout sum_k rho(E_k).
    """

    def __init__(self, n_blocks: int, hidden: int, head_hidden: int, head_depth: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.hidden = hidden
        self.phi = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = MLP(n_blocks * hidden, head_hidden, head_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, n)
        emb = self.phi(x.unsqueeze(-1)).sum(dim=2)
        return self.head(emb.reshape(x.shape[0], self.n_blocks * self.hidden))


class LearnedPhiVectorMarginalCS(nn.Module):
    """E_k=sum_i phi_psi(s_ki), followed by a linear head over (E_1,...,E_K).

    The residual MLP is initialized at zero, so phi_psi(s)=s at initialization
    and the whole model starts exactly at the linear block-CS ridge baseline.
    """

    def __init__(self, n_blocks: int, hidden: int, linear_coef: np.ndarray):
        super().__init__()
        coef = np.asarray(linear_coef, dtype=np.float32)
        if coef.shape != (1 + n_blocks,):
            raise ValueError(f"linear_coef must have shape {(1 + n_blocks,)}, got {coef.shape}")
        self.intercept = nn.Parameter(torch.tensor(coef[0], dtype=torch.float32))
        self.weights = nn.Parameter(torch.tensor(coef[1:], dtype=torch.float32))
        self.residual = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, n)
        phi_x = x + self.residual(x.unsqueeze(-1)).squeeze(-1)
        block_features = phi_x.sum(dim=2)
        return self.intercept + block_features @ self.weights


class OddMonotoneResidualMap(nn.Module):
    """Odd, sign-preserving monotone scalar map initialized at identity.

    phi(s) = sign(s) h(|s|), where
        h(r) = a0 r + sum_j a_j {softplus(b_j + c_j r) - softplus(b_j)}.
    The positive coefficients imply h'(r) > 0 and h(0)=0. This keeps the
    subscore direction but allows a more flexible monotone amplitude transform
    than a single multiplicative radial gate.
    """

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


class OddMonotoneVectorMarginalCS(nn.Module):
    """E_k=sum_i phi_psi(s_ki), followed by a linear head over block features."""

    def __init__(self, n_blocks: int, n_basis: int, linear_coef: np.ndarray):
        super().__init__()
        coef = np.asarray(linear_coef, dtype=np.float32)
        if coef.shape != (1 + n_blocks,):
            raise ValueError(f"linear_coef must have shape {(1 + n_blocks,)}, got {coef.shape}")
        self.intercept = nn.Parameter(torch.tensor(coef[0], dtype=torch.float32))
        self.weights = nn.Parameter(torch.tensor(coef[1:], dtype=torch.float32))
        self.phi = OddMonotoneResidualMap(n_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, n)
        block_features = self.phi(x).sum(dim=2)
        return self.intercept + block_features @ self.weights


class PhiEmbeddingVectorMarginalCS(nn.Module):
    """E_k=sum_i phi_psi(s_ki) with phi_psi: R -> R^d and one linear head.

    This is the clean non-radial, non-direction-preserving vector design:
    phi is a generic neural feature map, block order is kept by flattening
    (E_1, ..., E_K), and the final readout is linear.
    """

    def __init__(self, n_blocks: int, hidden: int, phi_dim: int):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.phi_dim = int(phi_dim)
        self.phi = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, phi_dim),
        )
        self.head = nn.Linear(self.n_blocks * self.phi_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, n)
        emb = self.phi(x.unsqueeze(-1)).sum(dim=2)
        return self.head(emb.reshape(x.shape[0], self.n_blocks * self.phi_dim)).squeeze(-1)


class AugmentedPhiEmbeddingVectorMarginalCS(nn.Module):
    """Linear block CS plus learned phi embeddings, with a single linear head.

    Feature vector is (sum_i s_ki)_k concatenated with
    (sum_i phi_psi(s_ki))_k. The linear part is initialized at the ridge
    baseline and the nonlinear embedding weights start at zero.
    """

    def __init__(self, n_blocks: int, hidden: int, phi_dim: int, linear_coef: np.ndarray):
        super().__init__()
        self.n_blocks = int(n_blocks)
        self.phi_dim = int(phi_dim)
        coef = np.asarray(linear_coef, dtype=np.float32)
        if coef.shape != (1 + n_blocks,):
            raise ValueError(f"linear_coef must have shape {(1 + n_blocks,)}, got {coef.shape}")
        self.phi = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, phi_dim),
        )
        self.intercept = nn.Parameter(torch.tensor(coef[0], dtype=torch.float32))
        weights = torch.zeros(n_blocks + n_blocks * phi_dim, dtype=torch.float32)
        weights[:n_blocks] = torch.as_tensor(coef[1:], dtype=torch.float32)
        self.weights = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, n)
        block_linear = x.sum(dim=2)
        emb = self.phi(x.unsqueeze(-1)).sum(dim=2)
        features = torch.cat([block_linear, emb.reshape(x.shape[0], self.n_blocks * self.phi_dim)], dim=1)
        return self.intercept + features @ self.weights


class RhoOnlyMarginalBlockSums(nn.Module):
    """Ablation: block sums first, then a shared nonlinear rho per block."""

    def __init__(self, hidden: int):
        super().__init__()
        self.rho = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, n)
        block_sum = x.sum(dim=2, keepdim=True)
        block_score = self.rho(block_sum).squeeze(-1)
        return block_score.sum(dim=1)


class PhiOnlyMarginalDeepSets(nn.Module):
    """Ablation: nonlinear phi before block sum, then only a linear rho."""

    def __init__(self, hidden: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.rho = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, n)
        emb = self.phi(x.unsqueeze(-1)).sum(dim=2)
        block_score = self.rho(emb).squeeze(-1)
        return block_score.sum(dim=1)


class ResidualPhiRhoMarginalDeepSets(nn.Module):
    """Nested ablation: rho-only block sums plus a zero-gated phi/rho residual.

    This model starts exactly as a rho-only block-sum model when residual_scale=0,
    so adding subscore-level nonlinear structure cannot damage the initialization.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.base = RhoOnlyMarginalBlockSums(hidden)
        self.phi = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.rho_residual = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, K, n)
        base_score = self.base(x)
        emb = self.phi(x.unsqueeze(-1)).sum(dim=2)
        residual = self.rho_residual(emb).squeeze(-1).sum(dim=1)
        return base_score + self.residual_scale * residual


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
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state = None
    bad = 0
    trace: list[dict[str, float]] = []
    step = 0

    with torch.no_grad():
        initial_val_loss = torch.mean((model(x_val_t) - y_val_t) ** 2).item()
    if not np.isfinite(initial_val_loss):
        raise RuntimeError(f"{name} produced non-finite initial validation loss: {initial_val_loss}")
    best_loss = initial_val_loss
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    trace.append({"step": 0.0, "train_loss": float("nan"), "val_loss": float(initial_val_loss)})
    print(f"{name} step 0/{iters}: train=nan, val={initial_val_loss:.4g}")

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
                with torch.no_grad():
                    val_loss = torch.mean((model(x_val_t) - y_val_t) ** 2).item()
                if not np.isfinite(val_loss):
                    raise RuntimeError(f"{name} produced non-finite validation loss at step {step}: {val_loss}")
                trace.append({"step": float(step), "train_loss": float(loss.item()), "val_loss": float(val_loss)})
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


def predict_torch(model: nn.Module, x: np.ndarray, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    target_var = float(np.var(y_true) + 1e-12)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 1e-12 else float("nan")
    denom = np.linalg.norm(y_true) * np.linalg.norm(y_pred)
    cosine = float(np.dot(y_true, y_pred) / denom) if denom > 1e-12 else float("nan")
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "std_mse": mse / target_var,
        "std_rmse": math.sqrt(mse / target_var),
        "corr": corr,
        "cosine": cosine,
    }


def smoothed_fsm_target_from_log_ratio(
    log_r: np.ndarray,
    u0: float,
    sigma_q: float,
    *,
    grid_size: int,
    grid_width: float,
    chunk_size: int,
) -> np.ndarray:
    """Compute the exact one-dimensional FSM smoothed target for this toy model.

    The local FSM regression target is tau=(u-u0)/sigma_q^2.  Conditional on an
    observed data set Y, the population optimum is

        E_q[tau | Y] =
          int tau(u) p(Y|u) q(u|u0) du / int p(Y|u) q(u|u0) du.

    For Model 1 the likelihood p(Y|u) is available up to constants, so this
    one-dimensional integral is a deterministic diagnostic.  It is not used for
    training.
    """
    if sigma_q <= 0:
        raise ValueError("sigma_q must be positive")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    log_r = np.asarray(log_r, dtype=np.float64)
    if log_r.ndim != 3:
        raise ValueError(f"log_r must have shape (n, K, m), got {log_r.shape}")

    z_grid = np.linspace(-float(grid_width), float(grid_width), int(grid_size))
    u_grid = float(u0) + float(sigma_q) * z_grid
    pi_grid = np.clip(pi_from_u(u_grid), 1e-12, 1.0 - 1e-12)
    log_prior = -0.5 * z_grid**2
    tau_grid = (u_grid - float(u0)) / (float(sigma_q) ** 2)
    log_pi = np.log(pi_grid)
    log_one_minus_pi = np.log1p(-pi_grid)

    block_log_r = log_r.sum(axis=2)
    out = np.empty(block_log_r.shape[0], dtype=np.float64)
    for start in range(0, block_log_r.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), block_log_r.shape[0])
        lr = block_log_r[start:stop]
        # shape: chunk x grid x blocks
        log_terms = np.logaddexp(
            log_one_minus_pi[None, :, None],
            log_pi[None, :, None] + lr[:, None, :],
        )
        logw = log_terms.sum(axis=2) + log_prior[None, :]
        logw -= np.max(logw, axis=1, keepdims=True)
        w = np.exp(logw)
        out[start:stop] = np.sum(w * tau_grid[None, :], axis=1) / np.sum(w, axis=1)
    return out


def target_decomposition_rows(
    predictions: dict[str, np.ndarray],
    exact_score: np.ndarray,
    smoothed_target: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    bias_row: dict[str, object] = {
        "method": "FSM smoothed target",
        "comparison": "smoothed_target_vs_exact_score",
    }
    bias_row.update(metrics(exact_score, smoothed_target))
    rows.append(bias_row)

    for method, pred in predictions.items():
        exact_row: dict[str, object] = {
            "method": method,
            "comparison": "prediction_vs_exact_score",
        }
        exact_row.update(metrics(exact_score, pred))
        rows.append(exact_row)

        smooth_row: dict[str, object] = {
            "method": method,
            "comparison": "prediction_vs_smoothed_target",
        }
        smooth_row.update(metrics(smoothed_target, pred))
        rows.append(smooth_row)
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
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pi0 = args.pi
    u0 = logit(pi0)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"device: {device}")
    print(f"pi0={pi0}, u0={u0:.4f}, tau={args.tau}, K={args.n_blocks}, block_size={args.block_size}")
    print(f"sigma_q={args.sigma_q}")

    y_train, log_r_train, target_train = make_fsm_data(
        rng, args.n_train, u0, args.sigma_q, args.n_blocks, args.block_size, args.tau
    )
    y_val, log_r_val, target_val = make_fsm_data(
        rng, args.n_val, u0, args.sigma_q, args.n_blocks, args.block_size, args.tau
    )
    y_test, log_r_test = simulate_center(
        rng, args.n_test, pi0, args.n_blocks, args.block_size, args.tau
    )
    full_test = score_u_from_log_ratio(log_r_test.sum(axis=2), pi0).sum(axis=1)

    s_train = score_u_from_log_ratio(log_r_train, pi0)
    s_val = score_u_from_log_ratio(log_r_val, pi0)
    s_test = score_u_from_log_ratio(log_r_test, pi0)
    evidence_train = log_ratio_from_score_u(s_train, pi0).sum(axis=2)
    evidence_val = log_ratio_from_score_u(s_val, pi0).sum(axis=2)
    evidence_test = log_ratio_from_score_u(s_test, pi0).sum(axis=2)
    block_score_train = score_u_from_log_ratio(evidence_train, pi0)
    block_score_val = score_u_from_log_ratio(evidence_val, pi0)
    block_score_test = score_u_from_log_ratio(evidence_test, pi0)

    raw_train, raw_val, raw_test = standardize_train_eval(
        y_train.reshape(args.n_train, -1),
        y_val.reshape(args.n_val, -1),
        y_test.reshape(args.n_test, -1),
    )
    flat_train, flat_val, flat_test = standardize_train_eval(
        s_train.reshape(args.n_train, -1),
        s_val.reshape(args.n_val, -1),
        s_test.reshape(args.n_test, -1),
    )
    struct_train, struct_val, struct_test = standardize_scalar_subscores(s_train, s_val, s_test)
    block_train = struct_train.sum(axis=2)
    block_val = struct_val.sum(axis=2)
    block_test = struct_test.sum(axis=2)
    evidence_train_std, evidence_val_std, evidence_test_std = standardize_train_eval(
        evidence_train, evidence_val, evidence_test
    )
    block_score_train_std, block_score_val_std, block_score_test_std = standardize_train_eval(
        block_score_train, block_score_val, block_score_test
    )
    block_train_mlp, block_val_mlp, block_test_mlp = standardize_train_eval(
        block_train, block_val, block_test
    )
    block_mean = block_train.mean(axis=0, keepdims=True)
    block_sd = block_train.std(axis=0, keepdims=True)
    block_sd = np.where(block_sd < 1e-8, 1.0, block_sd)

    predictions: dict[str, np.ndarray] = {}
    traces: list[dict[str, object]] = []

    # Default comparison: all default methods use a linear final head.
    predictions["raw linear ridge FSM"] = predict_linear(
        fit_linear_ridge(raw_train, target_train, ridge=args.ridge), raw_test
    )

    # Linear batched CS baseline: one feature per block, then a linear head.
    coef = fit_linear_ridge(flat_train, target_train, ridge=args.ridge)
    block_coef = fit_linear_ridge(block_train, target_train, ridge=args.ridge)
    predictions["linear block marginal CS ridge FSM"] = predict_linear(block_coef, block_test)

    if args.include_clean_five:
        gate_label = "polynomial" if args.clean_gate_kind == "log_polynomial" else "MLP"

        rho_only_model, trace = train_torch_model(
            RhoOnlyMarginalBlockSums(args.hidden),
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
            name="linear CS DeepSets FSM",
        )
        predictions["linear CS DeepSets FSM"] = predict_torch(rho_only_model, struct_test, device)
        traces += [{"method": "linear CS DeepSets FSM", **row} for row in trace]

        radial_model, trace = train_torch_model(
            RadialGateMarginalCS(
                args.n_blocks,
                args.gate_hidden,
                block_coef,
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
            radial_model, struct_test, device
        )
        traces += [{"method": f"{gate_label} radial-gate CS linear FSM", **row} for row in trace]

        radial_ds_model, trace = train_torch_model(
            RadialGateMarginalDeepSets(
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
            name=f"{gate_label} radial-gate CS DeepSets FSM",
        )
        predictions[f"{gate_label} radial-gate CS DeepSets FSM"] = predict_torch(
            radial_ds_model, struct_test, device
        )
        traces += [{"method": f"{gate_label} radial-gate CS DeepSets FSM", **row} for row in trace]

    if args.include_nonlinear_vector_ablations:
        evidence_coef = fit_linear_ridge(evidence_train_std, target_train, ridge=args.ridge)
        predictions["oracle nonlinear evidence-vector CS ridge FSM"] = predict_linear(
            evidence_coef, evidence_test_std
        )
        block_score_coef = fit_linear_ridge(block_score_train_std, target_train, ridge=args.ridge)
        predictions["oracle nonlinear block-score vector ridge FSM"] = predict_linear(
            block_score_coef, block_score_test_std
        )

    if args.include_learned_phi_vector:
        phi_vector_model, trace = train_torch_model(
            LearnedPhiVectorMarginalCS(args.n_blocks, args.hidden, block_coef),
            struct_train.astype(np.float32),
            target_train,
            struct_val.astype(np.float32),
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="learned phi-vector marginal CS linear-head FSM",
        )
        predictions["learned phi-vector marginal CS linear-head FSM"] = predict_torch(
            phi_vector_model, struct_test.astype(np.float32), device
        )
        traces += [{"method": "learned phi-vector marginal CS linear-head FSM", **row} for row in trace]

    if args.include_odd_monotone_vector:
        odd_model, trace = train_torch_model(
            OddMonotoneVectorMarginalCS(args.n_blocks, args.hidden, block_coef),
            struct_train.astype(np.float32),
            target_train,
            struct_val.astype(np.float32),
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="odd-monotone phi-vector marginal CS linear-head FSM",
        )
        predictions["odd-monotone phi-vector marginal CS linear-head FSM"] = predict_torch(
            odd_model, struct_test.astype(np.float32), device
        )
        traces += [
            {"method": "odd-monotone phi-vector marginal CS linear-head FSM", **row}
            for row in trace
        ]

    if args.include_phi_embedding_vector:
        phi_embedding_model, trace = train_torch_model(
            PhiEmbeddingVectorMarginalCS(args.n_blocks, args.hidden, args.phi_dim),
            struct_train.astype(np.float32),
            target_train,
            struct_val.astype(np.float32),
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="phi-embedding vector marginal CS linear-head FSM",
        )
        predictions["phi-embedding vector marginal CS linear-head FSM"] = predict_torch(
            phi_embedding_model, struct_test.astype(np.float32), device
        )
        traces += [{"method": "phi-embedding vector marginal CS linear-head FSM", **row} for row in trace]

    if args.include_extra_linear_ablations:
        predictions["linear marginal CS ridge FSM"] = predict_linear(coef, flat_test)

    if args.include_nonlinear_head_methods:
        block_mlp_model, trace = train_torch_model(
            MLP(block_train_mlp.shape[1], args.hidden, args.depth),
            block_train_mlp,
            target_train,
            block_val_mlp,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear block marginal CS MLP head FSM",
        )
        predictions["linear block marginal CS MLP head FSM"] = predict_torch(
            block_mlp_model, block_test_mlp, device
        )
        traces += [{"method": "linear block marginal CS MLP head FSM", **row} for row in trace]

        for gate_kind, method_name in [
            ("monotone_mlp", "MLP radial-gate marginal CS MLP head FSM"),
            ("positive_mlp", "positive-MLP radial-gate marginal CS MLP head FSM"),
            ("log_polynomial", "polynomial radial-gate marginal CS MLP head FSM"),
        ]:
            radial_mlp_model, trace = train_torch_model(
                RadialGateMarginalCSMLPHead(
                    args.n_blocks,
                    args.gate_hidden,
                    args.hidden,
                    args.depth,
                    block_mean,
                    block_sd,
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

        flat_model, trace = train_torch_model(
            MLP(flat_train.shape[1], args.hidden, args.depth),
            flat_train,
            target_train,
            flat_val,
            target_val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            device=device,
            name="linear marginal CS MLP FSM",
        )
        predictions["linear marginal CS MLP FSM"] = predict_torch(flat_model, flat_test, device)
        traces += [{"method": "linear marginal CS MLP FSM", **row} for row in trace]

        ds_model, trace = train_torch_model(
            StructuredMarginalDeepSets(args.hidden),
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
            name="structured marginal CS FSM",
        )
        predictions["structured marginal CS FSM"] = predict_torch(ds_model, struct_test, device)
        traces += [{"method": "structured marginal CS FSM", **row} for row in trace]

    if args.include_block_vector_head_ablation:
        block_vector_model, trace = train_torch_model(
            BlockVectorStructuredMLP(args.n_blocks, args.hidden, args.hidden, args.depth),
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
            name="block-vector structured CS MLP FSM",
        )
        predictions["block-vector structured CS MLP FSM"] = predict_torch(
            block_vector_model, struct_test, device
        )
        traces += [{"method": "block-vector structured CS MLP FSM", **row} for row in trace]

    if args.include_deepset_factor_ablations and not args.include_clean_five:
        rho_only_model, trace = train_torch_model(
            RhoOnlyMarginalBlockSums(args.hidden),
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
            name="rho-only block-sum CS FSM",
        )
        predictions["rho-only block-sum CS FSM"] = predict_torch(rho_only_model, struct_test, device)
        traces += [{"method": "rho-only block-sum CS FSM", **row} for row in trace]

        for model, method_name in [
            (PhiOnlyMarginalDeepSets(args.hidden), "phi-only subscore CS FSM"),
            (StructuredMarginalDeepSets(args.hidden), "phi+rho structured marginal CS FSM"),
        ]:
            ablation_model, trace = train_torch_model(
                model,
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
            predictions[method_name] = predict_torch(ablation_model, struct_test, device)
            traces += [{"method": method_name, **row} for row in trace]

        residual_model = ResidualPhiRhoMarginalDeepSets(args.hidden)
        residual_model.base.load_state_dict(
            {k: v.detach().cpu().clone() for k, v in rho_only_model.state_dict().items()}
        )
        residual_model, trace = train_torch_model(
            residual_model,
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
            name="rho-only + residual phi-rho CS FSM",
        )
        predictions["rho-only + residual phi-rho CS FSM"] = predict_torch(
            residual_model, struct_test, device
        )
        traces += [{"method": "rho-only + residual phi-rho CS FSM", **row} for row in trace]

    if not args.skip_radial_gates and not args.include_clean_five:
        for gate_kind, method_name in [
            ("monotone_mlp", "MLP radial-gate marginal CS FSM"),
            ("positive_mlp", "positive-MLP radial-gate marginal CS FSM"),
            ("log_polynomial", "polynomial radial-gate marginal CS FSM"),
        ]:
            radial_model, trace = train_torch_model(
                RadialGateMarginalCS(
                    args.n_blocks,
                    args.gate_hidden,
                    block_coef,
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

    rows = []
    for method, pred in predictions.items():
        row: dict[str, object] = {
            "method": method,
            "n_test": args.n_test,
            "pi": pi0,
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

    decomposition_rows: list[dict[str, object]] = []
    if args.compute_smoothed_target_diagnostics:
        print("\nComputing exact one-dimensional FSM smoothed target diagnostic...")
        smoothed_test = smoothed_fsm_target_from_log_ratio(
            log_r_test,
            u0,
            args.sigma_q,
            grid_size=args.smooth_grid_size,
            grid_width=args.smooth_grid_width,
            chunk_size=args.smooth_chunk_size,
        )
        decomposition_rows = target_decomposition_rows(predictions, full_test, smoothed_test)
        write_csv(out_dir / "score_target_decomposition.csv", decomposition_rows)

    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n==== Direct score error (true score used only for diagnostics) ====")
    print(f"{'method':<36}{'std_mse':>10}{'mse':>12}{'corr':>10}{'cosine':>10}")
    print("-" * 78)
    for row in rows:
        print(
            f"{row['method']:<36}"
            f"{float(row['std_mse']):>10.4g}"
            f"{float(row['mse']):>12.4g}"
            f"{float(row['corr']):>10.4g}"
            f"{float(row['cosine']):>10.4g}"
        )
    print("\nSaved:")
    print(out_dir / "score_summary_direct.csv")
    print(out_dir / "training_trace.csv")
    if decomposition_rows:
        print(out_dir / "score_target_decomposition.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=20_000)
    parser.add_argument("--n-val", type=int, default=2_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--n-blocks", type=int, default=20)
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
    parser.add_argument("--iters", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--include-nonlinear-vector-ablations", action="store_true")
    parser.add_argument("--include-learned-phi-vector", action="store_true")
    parser.add_argument("--include-odd-monotone-vector", action="store_true")
    parser.add_argument("--include-phi-embedding-vector", action="store_true")
    parser.add_argument("--include-nonlinear-head-methods", action="store_true")
    parser.add_argument("--include-extra-linear-ablations", action="store_true")
    parser.add_argument("--include-mlp-head-ablations", action="store_true")
    parser.add_argument("--include-block-vector-head-ablation", action="store_true")
    parser.add_argument("--include-deepset-factor-ablations", action="store_true")
    parser.add_argument("--include-clean-five", action="store_true")
    parser.add_argument(
        "--compute-smoothed-target-diagnostics",
        action="store_true",
        help="After training, compute exact-score MSE and the one-dimensional FSM smoothed-target decomposition. Diagnostic only.",
    )
    parser.add_argument("--smooth-grid-size", type=int, default=2001)
    parser.add_argument("--smooth-grid-width", type=float, default=8.0)
    parser.add_argument("--smooth-chunk-size", type=int, default=256)
    parser.add_argument(
        "--clean-gate-kind",
        choices=["monotone_mlp", "positive_mlp", "log_polynomial"],
        default="monotone_mlp",
    )
    parser.add_argument(
        "--gate-input",
        choices=["abs", "signed"],
        default="abs",
        help="Input to the positive multiplier in phi(s)=s*m(input): abs gives the old odd radial map; signed allows an asymmetric positive multiplier.",
    )
    parser.add_argument("--skip-radial-gates", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="runs/blockwise_mean_shift_fsm")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.pi < 1.0):
        raise ValueError("--pi must be in (0, 1)")
    if args.sigma_q <= 0:
        raise ValueError("--sigma-q must be positive")
    run(args)


if __name__ == "__main__":
    main()
