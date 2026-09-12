#!/usr/bin/env python3
"""Simulator and closed-form scores for the p=3 mean-shift block mixture.

The DGP generalizes Model 1 by freeing the shift and the noise scale::

    B_k ~ Bernoulli(pi),    Y_kj = B_k * tau + sigma * eps_kj,   eps ~ N(0, 1)

and works in the unconstrained parameter

    beta = (u, tau, lam) = (logit pi, tau, log sigma) in R^3.

Both the per-observation local score and the exact full-data score are
available in closed form, so the frozen exact-score evaluation used for the
p=1 experiments carries over without approximation. ``local_scores`` supplies
the network's inputs; ``exact_score`` is evaluation-only and is never shown to
a training loop.
"""

from __future__ import annotations

import math

import numpy as np


N_PARAMS = 3
PARAM_NAMES = ("u", "tau", "log_sigma")
PARAM_LABELS = ("d/du", "d/dtau", "d/dlog_sigma")


def sigmoid_np(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit_np(p: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def split_beta(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, tau, log sigma) columns of an (n, 3) parameter array."""
    beta = np.asarray(beta, dtype=np.float64)
    if beta.ndim != 2 or beta.shape[1] != N_PARAMS:
        raise ValueError(f"beta must have shape (n, {N_PARAMS}), got {beta.shape}")
    return beta[:, 0], beta[:, 1], beta[:, 2]


def simulate(rng: np.random.Generator, beta: np.ndarray, n_blocks: int, block_size: int) -> np.ndarray:
    """Simulate Y with shape (n, n_blocks, block_size)."""
    u, tau, lam = split_beta(beta)
    pi = sigmoid_np(u)
    sigma = np.exp(lam)
    active = rng.binomial(1, pi[:, None], size=(u.shape[0], n_blocks)).astype(np.float64)
    eps = rng.normal(size=(u.shape[0], n_blocks, block_size))
    return active[:, :, None] * tau[:, None, None] + sigma[:, None, None] * eps


def _broadcast(beta: np.ndarray, ndim: int) -> tuple[np.ndarray, ...]:
    u, tau, lam = split_beta(beta)
    shape = (-1,) + (1,) * (ndim - 1)
    return u.reshape(shape), tau.reshape(shape), lam.reshape(shape)


def local_scores(y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Per-observation marginal mixture scores, shape (n, K, m, 3).

    Each observation is treated as its own two-component mixture, which is the
    local score the aggregation architecture is allowed to see.
    """
    y = np.asarray(y, dtype=np.float64)
    u, tau, lam = _broadcast(beta, y.ndim)
    pi, sigma2 = sigmoid_np(u), np.exp(2.0 * lam)
    ratio = (tau * y - 0.5 * tau**2) / sigma2
    w = sigmoid_np(u + ratio)
    s_u = w - pi
    s_tau = w * (y - tau) / sigma2
    s_lam = -1.0 + ((1.0 - w) * y**2 + w * (y - tau) ** 2) / sigma2
    return np.stack([s_u, s_tau, s_lam], axis=-1)


def exact_score(y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Exact full-data score d/dbeta log p(Y | beta), shape (n, 3).

    Blocks are independent and the activation indicator is shared within a
    block, so the block likelihood ratio enters through one posterior weight.
    Evaluation only: never used for training or checkpoint selection.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"y must have shape (n, K, m), got {y.shape}")
    u, tau, lam = _broadcast(beta, 2)
    pi, sigma2 = sigmoid_np(u), np.exp(2.0 * lam)
    block_sum = y.sum(axis=2)
    block_sq = (y**2).sum(axis=2)
    m = y.shape[2]
    log_ratio = (tau * block_sum - 0.5 * m * tau**2) / sigma2
    w = sigmoid_np(u + log_ratio)
    s_u = (w - pi).sum(axis=1)
    s_tau = (w * (block_sum - m * tau) / sigma2).sum(axis=1)
    inactive = -m + block_sq / sigma2
    active = -m + (block_sq - 2.0 * tau * block_sum + m * tau**2) / sigma2
    s_lam = ((1.0 - w) * inactive + w * active).sum(axis=1)
    return np.stack([s_u, s_tau, s_lam], axis=-1)


def log_likelihood(y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """log p(Y | beta), shape (n,). Used only to validate ``exact_score``."""
    y = np.asarray(y, dtype=np.float64)
    u, tau, lam = _broadcast(beta, 2)
    pi, sigma2 = sigmoid_np(u), np.exp(2.0 * lam)
    m = y.shape[2]
    base = -m * lam - 0.5 * m * math.log(2.0 * math.pi) - (y**2).sum(axis=2) / (2.0 * sigma2)
    log_ratio = (tau * y.sum(axis=2) - 0.5 * m * tau**2) / sigma2
    per_block = base + np.logaddexp(np.log1p(-pi), np.log(pi) + log_ratio)
    return per_block.sum(axis=1)
