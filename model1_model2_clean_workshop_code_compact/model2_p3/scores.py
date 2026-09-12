#!/usr/bin/env python3
"""Simulator and closed-form scores for the p=3 common-factor variance mixture.

The DGP generalizes Model 2 by freeing the factor loading and the noise scale::

    B_k ~ Bernoulli(pi),   Y_kj = sigma * eps_kj + B_k * tau * Z_k

so an inactive block is ``N(0, sigma^2 I_m)`` and an active block is
``N(0, sigma^2 I_m + tau^2 11')``. The parameter is

    beta = (u, tau, lam) = (logit pi, tau, log sigma) in R^3.

Every log-likelihood ratio in this model -- marginal (r=1), within-block pair
(r=2) and full block (r=m) -- belongs to one family indexed by the subset size
``r``::

    L_r(S) = c_r + d_r S^2,    S = sum of the r observations,
    c_r = -log(1 + r tau^2 / A) / 2,     d_r = tau^2 / (2 A (A + r tau^2)),
    A = sigma^2,

because summing r jointly Gaussian coordinates leaves a scalar sufficient
statistic. ``local_scores`` supplies the marginal and pairwise channels the
network sees; ``exact_score`` is evaluation-only.
"""

from __future__ import annotations

import math

import numpy as np

N_PARAMS = 3
PARAM_NAMES = ("u", "tau", "log_sigma")


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def logit_np(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def split_beta(beta: np.ndarray):
    beta = np.asarray(beta, dtype=np.float64)
    if beta.ndim != 2 or beta.shape[1] != N_PARAMS:
        raise ValueError(f"beta must have shape (n, {N_PARAMS}), got {beta.shape}")
    return beta[:, 0], beta[:, 1], beta[:, 2]


def simulate(rng, beta: np.ndarray, n_blocks: int, block_size: int) -> np.ndarray:
    u, tau, lam = split_beta(beta)
    pi, sigma = sigmoid_np(u), np.exp(lam)
    active = rng.binomial(1, pi[:, None], size=(u.shape[0], n_blocks)).astype(np.float64)
    eps = rng.normal(size=(u.shape[0], n_blocks, block_size))
    z = rng.normal(size=(u.shape[0], n_blocks, 1))
    return sigma[:, None, None] * eps + active[:, :, None] * tau[:, None, None] * z


def _cd(tau, A, r):
    """(c_r, d_r) and their tau / log-sigma derivatives for subset size r."""
    denom = A + r * tau**2
    c = -0.5 * np.log1p(r * tau**2 / A)
    d = tau**2 / (2.0 * A * denom)
    dc_dtau = -r * tau / denom
    dd_dtau = tau / denom**2
    dc_dlam = 1.0 - A / denom
    dd_dlam = -(tau**2) * (2.0 * A + r * tau**2) / (A * denom**2)
    return c, d, dc_dtau, dd_dtau, dc_dlam, dd_dlam


def _subset_scores(S, Q, r, u, tau, lam):
    """Mixture score of one subset of size r with sum S and sum-of-squares Q."""
    A = np.exp(2.0 * lam)
    pi = sigmoid_np(u)
    c, d, dc_t, dd_t, dc_l, dd_l = _cd(tau, A, r)
    ratio = c + d * S**2
    w = sigmoid_np(u + ratio)
    s_u = w - pi
    s_tau = w * (dc_t + dd_t * S**2)
    # d log f0 / d lam for the inactive component, plus the weighted ratio term.
    s_lam = (-r + Q / A) + w * (dc_l + dd_l * S**2)
    return np.stack([s_u, s_tau, s_lam], axis=-1)


def _bcast(beta, ndim):
    u, tau, lam = split_beta(beta)
    shape = (-1,) + (1,) * (ndim - 1)
    return u.reshape(shape), tau.reshape(shape), lam.reshape(shape)


def marginal_scores(y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Per-observation scores, shape (n, K, m, 3)."""
    y = np.asarray(y, dtype=np.float64)
    u, tau, lam = _bcast(beta, y.ndim)
    return _subset_scores(y, y**2, 1, u, tau, lam)


def pairwise_scores(y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Within-block pair scores, shape (n, K, m(m-1)/2, 3)."""
    y = np.asarray(y, dtype=np.float64)
    i, j = np.triu_indices(y.shape[2], k=1)
    S = y[:, :, i] + y[:, :, j]
    Q = y[:, :, i] ** 2 + y[:, :, j] ** 2
    u, tau, lam = _bcast(beta, 3)
    return _subset_scores(S, Q, 2, u, tau, lam)


def exact_score(y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Exact full-data score d/dbeta log p(Y | beta), shape (n, 3)."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 3:
        raise ValueError(f"y must have shape (n, K, m), got {y.shape}")
    m = y.shape[2]
    S, Q = y.sum(axis=2), (y**2).sum(axis=2)
    u, tau, lam = _bcast(beta, 2)
    return _subset_scores(S, Q, m, u, tau, lam).sum(axis=1)


def log_likelihood(y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """log p(Y | beta); used only to validate ``exact_score``."""
    y = np.asarray(y, dtype=np.float64)
    m = y.shape[2]
    S, Q = y.sum(axis=2), (y**2).sum(axis=2)
    u, tau, lam = _bcast(beta, 2)
    A, pi = np.exp(2.0 * lam), sigmoid_np(u)
    c, d = _cd(tau, A, m)[:2]
    base = -0.5 * m * math.log(2.0 * math.pi) - m * lam - Q / (2.0 * A)
    return (base + np.logaddexp(np.log1p(-pi), np.log(pi) + c + d * S**2)).sum(axis=1)
