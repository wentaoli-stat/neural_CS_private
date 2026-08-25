#!/usr/bin/env python3
"""Exact and smoothed toy references for Model 1 Mode A diagnostics."""

from __future__ import annotations

import numpy as np

import run_blockwise_mean_shift_amortized_fsm_experiment as stage1


def logsumexp_np(value: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(value, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(value - maximum), axis=axis)
    )


def smoothed_fsm_score_from_full_lr(
    full_lr: np.ndarray,
    u0: float,
    sigma_q: float,
    grid_size: int,
    grid_radius: float,
    chunk_size: int,
) -> np.ndarray:
    """Return the Bayes-optimal local-FSM target at an anchor ``u0``.

    The input contains one exact block log-likelihood ratio per block. The
    returned quantity is

        E[(u - u0) / sigma_q**2 | Y, u ~ N(u0, sigma_q**2)].

    Exact likelihood information is used only by diagnostics, never by FSM
    training or deployment.
    """
    if sigma_q <= 0.0:
        raise ValueError("sigma_q must be positive")
    ratios = np.asarray(full_lr, dtype=np.float64)
    if ratios.ndim != 2:
        raise ValueError("full_lr must have shape (n_data, n_blocks)")

    u_grid = np.linspace(
        float(u0) - float(grid_radius) * float(sigma_q),
        float(u0) + float(grid_radius) * float(sigma_q),
        int(grid_size),
        dtype=np.float64,
    )
    pi_grid = stage1.sigmoid_np(u_grid)
    log_pi = np.log(np.clip(pi_grid, 1e-300, 1.0))
    log_one_minus_pi = np.log(np.clip(1.0 - pi_grid, 1e-300, 1.0))
    log_q_kernel = -0.5 * ((u_grid - float(u0)) / float(sigma_q)) ** 2
    proposal_score = (u_grid - float(u0)) / float(sigma_q) ** 2

    output = np.empty(ratios.shape[0], dtype=np.float64)
    for start in range(0, ratios.shape[0], max(1, int(chunk_size))):
        stop = min(start + max(1, int(chunk_size)), ratios.shape[0])
        batch = ratios[start:stop]
        block_log_mix = np.logaddexp(
            log_one_minus_pi[None, :, None],
            log_pi[None, :, None] + batch[:, None, :],
        )
        log_weight = log_q_kernel[None, :] + np.sum(block_log_mix, axis=2)
        log_norm = logsumexp_np(log_weight, axis=1)
        weight = np.exp(log_weight - log_norm[:, None])
        output[start:stop] = np.sum(weight * proposal_score[None, :], axis=1)
    return output
