"""One-step evaluation point for a frozen Model 1 score field.

For a learned score S(Y, beta), estimated from simulations only,

    b(beta) = E_{Y ~ p(. | beta)} S(Y, beta)                             (centering)
    H(beta) = d/dtheta E_{Y ~ p(. | theta)} S(Y, beta) at theta = beta   (sensitivity)

and, for data Y with data-only pilot u_hat,

    beta_1 = u_hat + {S(Y, u_hat) - b(u_hat)} / H(u_hat),

clipped to a maximum step and to the prior's u-range. For the exact score b = 0 and H is the
Fisher information, so beta_1 is a Newton step toward the MLE. A score multiplied by a smooth
function of beta (the tube's shrinkage) has b and H multiplied by the same factor, so the step is
unchanged. The generating parameter of test data is never used.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from model1 import stage1

ScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def sensitivity_grid(score_fn: ScoreFn, n_blocks: int, block_size: int, tau: float,
                     u_grid: np.ndarray, n: int, delta: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Centering b and sensitivity H of score_fn on u_grid, by central differences.

    The three data sets at theta = beta - delta, beta, beta + delta share their noise and
    block-activity uniforms, so the difference of means has little Monte Carlo error.
    """
    b = np.empty(len(u_grid))
    H = np.empty(len(u_grid))
    for i, u in enumerate(np.asarray(u_grid, dtype=np.float64)):
        rng = np.random.default_rng(seed + i)
        uniforms = rng.random((n, n_blocks))
        noise = rng.standard_normal((n, n_blocks, block_size))
        at = np.full(n, u)
        means = []
        for theta in (u - delta, u, u + delta):
            active = uniforms < float(stage1.sigmoid_np(theta))
            y = noise + tau * active[:, :, None]
            means.append(float(np.mean(score_fn(y, at))))
        b[i] = means[1]
        H[i] = (means[2] - means[0]) / (2.0 * delta)
    return b, H


def one_step(score_fn: ScoreFn, y: np.ndarray, pilot_u: np.ndarray, u_grid: np.ndarray,
             b: np.ndarray, H: np.ndarray, u_lo: float, u_hi: float,
             max_step: float = 3.0) -> tuple[np.ndarray, dict[str, float]]:
    pilot_u = np.asarray(pilot_u, dtype=np.float64)
    score = np.asarray(score_fn(y, pilot_u), dtype=np.float64).reshape(-1)
    b_at = np.interp(pilot_u, u_grid, b)
    h_at = np.interp(pilot_u, u_grid, H)
    usable = h_at > 1e-6
    step = np.where(usable, (score - b_at) / np.where(usable, h_at, 1.0), 0.0)
    step = np.clip(step, -max_step, max_step)
    beta1 = np.clip(pilot_u + step, u_lo, u_hi)
    return beta1, {
        "at_prior_bound": float(np.mean((beta1 <= u_lo) | (beta1 >= u_hi))),
        "step_capped": float(np.mean(np.abs(step) >= max_step)),
        "nonpositive_sensitivity": float(np.mean(~usable)),
        "mean_abs_step": float(np.mean(np.abs(beta1 - pilot_u))),
    }
