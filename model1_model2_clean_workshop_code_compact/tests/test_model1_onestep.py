"""One-step evaluation point: sensitivity of the exact score, scale invariance, and direction."""
from __future__ import annotations

import numpy as np

from model1 import onestep, stage1

K, M, TAU = 20, 5, 1.0


def exact_score(y, u):
    return stage1.full_score_u(y, stage1.sigmoid_np(u), TAU)


def test_exact_score_is_centred_with_sensitivity_equal_to_fisher_information():
    grid = stage1.logit_np(np.array([0.2, 0.5]))
    b, H = onestep.sensitivity_grid(exact_score, K, M, TAU, grid, n=6000, delta=0.05, seed=3)
    for i, u in enumerate(grid):
        y = stage1.simulate_mean_shift(np.random.default_rng(11), np.full(6000, u), K, M, TAU)
        info = float(np.var(exact_score(y, np.full(6000, u))))
        assert abs(b[i]) < 4 * np.sqrt(info / 6000)
        assert abs(H[i] / info - 1) < 0.08


def test_one_step_does_not_change_when_the_score_is_rescaled():
    grid = stage1.logit_np(np.linspace(0.1, 0.6, 6))
    scaled = lambda y, u: 0.3 * exact_score(y, u)  # noqa: E731
    b, H = onestep.sensitivity_grid(exact_score, K, M, TAU, grid, 800, 0.05, seed=5)
    bs, Hs = onestep.sensitivity_grid(scaled, K, M, TAU, grid, 800, 0.05, seed=5)
    y = stage1.simulate_mean_shift(np.random.default_rng(2), np.full(50, stage1.logit_np(0.3)), K, M, TAU)
    pilot = np.full(50, stage1.logit_np(0.3) + 0.3)
    lo, hi = grid[0], grid[-1]
    beta1, _ = onestep.one_step(exact_score, y, pilot, grid, b, H, lo, hi)
    beta1_scaled, _ = onestep.one_step(scaled, y, pilot, grid, bs, Hs, lo, hi)
    np.testing.assert_allclose(beta1_scaled, beta1, atol=1e-9)


def test_one_step_moves_a_displaced_pilot_toward_the_mle():
    # A Newton step targets each dataset's MLE, whose own sampling error can exceed the
    # pilot's displacement from the truth, so the comparison is with the exact MLE.
    grid = stage1.logit_np(np.linspace(0.1, 0.6, 11))
    b, H = onestep.sensitivity_grid(exact_score, 40, M, TAU, grid, 1000, 0.05, seed=7)
    truth = stage1.logit_np(0.3)
    y = stage1.simulate_mean_shift(np.random.default_rng(4), np.full(400, truth), 40, M, TAU)
    u_fine = np.linspace(grid[0], grid[-1], 4001)
    L = stage1.full_block_log_ratio(y, TAU)
    loglik = np.logaddexp(0.0, u_fine[None, :, None] + L[:, None, :]).sum(axis=2) - 40 * np.logaddexp(0.0, u_fine)
    mle = u_fine[np.argmax(loglik, axis=1)]
    interior = (mle > grid[0] + 0.3) & (mle < grid[-1] - 0.3)
    pilot = np.clip(mle + 0.5, grid[0], grid[-1])
    beta1, diag = onestep.one_step(exact_score, y, pilot, grid, b, H, grid[0], grid[-1])
    assert np.mean(np.abs(beta1 - mle)[interior]) < 0.5 * np.mean(np.abs(pilot - mle)[interior])
    assert diag["nonpositive_sensitivity"] == 0.0
