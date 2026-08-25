from __future__ import annotations

import numpy as np

from khoo_vs_jiang.block_mixture_jiang import (
    BlockMixtureConfig,
    block_log_likelihood_ratio,
    exact_mle,
    exact_score_p,
    simulate_raw_blocks,
)


def _config(model: str) -> BlockMixtureConfig:
    return BlockMixtureConfig(
        model=model,
        block_size=20,
        tau=0.5 if model == "model1" else 1.0,
        p_min=0.05,
        p_max=0.70,
        p_true=0.30,
        n_observations=20 if model == "model1" else 40,
    )


def test_block_shapes_and_reproducibility() -> None:
    for model in ("model1", "model2"):
        config = _config(model)
        p = np.linspace(0.1, 0.6, 7)
        left = simulate_raw_blocks(np.random.default_rng(7), p, config)
        right = simulate_raw_blocks(np.random.default_rng(7), p, config)
        assert left.shape == (7, 2, 10)
        np.testing.assert_array_equal(left, right)


def test_exact_score_matches_finite_difference() -> None:
    for model in ("model1", "model2"):
        config = _config(model)
        y = simulate_raw_blocks(np.random.default_rng(11), 0.3, config, n=23)
        p = 0.31
        h = 1e-6
        log_ratio = block_log_likelihood_ratio(y, config)

        def log_likelihood(value: float) -> np.ndarray:
            return np.logaddexp(np.log1p(-value), np.log(value) + log_ratio)

        finite_difference = (log_likelihood(p + h) - log_likelihood(p - h)) / (2 * h)
        np.testing.assert_allclose(
            exact_score_p(y, p, config).reshape(-1),
            finite_difference,
            rtol=2e-5,
            atol=2e-6,
        )


def test_exact_mle_has_near_zero_score_when_interior() -> None:
    for model in ("model1", "model2"):
        config = _config(model)
        y = simulate_raw_blocks(
            np.random.default_rng(17), config.p_true, config, n=2_000
        )
        estimate = exact_mle(y, config)
        assert config.p_min <= estimate <= config.p_max
        if config.p_min + 1e-5 < estimate < config.p_max - 1e-5:
            assert abs(float(exact_score_p(y, estimate, config).sum())) < 2e-4
