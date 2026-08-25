import numpy as np

from khoo_vs_jiang.block_mixture_compare import _config
from khoo_vs_jiang.block_mixture_jiang import simulate_raw_blocks
from khoo_vs_jiang.block_mixture_stage2_npe import (
    _simulate_full_datasets,
    exact_posterior,
    stratified_prior,
)


def test_stratified_prior_stays_inside_support_and_fills_strata():
    rng = np.random.default_rng(7)
    values = stratified_prior(rng, 20, 0.05, 0.70)
    assert values.shape == (20,)
    assert np.all(values >= 0.05)
    assert np.all(values <= 0.70)
    normalized = np.sort((values - 0.05) / (0.70 - 0.05))
    assert np.all(normalized >= np.arange(20) / 20)
    assert np.all(normalized <= (np.arange(20) + 1) / 20)


def test_full_dataset_simulator_returns_outer_dataset_and_block_axes():
    config = _config("model2")
    p = np.array([0.1, 0.3, 0.6])
    raw, container = _simulate_full_datasets(
        np.random.default_rng(11), p, config
    )
    assert raw.shape == (3, config.n_observations, config.block_size)
    assert container.shape == (
        3,
        config.n_observations,
        2,
        config.block_size // 2,
    )
    np.testing.assert_allclose(raw, container.reshape(raw.shape))


def test_exact_posterior_is_normalized_and_matches_prior_for_no_blocks():
    config = _config("model1")
    empty = np.empty((0, 2, config.block_size // 2), dtype=np.float64)
    axis, cdf, stats = exact_posterior(empty, config, 1001)
    assert axis.shape == cdf.shape == (1001,)
    assert np.isclose(cdf[-1], 1.0)
    assert np.isclose(stats["posterior_mean"], 0.5 * (config.p_min + config.p_max))
    assert abs(stats["q50"] - 0.5 * (config.p_min + config.p_max)) < 1e-3


def test_exact_posterior_changes_with_informative_data():
    config = _config("model1")
    low = simulate_raw_blocks(
        np.random.default_rng(19), 0.08, config, n=200
    )
    high = simulate_raw_blocks(
        np.random.default_rng(23), 0.65, config, n=200
    )
    _, _, low_stats = exact_posterior(low, config, 2001)
    _, _, high_stats = exact_posterior(high, config, 2001)
    assert low_stats["posterior_mean"] < high_stats["posterior_mean"]
