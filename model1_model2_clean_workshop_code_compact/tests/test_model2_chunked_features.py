from __future__ import annotations

import numpy as np
import pytest

from model2 import stage1


def _bank(n: int = 9, seed: int = 4):
    rng = np.random.default_rng(seed)
    anchor_pi = rng.uniform(0.05, 0.70, size=n)
    y = stage1.simulate_common_factor(rng, stage1.logit_np(anchor_pi), 3, 6, 1.0)
    return y, anchor_pi


@pytest.mark.parametrize("scalar_pi", [False, True])
def test_chunked_featurize_is_bit_identical(scalar_pi: bool) -> None:
    y, anchor_pi = _bank()
    pi = 0.3 if scalar_pi else anchor_pi
    stats = stage1.make_feature_stats(y, 1.0, pi)
    one_shot = stage1.featurize(y, 1.0, pi, stats)
    chunked = stage1.featurize(y, 1.0, pi, stats, chunk_rows=4)
    assert set(one_shot) == set(chunked)
    for key in one_shot:
        assert one_shot[key].dtype == chunked[key].dtype
        np.testing.assert_array_equal(one_shot[key], chunked[key])


@pytest.mark.parametrize("scalar_pi", [False, True])
def test_chunked_feature_stats_match_one_shot(scalar_pi: bool) -> None:
    y, anchor_pi = _bank()
    pi = 0.3 if scalar_pi else anchor_pi
    one_shot = stage1.make_feature_stats(y, 1.0, pi)
    chunked = stage1.make_feature_stats(y, 1.0, pi, chunk_rows=4)
    assert set(one_shot) == set(chunked)
    for key in one_shot:
        assert np.shape(one_shot[key]) == np.shape(chunked[key])
        np.testing.assert_allclose(chunked[key], one_shot[key], rtol=1e-12, atol=1e-14)


def test_default_path_is_unchanged_for_small_banks() -> None:
    y, anchor_pi = _bank(n=5)
    assert y.shape[0] <= stage1.FEATURE_CHUNK_ROWS
    stats = stage1.make_feature_stats(y, 1.0, anchor_pi)
    s1 = stage1.score_u_from_log_ratio(stage1.marginal_log_ratio(y, 1.0), anchor_pi)
    assert float(stats["s1_mean"]) == float(s1.mean())
    assert float(stats["s1_sd"]) == float(s1.std())
