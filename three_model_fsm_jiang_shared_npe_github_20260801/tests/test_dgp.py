import numpy as np

from khoo_vs_jiang.dgp import HMMConfig, exact_score_p, flatten_trajectories, simulate_trajectories
from khoo_vs_jiang.upstreams import load_upstreams


def test_iid_trajectory_shapes_and_exact_score() -> None:
    upstreams = load_upstreams()
    config = HMMConfig(length=8, block_size=4, p_true=0.94)
    y = simulate_trajectories(
        np.random.default_rng(7), config.p_true, config, upstreams, n=6
    )
    assert y.shape == (6, 8, 4)
    assert flatten_trajectories(y, config).shape == (6, 32)
    score = exact_score_p(y, config.p_true, config, upstreams)
    assert score.shape == (6, 1)
    assert np.all(np.isfinite(score))

