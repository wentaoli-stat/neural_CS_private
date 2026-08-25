from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from model1 import npe_utils as model1_npe
from model1 import stage1 as model1
from model2 import npe_utils as model2_npe
from model2 import stage1 as model2


@pytest.mark.parametrize(
    ("stage1", "npe", "shape", "tau"),
    [
        (model1, model1_npe, (3, 5), 0.5),
        (model2, model2_npe, (3, 5), 1.0),
    ],
)
def test_exact_posterior_matches_direct_mixture_likelihood(stage1, npe, shape, tau) -> None:
    rng = np.random.default_rng(123)
    y = rng.normal(size=shape)
    args = Namespace(pi_prior_min=0.05, pi_prior_max=0.70, grid_size=301, tau=tau)
    grid, weights, cdf = npe.exact_posterior_grid_arrays(y, args)
    log_ratio = stage1.full_block_log_ratio(y, tau)
    direct = np.asarray(
        [
            np.logaddexp(np.log1p(-pi), np.log(pi) + log_ratio).sum()
            for pi in grid
        ]
    )
    direct = np.exp(direct - direct.max())
    direct /= direct.sum()
    np.testing.assert_allclose(weights, direct, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(cdf, np.cumsum(direct), rtol=0, atol=1e-15)

