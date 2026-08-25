"""Model 1 exact-posterior reference plus shared NPE helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.npe import (
    exact_cdf_at,
    exact_sample_w1,
    import_sbi,
    resolve_device,
    sample_sbi_posterior,
    summarize_full_posterior,
    train_sbi_npe_method,
)
from model1 import stage1


def exact_posterior_grid_arrays(
    y: np.ndarray,
    args: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform-prior exact posterior grid, used for evaluation only."""
    y = np.asarray(y, dtype=np.float64)
    grid = np.linspace(float(args.pi_prior_min), float(args.pi_prior_max), int(args.grid_size))
    log_ratio = stage1.full_block_log_ratio(y, float(args.tau))
    log_weight = np.stack(
        [
            np.logaddexp(np.log1p(-pi), np.log(pi) + log_ratio).sum()
            for pi in grid
        ]
    )
    log_weight -= np.max(log_weight)
    weight = np.exp(log_weight)
    weight /= weight.sum()
    return grid, weight, np.cumsum(weight)


__all__ = [
    "exact_cdf_at",
    "exact_posterior_grid_arrays",
    "exact_sample_w1",
    "import_sbi",
    "resolve_device",
    "sample_sbi_posterior",
    "summarize_full_posterior",
    "train_sbi_npe_method",
]

