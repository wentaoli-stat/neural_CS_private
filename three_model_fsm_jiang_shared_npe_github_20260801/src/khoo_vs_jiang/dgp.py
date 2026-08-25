"""Common iid-trajectory HMM data-generating process."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .upstreams import UpstreamModules


@dataclass(frozen=True)
class HMMConfig:
    length: int = 100
    block_size: int = 10
    tau: float = 0.5
    fixed_p01: float = 0.06
    init_prob: float = 0.5
    p_min: float = 0.80
    p_max: float = 0.99
    p_true: float = 0.94

    def validate(self) -> None:
        if self.length < 2 or self.block_size < 2:
            raise ValueError("length and block_size must both be at least two")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if not 0.0 < self.fixed_p01 < 1.0:
            raise ValueError("fixed_p01 must lie in (0,1)")
        if not 0.0 < self.init_prob < 1.0:
            raise ValueError("init_prob must lie in (0,1)")
        if not 0.0 < self.p_min < self.p_true < self.p_max < 1.0:
            raise ValueError("require 0 < p_min < p_true < p_max < 1")

    @property
    def x_dim(self) -> int:
        return self.length * self.block_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def logit(probability: np.ndarray | float) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    return np.log(probability) - np.log1p(-probability)


def simulate_trajectories(
    rng: np.random.Generator,
    p11: np.ndarray | float,
    config: HMMConfig,
    upstreams: UpstreamModules,
    *,
    n: int | None = None,
    return_states: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate independent complete trajectories.

    Each returned row is one outer observation in Jiang's additive model.
    Dependence is confined to coordinates within that row.
    """
    config.validate()
    values = np.asarray(p11, dtype=np.float64).reshape(-1)
    if values.size == 1 and n is not None:
        values = np.full(int(n), float(values[0]), dtype=np.float64)
    elif n is not None and values.size != int(n):
        raise ValueError("p11 and n imply inconsistent sample sizes")
    if values.size < 1 or np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("all p11 values must lie in (0,1)")
    transition_u = np.stack(
        [
            np.full(values.size, float(logit(config.fixed_p01))),
            logit(values),
        ],
        axis=1,
    )
    y, states = upstreams.khoo_fixed.simulate_hmm_common_factor(
        rng,
        transition_u,
        int(config.length),
        int(config.block_size),
        float(config.tau),
        float(config.init_prob),
    )
    y = np.asarray(y, dtype=np.float32)
    if return_states:
        return y, np.asarray(states)
    return y


def flatten_trajectories(y: np.ndarray, config: HMMConfig) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    expected = (config.length, config.block_size)
    if y.ndim != 3 or y.shape[1:] != expected:
        raise ValueError(f"expected y shape (n,{expected[0]},{expected[1]}), got {y.shape}")
    return np.ascontiguousarray(y.reshape(y.shape[0], config.x_dim))


def exact_score_u(
    y: np.ndarray,
    p11: float,
    config: HMMConfig,
    upstreams: UpstreamModules,
) -> np.ndarray:
    """Exact score in u=logit(p11), for diagnostics only."""
    return np.asarray(
        upstreams.khoo_oneparam.exact_score_u(
            np.asarray(y, dtype=np.float64),
            float(config.fixed_p01),
            float(p11),
            float(config.tau),
            float(config.init_prob),
        ),
        dtype=np.float64,
    )


def exact_score_p(
    y: np.ndarray,
    p11: float,
    config: HMMConfig,
    upstreams: UpstreamModules,
) -> np.ndarray:
    """Exact score in the physical p11 coordinate, for diagnostics only."""
    return exact_score_u(y, p11, config, upstreams) / (p11 * (1.0 - p11))

