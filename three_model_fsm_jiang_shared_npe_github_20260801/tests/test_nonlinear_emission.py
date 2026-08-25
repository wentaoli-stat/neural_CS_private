from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from khoo_vs_jiang.dgp import HMMConfig
from khoo_vs_jiang.jiang_official import JiangFitted, JiangTrainingConfig
from khoo_vs_jiang.nonlinear_emission_screen import (
    SparseScaleMixtureEmission,
    coordinate_log_ratio,
    exact_score_u,
    full_emission_log_ratio,
    pairwise_emission_log_ratio,
    raw_subscores,
    simulate_trajectories,
)
from khoo_vs_jiang.nonlinear_emission_repeated import exact_loglik_relative
from khoo_vs_jiang.nonlinear_emission_stage2 import hmm_loglik_axis
from khoo_vs_jiang.nonlinear_emission_stage2_ablation import (
    jiang_score_u_at_pilot,
)
from khoo_vs_jiang.nonlinear_emission_twoparam import (
    GatedLocalGRU2D,
    LinearLocalGRU2D,
    TwoTransitionHMMConfig,
    exact_score_u_2d,
    sample_amortized_2d,
    simulate_trajectories_2d,
)
from khoo_vs_jiang.nonlinear_emission_twoparam_eval import (
    exact_loglik_relative_2d,
    solve_score_root_2d,
)
from khoo_vs_jiang.nonlinear_emission_twoparam_stage2 import (
    exact_posterior_grid_2d,
    soft_transition_pilot,
)
from khoo_vs_jiang.upstreams import load_upstreams


def test_sparse_scale_mixture_log_ratios_match_density_definition() -> None:
    emission = SparseScaleMixtureEmission(rho=0.2, scale=3.0)
    y = np.array([-2.0, 0.0, 1.5])
    ratio = (1.0 - emission.rho) + emission.rho / emission.scale * np.exp(
        0.5 * y**2 * (1.0 - 1.0 / emission.scale**2)
    )
    np.testing.assert_allclose(coordinate_log_ratio(y, emission), np.log(ratio))


def test_nonlinear_emission_shapes_and_exact_score() -> None:
    upstreams = load_upstreams()
    hmm = HMMConfig(length=8, block_size=4)
    emission = SparseScaleMixtureEmission(rho=0.1, scale=5.0)
    y = simulate_trajectories(
        np.random.default_rng(9), hmm.p_true, hmm, emission, n=6
    )
    coordinate = coordinate_log_ratio(y, emission)
    pairwise = pairwise_emission_log_ratio(coordinate)
    full = full_emission_log_ratio(y, emission)
    s1, s2 = raw_subscores(y, emission, 0.3, upstreams)
    score = exact_score_u(y, hmm.p_true, hmm, emission, upstreams)

    assert y.shape == (6, 8, 4)
    assert coordinate.shape == (6, 8, 4)
    assert pairwise.shape == (6, 8, 6)
    assert full.shape == (6, 8)
    np.testing.assert_allclose(full, coordinate.sum(axis=2))
    assert s1.shape == coordinate.shape
    assert s2.shape == pairwise.shape
    assert score.shape == (6, 1)
    assert np.all(np.isfinite(score))

    class Fitted:
        pass

    fitted = Fitted()
    fitted.hmm = hmm
    fitted.emission = emission
    epsilon = 1e-5
    u = np.log(hmm.p_true) - np.log1p(-hmm.p_true)
    p_plus = 1.0 / (1.0 + np.exp(-(u + epsilon)))
    p_minus = 1.0 / (1.0 + np.exp(-(u - epsilon)))
    finite_difference = (
        exact_loglik_relative(fitted, y, p_plus)
        - exact_loglik_relative(fitted, y, p_minus)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(finite_difference, score.sum(), rtol=2e-5, atol=2e-5)

    axis = np.array([0.85, hmm.p_true, 0.97])
    stage2_loglik = hmm_loglik_axis(y[0], axis, fitted)
    repeated_loglik = np.array(
        [exact_loglik_relative(fitted, y[:1], value) for value in axis]
    )
    np.testing.assert_allclose(stage2_loglik, repeated_loglik)


def test_jiang_stage2_feature_is_converted_from_p_to_logit_score() -> None:
    class IdentityTheta(nn.Module):
        def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            del x
            return theta

    class ZeroDebias(nn.Module):
        def forward(self, theta: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(theta)

    hmm = HMMConfig(length=2, block_size=1)
    fitted = JiangFitted(
        score_model=IdentityTheta(),
        debias_model=ZeroDebias(),
        scale_theta=2.0,
        hmm=hmm,
        training=JiangTrainingConfig.smoke(),
        proposal=None,
        round_id=1,
        diagnostics={},
        score_architecture="raw_sequence_gru",
    )
    p = np.array([0.85, 0.94], dtype=np.float64)
    u = np.log(p / (1.0 - p))
    y = np.zeros((2, hmm.length, hmm.block_size), dtype=np.float32)
    score_u = jiang_score_u_at_pilot(fitted, y, u, device="cpu")

    # Native network output is score_v=v=2p. Since v=2p, score_p=2*score_v=4p;
    # the logit-coordinate score is score_u=p(1-p)score_p.
    np.testing.assert_allclose(score_u, 4.0 * p**2 * (1.0 - p), rtol=1e-6)


def test_two_parameter_simulation_and_direct_fsm_shapes() -> None:
    hmm = TwoTransitionHMMConfig(length=8, block_size=4)
    emission = SparseScaleMixtureEmission(rho=0.2, scale=3.0)
    raw = sample_amortized_2d(
        np.random.default_rng(17), 12, hmm, emission, np.array([0.2, 0.3])
    )
    assert raw["y"].shape == (12, 8, 4)
    assert raw["anchor_u"].shape == (12, 2)
    assert raw["theta"].shape == (12, 2)
    assert raw["target"].shape == (12, 2)
    assert np.all(np.isfinite(raw["target"]))


def test_two_parameter_exact_score_matches_loglik_finite_difference() -> None:
    upstreams = load_upstreams()
    hmm = TwoTransitionHMMConfig(length=8, block_size=4)
    emission = SparseScaleMixtureEmission(rho=0.2, scale=3.0)
    theta = hmm.truth
    y = simulate_trajectories_2d(
        np.random.default_rng(23), theta, hmm, emission, n=7
    )
    score = exact_score_u_2d(y, theta, hmm, emission, upstreams)

    class Fitted:
        pass

    fitted = Fitted()
    fitted.hmm = hmm
    fitted.emission = emission
    u = np.log(theta) - np.log1p(-theta)
    epsilon = 1e-5
    finite_difference = np.zeros(2)
    for coordinate in range(2):
        direction = np.zeros(2)
        direction[coordinate] = epsilon
        plus = 1.0 / (1.0 + np.exp(-(u + direction)))
        minus = 1.0 / (1.0 + np.exp(-(u - direction)))
        finite_difference[coordinate] = (
            exact_loglik_relative_2d(fitted, y, plus)
            - exact_loglik_relative_2d(fitted, y, minus)
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        finite_difference, score.sum(axis=0), rtol=3e-5, atol=3e-5
    )


def test_two_parameter_gate_strictly_nests_linear_model_at_initialization() -> None:
    upstreams = load_upstreams()
    time_mean = np.zeros((1, 1, 2), dtype=np.float32)
    time_sd = np.ones((1, 1, 2), dtype=np.float32)
    linear = LinearLocalGRU2D(8, time_mean, time_sd, upstreams)
    gate = GatedLocalGRU2D(8, 4, time_mean, time_sd, upstreams)
    gate.core.load_state_dict(linear.core.state_dict())
    generator = torch.Generator().manual_seed(29)
    s1 = torch.randn(5, 7, 3, generator=generator)
    s2 = torch.randn(5, 7, 6, generator=generator)
    anchor = torch.randn(5, 2, generator=generator)
    with torch.no_grad():
        np.testing.assert_allclose(
            linear(s1, s2, anchor).numpy(),
            gate(s1, s2, anchor).numpy(),
            rtol=0.0,
            atol=1e-7,
        )


def test_two_parameter_stage2_pilot_and_exact_grid_are_finite() -> None:
    class Fitted:
        pass

    fitted = Fitted()
    fitted.hmm = TwoTransitionHMMConfig(length=8, block_size=4)
    fitted.emission = SparseScaleMixtureEmission(rho=0.2, scale=3.0)
    y = simulate_trajectories_2d(
        np.random.default_rng(31),
        fitted.hmm.truth,
        fitted.hmm,
        fitted.emission,
        n=3,
    )
    pilot_u, status = soft_transition_pilot(y, fitted)
    assert pilot_u.shape == (3, 2)
    assert status.shape == (3,)
    assert np.all(np.isfinite(pilot_u))

    exact = exact_posterior_grid_2d(y[0], fitted, grid_size=21)
    assert exact["weight"].shape == (21, 21)
    np.testing.assert_allclose(exact["weight"].sum(), 1.0)
    assert exact["mean"].shape == (2,)
    assert exact["covariance"].shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(exact["covariance"]) >= -1e-12)


def test_two_parameter_root_solver_resolves_float32_network() -> None:
    upstreams = load_upstreams()
    hmm = TwoTransitionHMMConfig(length=4, block_size=2)
    emission = SparseScaleMixtureEmission(rho=0.2, scale=3.0)
    target_u = torch.as_tensor(
        np.log(hmm.truth) - np.log1p(-hmm.truth), dtype=torch.float32
    )

    class CenteredField(nn.Module):
        def forward(
            self, s1: torch.Tensor, s2: torch.Tensor, anchor_z: torch.Tensor
        ) -> torch.Tensor:
            del s1, s2
            return anchor_z - target_u.to(anchor_z.device)

    fitted = SimpleNamespace(
        linear_model=CenteredField(),
        gate_model=CenteredField(),
        stats={
            "s1_mean": np.asarray(0.0),
            "s1_sd": np.asarray(1.0),
            "s2_mean": np.asarray(0.0),
            "s2_sd": np.asarray(1.0),
            "anchor_u_mean": np.zeros(2),
            "anchor_u_sd": np.ones(2),
        },
        hmm=hmm,
        emission=emission,
        training=SimpleNamespace(pi_ref=0.3),
    )
    y = simulate_trajectories_2d(
        np.random.default_rng(37), hmm.truth, hmm, emission, n=3
    )
    root = solve_score_root_2d(
        fitted,
        upstreams,
        y,
        "linear",
        device="cpu",
        max_nfev=50,
        max_initializations=1,
    )
    np.testing.assert_allclose(root["theta_hat"], hmm.truth, atol=2e-5)
    assert root["nfev"] > 1
    assert root["score_converged"]
