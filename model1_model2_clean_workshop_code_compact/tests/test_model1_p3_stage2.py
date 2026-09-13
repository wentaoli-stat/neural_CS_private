from __future__ import annotations

import csv
import inspect
import math

import numpy as np
import pytest

from model1_p3 import scores as sc
from model1_p3 import stage1
from model1_p3 import stage2_npe

LOW = np.asarray([sc.logit_np(0.05), 0.5, math.log(0.7)])
HIGH = np.asarray([sc.logit_np(0.70), 2.0, math.log(1.4)])


def test_pilot_signature_has_no_oracle_parameter() -> None:
    forbidden = {"beta", "beta_true", "theta", "pi", "pi_true"}
    assert not forbidden.intersection(inspect.signature(stage2_npe.data_only_pilot).parameters)


def test_grid_loglik_matches_closed_form_likelihood() -> None:
    rng = np.random.default_rng(3)
    beta = np.asarray([[sc.logit_np(0.3), 1.1, math.log(0.9)]])
    y = sc.simulate(rng, beta, 6, 5)[0]
    axes = (np.asarray([-1.0, 0.2]), np.asarray([0.7, 1.3, 1.9]), np.asarray([-0.2, 0.1]))
    grid = stage2_npe._grid_loglik(y.sum(axis=1), (y**2).sum(axis=1), y.shape[1], axes)
    for i, u in enumerate(axes[0]):
        for j, tau in enumerate(axes[1]):
            for k, lam in enumerate(axes[2]):
                expected = sc.log_likelihood(y[None], np.asarray([[u, tau, lam]]))[0]
                assert grid[i, j, k] == pytest.approx(expected, rel=1e-10, abs=1e-8)


def test_exact_posterior_grid_agrees_with_brute_force_mean() -> None:
    rng = np.random.default_rng(5)
    beta = np.asarray([[sc.logit_np(0.4), 1.2, math.log(1.0)]])
    y = sc.simulate(rng, beta, 20, 20)[0]
    exact = stage2_npe.exact_posterior(y, LOW, HIGH, coarse=40, fine=80)
    # Independent reference: a uniform 120^3 grid over the whole box.
    axes = tuple(np.linspace(LOW[i], HIGH[i], 120) for i in range(3))
    weights = stage2_npe._normalize(stage2_npe._grid_loglik(y.sum(axis=1), (y**2).sum(axis=1), 20, axes))
    mean, sd = stage2_npe._marginal_moments(axes, weights)
    assert np.all(np.abs(exact["mean"] - mean) <= 0.25 * sd)
    np.testing.assert_allclose(exact["sd"], sd, rtol=0.1)
    assert exact["window_edge_mass"] < 1e-6


def test_pilot_is_deterministic_and_recovers_beta_on_large_blocks() -> None:
    rng = np.random.default_rng(11)
    beta = np.asarray([[sc.logit_np(0.5), 1.5, math.log(0.9)], [sc.logit_np(0.3), 1.8, math.log(1.2)]])
    y = sc.simulate(rng, beta, 60, 60)
    args = stage2_npe.build_parser().parse_args(["--stage1-run-dir", "unused"])
    first, status = stage2_npe.data_only_pilot(y, LOW, HIGH, args)
    second, _ = stage2_npe.data_only_pilot(y, LOW, HIGH, args)
    np.testing.assert_array_equal(first, second)
    assert set(status) <= {"interior", "boundary"}
    assert np.all(np.abs(first - beta) <= np.asarray([0.35, 0.15, 0.05])), (first, beta)


def test_sliced_w1_is_zero_for_matching_distribution_and_positive_for_shift() -> None:
    rng = np.random.default_rng(2)
    beta = np.asarray([[sc.logit_np(0.3), 1.0, math.log(1.0)]])
    y = sc.simulate(rng, beta, 20, 20)[0]
    exact = stage2_npe.exact_posterior(y, LOW, HIGH, coarse=30, fine=40)
    mesh = np.meshgrid(*exact["axes"], indexing="ij")
    points = np.stack([g.reshape(-1) for g in mesh], axis=1)
    draws = points[rng.choice(points.shape[0], size=20_000, p=exact["weights"].reshape(-1))]
    shifted = draws + np.asarray([0.0, 0.2, 0.0])
    dirs = stage2_npe.fixed_directions(20, 1)
    result = stage2_npe.sliced_w1(exact, {"same": draws, "shift": shifted}, LOW, HIGH, dirs)
    assert result["same"] < 0.01
    assert result["shift"] > 5 * result["same"]


def test_tiny_p3_checkpoint_runs_through_stage2(tmp_path) -> None:
    stage1_dir = tmp_path / "stage1"
    stage1.run(stage1.build_parser().parse_args([
        "--n-train", "64", "--n-val", "32", "--n-blocks", "6", "--block-size", "5",
        "--methods", "linear,gate,stacked", "--m-dim", "2", "--iters", "3", "--batch-size", "16",
        "--lr-schedule", "constant", "--print-every", "1", "--device", "cpu",
        "--output-dir", str(stage1_dir),
    ]))
    out_dir = tmp_path / "stage2"
    stage2_npe.run(stage2_npe.build_parser().parse_args([
        "--stage1-run-dir", str(stage1_dir), "--n-sbi-train", "300",
        "--sbi-hidden-features", "8", "--sbi-num-components", "2", "--sbi-batch-size", "64",
        "--max-epochs", "2", "--stop-after-epochs", "1",
        "--test-betas", "0.3:1.0:1.0;0.6:1.5:0.8", "--test-seeds", "100,101",
        "--posterior-n", "200", "--coarse-grid-size", "12", "--grid-size", "16",
        "--sliced-directions", "8", "--pilot-steps", "20", "--pilot-sanity-n", "16",
        "--min-pilot-correlation", "-1", "--output-dir", str(out_dir),
    ]))
    with (out_dir / "posterior_by_seed.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 2 * 5
    methods = {row["method"] for row in rows}
    assert {stage2_npe.METHOD_LABELS[m] for m in stage2_npe.METHOD_ORDER} <= methods
    for row in rows:
        assert np.isfinite(float(row["sliced_w1"]))
        for c in stage2_npe.COORDS:
            assert np.isfinite(float(row[f"w1_{c}"]))
    with (out_dir / "paired_sliced_w1.csv").open() as handle:
        pairs = {(r["reference"], r["competitor"]) for r in csv.DictReader(handle)}
    label = stage2_npe.METHOD_LABELS
    assert (label["stacked"], label["gate"]) in pairs and (label["gate"], label["linear"]) in pairs
    assert (out_dir / "npe_stacked_state.pt").exists()
