from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from model1_p3 import evaluate_stage1, scores, stage1


def random_beta(rng, n):
    return np.stack([rng.uniform(-3, 1, n), rng.uniform(0.4, 2.0, n), rng.uniform(-0.4, 0.4, n)], axis=1)


def test_exact_score_matches_finite_differences():
    rng = np.random.default_rng(0)
    beta = random_beta(rng, 48)
    y = scores.simulate(rng, beta, 6, 8)
    h, approx = 1e-6, np.zeros_like(beta)
    for i in range(scores.N_PARAMS):
        up, down = beta.copy(), beta.copy()
        up[:, i] += h
        down[:, i] -= h
        approx[:, i] = (scores.log_likelihood(y, up) - scores.log_likelihood(y, down)) / (2 * h)
    exact = scores.exact_score(y, beta)
    assert np.max(np.abs(approx - exact) / (np.abs(exact) + 1.0)) < 1e-5


def test_local_scores_match_the_single_observation_mixture():
    rng = np.random.default_rng(1)
    beta = random_beta(rng, 12)
    y = scores.simulate(rng, beta, 3, 5)
    local = scores.local_scores(y, beta)
    # Each local score must equal the exact score of its own one-observation block.
    for j in range(y.shape[2]):
        single = scores.exact_score(y[:, :1, j : j + 1], beta)
        np.testing.assert_allclose(local[:, 0, j, :], single, rtol=1e-9, atol=1e-9)


def test_exact_score_is_additive_over_blocks():
    rng = np.random.default_rng(2)
    beta = random_beta(rng, 8)
    y = scores.simulate(rng, beta, 5, 4)
    per_block = sum(scores.exact_score(y[:, k : k + 1, :], beta) for k in range(y.shape[1]))
    np.testing.assert_allclose(per_block, scores.exact_score(y, beta), rtol=1e-9, atol=1e-9)


def matched_trio():
    torch.manual_seed(7)
    stats = {"block_mean": np.zeros(3), "block_sd": np.ones(3)}
    cfg = {"hidden": 32, "depth": 2, "gate_hidden": 16, "m_dim": 2, "include_constant_channel": 1}
    linear = stage1.build_model("linear", cfg, stats)
    state = linear.rho.state_dict()
    gate = stage1.build_model("gate", cfg, stats)
    gate.rho.load_state_dict(state)
    stacked = stage1.build_model("stacked", cfg, stats)
    stage1.load_matched_linear_rho(stacked.rho, state, stacked.linear_column_map, constant_column=0)
    return linear, gate, stacked


@pytest.mark.parametrize("method", ["gate", "stacked"])
def test_nonlinear_maps_nest_ilsa_exactly(method):
    linear, gate, stacked = matched_trio()
    s = torch.randn(6, 7, 9, 3)
    anchor = torch.randn(6, 3)
    block = s.mean(dim=2)
    reference = linear(block, anchor)
    assert reference.shape == (6, 3)
    out = gate(s, anchor) if method == "gate" else stacked(block, s, anchor)
    torch.testing.assert_close(out, reference, rtol=0, atol=1e-6)


def test_stacked_learned_channel_starts_at_zero_and_keeps_gradients():
    _, _, stacked = matched_trio()
    s = torch.randn(4, 5, 6, 3)
    anchor = torch.randn(4, 3)
    block = s.mean(dim=2)
    assert torch.count_nonzero(stacked.local_features(s, stage1.expand_anchor(anchor, s))) == 0
    stacked(block, s, anchor).sum().backward()
    assert stacked.local_features.net[-1].weight.grad.abs().max() > 0
    # The learned channel's readout columns must not be zeroed, or it is dead.
    start = 1 + scores.N_PARAMS
    assert torch.count_nonzero(stacked.rho[0].weight[:, start : start + stacked.m_dim]) > 0


@pytest.mark.parametrize("method", ["linear", "gate", "stacked"])
def test_permutation_invariance(method):
    linear, gate, stacked = matched_trio()
    torch.nn.init.normal_(stacked.local_features.net[-1].weight, std=0.1)
    torch.nn.init.normal_(gate.gate.net[-1].weight, std=0.1)
    s = torch.randn(5, 4, 7, 3)
    anchor = torch.randn(5, 3)
    block = s.mean(dim=2)
    call = {"linear": lambda b, x: linear(b, anchor),
            "gate": lambda b, x: gate(x, anchor),
            "stacked": lambda b, x: stacked(b, x, anchor)}[method]
    base = call(block, s)
    torch.testing.assert_close(base, call(block, s.flip(2)))            # within block
    torch.testing.assert_close(base, call(block.flip(1), s.flip(1)))    # across blocks


def test_stage1_runs_and_evaluates(tmp_path):
    args = stage1.build_parser().parse_args([
        "--n-train", "192", "--n-val", "96", "--n-blocks", "3", "--block-size", "5",
        "--iters", "4", "--batch-size", "32", "--print-every", "2",
        "--lr-schedule", "constant", "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    stage1.run(args)
    info = json.loads((tmp_path / "training_info.json").read_text())
    assert set(info) == {"linear", "gate", "stacked"}
    assert len({i["minibatch_sha256"] for i in info.values()}) == 1
    assert {i["best_source"] for i in info.values()} == {"raw"}
    for method in ("gate", "stacked"):
        assert info[method]["nested_initialization_max_abs_diff"] < 1e-6
    evaluation = evaluate_stage1.build_parser().parse_args([
        "--run-dir", str(tmp_path), "--n-test", "64",
        "--beta-values", "0.30:1.0:1.0", "--device", "cpu",
    ])
    evaluate_stage1.run(evaluation)
    assert (tmp_path / "score_summary_by_beta.csv").is_file()


def test_evaluation_rejects_points_outside_the_anchor_box(tmp_path):
    args = stage1.build_parser().parse_args([
        "--n-train", "96", "--n-val", "48", "--n-blocks", "3", "--block-size", "5",
        "--iters", "2", "--batch-size", "32", "--print-every", "2", "--methods", "linear",
        "--lr-schedule", "constant", "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    stage1.run(args)
    evaluation = evaluate_stage1.build_parser().parse_args([
        "--run-dir", str(tmp_path), "--n-test", "8",
        "--beta-values", "0.30:9.0:1.0", "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="outside the anchor box"):
        evaluate_stage1.run(evaluation)


def test_sigma_q_and_box_validation():
    with pytest.raises(ValueError, match="sigma-q"):
        stage1.parse_vector("0.2,0.1", "sigma-q")
    args = stage1.build_parser().parse_args(["--tau-min", "2.0", "--tau-max", "1.0"])
    with pytest.raises(ValueError, match="positive width"):
        stage1.anchor_box(args)


def test_blockstack_keeps_block_exchangeability_only():
    torch.manual_seed(3)
    cfg = {"hidden": 32, "depth": 2, "n_blocks": 4, "block_size": 6}
    model = stage1.build_model("blockstack", cfg, {})
    s = torch.randn(5, 4, 6, 3)
    anchor = torch.randn(5, 3)
    base = model(s, anchor)
    # Additive over blocks, so permuting blocks must not change the output.
    torch.testing.assert_close(base, model(s.flip(1), anchor))
    # But the within-block stack is ordered: that invariance is deliberately gone.
    assert not torch.allclose(base, model(s.flip(2), anchor), atol=1e-6)


@pytest.mark.parametrize("method", ["raw", "substack"])
def test_unstructured_baselines_have_no_permutation_invariance(method):
    torch.manual_seed(4)
    cfg = {"hidden": 32, "depth": 2, "n_blocks": 4, "block_size": 6}
    model = stage1.build_model(method, cfg, {})
    anchor = torch.randn(5, 3)
    x = torch.randn(5, 4, 6) if method == "raw" else torch.randn(5, 4, 6, 3)
    assert not torch.allclose(model(x, anchor), model(x.flip(1), anchor), atol=1e-6)


def test_baselines_are_not_warm_started_and_run_end_to_end(tmp_path):
    args = stage1.build_parser().parse_args([
        "--n-train", "192", "--n-val", "96", "--n-blocks", "3", "--block-size", "5",
        "--iters", "4", "--batch-size", "32", "--print-every", "2",
        "--methods", "linear,raw,substack,blockstack",
        "--lr-schedule", "constant", "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    stage1.run(args)
    info = json.loads((tmp_path / "training_info.json").read_text())
    assert set(info) == {"linear", "raw", "substack", "blockstack"}
    # None of the reference baselines nests ILSA, so none reports a nesting error.
    for method in ("raw", "substack", "blockstack"):
        assert "nested_initialization_max_abs_diff" not in info[method]
    assert len({i["minibatch_sha256"] for i in info.values()}) == 1
    evaluation = evaluate_stage1.build_parser().parse_args([
        "--run-dir", str(tmp_path), "--n-test", "64",
        "--beta-values", "0.30:1.0:1.0", "--device", "cpu",
    ])
    evaluate_stage1.run(evaluation)
    rows = list(csv.reader(open(tmp_path / "score_summary_by_beta.csv")))
    assert len(rows) == 5  # header plus one row per method


def test_raw_features_are_only_built_when_needed(tmp_path):
    stats_args = stage1.build_parser().parse_args(["--n-blocks", "3", "--block-size", "5"])
    rng = np.random.default_rng(0)
    sigma_q = stage1.parse_vector(stats_args.sigma_q, "sigma-q")
    low, high = stage1.anchor_box(stats_args)
    batch = stage1.sample_anchor_batch(rng, 16, low, high, sigma_q, 3, 5)
    stats = stage1.make_feature_stats(batch["y"], batch["anchor"])
    assert "y_z" not in stage1.featurize(batch["y"], batch["anchor"], stats)
    with_raw = stage1.featurize(batch["y"], batch["anchor"], stats, need_raw_y=True)
    assert with_raw["y_z"].shape == (16, 3, 5)
