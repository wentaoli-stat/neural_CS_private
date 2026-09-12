from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from model1.stage1 import load_matched_linear_rho
from model2_p3 import evaluate_stage1, scores, stage1

STATS = {"s1_mean": np.zeros(3), "s1_sd": np.ones(3), "s2_mean": np.zeros(3),
         "s2_sd": np.ones(3), "block_mean": np.zeros(6), "block_sd": np.ones(6)}
CFG = {"hidden": 32, "depth": 2, "gate_hidden": 16, "m_dim": 2, "include_constant_channel": 1}


def random_beta(rng, n):
    return np.stack([rng.uniform(-3, 1, n), rng.uniform(0.3, 2.0, n), rng.uniform(-0.4, 0.4, n)], 1)


def test_exact_score_matches_finite_differences():
    rng = np.random.default_rng(0)
    beta = random_beta(rng, 48)
    y = scores.simulate(rng, beta, 6, 8)
    h, approx = 1e-6, np.zeros_like(beta)
    for i in range(scores.N_PARAMS):
        up, dn = beta.copy(), beta.copy()
        up[:, i] += h
        dn[:, i] -= h
        approx[:, i] = (scores.log_likelihood(y, up) - scores.log_likelihood(y, dn)) / (2 * h)
    exact = scores.exact_score(y, beta)
    assert np.max(np.abs(approx - exact) / (np.abs(exact) + 1.0)) < 1e-5


def test_marginal_and_pairwise_are_the_r1_and_r2_members_of_the_family():
    """A marginal score is the exact score of a one-observation block; a pair
    score is the exact score of the corresponding two-observation block."""
    rng = np.random.default_rng(1)
    beta = random_beta(rng, 10)
    y = scores.simulate(rng, beta, 2, 4)
    marg = scores.marginal_scores(y, beta)
    for j in range(y.shape[2]):
        np.testing.assert_allclose(marg[:, 0, j, :], scores.exact_score(y[:, :1, j:j + 1], beta),
                                   rtol=1e-9, atol=1e-9)
    pair = scores.pairwise_scores(y, beta)
    i_idx, j_idx = np.triu_indices(y.shape[2], k=1)
    for p in range(len(i_idx)):
        sub = y[:, :1, [i_idx[p], j_idx[p]]]
        np.testing.assert_allclose(pair[:, 0, p, :], scores.exact_score(sub, beta),
                                   rtol=1e-9, atol=1e-9)


def test_exact_score_is_additive_over_blocks():
    rng = np.random.default_rng(2)
    beta = random_beta(rng, 8)
    y = scores.simulate(rng, beta, 5, 4)
    per_block = sum(scores.exact_score(y[:, k:k + 1, :], beta) for k in range(y.shape[1]))
    np.testing.assert_allclose(per_block, scores.exact_score(y, beta), rtol=1e-9, atol=1e-9)


def trio():
    torch.manual_seed(4)
    linear = stage1.build_model("linear", CFG, STATS)
    state = linear.rho.state_dict()
    built = {"linear": linear}
    for method in ("gate", "stacked_shared", "stacked_split"):
        model = stage1.build_model(method, CFG, STATS)
        if method in stage1.STACKED_METHODS:
            load_matched_linear_rho(model.rho, state, model.linear_column_map, constant_column=0)
        else:
            model.rho.load_state_dict(state)
        built[method] = model
    return built


@pytest.mark.parametrize("method", ["gate", "stacked_shared", "stacked_split"])
def test_nonlinear_maps_nest_ilsa_exactly(method):
    built = trio()
    s1, s2 = torch.randn(5, 4, 7, 3), torch.randn(5, 4, 11, 3)
    anchor = torch.randn(5, 3)
    block = torch.cat([s1.mean(2), s2.mean(2)], -1)
    reference = built["linear"](block, anchor)
    assert reference.shape == (5, 3)
    out = (built[method](s1, s2, anchor) if method == "gate"
           else built[method](block, s1, s2, anchor))
    torch.testing.assert_close(out, reference, rtol=0, atol=1e-6)


def test_shared_and_split_differ_only_by_one_local_mlp():
    built = trio()
    shared, split = built["stacked_shared"], built["stacked_split"]
    assert shared.local_features_pairwise is None
    assert split.local_features_pairwise is not None
    extra = sum(p.numel() for p in split.parameters()) - sum(p.numel() for p in shared.parameters())
    assert extra == sum(p.numel() for p in shared.local_features.parameters())


def test_channels_stay_separate_and_pooling_is_permutation_invariant():
    built = trio()
    model = built["stacked_shared"]
    torch.nn.init.normal_(model.local_features.net[-1].weight, std=0.2)
    s1, s2 = torch.randn(5, 4, 8, 3), torch.randn(5, 4, 8, 3)
    anchor = torch.randn(5, 3)
    block = torch.cat([s1.mean(2), s2.mean(2)], -1)
    base = model(block, s1, s2, anchor)
    torch.testing.assert_close(base, model(block, s1.flip(2), s2.flip(2), anchor))
    torch.testing.assert_close(base, model(block.flip(1), s1.flip(1), s2.flip(1), anchor))
    swapped = torch.cat([s2.mean(2), s1.mean(2)], -1)
    assert not torch.allclose(base, model(swapped, s2, s1, anchor), atol=1e-6)


def test_end_to_end_train_and_evaluate(tmp_path):
    args = stage1.build_parser().parse_args([
        "--n-train", "160", "--n-val", "80", "--n-blocks", "3", "--block-size", "5",
        "--iters", "4", "--batch-size", "32", "--print-every", "2", "--m-dim", "2",
        "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    stage1.run(args)
    info = json.loads((tmp_path / "training_info.json").read_text())
    assert set(info) == {"linear", "gate", "stacked_shared", "stacked_split"}
    assert len({i["minibatch_sha256"] for i in info.values()}) == 1
    for method in ("gate", "stacked_shared", "stacked_split"):
        assert info[method]["nested_initialization_max_abs_diff"] < 1e-5
    evaluation = evaluate_stage1.build_parser().parse_args([
        "--run-dir", str(tmp_path), "--n-test", "32",
        "--beta-values", "0.30:1.0:1.0", "--device", "cpu",
    ])
    evaluate_stage1.run(evaluation)
    assert (tmp_path / "score_summary_by_beta.csv").is_file()
