from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from model1.stage1 import load_matched_linear_rho
from model2 import runtime
from model2 import stage1 as m2
from model2 import stage2_npe as m2_stage2

STATS = dict(s1_mean=np.array(0.13), s1_sd=np.array(0.7),
             s2_mean=np.array(-0.08), s2_sd=np.array(1.2))


def matched(m_dim=2, shared=True, constant=True):
    torch.manual_seed(11)
    linear = m2.LinearCSBetaDeepSets(32, 2)
    stacked = m2.StackedCSBetaDeepSets(32, 2, 16, **STATS, m_dim=m_dim,
                                       include_constant_channel=constant,
                                       share_local_features=shared)
    load_matched_linear_rho(stacked.rho, linear.rho.state_dict(),
                            stacked.linear_column_map,
                            constant_column=0 if constant else None)
    return linear, stacked


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("m_dim", [1, 4])
def test_stacked_nests_ilsa_exactly(shared, m_dim):
    linear, stacked = matched(m_dim, shared)
    s1, s2 = torch.randn(5, 4, 7), torch.randn(5, 4, 11)
    anchor = torch.randn(5)
    block = torch.stack([s1.mean(2), s2.mean(2)], -1)
    torch.testing.assert_close(stacked(block, s1, s2, anchor), linear(block, anchor),
                               rtol=0, atol=1e-6)


def test_shared_and_split_differ_only_in_parameter_sharing():
    _, shared = matched(shared=True)
    _, split = matched(shared=False)
    assert shared.local_features_pairwise is None
    assert split.local_features_pairwise is not None
    # The split variant carries exactly one extra local MLP worth of parameters.
    extra = sum(p.numel() for p in split.parameters()) - sum(p.numel() for p in shared.parameters())
    assert extra == sum(p.numel() for p in shared.local_features.parameters())


def test_learned_channel_starts_at_zero_but_keeps_gradients():
    _, stacked = matched()
    s1, s2 = torch.randn(4, 3, 6), torch.randn(4, 3, 9)
    anchor = torch.randn(4)
    block = torch.stack([s1.mean(2), s2.mean(2)], -1)
    raw1 = s1 * stacked.s1_sd + stacked.s1_mean
    a1 = anchor.reshape(-1, 1, 1).expand_as(raw1)
    assert torch.count_nonzero(stacked.local_features(raw1, a1)) == 0
    stacked(block, s1, s2, anchor).sum().backward()
    assert stacked.local_features.net[-1].weight.grad.abs().max() > 0


def test_both_channels_remain_separate():
    """Swapping the two channels must change the output: they are not merged."""
    _, stacked = matched(m_dim=1)
    torch.nn.init.normal_(stacked.local_features.net[-1].weight, std=0.2)
    s1, s2 = torch.randn(5, 4, 8), torch.randn(5, 4, 8)
    anchor = torch.randn(5)
    block = torch.stack([s1.mean(2), s2.mean(2)], -1)
    swapped = torch.stack([s2.mean(2), s1.mean(2)], -1)
    assert not torch.allclose(stacked(block, s1, s2, anchor),
                              stacked(swapped, s2, s1, anchor), atol=1e-6)


def test_permutation_invariance_within_and_across_blocks():
    _, stacked = matched(m_dim=2)
    torch.nn.init.normal_(stacked.local_features.net[-1].weight, std=0.1)
    s1, s2 = torch.randn(5, 4, 7), torch.randn(5, 4, 11)
    anchor = torch.randn(5)
    block = torch.stack([s1.mean(2), s2.mean(2)], -1)
    base = stacked(block, s1, s2, anchor)
    torch.testing.assert_close(base, stacked(block, s1.flip(2), s2.flip(2), anchor))
    torch.testing.assert_close(base, stacked(block.flip(1), s1.flip(1), s2.flip(1), anchor))


def test_method_surface_and_defaults_are_backward_compatible():
    args = m2.build_parser().parse_args([])
    assert args.methods == "linear,shared_radial"      # packaged default unchanged
    assert (args.m_dim, args.include_constant_channel) == (1, 1)
    assert m2.STACKED_METHODS == {"stacked_shared", "stacked_split"}
    assert m2.NONLINEAR_METHODS == {"shared_radial", "stacked_shared", "stacked_split"}
    assert set(m2_stage2.METHOD_LABELS) == {
        "pilot", "linear", "shared_radial", "stacked_shared", "stacked_split"}
    # Existing seed indices must not shift, or archived NPE runs change.
    assert m2_stage2.METHOD_SEED_INDEX["linear"] == 0
    assert m2_stage2.METHOD_SEED_INDEX["shared_radial"] == 1
    assert m2_stage2.METHOD_SEED_INDEX["pilot"] == 2


@pytest.mark.parametrize("method", ["stacked_shared", "stacked_split"])
def test_end_to_end_train_and_frozen_runtime_replay(tmp_path, method):
    args = m2.build_parser().parse_args([
        "--n-train", "128", "--n-val", "64", "--n-blocks", "3", "--block-size", "5",
        "--tau", "1.0", "--methods", f"linear,{method}", "--m-dim", "2",
        "--iters", "4", "--batch-size", "32", "--print-every", "2",
        "--milestone-steps", "4", "--lr-schedule", "constant",
        "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    m2.run(args)
    info = json.loads((tmp_path / "training_info.json").read_text())
    assert info[method]["nested_initialization_max_abs_diff"] < 1e-6
    assert info[method]["local_map"] == "stacked_(1,s,m)"
    assert len({i["minibatch_indices_sha256"] for i in info.values()}) == 1
    score = runtime.AmortizedScoreRuntime(tmp_path, method, device="cpu")
    u = m2.logit_np(np.array([0.2, 0.4]))
    y = m2.simulate_common_factor(np.random.default_rng(3), u, 3, 5, 1.0)
    assert np.isfinite(score.score(y, u)).all()
