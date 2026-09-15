"""ILSA with local map (1, s): exact nesting in the gate and stacked maps, and legacy loading."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from model1 import runtime, stage1
from model1_p3 import stage1 as p3

P1_STATS = {"block_mean": np.zeros((1, 1, 1)), "block_sd": np.ones((1, 1, 1))}
P1_CFG = {"hidden": 32, "depth": 2, "gate_hidden": 8, "m_dim": 3,
          "include_constant_channel": 1, "gate_condition_on_anchor": True}


def test_p1_constant_ilsa_is_nested_by_gate_and_stacked():
    torch.manual_seed(3)
    cfg = {**P1_CFG, "linear_constant_channel": 1}
    linear = stage1.build_model("linear", cfg, P1_STATS)
    assert linear.rho[0].in_features == 3
    radial = stage1.build_model("radial", cfg, P1_STATS)
    radial.rho.load_state_dict(linear.rho.state_dict())
    stacked = stage1.build_model("stacked", cfg, P1_STATS)
    column_map, constant_column = stage1.stacked_linear_column_map(stacked, True)
    stage1.load_matched_linear_rho(stacked.rho, linear.rho.state_dict(), column_map,
                                   constant_column=constant_column)
    s = torch.randn(5, 4, 6)
    anchor = torch.randn(5)
    block = s.mean(dim=2, keepdim=True)
    reference = linear(block, anchor)
    torch.testing.assert_close(radial(s, anchor), reference, atol=1e-6, rtol=0)
    torch.testing.assert_close(stacked(block, s, anchor), reference, atol=1e-6, rtol=0)
    # The constant's readout weights carry over instead of starting at zero.
    torch.testing.assert_close(stacked.rho[0].weight[:, 0], linear.rho[0].weight[:, 0])


def test_p1_legacy_config_keeps_score_only_ilsa():
    assert stage1.build_model("linear", P1_CFG, P1_STATS).rho[0].in_features == 2
    assert stage1.build_model("radial", P1_CFG, P1_STATS).rho[0].in_features == 2


def test_p1_constant_ilsa_needs_a_stacked_constant():
    stacked = stage1.StackedCSBetaDeepSets(32, 2, 8, 1, include_constant_channel=False)
    with pytest.raises(ValueError, match="constant channel"):
        stage1.stacked_linear_column_map(stacked, True)
    args = stage1.build_parser().parse_args([
        "--methods", "linear,stacked", "--include-constant-channel", "0", "--output-dir", "unused",
    ])
    with pytest.raises(ValueError, match="linear-constant-channel"):
        stage1.run(args)


def test_constant_ilsa_is_the_default_for_new_runs():
    assert stage1.build_parser().parse_args([]).linear_constant_channel == 1
    assert p3.build_parser().parse_args([]).linear_constant_channel == 1


def test_p1_runtime_rejects_a_changed_linear_constant(tmp_path):
    args = stage1.build_parser().parse_args([
        "--n-train", "64", "--n-val", "32", "--n-blocks", "4", "--block-size", "7",
        "--methods", "linear", "--iters", "2", "--batch-size", "16", "--lr-schedule", "constant",
        "--print-every", "1", "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    stage1.run(args)
    runtime.AmortizedScoreRuntime(tmp_path, "linear", "cpu")
    config = json.loads((tmp_path / "config.json").read_text())
    config["linear_constant_channel"] = 0
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="linear_constant_channel"):
        runtime.AmortizedScoreRuntime(tmp_path, "linear", "cpu")


def test_p3_constant_ilsa_is_nested_by_gate_and_stacked():
    torch.manual_seed(5)
    stats = {"block_mean": np.zeros(3), "block_sd": np.ones(3)}
    cfg = {"hidden": 32, "depth": 2, "gate_hidden": 16, "m_dim": 2,
           "include_constant_channel": 1, "linear_constant_channel": 1}
    linear = p3.build_model("linear", cfg, stats)
    assert linear.rho[0].in_features == 1 + 2 * p3.P
    gate = p3.build_model("gate", cfg, stats)
    gate.rho.load_state_dict(linear.rho.state_dict())
    stacked = p3.build_model("stacked", cfg, stats)
    column_map, constant_column = p3.stacked_linear_column_map(stacked, True)
    p3.load_matched_linear_rho(stacked.rho, linear.rho.state_dict(), column_map,
                               constant_column=constant_column)
    s = torch.randn(6, 7, 9, 3)
    anchor = torch.randn(6, 3)
    block = s.mean(dim=2)
    reference = linear(block, anchor)
    torch.testing.assert_close(gate(s, anchor), reference, atol=1e-6, rtol=0)
    torch.testing.assert_close(stacked(block, s, anchor), reference, atol=1e-6, rtol=0)
    legacy = {key: value for key, value in cfg.items() if key != "linear_constant_channel"}
    assert p3.build_model("linear", legacy, stats).rho[0].in_features == 2 * p3.P
