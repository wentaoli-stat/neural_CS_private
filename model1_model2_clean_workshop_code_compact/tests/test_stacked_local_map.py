from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from model1 import evaluate_stage1, runtime, stage1


def matched_models(m_dim=1, constant=True):
    torch.manual_seed(19)
    linear = stage1.LinearCSBetaDeepSets(64, 2)
    stacked = stage1.StackedCSBetaDeepSets(64, 2, 16, m_dim, constant)
    stage1.load_matched_linear_rho(
        stacked.rho, linear.rho.state_dict(), stacked.linear_column_map,
        constant_column=0 if constant else None,
    )
    return linear, stacked


@pytest.mark.parametrize("m_dim", [1, 4])
@pytest.mark.parametrize("constant", [False, True])
def test_model1_stacked_identity_matches_linear(m_dim, constant):
    linear, stacked = matched_models(m_dim, constant)
    s = torch.randn(9, 20, 20)
    anchor = torch.randn(9)
    # Deliberately unrelated block inputs detect accidental recomputation.
    block = torch.randn(9, 20, 1)
    local = stacked.learned_features(s, anchor)
    assert local.shape == (9, 20, 20, m_dim)
    assert torch.count_nonzero(local) == 0
    inputs = stacked.readout_inputs(block, s, anchor)
    assert inputs.shape == (9, 20, int(constant) + 2 + m_dim)
    assert stacked.rho[0].in_features == inputs.shape[-1]
    torch.testing.assert_close(inputs[..., int(constant):int(constant)+1], block, rtol=0, atol=0)
    if constant:
        torch.testing.assert_close(inputs[..., 0], torch.ones(9, 20), rtol=0, atol=0)
        assert torch.count_nonzero(stacked.rho[0].weight[:, 0]) == 0
    torch.testing.assert_close(stacked(block, s, anchor), linear(block, anchor), atol=1e-6, rtol=0)
    start = int(constant) + 1
    assert torch.count_nonzero(stacked.rho[0].weight[:, start:start+m_dim]) > 0


def test_stacked_learns_after_zero_feature_initialization():
    _, model = matched_models()
    s = torch.randn(7, 4, 8)
    block = s.mean(2, keepdim=True)
    anchor = torch.randn(7)
    target = torch.randn(7)
    # No weight decay: updates must arise from the loss, not shrinkage.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    first_w = model.rho[0].weight[:, 2].detach().clone()
    for step in range(2):
        opt.zero_grad()
        (model(block, s, anchor)-target).square().mean().backward()
        assert model.local_features.net[-1].weight.grad.abs().max() > 0
        if step == 0:
            assert torch.count_nonzero(model.rho[0].weight.grad[:, 2]) == 0
            assert model.rho[0].weight.grad[:, 0].abs().max() > 0
        else:
            assert model.rho[0].weight.grad[:, 2].abs().max() > 0
            assert model.local_features.net[0].weight.grad.abs().max() > 0
        opt.step()
        assert model.local_features.net[-1].weight.abs().max() > 0
    assert not torch.equal(first_w, model.rho[0].weight[:, 2])


def test_stacked_block_and_local_permutation_invariance():
    _, model = matched_models(4)
    # Exercise the nonzero learned branch, not just the initial linear model.
    torch.nn.init.normal_(model.local_features.net[-1].weight, std=0.1)
    s = torch.randn(5, 4, 7)
    block = s.mean(2, keepdim=True)
    anchor = torch.randn(5)
    base = model(block, s, anchor)
    torch.testing.assert_close(base, model(block, s.flip(2), anchor))
    torch.testing.assert_close(base, model(block.flip(1), s.flip(1), anchor))


@pytest.mark.parametrize("m_dim,constant", [(1, 1), (4, 0)])
def test_stacked_runtime_roundtrip(tmp_path, m_dim, constant):
    args = stage1.build_parser().parse_args([
        "--n-train", "64", "--n-val", "32", "--n-blocks", "4", "--block-size", "7",
        "--methods", "linear,radial,stacked", "--iters", "3", "--batch-size", "16",
        "--lr-schedule", "constant", "--checkpoint-selection", "raw", "--print-every", "1",
        "--m-dim", str(m_dim), "--include-constant-channel", str(constant),
        "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    stage1.run(args)
    info = json.loads((tmp_path / "training_info.json").read_text())
    assert {i["best_source"] for i in info.values()} == {"raw"}
    assert len({i["minibatch_sha256"] for i in info.values()}) == 1
    assert info["stacked"]["nested_initialization_max_abs_diff"] < 1e-6
    score = runtime.AmortizedScoreRuntime(tmp_path, "stacked", "cpu")
    pi = np.array([0.1, 0.3, 0.5, 0.65])
    u = stage1.logit_np(pi)
    y = stage1.simulate_mean_shift(np.random.default_rng(29), u, 4, 7, 0.5)
    feat = stage1.featurize(y, 0.5, pi, score.stats)
    feat["anchor_z"] = ((u - score.anchor_mean) / score.anchor_sd).astype(np.float32)
    expected = stage1.predict(score.model, feat, "stacked", "cpu", 4)
    np.testing.assert_allclose(score.score(y, u), expected, rtol=0, atol=1e-6)
    _, derivative = score.score_and_derivative(y, u)
    assert np.isfinite(derivative).all()
    evaluation = evaluate_stage1.build_parser().parse_args([
        "--run-dir", str(tmp_path), "--n-test", "16", "--device", "cpu",
    ])
    evaluate_stage1.run(evaluation)
    with (tmp_path / "score_summary_by_pi.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert all(np.isfinite(float(row["mse"])) for row in rows)
    with (tmp_path / "training_trace.csv").open() as handle:
        trace = list(csv.DictReader(handle))
    stacked_trace = [r for r in trace if r["method"].startswith("stacked")]
    assert float(stacked_trace[0]["m_max_abs"]) == 0
    assert float(stacked_trace[-1]["m_max_abs"]) > 0
    assert all(np.isfinite(float(r["pooled_m_rms"])) for r in stacked_trace)


def test_readout_loader_rejects_overlapping_columns():
    linear, stacked = matched_models()
    with pytest.raises(ValueError, match="distinct"):
        stage1.load_matched_linear_rho(stacked.rho, linear.rho.state_dict(), (1, 1))
    with pytest.raises(ValueError, match="constant"):
        stage1.load_matched_linear_rho(stacked.rho, linear.rho.state_dict(), (1, 3), 1)


def test_raw_selection_ignores_lower_ema_loss(monkeypatch):
    linear, _ = matched_models()
    data = {"block": np.ones((3, 2, 1)), "anchor_z": np.zeros(3), "target": np.ones(3)}
    # Initial validation 10, then raw 5 and EMA 0. Raw-only must still pick raw.
    losses = iter([10.0, 5.0, 0.0])
    monkeypatch.setattr(stage1, "mse_over_tensors", lambda *a, **k: next(losses))
    _, _, info = stage1.train_model(
        linear, "linear", data, data, iters=1, batch_size=3, lr=1e-4,
        weight_decay=1e-3, patience=0, print_every=1, seed=7, grad_clip=5,
        ema_decay=0.995, gate_lr=1e-4, joint_rho_lr=1e-4, lr_schedule="constant",
        lr_decay_start_step=10000, lr_min_ratio=0.1, device="cpu", name="test",
        checkpoint_selection="raw",
    )
    assert info["best_source"] == "raw"
    assert info["best_val_loss"] == 5


def test_stacked_flags_preserve_legacy_defaults():
    args = stage1.build_parser().parse_args([])
    assert args.methods == "linear,radial"
    assert (args.m_dim, args.include_constant_channel) == (1, 1)
    assert args.checkpoint_selection == "raw_or_ema"
    assert args.m_pool_scale == "none"


BLOCK_SD = np.full((1, 1, 1), 0.3170857)


def scaled_stacked(m_dim=1, constant=True, m_pool_scale="block_sd"):
    torch.manual_seed(19)
    linear = stage1.LinearCSBetaDeepSets(64, 2)
    stacked = stage1.StackedCSBetaDeepSets(
        64, 2, 16, m_dim, constant, m_pool_scale=m_pool_scale, block_sd=BLOCK_SD,
    )
    stage1.load_matched_linear_rho(
        stacked.rho, linear.rho.state_dict(), stacked.linear_column_map,
        constant_column=0 if constant else None,
    )
    return linear, stacked


@pytest.mark.parametrize("m_dim", [1, 4])
def test_pooled_scaling_preserves_ilsa_nesting(m_dim):
    linear, stacked = scaled_stacked(m_dim)
    s = torch.randn(9, 20, 20)
    anchor = torch.randn(9)
    block = torch.randn(9, 20, 1)
    torch.testing.assert_close(stacked(block, s, anchor), linear(block, anchor), atol=1e-6, rtol=0)


def test_pooled_scaling_divides_the_learned_channel_only():
    _, plain = scaled_stacked(m_pool_scale="none")
    _, scaled = scaled_stacked(m_pool_scale="block_sd")
    # Give both the same nonzero learned branch, then compare readout inputs.
    state = {k: v.clone() for k, v in plain.local_features.state_dict().items()}
    torch.nn.init.normal_(state["net.4.weight"], std=0.1)
    plain.local_features.load_state_dict(state)
    scaled.local_features.load_state_dict(state)
    s = torch.randn(5, 4, 7)
    block = torch.randn(5, 4, 1)
    anchor = torch.randn(5)
    a, b = plain.readout_inputs(block, s, anchor), scaled.readout_inputs(block, s, anchor)
    # Constant, identity and anchor columns are untouched; only column 2 scales.
    for column in (0, 1, 3):
        torch.testing.assert_close(a[..., column], b[..., column], rtol=0, atol=0)
    torch.testing.assert_close(b[..., 2] * BLOCK_SD.item(), a[..., 2], rtol=1e-6, atol=0)
    assert not torch.allclose(a[..., 2], b[..., 2])


def test_pooled_scaling_requires_the_statistic_and_a_known_mode():
    with pytest.raises(ValueError, match="block_sd"):
        stage1.StackedCSBetaDeepSets(64, 2, 16, m_pool_scale="block_sd", block_sd=None)
    with pytest.raises(ValueError, match="m_pool_scale"):
        stage1.StackedCSBetaDeepSets(64, 2, 16, m_pool_scale="running")


def test_default_scaling_keeps_the_pre_flag_state_dict():
    plain = stage1.StackedCSBetaDeepSets(64, 2, 16)
    scaled = stage1.StackedCSBetaDeepSets(64, 2, 16, m_pool_scale="block_sd", block_sd=BLOCK_SD)
    # Checkpoints written before the flag existed must still load strictly.
    assert "m_pool_divisor" not in plain.state_dict()
    assert "m_pool_divisor" in scaled.state_dict()
    scaled.load_state_dict(scaled.state_dict(), strict=True)


def test_verifier_excludes_run_outputs_but_not_archived_artifacts():
    from scripts.verify_package import ROOT, ignored_generated_file
    assert ignored_generated_file(ROOT / "runs" / "example" / "model_stacked.pt")
    assert ignored_generated_file(ROOT / ".DS_Store")
    assert not ignored_generated_file(ROOT / "artifacts" / "model1" / "model_linear.pt")
    assert not ignored_generated_file(ROOT / "model1" / "stage1.py")
