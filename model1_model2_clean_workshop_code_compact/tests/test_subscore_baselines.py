"""All-subscores baselines for the Model 1 Stage-2 input comparison."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from common import raw_data_npe, raw_fsm
from model1 import stage1 as m1
from model1 import stage2_npe as m1_stage2


def analytic_subscores(y, u):
    return m1.score_u_from_log_ratio(m1.marginal_log_ratio(y, 0.5), m1.sigmoid_np(u))


def subscore_model(n_blocks=4, block_size=6):
    torch.manual_seed(3)
    return raw_fsm.SubscoreFlatMLP(
        n_blocks=n_blocks, block_size=block_size, hidden=12, depth=2, tau=0.5,
        anchor_mean=-0.6, anchor_sd=0.9, s_mean=0.01, s_sd=0.2,
    ).eval()


def test_subscore_fsm_recomputes_the_analytic_local_scores_at_the_anchor():
    model = subscore_model()
    rng = np.random.default_rng(0)
    u = rng.uniform(-2.5, 0.5, 5)
    y = m1.simulate_mean_shift(rng, u, 4, 6, 0.5)
    anchor_z = torch.as_tensor((u - (-0.6)) / 0.9, dtype=torch.float32)
    got = model.local_scores(torch.as_tensor(y, dtype=torch.float32), anchor_z).numpy()
    np.testing.assert_allclose(got, analytic_subscores(y, u), rtol=0, atol=2e-6)


def test_subscore_fsm_is_unstructured_and_anchor_conditioned():
    model = subscore_model()
    y = torch.randn(5, 4, 6)
    anchor = torch.linspace(-1.0, 1.0, 5)
    base = model(y, anchor)
    assert base.shape == (5,)
    assert not torch.allclose(base, model(y[:, [2, 0, 3, 1]], anchor))
    assert not torch.allclose(base, model(y[:, :, [4, 0, 2, 5, 1, 3]], anchor))
    assert not torch.allclose(base, model(y, anchor + 0.5))


def test_raw_and_subscore_fsm_differ_only_in_input():
    """At Model 1 geometry both unstructured Fisher-score baselines have 29,953 parameters."""
    raw = raw_fsm.DirectRawMLP(n_blocks=20, block_size=20, hidden=64, depth=2, y_mean=0.0, y_sd=1.0)
    sub = raw_fsm.SubscoreFlatMLP(n_blocks=20, block_size=20, hidden=64, depth=2, tau=0.5,
                                  anchor_mean=0.0, anchor_sd=1.0, s_mean=0.0, s_sd=1.0)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(raw) == count(sub) == 29_953


def test_subscore_architecture_contract_and_required_statistics():
    assert "subscore_flat_mlp" in raw_fsm.ARCHITECTURES
    method, label, context = raw_fsm.architecture_contract("subscore_flat_mlp")
    assert (method, label, context) == (raw_fsm.SUBSCORE_METHOD, raw_fsm.SUBSCORE_METHOD_LABEL,
                                        raw_fsm.SUBSCORE_CONTEXT_DEFINITION)
    assert method not in (raw_fsm.METHOD, raw_fsm.DIRECT_METHOD)
    with pytest.raises(ValueError, match="statistics"):
        raw_fsm.build_score_model("subscore_flat_mlp", n_blocks=2, block_size=3, phi_hidden=4,
                                  hidden=4, depth=1, y_mean=0.0, y_sd=1.0)


def test_subscore_fsm_trains_selects_raw_and_replays_through_the_runtime(tmp_path):
    out = tmp_path / "s1"
    args = raw_fsm.parser().parse_args([
        "stage1", "--model", "model1", "--architecture", "subscore_flat_mlp",
        "--n-blocks", "3", "--block-size", "4", "--n-train", "64", "--n-val", "32",
        "--iters", "3", "--batch-size", "16", "--print-every", "1", "--hidden", "8",
        "--checkpoint-selection", "raw", "--skip-exact-diagnostics",
        "--device", "cpu", "--output-dir", str(out),
    ])
    raw_fsm.train_stage1(args)
    info = json.loads((out / "training_info.json").read_text())
    assert info["best_source"] == "raw"
    stats = np.load(out / "feature_stats.npz")
    assert {"s_mean", "s_sd"} <= set(stats.files)
    runtime = raw_fsm.RawFSMRuntime(out, "cpu")
    assert runtime.method == raw_fsm.SUBSCORE_METHOD
    rng = np.random.default_rng(9)
    u = np.full(6, -0.7)
    y = m1.simulate_mean_shift(rng, u, 3, 4, 0.5)
    anchor_z = ((u - runtime.anchor_mean) / runtime.anchor_sd).astype(np.float32)
    expected = raw_fsm.predict(runtime.model, y.astype(np.float32), anchor_z, device="cpu", batch_size=4)
    np.testing.assert_allclose(runtime.score(y, u), expected, rtol=0, atol=1e-6)
    assert runtime.sanity_checks()["passed"]


def test_raw_fsm_default_selection_rule_is_unchanged():
    args = raw_fsm.parser().parse_args(["stage1", "--model", "model1", "--output-dir", "x"])
    assert args.checkpoint_selection == "raw_or_ema"


def test_raw_fsm_pilot_cache_must_match_the_training_bank(tmp_path):
    cache = tmp_path / "pilot_cache.npz"
    np.savez(cache, pi_train=np.array([0.1, 0.2], dtype=np.float32), pilot_u=np.zeros(2),
             pilot_status=np.array(["interior"] * 2), pilot_grid_size=np.asarray(201),
             tau=np.asarray(0.5), pilot_mode=np.asarray("marginal"))

    class Stub:
        model_name, tau = "model1", 0.5

    args = raw_fsm.parser().parse_args(["stage2", "--model", "model1", "--stage1-run-dir", "x",
                                        "--output-dir", "y", "--pilot-cache", str(cache)])
    u, status = raw_fsm.load_or_compute_training_pilot(None, np.array([0.1, 0.2]), Stub(), args)
    assert u.shape == (2,) and list(status) == ["interior", "interior"]
    with pytest.raises(ValueError, match="does not match"):
        raw_fsm.load_or_compute_training_pilot(None, np.array([0.3, 0.2]), Stub(), args)


def test_subscore_context_is_pilot_plus_every_local_score_and_takes_no_parameter():
    rng = np.random.default_rng(1)
    y = m1.simulate_mean_shift(rng, np.array([-1.0, 0.2, -2.0]), 3, 4, 0.5)
    pilot = np.array([-0.9, 0.1, -1.7])
    context = raw_data_npe.subscore_context(y, pilot, 0.5)
    assert context.shape == (3, 13) and context.dtype == np.float32
    np.testing.assert_allclose(context[:, 0], pilot, rtol=0, atol=1e-6)
    np.testing.assert_allclose(context[:, 1:], analytic_subscores(y, pilot).reshape(3, -1),
                               rtol=0, atol=1e-6)
    forbidden = {"pi", "pi_true", "theta", "u_true"}
    assert not forbidden.intersection(inspect.signature(raw_data_npe.subscore_context).parameters)


def test_subscore_context_uses_the_shared_data_only_pilot():
    args = raw_data_npe.build_parser().parse_args([
        "--model", "model1", "--context", "subscores_at_pilot", "--n-blocks", "3",
        "--block-size", "4", "--device", "cpu", "--output-dir", "unused",
    ])
    raw_data_npe.resolve_model_args(args)
    rng = np.random.default_rng(2)
    y = raw_data_npe.simulate(rng, np.array([0.1, 0.4, 0.6]), args)
    context = raw_data_npe.build_context(y, args)
    metadata = {"n_blocks": 3, "block_size": 4, "tau": 0.5}
    pilot, _ = m1_stage2.data_only_pilot(y, metadata, argparse_namespace(args))
    np.testing.assert_allclose(context[:, 0], pilot, rtol=0, atol=1e-6)


def argparse_namespace(args):
    import argparse
    return argparse.Namespace(
        pi_prior_min=args.pi_prior_min, pi_prior_max=args.pi_prior_max,
        pilot_grid_size=args.pilot_grid_size, pilot_batch_size=args.pilot_batch_size,
        pilot_grid_chunk_size=args.pilot_grid_chunk_size,
        pilot_progress_every=args.pilot_progress_every, device="cpu",
    )


def test_raw_npe_defaults_and_packaged_contract_are_unchanged():
    args = raw_data_npe.build_parser().parse_args(["--model", "model1", "--output-dir", "x"])
    assert args.context == "raw"
    assert raw_data_npe.CONTEXT_DEFINITION == "flatten(Y) in canonical block/coordinate order"
    assert raw_data_npe.context_contract(args) == (raw_data_npe.METHOD_LABEL,
                                                    raw_data_npe.CONTEXT_DEFINITION)
    args2 = raw_data_npe.build_parser().parse_args(
        ["--model", "model2", "--context", "subscores_at_pilot", "--output-dir", "x"])
    with pytest.raises(ValueError, match="model1 only"):
        raw_data_npe.context_contract(args2)


def test_raw_fsm_model1_pilot_runs_without_a_cache():
    """Regression: pilot_args must carry the ``device`` that data_only_pilot reads.

    Stage 2 loads the training pilot from a cache but always computes the pilot
    for each test dataset, so this path runs in every Stage-2 evaluation.
    """
    args = raw_fsm.parser().parse_args(["stage2", "--model", "model1", "--stage1-run-dir", "x",
                                        "--output-dir", "y", "--pilot-device", "cpu",
                                        "--pilot-progress-every", "0"])
    raw_fsm.model_defaults(args)

    class Stub:
        model_name, tau = "model1", 0.5

        def metadata(self):
            return {"n_blocks": 3, "block_size": 4, "tau": 0.5}

    rng = np.random.default_rng(4)
    y = m1.simulate_mean_shift(rng, m1.logit_np(np.array([0.1, 0.4, 0.6])), 3, 4, 0.5)
    pilot_u, status = raw_fsm.compute_pilot(y, Stub(), args)
    expected, _ = m1_stage2.data_only_pilot(y, Stub().metadata(), argparse_namespace(args))
    np.testing.assert_allclose(pilot_u, expected, rtol=0, atol=1e-9)
    assert len(status) == 3
