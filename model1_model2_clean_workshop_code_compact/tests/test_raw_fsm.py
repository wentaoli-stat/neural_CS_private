from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from common import raw_fsm


ROOT = Path(__file__).resolve().parents[1]


def test_raw_block_fsm_is_permutation_invariant_and_anchor_conditioned() -> None:
    torch.manual_seed(7)
    model = raw_fsm.RawBlockDeepSets(
        phi_hidden=8, hidden=12, depth=2, y_mean=0.1, y_sd=1.3
    ).eval()
    y = torch.randn(5, 4, 6)
    anchor = torch.linspace(-1.0, 1.0, 5)
    base = model(y, anchor)
    torch.testing.assert_close(base, model(y[:, [2, 0, 3, 1]], anchor))
    torch.testing.assert_close(base, model(y[:, :, [4, 0, 2, 5, 1, 3]], anchor))
    assert not torch.allclose(base, model(y, anchor + 0.5))


def test_direct_raw_fsm_is_one_unstructured_mlp_on_flat_y_and_anchor() -> None:
    torch.manual_seed(11)
    model = raw_fsm.DirectRawMLP(
        n_blocks=4, block_size=6, hidden=12, depth=2, y_mean=0.1, y_sd=1.3
    ).eval()
    y = torch.randn(5, 4, 6)
    anchor = torch.linspace(-1.0, 1.0, 5)
    base = model(y, anchor)
    assert base.shape == (5,)
    assert not torch.allclose(base, model(y[:, [2, 0, 3, 1]], anchor))
    assert not torch.allclose(base, model(y[:, :, [4, 0, 2, 5, 1, 3]], anchor))
    assert not torch.allclose(base, model(y, anchor + 0.5))
    assert not any("phi" in name or "gate" in name for name, _ in model.named_parameters())


def test_direct_raw_fsm_parameter_counts_are_explicit() -> None:
    expected = {"model1": 29_953, "model2": 106_753}
    for name, (n_blocks, block_size) in {
        "model1": (20, 20), "model2": (40, 40)
    }.items():
        model = raw_fsm.DirectRawMLP(
            n_blocks=n_blocks, block_size=block_size,
            hidden=64, depth=2, y_mean=0.0, y_sd=1.0,
        )
        assert sum(value.numel() for value in model.parameters()) == expected[name]


def test_direct_architecture_contract_is_distinct_from_raw_deepsets() -> None:
    method, label, context = raw_fsm.architecture_contract("direct_flat_mlp")
    assert method == raw_fsm.DIRECT_METHOD
    assert label == raw_fsm.DIRECT_METHOD_LABEL
    assert context == raw_fsm.DIRECT_CONTEXT_DEFINITION


def test_model_defaults_match_locked_geometries_and_proposals() -> None:
    for name, expected in (
        ("model1", (20, 20, 0.5, 0.20)),
        ("model2", (40, 40, 1.0, 0.15)),
    ):
        args = raw_fsm.parser().parse_args(
            ["stage1", "--model", name, "--output-dir", "unused"]
        )
        raw_fsm.model_defaults(args)
        assert (args.n_blocks, args.block_size, args.tau, args.sigma_q) == expected


def test_stage2_does_not_expose_unused_proposal_or_fake_pilot_threshold() -> None:
    args = raw_fsm.parser().parse_args(
        [
            "stage2",
            "--model",
            "model1",
            "--stage1-run-dir",
            "unused",
            "--output-dir",
            "unused",
        ]
    )
    raw_fsm.model_defaults(args)
    assert not hasattr(args, "sigma_q")
    assert not hasattr(args, "min_pilot_correlation")


def test_model2_raw_fsm_uses_data_only_equal_channel_pilot() -> None:
    class TinyRuntime:
        model_name = "model2"
        n_blocks = 2
        block_size = 3
        tau = 1.0
        pi_min = 0.05
        pi_max = 0.70
        u_min = float(raw_fsm.model2_stage1.logit_np(pi_min))
        u_max = float(raw_fsm.model2_stage1.logit_np(pi_max))
        device = "cpu"

        @staticmethod
        def validate_y(y: np.ndarray) -> np.ndarray:
            return np.asarray(y, dtype=np.float32)

        def metadata(self):
            return {
                "n_blocks": self.n_blocks,
                "block_size": self.block_size,
                "tau": self.tau,
                "pi_min": self.pi_min,
                "pi_max": self.pi_max,
            }

    args = raw_fsm.parser().parse_args(
        [
            "stage2", "--model", "model2", "--stage1-run-dir", "unused",
            "--output-dir", "unused", "--pilot-grid-size", "21",
            "--pilot-batch-size", "4", "--pilot-grid-chunk-size", "3",
            "--pilot-backend", "numpy", "--pilot-progress-every", "0",
        ]
    )
    raw_fsm.model_defaults(args)
    runtime = TinyRuntime()
    rng = np.random.default_rng(9)
    y = raw_fsm.model2_stage1.simulate_common_factor(
        rng, np.zeros(8), runtime.n_blocks, runtime.block_size, runtime.tau
    )
    root, status = raw_fsm.compute_pilot(y, runtime, args)
    assert root.shape == (8,)
    assert status.shape == (8,)
    assert np.all(np.isfinite(root))
    assert np.all((root >= runtime.u_min) & (root <= runtime.u_max))


def test_packaged_raw_fsm_runs_have_complete_stage1_and_stage2_outputs() -> None:
    for relative, model in (
        ("artifacts/model1/raw_fsm", "model1"),
        ("artifacts/model2/k40_m40/raw_fsm", "model2"),
    ):
        directory = ROOT / relative
        stage1 = directory / "stage1"
        stage2 = directory / "stage2_npe_50k"
        stage1_config = json.loads((stage1 / "config.json").read_text(encoding="utf-8"))
        stage2_config = json.loads((stage2 / "config.json").read_text(encoding="utf-8"))
        assert stage1_config["model"] == model
        assert stage1_config["exact_score_role"] == "post-training diagnostics only"
        assert stage2_config["model"] == model
        assert stage2_config["n_sbi_train"] == 50_000
        assert stage2_config["context_definition"] == raw_fsm.CONTEXT_DEFINITION
        with (stage2 / "posterior_by_seed.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 120
        assert sum(row["method"] == raw_fsm.METHOD_LABEL for row in rows) == 60


def test_packaged_direct_raw_fsm_is_flat_unstructured_and_complete() -> None:
    for relative, model, parameters in (
        ("artifacts/model1/direct_raw_fsm", "model1", 29_953),
        ("artifacts/model2/k40_m40/direct_raw_fsm", "model2", 106_753),
    ):
        directory = ROOT / relative
        stage1 = directory / "stage1"
        stage2 = directory / "stage2"
        stage1_config = json.loads((stage1 / "config.json").read_text(encoding="utf-8"))
        stage2_config = json.loads((stage2 / "config.json").read_text(encoding="utf-8"))
        assert stage1_config["model"] == model
        assert stage1_config["architecture"] == "direct_flat_mlp"
        assert stage1_config["method"] == raw_fsm.DIRECT_METHOD
        assert stage1_config["trainable_parameters"] == parameters
        assert not stage1_config["uses_phi"]
        assert not stage1_config["uses_gate"]
        assert not stage1_config["uses_pooling"]
        assert not stage1_config["uses_deepsets"]
        assert stage2_config["n_sbi_train"] == 50_000
        assert stage2_config["context_definition"] == raw_fsm.DIRECT_CONTEXT_DEFINITION
        with (stage2 / "posterior_by_seed.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 120
        assert sum(row["method"] == raw_fsm.DIRECT_METHOD_LABEL for row in rows) == 60
