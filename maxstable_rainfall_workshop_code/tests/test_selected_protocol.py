from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from run_stage2 import (
    CONTEXT_PROVENANCE_KEY,
    CONTEXT_PROVENANCE_SCHEMA_VERSION,
    NPE_MODEL,
    context_cache_protocol,
    decode_context_provenance,
    pilot_generation_protocol,
    validate_reused_context_provenance,
)
from maxstable_rainfall79.core import (
    DistanceBinPairMLPFullDatasetScore,
    FormalConfig,
    PositiveScoreAnchorPairGate,
    SELECTED_PROTOCOL_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]


def test_selected_protocol_name_matches_delivery_config() -> None:
    delivery = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert delivery["protocol"] == SELECTED_PROTOCOL_VERSION


def test_selected_stage2_interface_is_mdn_only() -> None:
    delivery = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    source = (ROOT / "code" / "runners" / "run_stage2.py").read_text(
        encoding="utf-8"
    )
    assert NPE_MODEL == "mdn"
    assert delivery["stage2"]["npe"] == "MDN"
    assert "--npe-model" not in source
    assert "--npe-transforms" not in source
    assert "--npe-bins" not in source


def test_stage1_float32_cache_contract_has_no_opt_out() -> None:
    source = (ROOT / "code" / "runners" / "train_stage1.py").read_text(
        encoding="utf-8"
    )
    assert "--require-float32-cache" not in source
    assert 'metadata.get("individual_pair_storage_dtype") != "float32"' in source


def test_legacy_context_cache_provenance_is_honestly_unknown(tmp_path: Path) -> None:
    path = tmp_path / "contexts.npz"
    np.savez_compressed(path, theta_train=np.zeros((2, 3), dtype=np.float32))
    with np.load(path, allow_pickle=False) as saved:
        provenance = decode_context_provenance(saved)
    assert provenance["status"] == "unknown_legacy_cache"
    assert set(pilot_generation_protocol(provenance).values()) == {"unknown"}
    cache_protocol = context_cache_protocol(
        provenance, reused=True, validated=False
    )
    assert cache_protocol["generation_parameters_source"] == "unknown_legacy_cache"
    assert cache_protocol["stage1_checkpoint_match"] == "unknown"
    assert not validate_reused_context_provenance(
        provenance,
        stage1_checkpoint_sha256="current",
        methods=("pilot", "linear", "positive"),
        n_train=2,
        n_test=1,
    )
    upgraded = tmp_path / "upgraded_contexts.npz"
    np.savez_compressed(
        upgraded,
        **{CONTEXT_PROVENANCE_KEY: np.asarray(json.dumps(provenance))},
    )
    with np.load(upgraded, allow_pickle=False) as saved:
        assert decode_context_provenance(saved) == provenance


def test_versioned_context_provenance_round_trip_and_validation(
    tmp_path: Path,
) -> None:
    methods = ("pilot", "linear", "positive")
    provenance = {
        "schema_version": CONTEXT_PROVENANCE_SCHEMA_VERSION,
        "status": "known",
        "stage1": {"checkpoint_sha256": "stage1-digest"},
        "method_labels": list(methods),
        "simulation": {"n_train": 10, "n_test": 4},
        "pilot": {
            "full_pair_count": 3081,
            "pairs_per_group": 7,
            "selected_pair_count": 140,
            "pair_selection_seed": 91,
            "pair_selection": "saved selection",
            "iterations": 3,
            "n_starts": 5,
            "n_refine_starts": 2,
            "start_radius": 1.25,
            "backtracking_steps": 4,
            "tolerance": 0.02,
        },
    }
    path = tmp_path / "contexts.npz"
    np.savez_compressed(
        path,
        **{CONTEXT_PROVENANCE_KEY: np.asarray(json.dumps(provenance))},
    )
    with np.load(path, allow_pickle=False) as saved:
        decoded = decode_context_provenance(saved)
    assert pilot_generation_protocol(decoded)["iterations"] == 3
    assert validate_reused_context_provenance(
        decoded,
        stage1_checkpoint_sha256="stage1-digest",
        methods=methods,
        n_train=10,
        n_test=4,
    )
    with pytest.raises(ValueError, match="different Stage-1 checkpoint"):
        validate_reused_context_provenance(
            decoded,
            stage1_checkpoint_sha256="other-digest",
            methods=methods,
            n_train=10,
            n_test=4,
        )


def test_locked_shape_and_parameterization() -> None:
    config = FormalConfig()
    config.validate()
    assert (config.n_years, config.n_sites, config.theta_dim) == (47, 79, 3)
    assert config.n_sites * (config.n_sites - 1) // 2 == 3081


def test_positive_anchor_is_exactly_nested_at_initialization() -> None:
    theta_dim = 3
    geometry = np.zeros((2, 3), dtype=np.float32)
    counts = np.ones(2, dtype=np.float32)
    group_ids = np.asarray((0, 0, 1, 1), dtype=np.int64)
    normalizer = np.ones((1, 1, 2, theta_dim), dtype=np.float32)
    linear = DistanceBinPairMLPFullDatasetScore(
        theta_dim,
        8,
        1,
        geometry,
        counts,
        group_ids,
        block_mean=np.zeros_like(normalizer),
        block_sd=normalizer,
    )
    positive = PositiveScoreAnchorPairGate(
        linear,
        4,
        np.zeros_like(normalizer),
        normalizer,
        normalizer,
        np.zeros(theta_dim, dtype=np.float32),
        np.ones(theta_dim, dtype=np.float32),
    )
    assert positive.gate[0].in_features == 2 * theta_dim
    score = torch.randn(3, 5, 4, theta_dim)
    anchor = torch.randn(3, theta_dim)
    with torch.no_grad():
        assert torch.equal(linear(score, anchor), positive(score, anchor))


def test_selected_results_exclude_old_positive_arm() -> None:
    for label in ("low", "center", "high"):
        summary = json.loads(
            (ROOT / "results" / "fixed_truth" / label / "summary.json").read_text()
        )
        names = {row["method"] for row in summary["methods"]}
        assert "Positive(s,w_a) NPE" in names
        assert "rough Positive NPE" not in names
        assert summary["old_unconditioned_positive_removed"] is True
