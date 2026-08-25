from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model1_feature_statistics_are_shared() -> None:
    linear = ROOT / "artifacts/model1/stage1/linear_validation_best/feature_stats.npz"
    radial = ROOT / "artifacts/model1/stage1/radial_matched_validation_best/feature_stats.npz"
    assert sha256(linear) == sha256(radial)


def test_model2_stage2_parents_match_packaged_checkpoints() -> None:
    stage1 = ROOT / "artifacts/model2/k40_m40/stage1/seed_20260709"
    fixed = json.loads(
        (ROOT / "artifacts/model2/k40_m40/stage2/npe_fixed20k_ema_seed_20260709/config.json")
        .read_text(encoding="utf-8")
    )
    valbest = json.loads(
        (ROOT / "artifacts/model2/k40_m40/stage2/npe_validation_best_seed_20260709/config.json")
        .read_text(encoding="utf-8")
    )
    assert fixed["stage1_runtime_metadata"]["linear"]["checkpoint_sha256"] == sha256(
        stage1 / "milestones/model_linear_step20000_ema.pt"
    )
    assert fixed["stage1_runtime_metadata"]["shared_radial"]["checkpoint_sha256"] == sha256(
        stage1 / "milestones/model_shared_radial_step20000_ema.pt"
    )
    assert valbest["stage1_runtime_metadata"]["linear"]["checkpoint_sha256"] == sha256(
        stage1 / "model_linear.pt"
    )
    assert valbest["stage1_runtime_metadata"]["shared_radial"]["checkpoint_sha256"] == sha256(
        stage1 / "model_shared_radial.pt"
    )
