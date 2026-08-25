from __future__ import annotations

from pathlib import Path

import pytest

from model1.runtime import AmortizedScoreRuntime as Model1Runtime
from model2.runtime import AmortizedScoreRuntime as Model2Runtime


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("directory", "method"),
    [
        ("artifacts/model1/stage1/linear_validation_best", "linear"),
        ("artifacts/model1/stage1/radial_matched_validation_best", "radial"),
    ],
)
def test_model1_frozen_runtime_replay(directory: str, method: str) -> None:
    runtime = Model1Runtime(ROOT / directory, method, device="cpu")
    assert runtime.run_sanity_checks()["passed"]


@pytest.mark.parametrize("method", ["linear", "shared_radial"])
def test_model2_validation_best_runtime_replay(method: str) -> None:
    directory = ROOT / "artifacts/model2/k40_m40/stage1/seed_20260709"
    runtime = Model2Runtime(directory, method, device="cpu")
    assert runtime.run_sanity_checks()["passed"]


@pytest.mark.parametrize("method", ["linear", "shared_radial"])
def test_model2_fixed20k_runtime_contract(method: str) -> None:
    directory = ROOT / "artifacts/model2/k40_m40/stage1/seed_20260709"
    checkpoint = directory / "milestones" / f"model_{method}_step20000_ema.pt"
    runtime = Model2Runtime(
        directory,
        method,
        device="cpu",
        checkpoint_path=checkpoint,
        require_fixed_final_ema=True,
    )
    assert runtime.selection == "fixed_milestone"
    assert runtime.checkpoint_source == "ema"
    assert runtime.checkpoint_step == 20000
    assert runtime.run_sanity_checks()["passed"]


def test_validation_best_cannot_masquerade_as_fixed20k() -> None:
    directory = ROOT / "artifacts/model2/k40_m40/stage1/seed_20260709"
    with pytest.raises(ValueError, match="selection='fixed_milestone'"):
        Model2Runtime(
            directory,
            "linear",
            device="cpu",
            checkpoint_path=directory / "model_linear.pt",
            require_fixed_final_ema=True,
        )

