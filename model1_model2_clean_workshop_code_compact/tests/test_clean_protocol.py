from __future__ import annotations

import ast
import inspect
from pathlib import Path

from model1 import evaluate_stage1 as model1_evaluate_stage1
from model1 import stage1 as model1_stage1
from model1 import stage2_npe as model1_stage2
from model2 import evaluate_stage1 as model2_evaluate_stage1
from model2 import stage1 as model2_stage1
from model2 import stage2_npe as model2_stage2


ROOT = Path(__file__).resolve().parents[1]


def test_public_method_surface_is_current_only() -> None:
    assert set(model1_stage1.ARCHITECTURES) == {"linear", "radial", "stacked"}
    assert set(model1_stage2.METHOD_LABELS) == {"pilot", "linear", "radial"}
    assert {"linear"} | model2_stage1.GATED_METHODS == {"linear", "shared_radial"}
    assert model2_stage1.STACKED_METHODS == {"stacked_shared", "stacked_split"}
    assert set(model2_stage2.METHOD_LABELS) == {
        "pilot", "linear", "shared_radial", "stacked_shared", "stacked_split"}


def test_formal_defaults() -> None:
    m1 = model1_stage1.build_parser().parse_args([])
    m2 = model2_stage1.build_parser().parse_args([])
    assert (m1.n_blocks, m1.block_size, m1.tau, m1.sigma_q) == (20, 20, 0.5, 0.2)
    assert m1.methods == "linear,radial"
    assert model1_stage1.SELECTED_RADIAL_INIT == "matched_random"
    assert model1_stage1.SELECTED_GATE_CONDITION_ON_ANCHOR is True
    assert model1_stage1.SELECTED_GATE_ONLY_STEPS == 0
    assert not hasattr(m1, "pi_ref")
    assert not hasattr(m1, "feature_pi_mode")
    assert not hasattr(m1, "gate_condition_on_anchor")
    assert not hasattr(m1, "radial_init")
    assert not hasattr(m1, "gate_only_steps")
    assert not hasattr(m1, "validation_pi_values")
    assert not hasattr(m1, "n_validation_per_anchor")
    assert not hasattr(m1, "linear_iters")
    assert not hasattr(m1, "radial_iters")
    assert not hasattr(m1, "n_test")
    assert not hasattr(m1, "diagnostic_pi_values")
    assert (m2.n_blocks, m2.block_size, m2.tau, m2.sigma_q) == (40, 40, 1.0, 0.15)
    assert m2.methods == "linear,shared_radial"
    assert model2_stage1.SELECTED_RADIAL_INIT == "matched_random"
    assert model2_stage1.SELECTED_GATE_CONDITION_ON_ANCHOR is True
    assert model2_stage1.SELECTED_GATE_ONLY_STEPS == 0
    assert not hasattr(m2, "pi_ref")
    assert not hasattr(m2, "feature_pi_mode")
    assert not hasattr(m2, "gate_condition_on_anchor")
    assert not hasattr(m2, "radial_init")
    assert not hasattr(m2, "gate_only_steps")
    assert not hasattr(m2, "validation_pi_values")
    assert not hasattr(m2, "n_validation_per_anchor")
    assert not hasattr(m2, "n_test")
    assert not hasattr(m2, "diagnostic_pi_values")
    assert not hasattr(m2, "skip_exact_diagnostics")
    assert m2.save_final_ema_validation is False
    m2_with_final_ema = model2_stage1.build_parser().parse_args(
        ["--save-final-ema-validation"]
    )
    assert m2_with_final_ema.save_final_ema_validation is True

    m1_stage2 = model1_stage2.build_parser().parse_args(
        ["--stage1-run-dir", "stage1"]
    )
    m2_stage2 = model2_stage2.build_parser().parse_args(
        ["--stage1-run-dir", "stage1"]
    )
    assert (m1_stage2.sbi_train_seed, m1_stage2.sbi_seed) == (20260723, 54000)
    assert not hasattr(m1_stage2, "stage1_run_dirs")
    assert not hasattr(m2_stage2, "stage1_run_dirs")
    assert m1_stage2.context_score_batch_size == 256
    assert m2_stage2.context_score_batch_size == 256
    assert hasattr(m1_stage2, "pilot_cache")
    assert hasattr(m2_stage2, "pilot_cache")
    assert not hasattr(m1_stage2, "score_batch_size")
    assert not hasattr(m1_stage2, "pilot_device")
    assert not hasattr(m2_stage2, "score_batch_size")
    assert not hasattr(m2_stage2, "stage1_checkpoints")
    assert not hasattr(m2_stage2, "pilot_mode")
    assert not hasattr(m2_stage2, "pilot_backend")
    assert not hasattr(m1_stage2, "min_mean_abs_score")
    assert not hasattr(m2_stage2, "min_mean_abs_score")
    assert model1_stage2.SELECTED_PILOT_MODE == "marginal"
    assert model2_stage2.SELECTED_PILOT_MODE == "equal_channel"
    assert model1_stage2.SELECTED_PILOT_BACKEND == "torch"
    assert model2_stage2.SELECTED_PILOT_BACKEND == "torch"

    m1_evaluation = model1_evaluate_stage1.build_parser().parse_args(
        ["--run-dir", "stage1"]
    )
    m2_evaluation = model2_evaluate_stage1.build_parser().parse_args(
        ["--run-dir", "stage1"]
    )
    assert (m1_evaluation.n_test, m1_evaluation.pi_values) == (
        5_000,
        "0.10,0.30,0.50,0.65",
    )
    assert (m2_evaluation.n_test, m2_evaluation.pi_values) == (
        5_000,
        "0.07,0.10,0.30,0.50,0.65,0.68",
    )


def test_selected_launchers_do_not_repeat_shadowed_options() -> None:
    model1_stage1_launcher = (ROOT / "scripts/run_model1_stage1.sh").read_text(
        encoding="utf-8"
    )
    assert "--iters 20000" in model1_stage1_launcher
    assert "--linear-iters" not in model1_stage1_launcher
    assert "--radial-iters" not in model1_stage1_launcher
    assert "-m model1.evaluate_stage1" in model1_stage1_launcher

    model2_stage1_launcher = (ROOT / "scripts/run_model2_stage1_40x40.sh").read_text(
        encoding="utf-8"
    )
    assert "--lr-schedule constant" in model2_stage1_launcher
    assert "--lr-decay-start-step" not in model2_stage1_launcher
    assert "--lr-min-ratio" not in model2_stage1_launcher
    assert "--save-final-ema-validation" in model2_stage1_launcher
    assert "-m model2.evaluate_stage1" in model2_stage1_launcher


def test_stage1_training_does_not_run_exact_evaluation() -> None:
    for module in (model1_stage1, model2_stage1):
        source = inspect.getsource(module.run)
        assert "build_eval_data(" not in source
        assert "full_score_u(" not in source
        assert "score_summary_by_pi.csv" not in source

    model1_stage2_launcher = (ROOT / "scripts/run_model1_stage2_npe.sh").read_text(
        encoding="utf-8"
    )
    assert "--stage1-run-dirs" not in model1_stage2_launcher

    launcher = (ROOT / "scripts/run_model2_stage2_validation_best_npe.sh").read_text(
        encoding="utf-8"
    )
    assert "--stage1-run-dirs" not in launcher
    assert "--stage1-checkpoints" not in launcher
    assert "--pilot-mode" not in launcher
    assert "--pilot-backend" not in launcher
    assert not (ROOT / "scripts/run_model2_stage2_fixed20k_npe.sh").exists()


def test_stage2_pilot_signatures_have_no_oracle_parameter() -> None:
    forbidden = {"pi", "pi_true", "theta", "u_true"}
    assert not forbidden.intersection(inspect.signature(model1_stage2.data_only_pilot).parameters)
    assert not forbidden.intersection(inspect.signature(model2_stage2.pilot_score_contexts).parameters)


def test_no_legacy_local_imports() -> None:
    forbidden_prefixes = (
        "run_blockwise_",
        "run_model1_mode_a",
        "run_model2_mode_a",
        "model1_mode_a",
        "model2_jiang",
    )
    for path in [*ROOT.glob("model1/*.py"), *ROOT.glob("model2/*.py"), *ROOT.glob("common/*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        assert not [module for module in modules if module.startswith(forbidden_prefixes)], path
