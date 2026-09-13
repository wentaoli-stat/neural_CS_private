from __future__ import annotations

import csv

import numpy as np
import pytest

from model1 import stage1
from model1 import stage2_npe


def _rows(methods: list[str], w1: dict[str, list[float]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, seed in enumerate((100, 101, 102)):
        rows.append({"method": "exact likelihood grid", "pi_true": 0.3, "seed": seed, "w1_to_exact": 0.0})
        for method in methods:
            rows.append({
                "method": stage2_npe.METHOD_LABELS[method],
                "pi_true": 0.3,
                "seed": seed,
                "w1_to_exact": w1[method][index],
            })
    return rows


W1 = {
    "pilot": [0.050, 0.060, 0.070],
    "linear": [0.030, 0.020, 0.040],
    "radial": [0.020, 0.025, 0.030],
    "stacked": [0.010, 0.030, 0.020],
}


def test_paired_w1_without_stacked_is_radial_versus_all() -> None:
    out = stage2_npe.paired_w1_comparisons(_rows(["pilot", "linear", "radial"], W1))
    radial = stage2_npe.METHOD_LABELS["radial"]
    assert [(row["reference"], row["competitor"]) for row in out] == [
        (radial, stage2_npe.METHOD_LABELS["linear"]),
        (radial, stage2_npe.METHOD_LABELS["pilot"]),
    ]
    assert out[0]["mean_paired_difference"] == pytest.approx(np.mean([-0.01, 0.005, -0.01]))
    assert out[0]["reference_wins"] == 2


def test_paired_w1_with_stacked_reports_the_three_contrasts() -> None:
    out = stage2_npe.paired_w1_comparisons(_rows(["pilot", "linear", "radial", "stacked"], W1))
    label = stage2_npe.METHOD_LABELS
    pairs = {(row["reference"], row["competitor"]): row for row in out}
    assert len(pairs) == len(out) == 5
    assert (label["radial"], label["stacked"]) not in pairs
    for reference, competitor in (("stacked", "radial"), ("radial", "linear"), ("stacked", "linear")):
        row = pairs[(label[reference], label[competitor])]
        expected = np.subtract(W1[reference], W1[competitor])
        assert row["n_pairs"] == 3
        assert row["mean_paired_difference"] == pytest.approx(expected.mean())
        assert row["reference_wins"] == int(np.sum(expected < 0))


def test_tiny_stacked_checkpoint_runs_through_stage2(tmp_path) -> None:
    stage1_dir = tmp_path / "stage1"
    stage1.run(stage1.build_parser().parse_args([
        "--n-train", "64", "--n-val", "32", "--n-blocks", "4", "--block-size", "7",
        "--methods", "linear,radial,stacked", "--iters", "3", "--batch-size", "16",
        "--lr-schedule", "constant", "--checkpoint-selection", "raw", "--print-every", "1",
        "--m-dim", "2", "--device", "cpu", "--output-dir", str(stage1_dir),
    ]))
    out_dir = tmp_path / "stage2"
    stage2_npe.run(stage2_npe.build_parser().parse_args([
        "--stage1-run-dir", str(stage1_dir),
        "--methods", "pilot,linear,radial,stacked",
        "--n-sbi-train", "200", "--sbi-hidden-features", "8", "--sbi-num-components", "2",
        "--sbi-batch-size", "50", "--max-epochs", "2", "--stop-after-epochs", "1",
        "--test-pi-values", "0.3,0.5", "--test-seeds", "100,101",
        "--posterior-n", "64", "--grid-size", "200",
        "--pilot-grid-size", "21", "--pilot-sanity-n", "16", "--pilot-progress-every", "0",
        "--min-pilot-correlation", "-1",
        "--device", "cpu", "--data-device", "cpu", "--output-dir", str(out_dir),
    ]))
    assert (out_dir / "npe_stacked_state.pt").exists()
    with (out_dir / "posterior_by_seed.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    stacked = [row for row in rows if row["method"] == stage2_npe.METHOD_LABELS["stacked"]]
    assert len(stacked) == 4
    assert all(np.isfinite(float(row["w1_to_exact"])) for row in stacked)
    assert all(np.isfinite(float(row["score_context"])) for row in stacked)
    with (out_dir / "paired_w1_comparisons.csv").open() as handle:
        pairs = {(row["reference"], row["competitor"]) for row in csv.DictReader(handle)}
    label = stage2_npe.METHOD_LABELS
    assert {(label["stacked"], label["radial"]), (label["radial"], label["linear"]),
            (label["stacked"], label["linear"])} <= pairs
