from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path

import numpy as np

from common import raw_data_npe
from model1 import stage2_npe as model1_stage2
from model2 import stage2_npe as model2_stage2


ROOT = Path(__file__).resolve().parents[1]


def test_raw_context_is_literal_flatten_and_has_no_parameter_argument() -> None:
    y = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    context = raw_data_npe.raw_context(y)
    assert context.shape == (2, 12)
    assert context.dtype == np.float32
    np.testing.assert_array_equal(context, y.reshape(2, 12).astype(np.float32))
    forbidden = {"pi", "pi_true", "theta", "u_true"}
    assert not forbidden.intersection(inspect.signature(raw_data_npe.raw_context).parameters)


def test_raw_simulation_matches_formal_stage2_simulators() -> None:
    pi = np.asarray([0.1, 0.5], dtype=np.float64)
    for model, formal in (("model1", model1_stage2), ("model2", model2_stage2)):
        args = raw_data_npe.build_parser().parse_args(
            ["--model", model, "--output-dir", "unused"]
        )
        raw_data_npe.resolve_model_args(args)
        metadata = {
            "n_blocks": args.n_blocks,
            "block_size": args.block_size,
            "tau": args.tau,
        }
        expected = formal.simulate(np.random.default_rng(123), pi, metadata)
        actual = raw_data_npe.simulate(np.random.default_rng(123), pi, args)
        np.testing.assert_array_equal(actual, expected)


def test_raw_defaults_match_current_model_sizes() -> None:
    for model, expected in (
        ("model1", (20, 20, 0.5)),
        ("model2", (40, 40, 1.0)),
    ):
        args = raw_data_npe.build_parser().parse_args(
            ["--model", model, "--output-dir", "unused"]
        )
        raw_data_npe.resolve_model_args(args)
        assert (args.n_blocks, args.block_size, args.tau) == expected


def test_packaged_raw_baselines_have_complete_paired_evaluation() -> None:
    for relative, model, dimension in (
        ("artifacts/model1/stage2/raw_data_npe_50k", "model1", 400),
        ("artifacts/model2/k40_m40/stage2/raw_data_npe_50k", "model2", 1600),
    ):
        directory = ROOT / relative
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        assert config["model"] == model
        assert config["context_dim"] == dimension
        assert config["n_sbi_train"] == 50_000
        assert config["context_definition"] == raw_data_npe.CONTEXT_DEFINITION
        assert len(config["raw_context_sha256"]) == 64
        with (directory / "posterior_by_seed.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 120
        assert sum(row["method"] == raw_data_npe.METHOD_LABEL for row in rows) == 60
