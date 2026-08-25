from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import torch

from khoo_vs_jiang.dgp import HMMConfig
from khoo_vs_jiang.jiang_official import RawSequenceGRUScore
from khoo_vs_jiang.upstreams import load_upstreams


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_jiang_uses_official_raw_trajectory_network() -> None:
    upstreams = load_upstreams()
    official = upstreams.jiang_utils
    config = HMMConfig(length=8, block_size=4)
    model = official.ELU_single_LikeScoreMatchingNN(
        1, config.x_dim, 1, hidden_size=11, num_layers=2
    )

    assert model.__class__.__module__ == "MLE.utils_sm"
    assert model.x_dim == config.x_dim
    assert model.obs_size == 1
    assert model.layers[0].in_features == 1 + config.x_dim
    output = model(torch.zeros(3, 1), torch.zeros(3, config.x_dim))
    assert output.shape == (3, 1)


def test_jiang_gru_ablation_preserves_single_and_repeated_score_interfaces() -> None:
    config = HMMConfig(length=8, block_size=4)
    model = RawSequenceGRUScore(
        config,
        hidden_size=7,
        y_mean=np.zeros(config.block_size, dtype=np.float32),
        y_sd=np.ones(config.block_size, dtype=np.float32),
    )

    theta = torch.zeros(3, 1, requires_grad=True)
    single = torch.zeros(3, config.x_dim)
    repeated = torch.zeros(3, 5 * config.x_dim)
    assert model(theta, single).shape == (3, 1)
    assert model.cal_penalty(theta, repeated).shape == (3, 5, 1)
    assert model.x_dim == config.x_dim

    value = model(theta, single).sum()
    derivative = torch.autograd.grad(value, theta, create_graph=True)[0]
    derivative.square().sum().backward()


def test_khoo_uses_upstream_composite_gate_gru_class() -> None:
    upstreams = load_upstreams()
    one = upstreams.khoo_oneparam
    time_mean = np.zeros(2, dtype=np.float32)
    time_sd = np.ones(2, dtype=np.float32)
    model = one.GatedLocalGRU(7, 5, time_mean, time_sd)

    assert model.__class__.__module__ == (
        "run_hmm_oneparam_local_ncs_gru_experiment"
    )
    assert model.marginal_gate.__class__.__name__ == "PositiveMLPMultiplier"
    assert model.pairwise_gate.__class__.__name__ == "PositiveMLPMultiplier"
    assert model.core.__class__.__name__ == "AnchorGRUCore"

    # The two inputs are the upstream marginal and pairwise local composite
    # subscore tensors, not raw observations or a recovered likelihood.
    s1 = torch.zeros(4, 3, 6)
    s2 = torch.zeros(4, 3, 5)
    anchor = torch.zeros(4, 1)
    assert model(s1, s2, anchor).shape == (4, 1)


def test_stage1_source_has_no_npe_or_nle_imports() -> None:
    forbidden_fragments = (
        "sbi.inference",
        "nflows",
        "pyknos",
        "neural_posterior",
        "neural_likelihood",
    )
    imported: list[str] = []
    stage1_paths = (
        REPO_ROOT / "src" / "khoo_vs_jiang" / "jiang_official.py",
        REPO_ROOT / "src" / "khoo_vs_jiang" / "block_mixture_jiang.py",
        REPO_ROOT / "src" / "khoo_vs_jiang" / "nonlinear_emission_screen.py",
    )
    for path in stage1_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

    assert not any(
        fragment in module
        for module in imported
        for fragment in forbidden_fragments
    ), imported
