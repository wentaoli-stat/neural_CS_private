from __future__ import annotations

import numpy as np
import torch

from model1 import stage1 as model1
from model2 import stage1 as model2


def test_model1_radial_identity_matches_linear() -> None:
    torch.manual_seed(7)
    linear = model1.LinearCSBetaDeepSets(hidden=12, depth=2)
    radial = model1.RadialCSBetaDeepSets(
        hidden=12,
        depth=2,
        gate_hidden=8,
        gate_condition_on_anchor=True,
        block_mean=np.zeros((1, 1, 1)),
        block_sd=np.ones((1, 1, 1)),
    )
    radial.rho.load_state_dict(linear.rho.state_dict())
    score = torch.randn(5, 4, 7)
    anchor = torch.randn(5)
    block = score.mean(dim=2, keepdim=True)
    torch.testing.assert_close(radial(score, anchor), linear(block, anchor), atol=1e-6, rtol=0)


def test_model2_shared_identity_matches_linear() -> None:
    torch.manual_seed(11)
    linear = model2.LinearCSBetaDeepSets(hidden=12, depth=2)
    shared = model2.SharedRawRadialCSBetaDeepSets(
        hidden=12,
        depth=2,
        gate_hidden=8,
        gate_condition_on_anchor=True,
        s1_mean=np.array(0.13),
        s1_sd=np.array(0.7),
        s2_mean=np.array(-0.08),
        s2_sd=np.array(1.2),
        block_mean=np.zeros((1, 1, 2)),
        block_sd=np.ones((1, 1, 2)),
    )
    shared.rho.load_state_dict(linear.rho.state_dict())
    s1 = torch.randn(5, 4, 7)
    s2 = torch.randn(5, 4, 11)
    anchor = torch.randn(5)
    block = torch.stack([s1.mean(dim=2), s2.mean(dim=2)], dim=-1)
    torch.testing.assert_close(shared(s1, s2, anchor), linear(block, anchor), atol=1e-6, rtol=0)

