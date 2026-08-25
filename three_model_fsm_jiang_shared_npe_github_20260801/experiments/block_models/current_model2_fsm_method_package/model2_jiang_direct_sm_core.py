#!/usr/bin/env python3
"""Reusable one-dimensional Jiang direct score-matching training utilities.

The model-specific runner supplies a differentiable per-block score
``score_net(u, y_block)``.  This module owns only the loss identities and the
training protocol, so the Model-2 marginal/pairwise representation stays in
its own file.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn


def resolve_device(text: str) -> str:
    if text == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if text == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return text


def boundary_weight(
    u: torch.Tensor,
    u_min: float,
    u_max: float,
    scale_dist: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    midpoint = 0.5 * (u_min + u_max)
    distance = torch.minimum(u - u_min, u_max - u)
    g = distance / scale_dist
    g1 = torch.where(u < midpoint, torch.ones_like(u), -torch.ones_like(u))
    return g, g1 / scale_dist


def score_matching_loss(
    score_net: nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    prop_score: torch.Tensor,
    *,
    u_min: float,
    u_max: float,
    scale_dist: float,
    create_graph: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Jiang direct-SM loss in the physical unconstrained coordinate u.

    The derivative is taken through the uncentered network output, while the
    other terms use minibatch-centered scores.  This asymmetry follows the
    reference objective and must not be simplified away.
    """

    u_work = u.detach().clone().requires_grad_(True)
    raw = score_net(u_work, y)
    centered = raw - raw.mean(dim=0)
    derivative = torch.autograd.grad(
        raw.sum(), u_work, create_graph=create_graph
    )[0]
    g, g1 = boundary_weight(u_work, u_min, u_max, scale_dist)
    quadratic = 0.5 * g * centered.square()
    derivative_term = g * derivative
    boundary_term = g1 * centered
    proposal_term = g * centered * prop_score
    loss = (quadratic + derivative_term + boundary_term + proposal_term).mean()
    return loss, {
        "quadratic": quadratic.mean().detach(),
        "derivative": derivative_term.mean().detach(),
        "boundary": boundary_term.mean().detach(),
        "proposal": proposal_term.mean().detach(),
        "batch_bias": raw.mean().detach(),
    }


def fisher_penalty(
    score_net: nn.Module,
    u: torch.Tensor,
    y: torch.Tensor,
    *,
    u_min: float,
    u_max: float,
    scale_dist: float,
) -> torch.Tensor:
    """Weighted empirical Bartlett/curvature penalty on a reference table."""

    batch, obs_size, block_size = y.shape
    u_work = u.detach().clone().requires_grad_(True)
    u_repeated = u_work[:, None].expand(-1, obs_size).reshape(-1)
    scores = score_net(
        u_repeated, y.reshape(-1, block_size)
    ).reshape(batch, obs_size)
    mean_score = scores.mean(dim=1)
    mean_derivative = torch.autograd.grad(
        mean_score.sum(), u_work, create_graph=True
    )[0]
    variance = (scores - mean_score[:, None]).square().mean(dim=1)
    g, _ = boundary_weight(u_work, u_min, u_max, scale_dist)
    return (g * (variance + mean_derivative)).square().mean()


def evaluate_sm_loss(
    score_net: nn.Module,
    data: dict[str, np.ndarray],
    batch_size: int,
    device: str,
    u_min: float,
    u_max: float,
    scale_dist: float,
) -> float:
    losses: list[float] = []
    for start in range(0, data["u"].shape[0], batch_size):
        stop = min(start + batch_size, data["u"].shape[0])
        loss, _ = score_matching_loss(
            score_net,
            torch.as_tensor(data["u"][start:stop], device=device),
            torch.as_tensor(data["y"][start:stop], device=device),
            torch.as_tensor(data["prop_score"][start:stop], device=device),
            u_min=u_min,
            u_max=u_max,
            scale_dist=scale_dist,
            create_graph=False,
        )
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_score_phase(
    score_net: nn.Module,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    reference: dict[str, np.ndarray] | None,
    *,
    steps: int,
    batch_size: int,
    ref_batch_size: int,
    lr: float,
    weight_decay: float,
    lam_fisher: float,
    patience: int,
    print_every: int,
    seed: int,
    device: str,
    u_min: float,
    u_max: float,
    scale_dist: float,
    phase: str,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    optimizer = torch.optim.Adam(
        score_net.parameters(), lr=lr, weight_decay=weight_decay
    )
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(score_net.state_dict())
    best_val = evaluate_sm_loss(
        score_net, val, batch_size, device, u_min, u_max, scale_dist
    )
    best_step = 0
    bad = 0
    trace: list[dict[str, Any]] = []
    print(f"{phase} step 0/{steps}: val_sm={best_val:.6g}")
    for step in range(1, steps + 1):
        index = rng.integers(
            0, train["u"].shape[0], size=min(batch_size, train["u"].shape[0])
        )
        u = torch.as_tensor(train["u"][index], device=device)
        y = torch.as_tensor(train["y"][index], device=device)
        prop = torch.as_tensor(train["prop_score"][index], device=device)
        optimizer.zero_grad(set_to_none=True)
        sm_loss, terms = score_matching_loss(
            score_net,
            u,
            y,
            prop,
            u_min=u_min,
            u_max=u_max,
            scale_dist=scale_dist,
            create_graph=True,
        )
        penalty = torch.zeros((), device=device)
        if reference is not None and lam_fisher > 0:
            ridx = rng.integers(
                0,
                reference["u"].shape[0],
                size=min(ref_batch_size, reference["u"].shape[0]),
            )
            penalty = fisher_penalty(
                score_net,
                torch.as_tensor(reference["u"][ridx], device=device),
                torch.as_tensor(reference["y"][ridx], device=device),
                u_min=u_min,
                u_max=u_max,
                scale_dist=scale_dist,
            )
        objective = sm_loss + lam_fisher * penalty
        if not torch.isfinite(objective):
            raise RuntimeError(f"Non-finite {phase} objective at step {step}")
        objective.backward()
        if not all(
            p.grad is None or bool(torch.isfinite(p.grad).all())
            for p in score_net.parameters()
        ):
            raise RuntimeError(f"Non-finite {phase} gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % print_every == 0 or step == steps:
            val_loss = evaluate_sm_loss(
                score_net, val, batch_size, device, u_min, u_max, scale_dist
            )
            row = {
                "phase": phase,
                "step": step,
                "train_objective": float(objective.detach().cpu()),
                "train_sm": float(sm_loss.detach().cpu()),
                "train_fisher_penalty": float(penalty.detach().cpu()),
                "val_sm": val_loss,
                **{f"term_{key}": float(value.cpu()) for key, value in terms.items()},
            }
            trace.append(row)
            print(
                f"{phase} step {step}/{steps}: obj={row['train_objective']:.6g}, "
                f"sm={row['train_sm']:.6g}, pen={row['train_fisher_penalty']:.6g}, "
                f"val_sm={val_loss:.6g}, best={min(best_val, val_loss):.6g}"
            )
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = copy.deepcopy(score_net.state_dict())
                best_step = step
                bad = 0
            else:
                bad += 1
            if patience > 0 and bad >= patience:
                print(f"{phase}: early stop at step {step}")
                break
    score_net.load_state_dict(best_state)
    return score_net, trace, {"best_step": best_step, "best_val_sm": best_val}


@torch.no_grad()
def reference_mean_scores(
    score_net: nn.Module,
    table: dict[str, np.ndarray],
    anchor_batch: int,
    device: str,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, table["u"].shape[0], anchor_batch):
        stop = min(start + anchor_batch, table["u"].shape[0])
        y = torch.as_tensor(table["y"][start:stop], device=device)
        u = torch.as_tensor(table["u"][start:stop], device=device)
        n, m, d = y.shape
        score = score_net(
            u[:, None].expand(-1, m).reshape(-1), y.reshape(-1, d)
        ).reshape(n, m)
        output.append(score.mean(dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


class DebiasRegression(nn.Module):
    def __init__(self, hidden: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        width = 1
        for _ in range(depth):
            layers.extend([nn.Linear(width, hidden), nn.ELU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u.reshape(-1, 1)).squeeze(-1)


def train_debias(
    model: nn.Module,
    train_u: np.ndarray,
    train_target: np.ndarray,
    val_u: np.ndarray,
    val_target: np.ndarray,
    *,
    u_min: float,
    u_max: float,
    scale_dist: float,
    steps: int,
    curve_steps: int,
    batch_size: int,
    lr: float,
    curve_lr: float,
    lam_curve: float,
    patience: int,
    print_every: int,
    seed: int,
    device: str,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)

    def regression_loss(
        u_np: np.ndarray,
        target_np: np.ndarray,
        create_graph: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = torch.as_tensor(u_np, device=device)
        target = torch.as_tensor(target_np, device=device)
        if create_graph:
            u = u.detach().clone().requires_grad_(True)
        pred = model(u)
        g, _ = boundary_weight(u, u_min, u_max, scale_dist)
        return (g * (pred - target).square()).mean(), pred, target, u, g

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        best_val = float(regression_loss(val_u, val_target)[0].cpu())
    best_step = 0
    bad = 0
    trace: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        idx = rng.integers(
            0, train_u.shape[0], size=min(batch_size, train_u.shape[0])
        )
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _, _ = regression_loss(train_u[idx], train_target[idx])
        loss.backward()
        optimizer.step()
        if step == 1 or step % print_every == 0 or step == steps:
            with torch.no_grad():
                val_loss = float(regression_loss(val_u, val_target)[0].cpu())
            trace.append(
                {
                    "phase": "debias",
                    "step": step,
                    "train_loss": float(loss.detach().cpu()),
                    "val_loss": val_loss,
                }
            )
            if val_loss < best_val - 1e-8:
                best_val, best_step, bad = val_loss, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if patience > 0 and bad >= patience:
                break
    model.load_state_dict(best_state)

    initial_state = copy.deepcopy(model.state_dict())
    initial_val = best_val
    optimizer = torch.optim.Adam(model.parameters(), lr=curve_lr)
    curve_best_state: dict[str, torch.Tensor] | None = None
    curve_best_val = initial_val
    curve_best_step = 0
    for step in range(1, curve_steps + 1):
        idx = rng.integers(
            0, train_u.shape[0], size=min(batch_size, train_u.shape[0])
        )
        optimizer.zero_grad(set_to_none=True)
        reg, pred, target, u, g = regression_loss(
            train_u[idx], train_target[idx], create_graph=True
        )
        derivative = torch.autograd.grad(pred.sum(), u, create_graph=True)[0]
        curve = (
            g.square() * (pred.square() - derivative - 2.0 * target * pred).square()
        ).mean()
        objective = reg + lam_curve * curve
        if not torch.isfinite(objective):
            raise RuntimeError(f"Non-finite debias curve objective at step {step}")
        objective.backward()
        optimizer.step()
        if step == 1 or step % print_every == 0 or step == curve_steps:
            with torch.no_grad():
                val_loss = float(regression_loss(val_u, val_target)[0].cpu())
            trace.append(
                {
                    "phase": "debias_curve",
                    "step": step,
                    "train_loss": float(reg.detach().cpu()),
                    "curve_loss": float(curve.detach().cpu()),
                    "val_loss": val_loss,
                }
            )
            if val_loss < curve_best_val - 1e-8:
                curve_best_val = val_loss
                curve_best_step = step
                curve_best_state = copy.deepcopy(model.state_dict())
    if curve_best_state is None:
        model.load_state_dict(initial_state)
    else:
        model.load_state_dict(curve_best_state)
    return model, trace, {
        "best_step": best_step,
        "best_val_loss": min(initial_val, curve_best_val),
        "curve_kept": curve_best_state is not None,
        "curve_best_step": curve_best_step,
    }
