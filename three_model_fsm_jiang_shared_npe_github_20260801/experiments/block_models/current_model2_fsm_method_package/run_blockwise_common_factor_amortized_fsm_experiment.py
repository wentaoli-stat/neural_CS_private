#!/usr/bin/env python3
"""Amortized conditional FSM for the blockwise common-factor mixture.

This is a first, deliberately simple version of the amortized score field

    S(Y, beta_anchor) ~= d/dbeta log p_beta(Y)

for the Model 2 common-factor variance mixture. We work in the unconstrained
coordinate u=logit(pi), sample anchors u_a over a pi range, sample local
perturbations u | u_a ~ N(u_a, sigma_q^2), simulate Y ~ p_u, and regress

    S(Y, u_a) -> (u - u_a) / sigma_q^2.

The exact score is used only for diagnostics after training. By default, the
local composite-score features are evaluated at each sample's own anchor
pi_a, so the network really sees a beta-conditional approximation problem.
The old fixed-pi_ref feature mode is kept as an ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F


def sigmoid_np(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def logit_np(p: np.ndarray | float) -> np.ndarray:
    p_arr = np.asarray(p, dtype=np.float64)
    return np.log(p_arr) - np.log1p(-p_arr)


def softplus_inverse(x: float) -> float:
    return math.log(math.expm1(x))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def score_u_from_log_ratio(log_ratio: np.ndarray, pi: float | np.ndarray) -> np.ndarray:
    """Mixture score wrt u=logit(pi) from a likelihood ratio log f1/f0.

    For a local two-component mixture, posterior active probability is
    sigmoid(logit(pi) + log_ratio), hence the u-score is posterior_prob - pi.
    """
    log_ratio = np.asarray(log_ratio, dtype=np.float64)
    pi_arr = np.asarray(pi, dtype=np.float64)
    if pi_arr.ndim == 0:
        logit_pi = logit_np(float(pi_arr))
        w = sigmoid_np(logit_pi + log_ratio)
        return w - float(pi_arr)
    shape = (pi_arr.shape[0],) + (1,) * (log_ratio.ndim - 1)
    pi_b = pi_arr.reshape(shape)
    w = sigmoid_np(logit_np(pi_arr).reshape(shape) + log_ratio)
    return w - pi_b


def simulate_common_factor(
    rng: np.random.Generator,
    u: np.ndarray,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> np.ndarray:
    """Simulate Y_{ki} = eps_{ki} + B_k * tau * Z_k."""
    u = np.asarray(u, dtype=np.float64)
    pi = sigmoid_np(u)
    active = rng.binomial(1, pi[:, None], size=(u.shape[0], n_blocks)).astype(np.float64)
    eps = rng.normal(size=(u.shape[0], n_blocks, block_size))
    z = rng.normal(size=(u.shape[0], n_blocks, 1))
    return eps + active[:, :, None] * tau * z


def full_block_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    """Block likelihood ratio log f1(Y_k) / f0(Y_k)."""
    m = y.shape[-1]
    block_sum = y.sum(axis=-1)
    return -0.5 * math.log1p(m * tau**2) + (
        tau**2 / (2.0 * (1.0 + m * tau**2))
    ) * block_sum**2


def marginal_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    return -0.5 * math.log1p(tau**2) + (tau**2 / (2.0 * (1.0 + tau**2))) * y**2


def pairwise_log_ratio(y: np.ndarray, tau: float) -> np.ndarray:
    m = y.shape[-1]
    i_idx, j_idx = np.triu_indices(m, k=1)
    pair_sum = y[:, :, i_idx] + y[:, :, j_idx]
    return -0.5 * math.log1p(2.0 * tau**2) + (
        tau**2 / (2.0 * (1.0 + 2.0 * tau**2))
    ) * pair_sum**2


def full_score_u(y: np.ndarray, pi: float | np.ndarray, tau: float) -> np.ndarray:
    return score_u_from_log_ratio(full_block_log_ratio(y, tau), pi).sum(axis=1)


def sample_from_anchor_pi(
    rng: np.random.Generator,
    anchor_pi: np.ndarray,
    sigma_q: float,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> dict[str, np.ndarray]:
    """Simulate FSM pairs for a supplied vector of anchor probabilities."""
    anchor_pi = np.asarray(anchor_pi, dtype=np.float64)
    anchor_u = logit_np(anchor_pi)
    eps = rng.normal(size=anchor_pi.shape[0])
    u = anchor_u + sigma_q * eps
    pi = sigmoid_np(u)
    y = simulate_common_factor(rng, u, n_blocks, block_size, tau)
    target = (u - anchor_u) / (sigma_q**2)
    return {
        "y": y,
        "anchor_pi": anchor_pi,
        "anchor_u": anchor_u,
        "u": u,
        "pi": pi,
        "target": target,
    }


def sample_anchor_batch(
    rng: np.random.Generator,
    n: int,
    pi_min: float,
    pi_max: float,
    sigma_q: float,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> dict[str, np.ndarray]:
    """Sample continuous stratified anchors and their local FSM pairs."""
    grid = (np.arange(n, dtype=np.float64) + rng.uniform(size=n)) / max(n, 1)
    rng.shuffle(grid)
    anchor_pi = pi_min + (pi_max - pi_min) * grid
    return sample_from_anchor_pi(rng, anchor_pi, sigma_q, n_blocks, block_size, tau)


def sample_fixed_anchor_batch(
    rng: np.random.Generator,
    n: int,
    anchor_pi: float,
    sigma_q: float,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> dict[str, np.ndarray]:
    """Sample local FSM pairs at one fixed anchor for validation curves."""
    anchors = np.full(n, float(anchor_pi), dtype=np.float64)
    return sample_from_anchor_pi(rng, anchors, sigma_q, n_blocks, block_size, tau)


def feature_pi_for_raw(raw: dict[str, np.ndarray], pi_ref: float, mode: str) -> np.ndarray | float:
    if mode == "anchor":
        return raw["anchor_pi"]
    if mode == "fixed":
        return float(pi_ref)
    raise ValueError(f"Unknown feature pi mode: {mode}")


def make_feature_stats(
    y: np.ndarray,
    tau: float,
    feature_pi: np.ndarray | float,
) -> dict[str, np.ndarray]:
    s1 = score_u_from_log_ratio(marginal_log_ratio(y, tau), feature_pi)
    s2 = score_u_from_log_ratio(pairwise_log_ratio(y, tau), feature_pi)
    stats = {
        "s1_mean": np.array(float(s1.mean()), dtype=np.float64),
        "s1_sd": np.array(float(s1.std()), dtype=np.float64),
        "s2_mean": np.array(float(s2.mean()), dtype=np.float64),
        "s2_sd": np.array(float(s2.std()), dtype=np.float64),
    }
    for key in ("s1_sd", "s2_sd"):
        stats[key] = np.where(stats[key] < 1e-8, 1.0, stats[key])
    s1_z = (s1 - stats["s1_mean"]) / stats["s1_sd"]
    s2_z = (s2 - stats["s2_mean"]) / stats["s2_sd"]
    block_raw = np.stack([s1_z.mean(axis=2), s2_z.mean(axis=2)], axis=-1)
    stats["block_mean"] = block_raw.mean(axis=(0, 1), keepdims=True)
    stats["block_sd"] = block_raw.std(axis=(0, 1), keepdims=True)
    stats["block_sd"] = np.where(stats["block_sd"] < 1e-8, 1.0, stats["block_sd"])
    return stats


def featurize(
    y: np.ndarray,
    tau: float,
    feature_pi: np.ndarray | float,
    stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    s1 = score_u_from_log_ratio(marginal_log_ratio(y, tau), feature_pi)
    s2 = score_u_from_log_ratio(pairwise_log_ratio(y, tau), feature_pi)
    s1_z = ((s1 - stats["s1_mean"]) / stats["s1_sd"]).astype(np.float32)
    s2_z = ((s2 - stats["s2_mean"]) / stats["s2_sd"]).astype(np.float32)
    block_raw = np.stack([s1_z.mean(axis=2), s2_z.mean(axis=2)], axis=-1)
    block = ((block_raw - stats["block_mean"]) / stats["block_sd"]).astype(np.float32)
    return {"s1": s1_z, "s2": s2_z, "block": block}


def standardize_anchor(
    anchor_u_train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mean = float(anchor_u_train.mean())
    sd = float(anchor_u_train.std())
    if sd < 1e-8:
        sd = 1.0
    return tuple(((x - mean) / sd).astype(np.float32) for x in (anchor_u_train, *others))


class PositiveMLPMultiplier(nn.Module):
    """Positive scalar multiplier, initialized exactly at m=1.

    With ``input_dim=1`` this is m(s). With ``input_dim=2`` this is the
    beta-conditioned gate m(s, u_anchor), which is the default radial map in
    the amortized setting.
    """

    def __init__(self, hidden: int, input_dim: int = 1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.constant_(final.bias, softplus_inverse(1.0))

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        if len(features) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} gate features, got {len(features)}")
        x = torch.stack(features, dim=-1)
        return F.softplus(self.net(x).squeeze(-1))


def make_rho(in_dim: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.SiLU())
        d = hidden
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


class LinearCSBetaDeepSets(nn.Module):
    """DeepSets readout over block-level linear CS, conditioned on anchor beta."""

    def __init__(self, hidden: int, depth: int):
        super().__init__()
        self.rho = make_rho(3, hidden, depth)

    def forward(self, block: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        anchor = anchor_z.reshape(-1, 1, 1).expand(-1, block.shape[1], 1)
        x = torch.cat([block, anchor], dim=-1)
        return self.rho(x).squeeze(-1).sum(dim=1)


class RadialCSBetaDeepSets(nn.Module):
    """Signed positive radial-style gates followed by beta-conditioned DeepSets."""

    def __init__(
        self,
        hidden: int,
        depth: int,
        gate_hidden: int,
        gate_condition_on_anchor: bool,
        block_mean: np.ndarray,
        block_sd: np.ndarray,
    ):
        super().__init__()
        self.gate_condition_on_anchor = bool(gate_condition_on_anchor)
        gate_input_dim = 2 if self.gate_condition_on_anchor else 1
        self.marginal_gate = PositiveMLPMultiplier(gate_hidden, input_dim=gate_input_dim)
        self.pairwise_gate = PositiveMLPMultiplier(gate_hidden, input_dim=gate_input_dim)
        self.rho = make_rho(3, hidden, depth)
        self.register_buffer("block_mean", torch.as_tensor(block_mean, dtype=torch.float32))
        self.register_buffer("block_sd", torch.as_tensor(block_sd, dtype=torch.float32))

    def forward(self, s1: torch.Tensor, s2: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        if self.gate_condition_on_anchor:
            anchor_s1 = anchor_z.reshape(-1, 1, 1).expand_as(s1)
            anchor_s2 = anchor_z.reshape(-1, 1, 1).expand_as(s2)
            gated_s1 = s1 * self.marginal_gate(s1, anchor_s1)
            gated_s2 = s2 * self.pairwise_gate(s2, anchor_s2)
        else:
            gated_s1 = s1 * self.marginal_gate(s1)
            gated_s2 = s2 * self.pairwise_gate(s2)
        block_raw = torch.stack([gated_s1.mean(dim=2), gated_s2.mean(dim=2)], dim=-1)
        block = (block_raw - self.block_mean) / self.block_sd
        anchor = anchor_z.reshape(-1, 1, 1).expand(-1, block.shape[1], 1)
        x = torch.cat([block, anchor], dim=-1)
        return self.rho(x).squeeze(-1).sum(dim=1)


def tensors_for_method(data: dict[str, np.ndarray], method: str, device: str) -> dict[str, torch.Tensor]:
    if method == "linear":
        keys = ("block", "anchor_z")
    elif method == "radial":
        keys = ("s1", "s2", "anchor_z")
    else:
        raise ValueError(f"Unknown method: {method}")
    return {key: torch.as_tensor(data[key], dtype=torch.float32, device=device) for key in keys}


def forward_method(model: nn.Module, batch: dict[str, torch.Tensor], method: str) -> torch.Tensor:
    if method == "linear":
        return model(batch["block"], batch["anchor_z"])
    if method == "radial":
        return model(batch["s1"], batch["s2"], batch["anchor_z"])
    raise ValueError(f"Unknown method: {method}")


def mse_over_tensors(
    model: nn.Module,
    x: dict[str, torch.Tensor],
    y: torch.Tensor,
    method: str,
    batch_size: int,
) -> float:
    model.eval()
    sqerr = 0.0
    n_total = 0
    with torch.no_grad():
        n = int(y.shape[0])
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            xb = {k: v[sl] for k, v in x.items()}
            pred = forward_method(model, xb, method)
            err = pred - y[sl]
            sqerr += float(torch.sum(err**2).item())
            n_total += int(err.numel())
    model.train()
    return sqerr / max(n_total, 1)


def predict(
    model: nn.Module,
    data: dict[str, np.ndarray],
    method: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    x = tensors_for_method(data, method, device)
    n = int(next(iter(x.values())).shape[0])
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            xb = {k: v[sl] for k, v in x.items()}
            out.append(forward_method(model, xb, method).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def clone_state_dict(model: nn.Module, *, cpu: bool) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        cloned = value.detach().clone()
        state[key] = cloned.cpu() if cpu else cloned
    return state


@torch.no_grad()
def update_ema_state(
    ema_state: dict[str, torch.Tensor],
    model: nn.Module,
    decay: float,
) -> None:
    for key, value in model.state_dict().items():
        if torch.is_floating_point(value):
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
        else:
            ema_state[key].copy_(value.detach())


def train_model(
    model: nn.Module,
    method: str,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    *,
    iters: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    print_every: int,
    seed: int,
    grad_clip: float,
    ema_decay: float,
    gate_only_steps: int,
    gate_lr: float,
    joint_rho_lr: float,
    device: str,
    name: str,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    model = model.to(device)
    x_train = tensors_for_method(train, method, device)
    x_val = tensors_for_method(val, method, device)
    y_train = torch.as_tensor(train["target"], dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val["target"], dtype=torch.float32, device=device)
    if method == "radial":
        gate_params = list(model.marginal_gate.parameters()) + list(model.pairwise_gate.parameters())
        rho_params = list(model.rho.parameters())
        opt = torch.optim.AdamW(
            [
                {"params": gate_params, "lr": gate_lr, "name": "gate"},
                {
                    "params": rho_params,
                    "lr": 0.0 if gate_only_steps > 0 else joint_rho_lr,
                    "name": "rho",
                },
            ],
            weight_decay=weight_decay,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best_state = clone_state_dict(model, cpu=True)
    ema_state = clone_state_dict(model, cpu=False)
    best_val = mse_over_tensors(model, x_val, y_val, method, batch_size)
    if not np.isfinite(best_val):
        raise RuntimeError(f"{name} produced non-finite initial validation loss")
    best_step = 0
    best_source = "raw"
    trace: list[dict[str, Any]] = [
        {
            "method": name,
            "step": 0,
            "stage": "gate_only" if method == "radial" and gate_only_steps > 0 else "joint",
            "train_loss": best_val,
            "val_loss_raw": best_val,
            "val_loss_ema": best_val,
            "best_val_loss": best_val,
            "best_source": best_source,
        }
    ]
    bad = 0
    n_train = int(y_train.shape[0])
    print(f"{name} step 0/{iters}: val={best_val:.6g}")

    for step in range(1, iters + 1):
        if method == "radial" and gate_only_steps > 0 and step == gate_only_steps + 1:
            opt.param_groups[1]["lr"] = joint_rho_lr
            print(f"{name}: unfreezing rho at step {step} with lr={joint_rho_lr:g}")
        idx = torch.as_tensor(
            rng.integers(0, n_train, size=min(batch_size, n_train)),
            dtype=torch.long,
            device=device,
        )
        xb = {k: v.index_select(0, idx) for k, v in x_train.items()}
        yb = y_train.index_select(0, idx)
        opt.zero_grad(set_to_none=True)
        pred = forward_method(model, xb, method)
        loss = torch.mean((pred - yb) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name} produced non-finite training loss at step {step}")
        loss.backward()
        for param in model.parameters():
            if param.grad is not None and not torch.all(torch.isfinite(param.grad)):
                raise RuntimeError(f"{name} produced non-finite gradients at step {step}")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"{name} produced non-finite gradient norm at step {step}")
        opt.step()
        update_ema_state(ema_state, model, ema_decay)
        if step == 1 or step % print_every == 0 or step == iters:
            val_loss_raw = mse_over_tensors(model, x_val, y_val, method, batch_size)
            raw_state = clone_state_dict(model, cpu=False)
            model.load_state_dict(ema_state)
            val_loss_ema = mse_over_tensors(model, x_val, y_val, method, batch_size)
            model.load_state_dict(raw_state)
            if not np.isfinite(val_loss_raw) or not np.isfinite(val_loss_ema):
                raise RuntimeError(f"{name} produced non-finite validation loss at step {step}")
            if val_loss_ema < val_loss_raw:
                selected_val = val_loss_ema
                selected_source = "ema"
                selected_state = {key: value.detach().cpu().clone() for key, value in ema_state.items()}
            else:
                selected_val = val_loss_raw
                selected_source = "raw"
                selected_state = clone_state_dict(model, cpu=True)
            stage = "gate_only" if method == "radial" and step <= gate_only_steps else "joint"
            trace.append(
                {
                    "method": name,
                    "step": step,
                    "stage": stage,
                    "train_loss": float(loss.detach().cpu().item()),
                    "val_loss_raw": float(val_loss_raw),
                    "val_loss_ema": float(val_loss_ema),
                    "best_val_loss": float(min(best_val, selected_val)),
                    "best_source": selected_source if selected_val < best_val else best_source,
                    "grad_norm": float(grad_norm.detach().cpu().item()),
                }
            )
            print(
                f"{name} step {step}/{iters} [{stage}]: train={loss.item():.6g}, "
                f"val_raw={val_loss_raw:.6g}, val_ema={val_loss_ema:.6g}, "
                f"best={min(best_val, selected_val):.6g}"
            )
            if selected_val < best_val - 1e-6:
                best_val = selected_val
                best_state = selected_state
                best_step = step
                best_source = selected_source
                bad = 0
            else:
                bad += 1
            if patience > 0 and bad >= patience:
                print(f"{name}: early stopping at step {step}; best val={best_val:.6g}")
                break
    model.load_state_dict(best_state)
    info = {
        "best_step": best_step,
        "best_val_loss": float(best_val),
        "best_source": best_source,
        "stopped_step": int(trace[-1]["step"]),
    }
    return model, trace, info


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    var = float(np.var(y_true) + 1e-12)
    pred_sd = float(np.std(y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if pred_sd > 1e-12 else float("nan")
    denom = float(np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    cosine = float(np.dot(y_true, y_pred) / denom) if denom > 1e-12 else float("nan")
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "std_mse": mse / var,
        "std_rmse": math.sqrt(mse / var),
        "corr": corr,
        "cosine": cosine,
        "target_mean": float(np.mean(y_true)),
        "target_sd": float(np.std(y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_sd": pred_sd,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prepare_split(
    raw: dict[str, np.ndarray],
    tau: float,
    pi_ref: float,
    feature_pi_mode: str,
    stats: dict[str, np.ndarray],
    anchor_z: np.ndarray,
) -> dict[str, np.ndarray]:
    feature_pi = feature_pi_for_raw(raw, pi_ref, feature_pi_mode)
    feat = featurize(raw["y"], tau, feature_pi, stats)
    feat["anchor_z"] = anchor_z.astype(np.float32)
    feat["target"] = raw["target"].astype(np.float32)
    feat["anchor_pi"] = raw["anchor_pi"].astype(np.float64)
    return feat


def build_eval_data(
    rng: np.random.Generator,
    pi: float,
    n: int,
    n_blocks: int,
    block_size: int,
    tau: float,
    pi_ref: float,
    feature_pi_mode: str,
    stats: dict[str, np.ndarray],
    anchor_mean: float,
    anchor_sd: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    u = np.full(n, float(logit_np(pi)), dtype=np.float64)
    y = simulate_common_factor(rng, u, n_blocks, block_size, tau)
    feature_pi: np.ndarray | float
    if feature_pi_mode == "anchor":
        feature_pi = np.full(n, pi, dtype=np.float64)
    elif feature_pi_mode == "fixed":
        feature_pi = float(pi_ref)
    else:
        raise ValueError(f"Unknown feature pi mode: {feature_pi_mode}")
    feat = featurize(y, tau, feature_pi, stats)
    feat["anchor_z"] = ((u - anchor_mean) / anchor_sd).astype(np.float32)
    truth = full_score_u(y, pi, tau)
    return feat, truth


def run(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (0.0 < args.pi_ref < 1.0):
        raise ValueError("--pi-ref must be in (0, 1)")
    if args.feature_pi_mode not in {"anchor", "fixed"}:
        raise ValueError("--feature-pi-mode must be anchor or fixed")
    if not (0.0 < args.anchor_pi_min < args.anchor_pi_max < 1.0):
        raise ValueError("anchor pi range must lie inside (0, 1)")
    if args.sigma_q <= 0:
        raise ValueError("--sigma-q must be positive")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")
    if not (0.0 <= args.ema_decay < 1.0):
        raise ValueError("--ema-decay must lie in [0, 1)")
    if args.gate_only_steps < 0:
        raise ValueError("--gate-only-steps must be nonnegative")
    validation_pis = [float(value) for value in args.validation_pi_values.split(",") if value.strip()]
    if not validation_pis:
        raise ValueError("--validation-pi-values must contain at least one value")
    if any(pi < args.anchor_pi_min or pi > args.anchor_pi_max for pi in validation_pis):
        raise ValueError("validation pi values must lie inside the anchor training range")
    if args.n_validation_per_anchor <= 0:
        raise ValueError("--n-validation-per-anchor must be positive")

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"torch device: {device}")
    print(f"Output directory: {out_dir}")
    print(
        "setting:",
        {
            "n_blocks": args.n_blocks,
            "block_size": args.block_size,
            "tau": args.tau,
            "pi_ref": args.pi_ref,
            "feature_pi_mode": args.feature_pi_mode,
            "anchor_pi_range": [args.anchor_pi_min, args.anchor_pi_max],
            "sigma_q": args.sigma_q,
            "gate_condition_on_anchor": bool(args.gate_condition_on_anchor),
            "methods": args.methods,
        },
    )

    rng = np.random.default_rng(args.seed)
    print("\nGenerating amortized FSM data...")
    train_raw = sample_anchor_batch(
        rng,
        args.n_train,
        args.anchor_pi_min,
        args.anchor_pi_max,
        args.sigma_q,
        args.n_blocks,
        args.block_size,
        args.tau,
    )
    val_raw = sample_anchor_batch(
        rng,
        args.n_val,
        args.anchor_pi_min,
        args.anchor_pi_max,
        args.sigma_q,
        args.n_blocks,
        args.block_size,
        args.tau,
    )
    print("train y:", train_raw["y"].shape, "val y:", val_raw["y"].shape)

    if args.feature_pi_mode == "anchor":
        print("Computing anchor/beta-conditioned local CS features...")
    else:
        print("Computing fixed-pi_ref local CS features...")
    train_feature_pi = feature_pi_for_raw(train_raw, args.pi_ref, args.feature_pi_mode)
    stats = make_feature_stats(train_raw["y"], args.tau, train_feature_pi)
    anchor_z_train, anchor_z_val = standardize_anchor(train_raw["anchor_u"], val_raw["anchor_u"])
    anchor_mean = float(train_raw["anchor_u"].mean())
    anchor_sd = float(train_raw["anchor_u"].std())
    if anchor_sd < 1e-8:
        anchor_sd = 1.0
    train = prepare_split(train_raw, args.tau, args.pi_ref, args.feature_pi_mode, stats, anchor_z_train)
    val = prepare_split(val_raw, args.tau, args.pi_ref, args.feature_pi_mode, stats, anchor_z_val)
    print("linear block features:", train["block"].shape)
    print("local marginal/pairwise:", train["s1"].shape, train["s2"].shape)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    bad = sorted(set(methods) - {"linear", "radial"})
    if bad:
        raise ValueError(f"Unknown methods: {bad}")
    if "radial" in methods and "linear" not in methods:
        raise ValueError("The primary radial method requires linear in --methods for exact warm-start")

    config = vars(args).copy()
    config["device_resolved"] = device
    config["anchor_sampler"] = "continuous_stratified"
    config["sigma_q_mode"] = "fixed"
    config["block_aggregation"] = "mean"
    config["rho_activation"] = "silu"
    config["anchor_u_mean"] = anchor_mean
    config["anchor_u_sd"] = anchor_sd
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    np.savez(
        out_dir / "feature_stats.npz",
        **stats,
        anchor_u_mean=np.array(anchor_mean),
        anchor_u_sd=np.array(anchor_sd),
    )

    models: dict[str, tuple[str, nn.Module]] = {}
    all_trace: list[dict[str, Any]] = []
    training_info: dict[str, Any] = {}
    linear_model: LinearCSBetaDeepSets | None = None
    for method in (candidate for candidate in ("linear", "radial") if candidate in methods):
        if method == "linear":
            label = "linear CS beta-conditioned DeepSets FSM"
            set_global_seed(args.seed + 11)
            model = LinearCSBetaDeepSets(args.hidden, args.depth)
        else:
            if args.gate_condition_on_anchor:
                label = "radial CS beta-conditioned gate+DeepSets FSM"
            else:
                label = "radial CS beta-conditioned DeepSets FSM"
            if linear_model is None:
                raise RuntimeError("linear model must be trained before radial warm-start")
            set_global_seed(args.seed + 22)
            model = RadialCSBetaDeepSets(
                args.hidden,
                args.depth,
                args.gate_hidden,
                gate_condition_on_anchor=bool(args.gate_condition_on_anchor),
                block_mean=stats["block_mean"],
                block_sd=stats["block_sd"],
            )
            model.rho.load_state_dict(linear_model.rho.state_dict())
            model = model.to(device)
            check_n = min(256, int(train["target"].shape[0]))
            check_data = {key: value[:check_n] for key, value in train.items()}
            linear_check = predict(linear_model, check_data, "linear", device, args.batch_size)
            radial_check = predict(model, check_data, "radial", device, args.batch_size)
            nesting_error = float(np.max(np.abs(linear_check - radial_check)))
            print(f"nested warm-start max abs diff: {nesting_error:.3e}")
            if nesting_error > 1e-5:
                raise RuntimeError(f"radial warm-start does not reproduce linear output: {nesting_error:.3e}")
        print(f"\nTraining {label}...")
        model, trace, info = train_model(
            model,
            method,
            train,
            val,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            print_every=args.print_every,
            seed=args.seed + 30_001,
            grad_clip=args.grad_clip,
            ema_decay=args.ema_decay,
            gate_only_steps=args.gate_only_steps if method == "radial" else 0,
            gate_lr=args.gate_lr,
            joint_rho_lr=args.joint_rho_lr,
            device=device,
            name=label,
        )
        info["method"] = method
        info["label"] = label
        if method == "radial":
            info["nested_warm_start_max_abs_diff"] = nesting_error
        training_info[method] = info
        torch.save(
            {
                "state_dict": model.state_dict(),
                "method": method,
                "label": label,
                "config": config,
                "training_info": info,
            },
            out_dir / f"model_{method}.pt",
        )
        if method == "linear":
            linear_model = model
        models[label] = (method, model)
        all_trace.extend(trace)
    write_csv(out_dir / "training_trace.csv", all_trace)
    (out_dir / "training_info.json").write_text(json.dumps(training_info, indent=2), encoding="utf-8")

    print("\nComputing fixed-grid FSM validation curve...")
    grid_rows: list[dict[str, Any]] = []
    grid_rng = np.random.default_rng(args.seed + 100_003)
    target_variance = 1.0 / (args.sigma_q**2)
    for pi in validation_pis:
        grid_raw = sample_fixed_anchor_batch(
            grid_rng,
            args.n_validation_per_anchor,
            pi,
            args.sigma_q,
            args.n_blocks,
            args.block_size,
            args.tau,
        )
        grid_anchor_z = ((grid_raw["anchor_u"] - anchor_mean) / anchor_sd).astype(np.float32)
        grid_data = prepare_split(
            grid_raw,
            args.tau,
            args.pi_ref,
            args.feature_pi_mode,
            stats,
            grid_anchor_z,
        )
        for label, (method, model) in models.items():
            pred_grid = predict(model, grid_data, method, device, args.batch_size)
            fsm_mse = float(np.mean((pred_grid - grid_data["target"]) ** 2))
            grid_rows.append(
                {
                    "method": label,
                    "pi": pi,
                    "n": args.n_validation_per_anchor,
                    "sigma_q": args.sigma_q,
                    "target_variance": target_variance,
                    "fsm_mse": fsm_mse,
                    "fsm_relative_mse": fsm_mse / target_variance,
                }
            )
    write_csv(out_dir / "fixed_grid_fsm_loss.csv", grid_rows)

    print("\nExact-score diagnostics at selected pi values...")
    diag_rows: list[dict[str, Any]] = []
    for pi_str in args.diagnostic_pi_values.split(","):
        pi = float(pi_str)
        eval_feat, truth = build_eval_data(
            rng,
            pi,
            args.n_test,
            args.n_blocks,
            args.block_size,
            args.tau,
            args.pi_ref,
            args.feature_pi_mode,
            stats,
            anchor_mean,
            anchor_sd,
        )
        for label, (method, model) in models.items():
            pred = predict(model, eval_feat, method, device, args.batch_size)
            row = metric_row(truth, pred)
            row.update({"method": label, "pi": pi, "n": args.n_test})
            diag_rows.append(row)
    write_csv(out_dir / "score_summary_by_pi.csv", diag_rows)

    print("\n==== Exact-score diagnostics (true score used only for evaluation) ====")
    for row in diag_rows:
        print(
            f"pi={row['pi']:.3g} {row['method']:<45} "
            f"std_mse={row['std_mse']:.4g} corr={row['corr']:.4g} cosine={row['cosine']:.4g}"
        )
    print("\nSaved:")
    print(out_dir / "config.json")
    print(out_dir / "feature_stats.npz")
    print(out_dir / "training_trace.csv")
    print(out_dir / "training_info.json")
    print(out_dir / "fixed_grid_fsm_loss.csv")
    print(out_dir / "score_summary_by_pi.csv")
    for method in methods:
        print(out_dir / f"model_{method}.pt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=40_000)
    parser.add_argument("--n-val", type=int, default=8_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--n-blocks", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--pi-ref", type=float, default=0.3)
    parser.add_argument(
        "--feature-pi-mode",
        type=str,
        default="anchor",
        choices=("anchor", "fixed"),
        help="anchor evaluates local subscores at beta_anchor; fixed uses pi_ref ablation.",
    )
    parser.add_argument("--anchor-pi-min", type=float, default=0.05)
    parser.add_argument("--anchor-pi-max", type=float, default=0.7)
    parser.add_argument("--sigma-q", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--methods", type=str, default="linear,radial")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument(
        "--gate-condition-on-anchor",
        type=int,
        default=1,
        help="1: radial gate is m(s, anchor_u); 0: radial gate is m(s) ablation.",
    )
    parser.add_argument("--iters", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4, help="Linear-model learning rate.")
    parser.add_argument("--gate-lr", type=float, default=1e-4)
    parser.add_argument("--joint-rho-lr", type=float, default=1e-4)
    parser.add_argument("--gate-only-steps", type=int, default=400)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument(
        "--validation-pi-values",
        type=str,
        default="0.07,0.10,0.20,0.30,0.40,0.50,0.60,0.65,0.68",
        help="Fixed anchor grid used only for post-training FSM validation curves.",
    )
    parser.add_argument("--n-validation-per-anchor", type=int, default=1_000)
    parser.add_argument("--diagnostic-pi-values", type=str, default="0.1,0.3,0.5,0.65")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/blockwise_common_factor_amortized_fsm",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
