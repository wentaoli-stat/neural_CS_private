"""Thin, experiment-specific adapter around Jiang's official implementation.

The score network, direct score-matching losses, Fisher penalty, debias
network, and debias training functions are imported from the authors' frozen
repository.  This module only supplies the HMM simulator data and orchestration
that their example scripts hard-code for the Queuing model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
from scipy.special import ndtr, ndtri
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dgp import HMMConfig, flatten_trajectories, simulate_trajectories
from .upstreams import UpstreamModules


TrajectorySimulator = Callable[
    [np.random.Generator, np.ndarray, HMMConfig, UpstreamModules], np.ndarray
]


@dataclass(frozen=True)
class GaussianProposal:
    mean: float
    sd: float

    def validate(self, bounds: tuple[float, float]) -> None:
        lower, upper = bounds
        if not lower <= self.mean <= upper:
            raise ValueError("proposal mean must lie inside the parameter bounds")
        if not np.isfinite(self.sd) or self.sd <= 0.0:
            raise ValueError("proposal sd must be positive")


@dataclass(frozen=True)
class JiangTrainingConfig:
    hidden_size: int = 64
    num_layers: int = 2
    training_size: int = 10_000
    batch_size: int = 10
    coord_epochs: int = 100
    joint_epochs: int = 100
    fisher_epochs: int = 50
    coord_lr: float = 1e-3
    joint_lr: float = 1e-3
    fisher_lr: float = 1e-4
    early_stop_patience: int = 10
    scheduler_patience: int = 3
    extra_sample_size: int = 1_000
    extra_obs_size: int = 500
    extra_batch_size: int = 10
    lambda_fisher: float = 1e-1
    debias_hidden_size: int = 64
    debias_num_layers: int = 2
    debias_epochs: int = 500
    debias_lr: float = 1e-3
    debias_patience: int = 30
    debias_fisher_epochs: int = 50
    debias_fisher_lr: float = 1e-5
    lambda_debias_curve: float = 1e-4

    @classmethod
    def smoke(cls) -> "JiangTrainingConfig":
        """Tiny end-to-end integrity profile; never a headline result."""
        return cls(
            training_size=256,
            batch_size=16,
            coord_epochs=2,
            joint_epochs=2,
            fisher_epochs=1,
            early_stop_patience=3,
            extra_sample_size=24,
            extra_obs_size=12,
            extra_batch_size=6,
            debias_epochs=3,
            debias_patience=3,
            debias_fisher_epochs=1,
        )

    def validate(self) -> None:
        integer_fields = (
            self.hidden_size,
            self.num_layers,
            self.training_size,
            self.batch_size,
            self.coord_epochs,
            self.joint_epochs,
            self.fisher_epochs,
            self.extra_sample_size,
            self.extra_obs_size,
            self.extra_batch_size,
            self.debias_hidden_size,
            self.debias_num_layers,
            self.debias_epochs,
            self.debias_fisher_epochs,
        )
        if any(int(value) < 1 for value in integer_fields):
            raise ValueError("all Jiang size/epoch fields must be positive")


class RawSequenceGRUScore(nn.Module):
    """Architecture-only Jiang adaptation for an ordered HMM trajectory.

    The training objective, repeated-reference Fisher penalty, debiasing,
    root solver, and covariance formulas remain Jiang's.  Only the raw score
    network changes from the authors' flattened ELU MLP to the same simple
    raw-sequence GRU shape used by the earlier sequence-level comparison.
    """

    def __init__(
        self,
        hmm: HMMConfig,
        hidden_size: int,
        y_mean: np.ndarray | torch.Tensor,
        y_sd: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.hmm = hmm
        self.x_dim = int(hmm.x_dim)
        self.obs_size = 1
        self.gru = nn.GRU(
            int(hmm.block_size) + 1,
            int(hidden_size),
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ELU(),
            nn.Linear(int(hidden_size), 1),
        )
        mean = torch.as_tensor(y_mean, dtype=torch.float32).reshape(
            1, 1, int(hmm.block_size)
        )
        sd = torch.as_tensor(y_sd, dtype=torch.float32).reshape(
            1, 1, int(hmm.block_size)
        )
        self.register_buffer("y_mean", mean)
        self.register_buffer("y_sd", torch.clamp(sd, min=1e-8))

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.x_dim:
            raise ValueError(
                f"raw GRU expects one flattened trajectory of width {self.x_dim}, "
                f"got {tuple(x.shape)}"
            )
        sequence = x.reshape(
            x.shape[0], int(self.hmm.length), int(self.hmm.block_size)
        )
        standardized = (sequence - self.y_mean) / self.y_sd
        theta_time = theta[:, None, :].expand(-1, int(self.hmm.length), -1)
        _, hidden = self.gru(torch.cat((standardized, theta_time), dim=2))
        return self.head(hidden[-1])

    def cal_penalty(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Match the official MLP repeated-observation interface."""
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        obs_size = int(x.shape[1] // self.x_dim)
        if obs_size < 1 or x.shape[1] != obs_size * self.x_dim:
            raise ValueError("reference table width is not a multiple of x_dim")
        repeated_theta = theta.repeat_interleave(obs_size, dim=0)
        score = self.forward(repeated_theta, x.reshape(-1, self.x_dim))
        return score.reshape(theta.shape[0], obs_size, 1)

    def subtract_output_bias(self, value: torch.Tensor) -> None:
        with torch.no_grad():
            self.head[-1].bias.sub_(value.reshape_as(self.head[-1].bias))


@dataclass
class JiangFitted:
    score_model: nn.Module
    debias_model: nn.Module
    scale_theta: float
    hmm: HMMConfig
    training: JiangTrainingConfig
    proposal: GaussianProposal | None
    round_id: int
    diagnostics: dict[str, Any]
    score_architecture: str = "official_mlp"


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value


def _sample_parameter(
    rng: np.random.Generator,
    n: int,
    bounds: tuple[float, float],
    proposal: GaussianProposal | None,
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = bounds
    if proposal is None:
        value = rng.uniform(lower, upper, size=int(n))
        proposal_score = np.zeros_like(value)
    else:
        proposal.validate(bounds)
        lo = ndtr((lower - proposal.mean) / proposal.sd)
        hi = ndtr((upper - proposal.mean) / proposal.sd)
        uniform = rng.uniform(size=int(n))
        probability = (1.0 - uniform) * lo + uniform * hi
        value = proposal.mean + proposal.sd * ndtri(probability)
        proposal_score = (proposal.mean - value) / proposal.sd**2
    value = np.clip(value, lower, upper)
    return value.astype(np.float32), proposal_score.astype(np.float32)


def _score_table(
    rng: np.random.Generator,
    n: int,
    hmm: HMMConfig,
    upstreams: UpstreamModules,
    proposal: GaussianProposal | None,
    simulator: TrajectorySimulator | None = None,
) -> dict[str, torch.Tensor]:
    p, proposal_score = _sample_parameter(
        rng, int(n), (hmm.p_min, hmm.p_max), proposal
    )
    y = (
        simulate_trajectories(rng, p, hmm, upstreams)
        if simulator is None
        else simulator(rng, p, hmm, upstreams)
    )
    return {
        "p": torch.as_tensor(p[:, None], dtype=torch.float32),
        "x": torch.as_tensor(flatten_trajectories(y, hmm), dtype=torch.float32),
        "proposal_score": torch.as_tensor(
            proposal_score[:, None], dtype=torch.float32
        ),
    }


def _repeated_table(
    rng: np.random.Generator,
    n_anchor: int,
    obs_per_anchor: int,
    hmm: HMMConfig,
    upstreams: UpstreamModules,
    proposal: GaussianProposal | None,
    simulator: TrajectorySimulator | None = None,
) -> dict[str, torch.Tensor]:
    p, _ = _sample_parameter(
        rng, int(n_anchor), (hmm.p_min, hmm.p_max), proposal
    )
    repeated = np.repeat(p, int(obs_per_anchor))
    y = (
        simulate_trajectories(rng, repeated, hmm, upstreams)
        if simulator is None
        else simulator(rng, repeated, hmm, upstreams)
    )
    flat = flatten_trajectories(y, hmm).reshape(
        int(n_anchor), int(obs_per_anchor) * hmm.x_dim
    )
    return {
        "p": torch.as_tensor(p[:, None], dtype=torch.float32),
        "x": torch.as_tensor(flat, dtype=torch.float32),
    }


def _scaled_weight_functions(
    lower: float,
    upper: float,
    scale_dist: torch.Tensor,
) -> tuple[
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
]:
    def distance(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lo = torch.full_like(theta, float(lower))
        hi = torch.full_like(theta, float(upper))
        return torch.minimum(theta - lo, hi - theta), lo, hi

    def g(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del x
        return distance(theta)[0] / scale_dist

    def g1(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del x
        _, lo, hi = distance(theta)
        return (2 * (theta < (lo + hi) / 2) - 1) / scale_dist

    def g_coord(theta: torch.Tensor, x: torch.Tensor, id_dim: int) -> torch.Tensor:
        return g(theta, x)[:, id_dim]

    def g1_coord(theta: torch.Tensor, x: torch.Tensor, id_dim: int) -> torch.Tensor:
        return g1(theta, x)[:, id_dim]

    return g, g1, g_coord, g1_coord


def _make_loaders(
    table: dict[str, torch.Tensor],
    training_size: int,
    batch_size: int,
    *,
    scale: float,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor]:
    theta = table["p"] * float(scale)
    proposal_score = table["proposal_score"] / float(scale)
    train_theta = theta[:training_size]
    val_theta = theta[training_size:]
    train = TensorDataset(
        train_theta,
        table["x"][:training_size],
        proposal_score[:training_size],
    )
    val = TensorDataset(
        val_theta,
        table["x"][training_size:],
        proposal_score[training_size:],
    )
    return (
        DataLoader(train, batch_size=int(batch_size), shuffle=True),
        DataLoader(val, batch_size=int(batch_size), shuffle=False),
        train_theta,
        table["x"][:training_size],
        val_theta,
    )


def _score_matching_validation(
    official: Any,
    model: nn.Module,
    loader: DataLoader,
    g: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g1: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for theta, x, proposal_score in loader:
        loss, _ = official.Like_score_loss(
            model,
            theta.to(official.device),
            x.to(official.device),
            proposal_score.to(official.device),
            g,
            g1,
        )
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def _train_fisher_phase(
    official: Any,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    extra_train_loader: DataLoader,
    extra_val_loader: DataLoader,
    g: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g1: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    config: JiangTrainingConfig,
) -> tuple[nn.Module, torch.Tensor, dict[str, Any]]:
    """Generic transcription of official Queuing/train_sm_fisher.py."""
    best_val_sm = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer: dict[str, Any] | None = None
    best_epoch = 0
    extra_cycle: Any
    trace: list[dict[str, float]] = []
    for epoch in range(int(config.fisher_epochs)):
        model.train()
        extra_cycle = iter(extra_train_loader)
        train_total = train_sm = train_penalty = 0.0
        valid = 0
        for theta, x, proposal_score in train_loader:
            try:
                theta_extra, x_extra = next(extra_cycle)
            except StopIteration:
                extra_cycle = iter(extra_train_loader)
                theta_extra, x_extra = next(extra_cycle)
            optimizer.zero_grad()
            sm_loss, _ = official.Like_score_loss(
                model,
                theta.to(official.device),
                x.to(official.device),
                proposal_score.to(official.device),
                g,
                g1,
            )
            penalty = official.weighted_Fisher_penalty(
                model,
                theta_extra.to(official.device),
                x_extra.to(official.device),
                g,
            )
            objective = sm_loss + float(config.lambda_fisher) * penalty
            if not torch.isfinite(objective):
                continue
            objective.backward()
            optimizer.step()
            valid += 1
            train_total += float(objective.detach().cpu())
            train_sm += float(sm_loss.detach().cpu())
            train_penalty += float(penalty.detach().cpu())

        model.eval()
        val_extra_cycle = iter(extra_val_loader)
        val_total = val_sm = val_penalty = 0.0
        val_valid = 0
        for theta, x, proposal_score in val_loader:
            try:
                theta_extra, x_extra = next(val_extra_cycle)
            except StopIteration:
                val_extra_cycle = iter(extra_val_loader)
                theta_extra, x_extra = next(val_extra_cycle)
            sm_loss, _ = official.Like_score_loss(
                model,
                theta.to(official.device),
                x.to(official.device),
                proposal_score.to(official.device),
                g,
                g1,
            )
            penalty = official.weighted_Fisher_penalty(
                model,
                theta_extra.to(official.device),
                x_extra.to(official.device),
                g,
            )
            objective = sm_loss + float(config.lambda_fisher) * penalty
            if not torch.isfinite(objective):
                continue
            val_valid += 1
            val_total += float(objective.detach().cpu())
            val_sm += float(sm_loss.detach().cpu())
            val_penalty += float(penalty.detach().cpu())
        avg_val_sm = val_sm / max(val_valid, 1)
        if avg_val_sm < best_val_sm:
            best_val_sm = avg_val_sm
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
        row = {
            "epoch": float(epoch + 1),
            "train_total": train_total / max(valid, 1),
            "train_sm": train_sm / max(valid, 1),
            "train_fisher": train_penalty / max(valid, 1),
            "val_total": val_total / max(val_valid, 1),
            "val_sm": avg_val_sm,
            "val_fisher": val_penalty / max(val_valid, 1),
        }
        trace.append(row)
        print("official Jiang Fisher", row)
    if best_state is not None:
        model.load_state_dict(best_state)
        if best_optimizer is not None:
            optimizer.load_state_dict(best_optimizer)

    total_bias: torch.Tensor | float = 0.0
    for theta, x, proposal_score in train_loader:
        _, bias = official.Like_score_loss(
            model,
            theta.to(official.device),
            x.to(official.device),
            proposal_score.to(official.device),
            g,
            g1,
        )
        total_bias = total_bias + bias.detach()
    bias_lastlayer = total_bias / len(train_loader)
    return model, bias_lastlayer, {
        "best_epoch": best_epoch,
        "best_val_sm": best_val_sm,
        "trace": trace,
    }


def _mean_score_labels(
    model: nn.Module,
    theta: torch.Tensor,
    x: torch.Tensor,
    obs_per_anchor: int,
    batch_anchors: int,
    device: str,
) -> torch.Tensor:
    output = torch.empty(theta.shape[0], 1, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, theta.shape[0], int(batch_anchors)):
            stop = min(start + int(batch_anchors), theta.shape[0])
            output[start:stop] = (
                model.cal_penalty(theta[start:stop].to(device), x[start:stop].to(device))
                .mean(dim=1)
                .cpu()
            )
    return output


def _check_debias_loss(
    official: Any,
    model: nn.Module,
    loader: DataLoader,
    lambda_curve: float,
) -> dict[str, float]:
    loss_fn = nn.MSELoss()
    total_reg = total_curve = total = 0.0
    count = 0
    model.eval()
    for theta, target, weight in loader:
        theta = theta.to(official.device)
        target = target.to(official.device)
        root_weight = weight.to(official.device).sqrt()
        weight_tensor = root_weight.unsqueeze(2) @ root_weight.unsqueeze(1)
        prediction = model(theta)
        pred_outer = prediction.unsqueeze(2) @ prediction.unsqueeze(1)
        jacobian = official.Deb_curve(model, theta)
        reg = loss_fn(prediction * root_weight, target * root_weight)
        curve = loss_fn(
            pred_outer * weight_tensor,
            (
                jacobian
                + torch.einsum("bi,bj->bij", target, prediction)
                + torch.einsum("bi,bj->bij", prediction, target)
            )
            * weight_tensor,
        )
        objective = reg + float(lambda_curve) * curve
        batch_n = theta.shape[0]
        total += float(objective.detach().cpu()) * batch_n
        total_reg += float(reg.detach().cpu()) * batch_n
        total_curve += float(curve.detach().cpu()) * batch_n
        count += batch_n
    return {
        "total": total / max(count, 1),
        "reg": total_reg / max(count, 1),
        "curve": total_curve / max(count, 1),
    }


def fit_jiang_round(
    *,
    upstreams: UpstreamModules,
    hmm: HMMConfig,
    config: JiangTrainingConfig,
    proposal: GaussianProposal | None,
    round_id: int,
    seed: int,
    device: str = "auto",
    score_architecture: str = "official_mlp",
    simulator: TrajectorySimulator | None = None,
) -> JiangFitted:
    """Fit one complete official Jiang proposal round."""
    hmm.validate()
    config.validate()
    if score_architecture not in {"official_mlp", "raw_sequence_gru"}:
        raise ValueError(f"unknown Jiang score architecture: {score_architecture}")
    if score_architecture == "raw_sequence_gru" and torch.cuda.is_available():
        # Jiang's objective backpropagates through a score Jacobian.  PyTorch's
        # cuDNN RNN path does not support the required double backward, whereas
        # the native GRU implementation does.
        torch.backends.cudnn.enabled = False
    device = resolve_device(device)
    set_seed(seed)
    official = upstreams.jiang_utils
    official.device = torch.device(device)
    rng = np.random.default_rng(int(seed))
    start_time = time.perf_counter()
    score_n = int(np.ceil(1.2 * config.training_size))
    extra_n = int(np.ceil(1.2 * config.extra_sample_size))
    score_table = _score_table(
        rng, score_n, hmm, upstreams, proposal, simulator=simulator
    )

    # Official coordinate-wise scale calibration.  There is one coordinate in
    # this benchmark, but the authors still run this before the joint model.
    coord_loader, coord_val_loader, coord_theta, _, _ = _make_loaders(
        score_table,
        config.training_size,
        config.batch_size,
        scale=1.0,
    )
    coord_scale_dist = torch.minimum(
        coord_theta - hmm.p_min, hmm.p_max - coord_theta
    ).mean().to(device)
    _, _, g_coord, g1_coord = _scaled_weight_functions(
        hmm.p_min, hmm.p_max, coord_scale_dist
    )
    coord_model = official.ELU_single_LikeScoreMatchingNN_1d(
        1, hmm.x_dim, 1, config.hidden_size, config.num_layers
    )
    coord_optimizer = torch.optim.Adam(
        coord_model.parameters(), lr=config.coord_lr, weight_decay=1e-5
    )
    coord_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        coord_optimizer,
        mode="min",
        factor=0.5,
        patience=config.scheduler_patience,
        min_lr=1e-5,
    )
    _, best_coord_loss = official.train_deb_1d(
        coord_model,
        0,
        coord_optimizer,
        coord_loader,
        coord_val_loader,
        g_coord,
        g1_coord,
        config.coord_epochs,
        coord_scheduler,
        config.early_stop_patience,
    )
    scale_theta = float(np.sqrt(abs(float(best_coord_loss))))
    if not np.isfinite(scale_theta) or scale_theta < 1e-5:
        raise RuntimeError(f"invalid official scale_theta={scale_theta}")

    train_loader, val_loader, train_theta, train_x, _ = _make_loaders(
        score_table,
        config.training_size,
        config.batch_size,
        scale=scale_theta,
    )
    lower_scaled = hmm.p_min * scale_theta
    upper_scaled = hmm.p_max * scale_theta
    scale_dist = torch.minimum(
        train_theta - lower_scaled, upper_scaled - train_theta
    ).mean().to(device)
    g, g1, _, _ = _scaled_weight_functions(
        lower_scaled, upper_scaled, scale_dist
    )
    if score_architecture == "official_mlp":
        score_model = official.ELU_single_LikeScoreMatchingNN(
            1, hmm.x_dim, 1, config.hidden_size, config.num_layers
        )
    else:
        sequence = score_table["x"][: config.training_size].reshape(
            -1, int(hmm.length), int(hmm.block_size)
        )
        y_mean = sequence.mean(dim=(0, 1))
        y_sd = torch.clamp(sequence.std(dim=(0, 1)), min=1e-8)
        score_model = RawSequenceGRUScore(
            hmm,
            config.hidden_size,
            y_mean,
            y_sd,
        )
    joint_optimizer = torch.optim.Adam(
        score_model.parameters(), lr=config.joint_lr, weight_decay=1e-5
    )
    joint_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        joint_optimizer,
        mode="min",
        factor=0.5,
        patience=config.scheduler_patience,
        min_lr=1e-5,
    )
    official.train_deb(
        score_model,
        joint_optimizer,
        train_loader,
        val_loader,
        g,
        g1,
        config.joint_epochs,
        joint_scheduler,
        config.early_stop_patience,
    )
    initial_val_sm = _score_matching_validation(
        official, score_model, val_loader, g, g1
    )

    repeated = _repeated_table(
        rng,
        extra_n,
        config.extra_obs_size,
        hmm,
        upstreams,
        proposal,
        simulator=simulator,
    )
    repeated_theta = repeated["p"] * scale_theta
    repeated_train = TensorDataset(
        repeated_theta[: config.extra_sample_size],
        repeated["x"][: config.extra_sample_size],
    )
    repeated_val = TensorDataset(
        repeated_theta[config.extra_sample_size :],
        repeated["x"][config.extra_sample_size :],
    )
    extra_train_loader = DataLoader(
        repeated_train, batch_size=config.extra_batch_size, shuffle=True
    )
    extra_val_loader = DataLoader(
        repeated_val, batch_size=config.extra_batch_size, shuffle=False
    )
    fisher_optimizer = torch.optim.Adam(
        score_model.parameters(), lr=config.fisher_lr, weight_decay=1e-5
    )
    score_model, fisher_bias, fisher_info = _train_fisher_phase(
        official,
        score_model.to(device),
        fisher_optimizer,
        train_loader,
        val_loader,
        extra_train_loader,
        extra_val_loader,
        g,
        g1,
        config,
    )
    fisher_val_sm = _score_matching_validation(
        official, score_model, val_loader, g, g1
    )
    # The authors select the best checkpoint *within* the Fisher-continuation
    # epochs, but do not compare it with and roll back to the pre-Fisher model.
    # Keep that repository behavior even if its validation SM value is worse
    # than the initial checkpoint.
    bias_lastlayer = fisher_bias
    fisher_info["official_reverted_to_initial"] = False
    fisher_info["pre_fisher_validation_sm"] = initial_val_sm
    if isinstance(score_model, RawSequenceGRUScore):
        score_model.subtract_output_bias(bias_lastlayer.to(device))
    else:
        with torch.no_grad():
            score_model.layers[-1].bias.sub_(bias_lastlayer.to(device))

    train_anchor_theta = repeated_theta[: config.extra_sample_size]
    val_anchor_theta = repeated_theta[config.extra_sample_size :]
    train_anchor_x = repeated["x"][: config.extra_sample_size]
    val_anchor_x = repeated["x"][config.extra_sample_size :]
    train_mean_score = _mean_score_labels(
        score_model,
        train_anchor_theta,
        train_anchor_x,
        config.extra_obs_size,
        config.extra_batch_size,
        device,
    )
    val_mean_score = _mean_score_labels(
        score_model,
        val_anchor_theta,
        val_anchor_x,
        config.extra_obs_size,
        config.extra_batch_size,
        device,
    )
    # The official DebReg script recomputes the normalization of the boundary
    # weight on ref_R instead of reusing the score-table normalization.
    debias_scale_dist = torch.minimum(
        train_anchor_theta - lower_scaled, upper_scaled - train_anchor_theta
    ).mean().to(device)
    g_debias, _, _, _ = _scaled_weight_functions(
        lower_scaled, upper_scaled, debias_scale_dist
    )
    # The authors' boundary weight depends only on theta in this experiment.
    # Avoid moving the multi-gigabyte repeated-x table to GPU merely to pass an
    # unused positional argument; the evaluated function is unchanged.
    train_weight = g_debias(
        train_anchor_theta.to(device), torch.empty_like(train_anchor_theta, device=device)
    ).cpu()
    val_weight = g_debias(
        val_anchor_theta.to(device), torch.empty_like(val_anchor_theta, device=device)
    ).cpu()
    debias_train_loader = DataLoader(
        TensorDataset(train_anchor_theta, train_mean_score, train_weight),
        batch_size=config.batch_size,
        shuffle=True,
    )
    debias_val_loader = DataLoader(
        TensorDataset(val_anchor_theta, val_mean_score, val_weight),
        batch_size=config.batch_size,
        shuffle=False,
    )
    debias_model = official.Deb_ELU(
        1, 1, config.debias_hidden_size, config.debias_num_layers
    ).to(device)
    debias_optimizer = torch.optim.Adam(
        debias_model.parameters(), lr=config.debias_lr
    )
    debias_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        debias_optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )
    official.train_weighted_DebReg(
        debias_model,
        debias_optimizer,
        debias_train_loader,
        debias_val_loader,
        config.debias_epochs,
        debias_scheduler,
        config.debias_patience,
    )
    debias_before = _check_debias_loss(
        official, debias_model, debias_val_loader, config.lambda_debias_curve
    )
    debias_fisher_optimizer = torch.optim.Adam(
        debias_model.parameters(), lr=config.debias_fisher_lr
    )
    official.train_weighted_DebReg_fisher_crossterm(
        debias_model,
        debias_fisher_optimizer,
        debias_train_loader,
        debias_val_loader,
        config.debias_fisher_epochs,
        config.lambda_debias_curve,
        debias_before["reg"],
    )
    debias_after = _check_debias_loss(
        official, debias_model, debias_val_loader, config.lambda_debias_curve
    )
    diagnostics = {
        "round_id": int(round_id),
        "score_architecture": score_architecture,
        "seed": int(seed),
        "scale_theta": scale_theta,
        "best_coordinate_validation_loss": float(best_coord_loss),
        "initial_validation_sm": initial_val_sm,
        "fisher_validation_sm": fisher_val_sm,
        "fisher": fisher_info,
        "debias_validation_before_curve": debias_before,
        "debias_validation_after_curve": debias_after,
        "train_mean_score_abs": float(train_mean_score.abs().mean()),
        "val_mean_score_abs": float(val_mean_score.abs().mean()),
        "wall_clock_seconds": time.perf_counter() - start_time,
        "official_jiang_utils": str(Path(str(official.__file__)).resolve()),
    }
    return JiangFitted(
        score_model=score_model,
        debias_model=debias_model,
        scale_theta=scale_theta,
        hmm=hmm,
        training=config,
        proposal=proposal,
        round_id=int(round_id),
        diagnostics=diagnostics,
        score_architecture=score_architecture,
    )


def save_fitted(path: Path, fitted: JiangFitted) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "score_state_dict": fitted.score_model.state_dict(),
            "debias_state_dict": fitted.debias_model.state_dict(),
            "scale_theta": fitted.scale_theta,
            "hmm": fitted.hmm.to_dict(),
            "training": asdict(fitted.training),
            "proposal": asdict(fitted.proposal) if fitted.proposal else None,
            "round_id": fitted.round_id,
            "diagnostics": fitted.diagnostics,
            "score_architecture": fitted.score_architecture,
            "method": (
                "official_Jiang_ELU_MLP_additive_score"
                if fitted.score_architecture == "official_mlp"
                else "Jiang_official_objectives_raw_sequence_GRU_additive_score"
            ),
        },
        path,
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "scale_theta": fitted.scale_theta,
                "hmm": fitted.hmm.to_dict(),
                "training": asdict(fitted.training),
                "proposal": asdict(fitted.proposal) if fitted.proposal else None,
                "round_id": fitted.round_id,
                "diagnostics": fitted.diagnostics,
                "score_architecture": fitted.score_architecture,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_fitted(
    path: Path,
    upstreams: UpstreamModules,
    *,
    device: str = "auto",
) -> JiangFitted:
    device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    hmm = HMMConfig(**checkpoint["hmm"])
    training = JiangTrainingConfig(**checkpoint["training"])
    proposal = (
        GaussianProposal(**checkpoint["proposal"])
        if checkpoint["proposal"] is not None
        else None
    )
    official = upstreams.jiang_utils
    official.device = torch.device(device)
    score_architecture = checkpoint.get("score_architecture", "official_mlp")
    if score_architecture == "official_mlp":
        score_model = official.ELU_single_LikeScoreMatchingNN(
            1, hmm.x_dim, 1, training.hidden_size, training.num_layers
        ).to(device)
    elif score_architecture == "raw_sequence_gru":
        score_model = RawSequenceGRUScore(
            hmm,
            training.hidden_size,
            np.zeros(hmm.block_size, dtype=np.float32),
            np.ones(hmm.block_size, dtype=np.float32),
        ).to(device)
    else:
        raise ValueError(f"unknown checkpoint score architecture: {score_architecture}")
    score_model.load_state_dict(checkpoint["score_state_dict"])
    debias_model = official.Deb_ELU(
        1, 1, training.debias_hidden_size, training.debias_num_layers
    ).to(device)
    debias_model.load_state_dict(checkpoint["debias_state_dict"])
    return JiangFitted(
        score_model=score_model,
        debias_model=debias_model,
        scale_theta=float(checkpoint["scale_theta"]),
        hmm=hmm,
        training=training,
        proposal=proposal,
        round_id=int(checkpoint["round_id"]),
        diagnostics=checkpoint["diagnostics"],
        score_architecture=score_architecture,
    )


def _as_flat_x(y: np.ndarray, hmm: HMMConfig) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 3:
        return flatten_trajectories(y, hmm)
    if y.ndim == 2 and y.shape[1] == hmm.x_dim:
        return np.ascontiguousarray(y)
    raise ValueError(
        f"expected trajectories (n,{hmm.length},{hmm.block_size}) or "
        f"flattened (n,{hmm.x_dim}), got {y.shape}"
    )


def debiased_scores_tensor(
    fitted: JiangFitted,
    p: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Official debiased single-observation scores in the scaled coordinate."""
    if p.ndim == 0:
        p = p.reshape(1, 1)
    elif p.ndim == 1:
        p = p.reshape(-1, 1)
    if p.shape[0] == 1 and x.shape[0] != 1:
        p = p.expand(x.shape[0], 1)
    if p.shape[0] != x.shape[0]:
        raise ValueError("p and x batch sizes do not agree")
    scaled = p * float(fitted.scale_theta)
    return fitted.score_model(scaled, x) - fitted.debias_model(scaled)


@torch.no_grad()
def debiased_scores(
    fitted: JiangFitted,
    p: float,
    y: np.ndarray,
    *,
    device: str = "auto",
    batch_size: int = 512,
    physical_coordinate: bool = False,
) -> np.ndarray:
    """Evaluate all single-trajectory scores.

    The official network is trained in v=scale_theta*p.  Its native output is
    the v-score.  Multiplication by scale_theta returns the physical p-score.
    """
    device = resolve_device(device)
    flat = _as_flat_x(y, fitted.hmm)
    output: list[np.ndarray] = []
    fitted.score_model.eval()
    fitted.debias_model.eval()
    for start in range(0, flat.shape[0], int(batch_size)):
        stop = min(start + int(batch_size), flat.shape[0])
        x = torch.as_tensor(flat[start:stop], dtype=torch.float32, device=device)
        p_tensor = torch.full((stop - start, 1), float(p), device=device)
        value = debiased_scores_tensor(fitted, p_tensor, x)
        if physical_coordinate:
            value = value * float(fitted.scale_theta)
        output.append(value.cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float64)


def solve_score_root(
    fitted: JiangFitted,
    y: np.ndarray,
    *,
    device: str = "auto",
    initial_p: float | None = None,
    weights: np.ndarray | None = None,
    seed: int = 0,
    maxiter: int = 10_000,
    score_tol: float = 1e-3,
    step_tol: float = 1e-6,
    max_initializations: int = 10,
) -> dict[str, Any]:
    """Repository-faithful projected score-ascent root finder."""
    device = resolve_device(device)
    flat = _as_flat_x(y, fitted.hmm)
    n = flat.shape[0]
    if weights is None:
        weight = torch.ones(n, 1, dtype=torch.float32, device=device)
    else:
        values = np.asarray(weights, dtype=np.float32).reshape(-1)
        if values.size != n:
            raise ValueError("bootstrap weight count does not equal n")
        weight = torch.as_tensor(values[:, None], device=device)
    x = torch.as_tensor(flat, dtype=torch.float32, device=device)
    rng = np.random.default_rng(int(seed))
    lower_v = fitted.hmm.p_min * fitted.scale_theta
    upper_v = fitted.hmm.p_max * fitted.scale_theta
    step_size = 0.1 / n
    last: dict[str, Any] | None = None
    attempts = 1 if initial_p is not None else int(max_initializations)
    fitted.score_model.eval()
    fitted.debias_model.eval()
    with torch.no_grad():
        for attempt in range(1, attempts + 1):
            start_p = (
                float(initial_p)
                if initial_p is not None
                else float(rng.uniform(fitted.hmm.p_min, fitted.hmm.p_max))
            )
            v = torch.tensor([[start_p * fitted.scale_theta]], device=device)
            converged_step = False
            iteration = 0
            for iteration in range(int(maxiter)):
                p_tensor = (v / fitted.scale_theta).expand(n, 1)
                per_observation = debiased_scores_tensor(fitted, p_tensor, x)
                gradient = torch.sum(per_observation * weight).reshape(1, 1)
                updated = torch.clamp(
                    v + step_size * gradient,
                    min=float(lower_v),
                    max=float(upper_v),
                )
                if torch.linalg.vector_norm(updated - v) < float(step_tol):
                    v = updated
                    converged_step = True
                    break
                v = updated
            p_tensor = (v / fitted.scale_theta).expand(n, 1)
            score = torch.sum(
                debiased_scores_tensor(fitted, p_tensor, x) * weight
            )
            normalized = float((score.abs() / n).cpu())
            last = {
                "p_hat": float((v / fitted.scale_theta).cpu().item()),
                "scaled_score_sum": float(score.cpu()),
                "normalized_score_abs": normalized,
                "iterations": int(iteration),
                "attempts": int(attempt),
                "step_converged": bool(converged_step),
                "score_converged": bool(normalized < score_tol),
                "at_lower_bound": bool(abs(float(v.cpu()) - lower_v) < 1e-7),
                "at_upper_bound": bool(abs(float(v.cpu()) - upper_v) < 1e-7),
            }
            if normalized < float(score_tol):
                return last
    if last is None:
        raise RuntimeError("root solver made no attempts")
    return last


def covariance_estimates(
    fitted: JiangFitted,
    y: np.ndarray,
    p_hat: float,
    *,
    device: str = "auto",
) -> dict[str, float]:
    """Official Eq. (6), Eq. (5), and sandwich limit covariances in p."""
    device = resolve_device(device)
    flat = _as_flat_x(y, fitted.hmm)
    x = torch.as_tensor(flat, dtype=torch.float32, device=device)
    v = torch.tensor(
        [[float(p_hat) * fitted.scale_theta]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    p = (v / fitted.scale_theta).expand(x.shape[0], 1)
    score = debiased_scores_tensor(fitted, p, x)
    i_ss = torch.mean(score.square())
    mean_score = torch.mean(score)
    derivative = torch.autograd.grad(mean_score, v, create_graph=False)[0].reshape(())
    i_curv = -derivative
    scale_sq = float(fitted.scale_theta) ** 2

    def safe(value: torch.Tensor) -> float:
        result = float(value.detach().cpu())
        return result if np.isfinite(result) else float("nan")

    i_ss_value = safe(i_ss)
    i_curv_value = safe(i_curv)
    cov_ss = 1.0 / i_ss_value / scale_sq if i_ss_value > 0.0 else float("nan")
    cov_curv = (
        1.0 / i_curv_value / scale_sq if i_curv_value > 0.0 else float("nan")
    )
    cov_sand = (
        i_ss_value / (i_curv_value**2) / scale_sq
        if i_ss_value >= 0.0 and i_curv_value != 0.0
        else float("nan")
    )
    return {
        "I_ss_scaled": i_ss_value,
        "I_curv_scaled": i_curv_value,
        "Cov_ss": cov_ss,
        "Cov_curv": cov_curv,
        "Cov_sand": cov_sand,
        "mean_scaled_score": safe(mean_score),
    }


def normal_confidence_intervals(
    p_hat: float,
    covariance: dict[str, float],
    n: int,
    *,
    z_value: float = 1.959963984540054,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for key in ("Cov_ss", "Cov_curv", "Cov_sand"):
        value = float(covariance[key])
        half = z_value * np.sqrt(value / int(n)) if value >= 0.0 else float("nan")
        output[key] = {
            "lower": float(p_hat - half),
            "upper": float(p_hat + half),
            "width": float(2.0 * half),
        }
    return output


def multiplier_bootstrap(
    fitted: JiangFitted,
    y: np.ndarray,
    p_hat: float,
    *,
    num_bootstrap: int = 2_000,
    device: str = "auto",
    initial_p: float | None = None,
    seed: int = 0,
    maxiter: int = 10_000,
) -> dict[str, Any]:
    """Official Exp(1) weighted roots and repository basic interval."""
    flat = _as_flat_x(y, fitted.hmm)
    rng = np.random.default_rng(int(seed))
    roots = np.empty(int(num_bootstrap), dtype=np.float64)
    converged = np.zeros(int(num_bootstrap), dtype=bool)
    for index in range(int(num_bootstrap)):
        weights = rng.exponential(scale=1.0, size=flat.shape[0])
        result = solve_score_root(
            fitted,
            flat,
            device=device,
            initial_p=initial_p,
            weights=weights,
            seed=seed + index + 1,
            maxiter=maxiter,
        )
        roots[index] = result["p_hat"]
        converged[index] = result["score_converged"]
    centered = roots - float(p_hat)
    q025, q975 = np.quantile(centered, [0.025, 0.975])
    lower = float(p_hat - q975)
    upper = float(p_hat - q025)
    return {
        "method": "bootstrap_repo_basic",
        "lower": lower,
        "upper": upper,
        "width": upper - lower,
        "q025_centered": float(q025),
        "q975_centered": float(q975),
        "convergence_rate": float(np.mean(converged)),
        "roots": roots,
    }


def infer_dataset(
    fitted: JiangFitted,
    y: np.ndarray,
    *,
    previous_round_p: float | None = None,
    num_bootstrap: int = 0,
    seed: int = 0,
    device: str = "auto",
    maxiter: int = 10_000,
) -> dict[str, Any]:
    root = solve_score_root(
        fitted,
        y,
        device=device,
        initial_p=previous_round_p,
        seed=seed,
        maxiter=maxiter,
    )
    p_hat = float(root["p_hat"])
    covariance = covariance_estimates(fitted, y, p_hat, device=device)
    normal = normal_confidence_intervals(p_hat, covariance, _as_flat_x(y, fitted.hmm).shape[0])
    output: dict[str, Any] = {
        "root": root,
        "covariance": covariance,
        "normal_intervals": normal,
    }
    if int(num_bootstrap) > 0:
        bootstrap = multiplier_bootstrap(
            fitted,
            y,
            p_hat,
            num_bootstrap=num_bootstrap,
            device=device,
            initial_p=previous_round_p,
            seed=seed + 100_000,
            maxiter=maxiter,
        )
        output["bootstrap"] = {
            key: value for key, value in bootstrap.items() if key != "roots"
        }
        output["bootstrap_roots"] = bootstrap["roots"]
    return output


def score_diagnostic(
    fitted: JiangFitted,
    upstreams: UpstreamModules,
    p_values: list[float],
    *,
    n_per_value: int = 500,
    seed: int = 0,
    device: str = "auto",
) -> list[dict[str, float]]:
    from .dgp import exact_score_p

    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, float]] = []
    for p in p_values:
        y = simulate_trajectories(rng, p, fitted.hmm, upstreams, n=n_per_value)
        prediction = debiased_scores(
            fitted, p, y, device=device, physical_coordinate=True
        )
        truth = exact_score_p(y, p, fitted.hmm, upstreams)
        truth_sd = max(float(truth.std()), 1e-12)
        error = prediction - truth
        rows.append(
            {
                "p": float(p),
                "n": float(n_per_value),
                "mse": float(np.mean(error**2)),
                "std_mse": float(np.mean((error / truth_sd) ** 2)),
                "corr": float(np.corrcoef(prediction[:, 0], truth[:, 0])[0, 1]),
                "prediction_mean": float(prediction.mean()),
                "truth_mean": float(truth.mean()),
                "truth_sd": truth_sd,
            }
        )
    return rows
