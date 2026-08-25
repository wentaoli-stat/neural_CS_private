#!/usr/bin/env python3
"""Paired Jiang-loss ablation for Model 2 common-factor CS features.

Four arms share the same D^S/D^R cache and paired test draws:

* ``linear_cs_plain_base``: direct score matching only;
* ``structured_cs_plain``: strict nested gate, direct score matching only;
* ``linear_cs``: direct SM + Fisher penalty + both debias stages;
* ``structured_cs``: strict nested gate + the full Jiang recipe.

The representation is identical across the plain/full comparison.  It uses
separate marginal and same-block pairwise subscores, frozen local
standardization, within-block means, frozen two-channel block
standardization, and an anchor-conditioned rho readout.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import model2_jiang_direct_sm_core as core
import run_blockwise_common_factor_amortized_fsm_experiment as stage1


METHOD_LABELS = {
    "linear_cs_plain_base": "M2 J-linear-CS-plain-base",
    "structured_cs_plain": "M2 J-structured-CS-plain",
    "linear_cs": "M2 J-linear-CS-full",
    "structured_cs": "M2 J-structured-CS-full",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def stratified_pi(
    rng: np.random.Generator,
    n: int,
    pi_min: float,
    pi_max: float,
) -> np.ndarray:
    grid = (np.arange(n, dtype=np.float64) + rng.uniform(size=n)) / max(n, 1)
    rng.shuffle(grid)
    return pi_min + (pi_max - pi_min) * grid


def sample_single_blocks(
    rng: np.random.Generator,
    n: int,
    pi_min: float,
    pi_max: float,
    block_size: int,
    tau: float,
) -> dict[str, np.ndarray]:
    pi = stratified_pi(rng, n, pi_min, pi_max)
    u = stage1.logit_np(pi)
    y = stage1.simulate_common_factor(rng, u, 1, block_size, tau)[:, 0]
    return {
        "u": u.astype(np.float32),
        "pi": pi.astype(np.float32),
        "y": y.astype(np.float32),
        "prop_score": (1.0 - 2.0 * pi).astype(np.float32),
    }


def sample_reference_table(
    rng: np.random.Generator,
    n_anchor: int,
    obs_per_anchor: int,
    pi_min: float,
    pi_max: float,
    block_size: int,
    tau: float,
) -> dict[str, np.ndarray]:
    pi = stratified_pi(rng, n_anchor, pi_min, pi_max)
    u = stage1.logit_np(pi)
    repeated_u = np.repeat(u, obs_per_anchor)
    y = stage1.simulate_common_factor(
        rng, repeated_u, 1, block_size, tau
    )[:, 0]
    return {
        "u": u.astype(np.float32),
        "pi": pi.astype(np.float32),
        "y": y.reshape(n_anchor, obs_per_anchor, block_size).astype(np.float32),
    }


def make_or_load_shared_tables(
    cache_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    metadata_path = cache_path.with_suffix(".json")
    expected = {
        "schema": "model2_jiang_shared_v1",
        "seed": int(args.seed),
        "n_s_train": int(args.n_s_train),
        "n_s_val": int(args.n_s_val),
        "n_r_train": int(args.n_r_train),
        "n_r_val": int(args.n_r_val),
        "m_r": int(args.m_r),
        "pi_min": float(args.pi_min),
        "pi_max": float(args.pi_max),
        "block_size": int(args.block_size),
        "tau": float(args.tau),
    }
    if cache_path.exists() or metadata_path.exists():
        if not cache_path.exists() or not metadata_path.exists():
            raise RuntimeError("Shared cache is only partially present")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise RuntimeError(
                    f"Cache mismatch for {key}: expected {value!r}, "
                    f"found {metadata.get(key)!r}"
                )
        with np.load(cache_path) as loaded:
            tables = {
                split: {
                    key: np.array(loaded[f"{split}__{key}"], copy=True)
                    for key in metadata["table_keys"][split]
                }
                for split in ("ds_train", "ds_val", "dr_train", "dr_val")
            }
        print("Loaded shared D^S/D^R cache:", cache_path)
        return tables, metadata

    rng = np.random.default_rng(args.seed)
    tables = {
        "ds_train": sample_single_blocks(
            rng,
            args.n_s_train,
            args.pi_min,
            args.pi_max,
            args.block_size,
            args.tau,
        ),
        "ds_val": sample_single_blocks(
            rng,
            args.n_s_val,
            args.pi_min,
            args.pi_max,
            args.block_size,
            args.tau,
        ),
        "dr_train": sample_reference_table(
            rng,
            args.n_r_train,
            args.m_r,
            args.pi_min,
            args.pi_max,
            args.block_size,
            args.tau,
        ),
        "dr_val": sample_reference_table(
            rng,
            args.n_r_val,
            args.m_r,
            args.pi_min,
            args.pi_max,
            args.block_size,
            args.tau,
        ),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        f"{split}__{key}": value
        for split, table in tables.items()
        for key, value in table.items()
    }
    np.savez(cache_path, **flat)
    metadata = {
        **expected,
        "path": str(cache_path),
        "table_keys": {split: sorted(table) for split, table in tables.items()},
        "sha256_arrays": array_digest(flat),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved shared D^S/D^R cache:", cache_path)
    return tables, metadata


def _sum_sumsq(values: np.ndarray) -> tuple[int, float, float]:
    x = np.asarray(values, dtype=np.float64)
    return x.size, float(x.sum(dtype=np.float64)), float(np.square(x).sum(dtype=np.float64))


def _mean_sd(count: int, total: float, total_sq: float) -> tuple[float, float]:
    mean = total / max(count, 1)
    variance = max(total_sq / max(count, 1) - mean * mean, 0.0)
    return mean, max(math.sqrt(variance), 1e-8)


def make_cs_stats_streaming(
    y: np.ndarray,
    u: np.ndarray,
    tau: float,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Fit both frozen standardization layers without materializing all pairs."""

    pi = stage1.sigmoid_np(np.asarray(u, dtype=np.float64))
    counts = [0, 0]
    sums = [0.0, 0.0]
    sums_sq = [0.0, 0.0]
    for start in range(0, y.shape[0], chunk_size):
        stop = min(start + chunk_size, y.shape[0])
        block = np.asarray(y[start:stop], dtype=np.float64)[:, None, :]
        p = pi[start:stop]
        s1 = stage1.score_u_from_log_ratio(stage1.marginal_log_ratio(block, tau), p)
        s2 = stage1.score_u_from_log_ratio(stage1.pairwise_log_ratio(block, tau), p)
        for index, values in enumerate((s1, s2)):
            count, total, total_sq = _sum_sumsq(values)
            counts[index] += count
            sums[index] += total
            sums_sq[index] += total_sq
    s1_mean, s1_sd = _mean_sd(counts[0], sums[0], sums_sq[0])
    s2_mean, s2_sd = _mean_sd(counts[1], sums[1], sums_sq[1])

    block_count = 0
    block_sum = np.zeros(2, dtype=np.float64)
    block_sum_sq = np.zeros(2, dtype=np.float64)
    for start in range(0, y.shape[0], chunk_size):
        stop = min(start + chunk_size, y.shape[0])
        block = np.asarray(y[start:stop], dtype=np.float64)[:, None, :]
        p = pi[start:stop]
        s1 = stage1.score_u_from_log_ratio(stage1.marginal_log_ratio(block, tau), p)
        s2 = stage1.score_u_from_log_ratio(stage1.pairwise_log_ratio(block, tau), p)
        block_raw = np.stack(
            [
                ((s1 - s1_mean) / s1_sd).mean(axis=(1, 2)),
                ((s2 - s2_mean) / s2_sd).mean(axis=(1, 2)),
            ],
            axis=1,
        )
        block_count += block_raw.shape[0]
        block_sum += block_raw.sum(axis=0, dtype=np.float64)
        block_sum_sq += np.square(block_raw).sum(axis=0, dtype=np.float64)
    block_mean = block_sum / max(block_count, 1)
    block_var = np.maximum(
        block_sum_sq / max(block_count, 1) - np.square(block_mean), 0.0
    )
    block_sd = np.maximum(np.sqrt(block_var), 1e-8)
    anchor_mean = float(np.mean(u, dtype=np.float64))
    anchor_sd = max(float(np.std(u, dtype=np.float64)), 1e-8)
    return {
        "s1_mean": np.array(s1_mean, dtype=np.float64),
        "s1_sd": np.array(s1_sd, dtype=np.float64),
        "s2_mean": np.array(s2_mean, dtype=np.float64),
        "s2_sd": np.array(s2_sd, dtype=np.float64),
        "block_mean": block_mean.astype(np.float64),
        "block_sd": block_sd.astype(np.float64),
        "anchor_u_mean": np.array(anchor_mean, dtype=np.float64),
        "anchor_u_sd": np.array(anchor_sd, dtype=np.float64),
    }


class CSJiangCommonFactorBlockScore(nn.Module):
    """Differentiable Model-2 marginal/pairwise CS representation."""

    def __init__(
        self,
        tau: float,
        block_size: int,
        stats: dict[str, np.ndarray],
        hidden: int,
        depth: int,
        *,
        structured: bool,
        gate_hidden: int,
    ):
        super().__init__()
        self.tau = float(tau)
        self.block_size = int(block_size)
        self.structured = bool(structured)
        pair_i, pair_j = np.triu_indices(self.block_size, k=1)
        self.register_buffer("pair_i", torch.as_tensor(pair_i, dtype=torch.long))
        self.register_buffer("pair_j", torch.as_tensor(pair_j, dtype=torch.long))
        for name in (
            "s1_mean",
            "s1_sd",
            "s2_mean",
            "s2_sd",
            "block_mean",
            "block_sd",
            "anchor_u_mean",
            "anchor_u_sd",
        ):
            self.register_buffer(name, torch.as_tensor(stats[name], dtype=torch.float32))
        self.rho = stage1.make_rho(3, hidden, depth)
        self.marginal_gate = (
            stage1.PositiveMLPMultiplier(gate_hidden, input_dim=2)
            if structured
            else None
        )
        self.pairwise_gate = (
            stage1.PositiveMLPMultiplier(gate_hidden, input_dim=2)
            if structured
            else None
        )

    def local_standardized_subscores(
        self,
        physical_u: torch.Tensor,
        y_block: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = physical_u.reshape(-1)
        pi = torch.sigmoid(u)
        tau_sq = self.tau**2
        marginal_log_ratio = (
            -0.5 * math.log1p(tau_sq)
            + tau_sq / (2.0 * (1.0 + tau_sq)) * y_block.square()
        )
        pair_sum = y_block[:, self.pair_i] + y_block[:, self.pair_j]
        pairwise_log_ratio = (
            -0.5 * math.log1p(2.0 * tau_sq)
            + tau_sq / (2.0 * (1.0 + 2.0 * tau_sq)) * pair_sum.square()
        )
        s1 = torch.sigmoid(u[:, None] + marginal_log_ratio) - pi[:, None]
        s2 = torch.sigmoid(u[:, None] + pairwise_log_ratio) - pi[:, None]
        s1 = (s1 - self.s1_mean) / self.s1_sd
        s2 = (s2 - self.s2_mean) / self.s2_sd
        anchor_z = (u - self.anchor_u_mean) / self.anchor_u_sd
        return s1, s2, anchor_z

    def forward(self, physical_u: torch.Tensor, y_block: torch.Tensor) -> torch.Tensor:
        s1, s2, anchor_z = self.local_standardized_subscores(physical_u, y_block)
        if self.structured:
            if self.marginal_gate is None or self.pairwise_gate is None:
                raise RuntimeError("Structured model is missing its gates")
            s1 = s1 * self.marginal_gate(s1, anchor_z[:, None].expand_as(s1))
            s2 = s2 * self.pairwise_gate(s2, anchor_z[:, None].expand_as(s2))
        block_raw = torch.stack([s1.mean(dim=1), s2.mean(dim=1)], dim=1)
        block = (block_raw - self.block_mean) / self.block_sd
        return self.rho(torch.cat([block, anchor_z[:, None]], dim=1)).squeeze(-1)

    def copy_linear_readout_(self, linear: "CSJiangCommonFactorBlockScore") -> None:
        if not self.structured:
            raise RuntimeError("copy_linear_readout_ requires a structured model")
        self.rho.load_state_dict(copy.deepcopy(linear.rho.state_dict()))

    def output_layer(self) -> nn.Linear:
        layer = self.rho[-1]
        if not isinstance(layer, nn.Linear):
            raise TypeError("Expected a linear rho output layer")
        return layer


def check_proposal_score(pi: np.ndarray, prop_score: np.ndarray) -> float:
    expected = 1.0 - 2.0 * np.asarray(pi, dtype=np.float64)
    return float(np.max(np.abs(expected - np.asarray(prop_score, dtype=np.float64))))


def structural_sanity_checks(
    linear: CSJiangCommonFactorBlockScore,
    structured: CSJiangCommonFactorBlockScore,
    data: dict[str, np.ndarray],
    *,
    device: str,
    u_min: float,
    u_max: float,
    scale_dist: float,
) -> dict[str, Any]:
    linear.eval()
    structured.eval()
    n = min(24, data["u"].shape[0])
    u = torch.as_tensor(data["u"][:n], device=device)
    y = torch.as_tensor(data["y"][:n], device=device)
    prop = torch.as_tensor(data["prop_score"][:n], device=device)
    with torch.no_grad():
        linear_pred = linear(u, y)
        structured_pred = structured(u, y)
        nesting_error = float(torch.max(torch.abs(linear_pred - structured_pred)).cpu())
        permutation = torch.arange(y.shape[1] - 1, -1, -1, device=device)
        permutation_error = float(
            torch.max(torch.abs(structured(u, y[:, permutation]) - structured_pred)).cpu()
        )
    u_grad = u.detach().clone().requires_grad_(True)
    derivative = torch.autograd.grad(
        structured(u_grad, y).sum(), u_grad, create_graph=False
    )[0]
    h = 1e-3
    with torch.no_grad():
        finite_difference = (structured(u + h, y) - structured(u - h, y)) / (2.0 * h)
    derivative_error = float(torch.max(torch.abs(derivative - finite_difference)).cpu())
    derivative_scale = float(torch.max(torch.abs(finite_difference)).cpu())
    loss, _ = core.score_matching_loss(
        structured,
        u,
        y,
        prop,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
        create_graph=True,
    )
    structured.zero_grad(set_to_none=True)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in structured.parameters()
    )
    result = {
        "strict_nesting_max_abs": nesting_error,
        "permutation_max_abs": permutation_error,
        "autodiff_fd_max_abs": derivative_error,
        "autodiff_fd_relative": derivative_error / max(derivative_scale, 1e-8),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": gradients_finite,
    }
    result["passed"] = bool(
        nesting_error < 2e-6
        and permutation_error < 2e-6
        and result["autodiff_fd_relative"] < 2e-2
        and result["loss_finite"]
        and gradients_finite
    )
    structured.zero_grad(set_to_none=True)
    if not result["passed"]:
        raise RuntimeError(f"Structural sanity checks failed: {result}")
    return result


@torch.no_grad()
def subtract_global_bias(
    score_net: CSJiangCommonFactorBlockScore,
    data: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, data["u"].shape[0], batch_size):
        stop = min(start + batch_size, data["u"].shape[0])
        score = score_net(
            torch.as_tensor(data["u"][start:stop], device=device),
            torch.as_tensor(data["y"][start:stop], device=device),
        )
        total += float(score.sum().cpu())
        count += int(score.numel())
    bias = total / max(count, 1)
    layer = score_net.output_layer()
    layer.bias.sub_(
        torch.as_tensor(bias, dtype=layer.bias.dtype, device=layer.bias.device)
    )
    return bias


class PlainDatasetScoreWrapper(nn.Module):
    def __init__(self, score_net: nn.Module):
        super().__init__()
        self.score_net = score_net

    def forward(self, y: torch.Tensor, physical_u: torch.Tensor) -> torch.Tensor:
        n, n_blocks, block_size = y.shape
        repeated_u = physical_u.reshape(n, 1).expand(-1, n_blocks).reshape(-1)
        scores = self.score_net(
            repeated_u, y.reshape(-1, block_size)
        ).reshape(n, n_blocks)
        return scores.sum(dim=1)


class FullDatasetScoreWrapper(nn.Module):
    def __init__(self, score_net: nn.Module, debias_net: nn.Module):
        super().__init__()
        self.score_net = score_net
        self.debias_net = debias_net

    def forward(self, y: torch.Tensor, physical_u: torch.Tensor) -> torch.Tensor:
        n, n_blocks, block_size = y.shape
        repeated_u = physical_u.reshape(n, 1).expand(-1, n_blocks).reshape(-1)
        scores = self.score_net(
            repeated_u, y.reshape(-1, block_size)
        ).reshape(n, n_blocks)
        return scores.sum(dim=1) - n_blocks * self.debias_net(physical_u)


def train_phase(
    model: nn.Module,
    tables: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    *,
    phase: str,
    steps: int,
    lr: float,
    lam_fisher: float,
    seed: int,
    device: str,
    u_min: float,
    u_max: float,
    scale_dist: float,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    reference = tables["dr_train"] if lam_fisher > 0 else None
    trained, trace, info = core.train_score_phase(
        model,
        tables["ds_train"],
        tables["ds_val"],
        reference,
        steps=steps,
        batch_size=args.batch_size,
        ref_batch_size=args.ref_batch_size,
        lr=lr,
        weight_decay=args.weight_decay,
        lam_fisher=lam_fisher,
        patience=args.patience,
        print_every=args.print_every,
        seed=seed,
        device=device,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
        phase=phase,
    )
    if lam_fisher == 0 and any(
        abs(float(row["train_fisher_penalty"])) > 0 for row in trace
    ):
        raise RuntimeError(f"Plain phase {phase} unexpectedly used a penalty")
    return trained, trace, info


def fit_debias(
    score_net: nn.Module,
    tables: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    *,
    seed: int,
    device: str,
    u_min: float,
    u_max: float,
    scale_dist: float,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    target_train = core.reference_mean_scores(
        score_net, tables["dr_train"], args.ref_batch_size, device
    )
    target_val = core.reference_mean_scores(
        score_net, tables["dr_val"], args.ref_batch_size, device
    )
    model = core.DebiasRegression(args.debias_hidden, args.debias_depth).to(device)
    return core.train_debias(
        model,
        tables["dr_train"]["u"],
        target_train,
        tables["dr_val"]["u"],
        target_val,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
        steps=args.debias_steps,
        curve_steps=args.debias_curve_steps,
        batch_size=args.debias_batch_size,
        lr=args.debias_lr,
        curve_lr=args.debias_curve_lr,
        lam_curve=args.lam_curve,
        patience=args.patience,
        print_every=args.print_every,
        seed=seed,
        device=device,
    )


def gate_state_max_abs_difference(
    left: CSJiangCommonFactorBlockScore,
    right: CSJiangCommonFactorBlockScore,
) -> float:
    differences: list[float] = []
    for gate_name in ("marginal_gate", "pairwise_gate"):
        left_gate = getattr(left, gate_name)
        right_gate = getattr(right, gate_name)
        if left_gate is None or right_gate is None:
            raise RuntimeError("Gate-state comparison requires structured models")
        for key, value in left_gate.state_dict().items():
            differences.append(
                float(torch.max(torch.abs(value - right_gate.state_dict()[key])).cpu())
            )
    return max(differences, default=0.0)


def evaluate_methods(
    wrappers: dict[str, nn.Module],
    args: argparse.Namespace,
    out: Path,
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    mean_zero_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    pi_values = [float(item) for item in args.diagnostic_pi_values.split(",")]
    for index, pi in enumerate(pi_values):
        rng = np.random.default_rng(args.seed + 100_003 + index)
        u = np.full(args.n_test, float(stage1.logit_np(pi)), dtype=np.float64)
        y = stage1.simulate_common_factor(
            rng, u, args.n_blocks, args.block_size, args.tau
        )
        truth = stage1.full_score_u(y, pi, args.tau)
        predictions[f"pi{pi:g}__truth"] = truth.astype(np.float32)
        for method, wrapper in wrappers.items():
            wrapper.eval()
            parts: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, args.n_test, args.eval_batch_size):
                    stop = min(start + args.eval_batch_size, args.n_test)
                    parts.append(
                        wrapper(
                            torch.as_tensor(
                                y[start:stop], dtype=torch.float32, device=device
                            ),
                            torch.as_tensor(
                                u[start:stop], dtype=torch.float32, device=device
                            ),
                        ).cpu().numpy()
                    )
            pred = np.concatenate(parts).astype(np.float64)
            if not np.isfinite(pred).all():
                raise RuntimeError(f"Non-finite predictions for {method}, pi={pi}")
            predictions[f"pi{pi:g}__{method}"] = pred.astype(np.float32)
            row = stage1.metric_row(truth, pred)
            row.update(
                {
                    "seed": args.seed,
                    "method": METHOD_LABELS[method],
                    "method_id": method,
                    "pi": pi,
                    "n": args.n_test,
                }
            )
            rows.append(row)
            se = float(np.std(pred, ddof=1) / math.sqrt(pred.size))
            mean_zero_rows.append(
                {
                    "seed": args.seed,
                    "method": METHOD_LABELS[method],
                    "method_id": method,
                    "pi": pi,
                    "mean_score": float(np.mean(pred)),
                    "se": se,
                    "abs_mean_over_se": float(
                        abs(np.mean(pred)) / max(se, 1e-12)
                    ),
                    "passed_2se": bool(abs(np.mean(pred)) < 2.0 * se),
                }
            )
            print(
                f"{METHOD_LABELS[method]:34s} pi={pi:.3f}: "
                f"mse={row['mse']:.6g}, std_mse={row['std_mse']:.6g}, "
                f"corr={row['corr']:.5g}, mean={np.mean(pred):+.5g}"
            )
    np.savez(out / "paired_test_predictions.npz", **predictions)
    return rows, mean_zero_rows


def run(args: argparse.Namespace) -> None:
    started = time.time()
    stage1.set_global_seed(int(args.seed))
    device = core.resolve_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_path)
    u_min = float(stage1.logit_np(args.pi_min))
    u_max = float(stage1.logit_np(args.pi_max))
    print("device:", device)
    print("output:", out)
    print("cache:", cache_path)

    tables, cache_metadata = make_or_load_shared_tables(cache_path, args)
    proposal_error = check_proposal_score(
        tables["ds_train"]["pi"], tables["ds_train"]["prop_score"]
    )
    if proposal_error > 2e-7:
        raise RuntimeError(f"Proposal-score check failed: {proposal_error}")
    scale_dist = float(
        np.minimum(
            tables["ds_train"]["u"] - u_min,
            u_max - tables["ds_train"]["u"],
        ).mean()
    )
    if not np.isfinite(scale_dist) or scale_dist <= 0:
        raise RuntimeError("Invalid boundary-weight scale")

    print("Fitting frozen Model-2 CS standardization statistics...")
    cs_stats = make_cs_stats_streaming(
        tables["ds_train"]["y"],
        tables["ds_train"]["u"],
        args.tau,
        args.stats_chunk_size,
    )
    np.savez(out / "shared_cs_feature_stats.npz", **cs_stats)

    print("\nTraining shared direct-SM linear base...")
    linear_direct = CSJiangCommonFactorBlockScore(
        args.tau,
        args.block_size,
        cs_stats,
        args.jiang_hidden,
        args.jiang_depth,
        structured=False,
        gate_hidden=args.gate_hidden,
    ).to(device)
    linear_direct, linear_direct_trace, linear_direct_info = train_phase(
        linear_direct,
        tables,
        args,
        phase="linear_cs_shared_direct_sm",
        steps=args.init_steps,
        lr=args.init_lr,
        lam_fisher=0.0,
        seed=args.seed + 1101,
        device=device,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
    )
    linear_plain = copy.deepcopy(linear_direct)
    linear_full = copy.deepcopy(linear_direct)

    print("\nRefining full linear arm with Fisher curvature penalty...")
    linear_full, linear_fisher_trace, linear_fisher_info = train_phase(
        linear_full,
        tables,
        args,
        phase="linear_cs_full_fisher",
        steps=args.fisher_steps,
        lr=args.fisher_lr,
        lam_fisher=args.lam_fisher,
        seed=args.seed + 1202,
        device=device,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
    )
    linear_full_bias = subtract_global_bias(
        linear_full, tables["ds_train"], args.batch_size, device
    )

    # One template guarantees identical gate initialization in the plain/full arms.
    structured_template = CSJiangCommonFactorBlockScore(
        args.tau,
        args.block_size,
        cs_stats,
        args.jiang_hidden,
        args.jiang_depth,
        structured=True,
        gate_hidden=args.gate_hidden,
    ).to(device)
    structured_plain = copy.deepcopy(structured_template)
    structured_full = copy.deepcopy(structured_template)
    structured_plain.copy_linear_readout_(linear_plain)
    structured_full.copy_linear_readout_(linear_full)
    gate_initial_difference = gate_state_max_abs_difference(
        structured_plain, structured_full
    )
    plain_sanity = structural_sanity_checks(
        linear_plain,
        structured_plain,
        tables["ds_val"],
        device=device,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
    )
    full_sanity = structural_sanity_checks(
        linear_full,
        structured_full,
        tables["ds_val"],
        device=device,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
    )
    sanity = {
        "proposal_score_max_abs": proposal_error,
        "plain": plain_sanity,
        "full": full_sanity,
        "plain_full_gate_initial_max_abs": gate_initial_difference,
        "passed": bool(
            plain_sanity["passed"]
            and full_sanity["passed"]
            and gate_initial_difference == 0.0
        ),
    }
    print("Structural sanity checks:", sanity)
    if not sanity["passed"]:
        raise RuntimeError("Combined structural sanity checks failed")

    print("\nTraining plain structured arm with direct SM only...")
    structured_plain, structured_plain_trace, structured_plain_info = train_phase(
        structured_plain,
        tables,
        args,
        phase="structured_cs_plain_direct_sm",
        steps=args.structured_steps,
        lr=args.plain_structured_lr,
        lam_fisher=0.0,
        seed=args.seed + 2202,
        device=device,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
    )
    print("\nTraining full structured arm with Fisher curvature penalty...")
    structured_full, structured_full_trace, structured_full_info = train_phase(
        structured_full,
        tables,
        args,
        phase="structured_cs_full_fisher",
        steps=args.structured_steps,
        lr=args.full_structured_lr,
        lam_fisher=args.lam_fisher,
        seed=args.seed + 2202,
        device=device,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
    )
    structured_full_bias = subtract_global_bias(
        structured_full, tables["ds_train"], args.batch_size, device
    )

    print("\nFitting conditional debias regressions for the full arms...")
    linear_debias, linear_debias_trace, linear_debias_info = fit_debias(
        linear_full,
        tables,
        args,
        seed=args.seed + 1303,
        device=device,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
    )
    structured_debias, structured_debias_trace, structured_debias_info = fit_debias(
        structured_full,
        tables,
        args,
        seed=args.seed + 2303,
        device=device,
        u_min=u_min,
        u_max=u_max,
        scale_dist=scale_dist,
    )

    wrappers: dict[str, nn.Module] = {
        "linear_cs_plain_base": PlainDatasetScoreWrapper(linear_plain).to(device).eval(),
        "structured_cs_plain": PlainDatasetScoreWrapper(structured_plain).to(device).eval(),
        "linear_cs": FullDatasetScoreWrapper(linear_full, linear_debias).to(device).eval(),
        "structured_cs": FullDatasetScoreWrapper(
            structured_full, structured_debias
        ).to(device).eval(),
    }
    rows, mean_zero_rows = evaluate_methods(wrappers, args, out, device)
    write_csv(out / "score_summary_by_pi.csv", rows)
    write_csv(out / "mean_zero_checks.csv", mean_zero_rows)

    traces = [
        *(
            {"method": "linear_cs_plain_base", **row}
            for row in linear_direct_trace
        ),
        *(
            {"method": "linear_cs", **row}
            for row in linear_fisher_trace + linear_debias_trace
        ),
        *(
            {"method": "structured_cs_plain", **row}
            for row in structured_plain_trace
        ),
        *(
            {"method": "structured_cs", **row}
            for row in structured_full_trace + structured_debias_trace
        ),
    ]
    write_csv(out / "training_trace.csv", traces)

    plain_contract = {
        "direct_score_matching": True,
        "fisher_curvature_penalty": False,
        "global_bias_subtraction": False,
        "conditional_debias_regression": False,
        "reference_table_used": False,
        "training_table": "D^S only",
    }
    full_contract = {
        "direct_score_matching": True,
        "fisher_curvature_penalty": True,
        "global_bias_subtraction": True,
        "conditional_debias_regression": True,
        "reference_table_used": True,
        "training_table": "D^S + D^R",
    }
    method_contracts = {
        "linear_cs_plain_base": plain_contract,
        "structured_cs_plain": plain_contract,
        "linear_cs": full_contract,
        "structured_cs": full_contract,
    }
    training_info: dict[str, Any] = {
        "linear_cs_plain_base": {
            "shared_direct": linear_direct_info,
            "warm_start": None,
        },
        "structured_cs_plain": {
            "shared_direct": linear_direct_info,
            "warm_start": "linear_cs_plain_base",
            "structured": structured_plain_info,
        },
        "linear_cs": {
            "shared_direct": linear_direct_info,
            "fisher": linear_fisher_info,
            "bias_subtracted": linear_full_bias,
            "debias": linear_debias_info,
        },
        "structured_cs": {
            "shared_direct": linear_direct_info,
            "warm_start": "post-fisher/post-global-bias linear_cs",
            "structured": structured_full_info,
            "bias_subtracted": structured_full_bias,
            "debias": structured_debias_info,
        },
        "seconds": time.time() - started,
    }
    config = vars(args).copy()
    config.update(
        {
            "device_resolved": device,
            "parameter_input": "physical_u",
            "proposal_distribution": "uniform_pi_stratified",
            "proposal_score": "1-2*pi(u)",
            "feature_pipeline": (
                "separate marginal/pairwise local zscore -> identity/signed-positive "
                "anchor-conditioned gates -> within-block means -> frozen two-channel "
                "block zscore -> rho([marginal,pairwise,physical-u-derived anchor_z])"
            ),
            "same_block_pairs_only": True,
            "shared_cache": cache_metadata,
            "shared_cache_path": str(cache_path),
            "scale_dist": scale_dist,
            "loss_contracts": method_contracts,
        }
    )
    for method, wrapper in wrappers.items():
        torch.save(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "wrapper_class": type(wrapper).__name__,
                "state_dict": wrapper.state_dict(),
                "config": config,
                "training_info": training_info[method],
                "cs_feature_stats": cs_stats,
                "loss_contract": method_contracts[method],
            },
            out / f"model_{method}.pt",
        )
    (out / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (out / "training_info.json").write_text(
        json.dumps(training_info, indent=2), encoding="utf-8"
    )
    (out / "sanity_checks.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )
    print("Saved Model-2 four-arm ablation to", out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--n-s-train", type=int, default=600_000)
    parser.add_argument("--n-s-val", type=int, default=120_000)
    parser.add_argument("--n-r-train", type=int, default=2_000)
    parser.add_argument("--n-r-val", type=int, default=400)
    parser.add_argument("--m-r", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--n-blocks", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--pi-min", type=float, default=0.05)
    parser.add_argument("--pi-max", type=float, default=0.70)
    parser.add_argument("--diagnostic-pi-values", default="0.10,0.30,0.50,0.65")
    parser.add_argument("--jiang-hidden", type=int, default=64)
    parser.add_argument("--jiang-depth", type=int, default=2)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--init-steps", type=int, default=3_000)
    parser.add_argument("--fisher-steps", type=int, default=2_000)
    parser.add_argument("--structured-steps", type=int, default=2_000)
    parser.add_argument("--debias-steps", type=int, default=2_000)
    parser.add_argument("--debias-curve-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--ref-batch-size", type=int, default=16)
    parser.add_argument("--debias-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--stats-chunk-size", type=int, default=2_048)
    parser.add_argument("--init-lr", type=float, default=1e-3)
    parser.add_argument("--fisher-lr", type=float, default=1e-4)
    parser.add_argument("--plain-structured-lr", type=float, default=1e-3)
    parser.add_argument("--full-structured-lr", type=float, default=1e-4)
    parser.add_argument("--debias-lr", type=float, default=1e-3)
    parser.add_argument("--debias-curve-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lam-fisher", type=float, default=1e-2)
    parser.add_argument("--lam-curve", type=float, default=1e-4)
    parser.add_argument("--debias-hidden", type=int, default=32)
    parser.add_argument("--debias-depth", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", default="runs/model2_jiang_cs_loss_ablation"
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
