#!/usr/bin/env python3
"""Load a current Model 2 amortized-FSM checkpoint and evaluate its field.

The runtime deliberately reuses the Stage-1 model classes and frozen feature
statistics. It exposes the frozen score needed by the data-only Stage-2 context,

    S(Y, u_eval),  u_eval = logit(pi_eval),

for data generated at any parameter value.  Feature construction remains
differentiable with respect to ``u_eval`` for runtime validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model2 import stage1


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def parse_method_list(text: str) -> list[str]:
    methods = [part.strip() for part in str(text).split(",") if part.strip()]
    bad = sorted(set(methods) - {"linear", "shared_radial"})
    if bad:
        raise ValueError(f"Unknown methods: {bad}")
    if not methods:
        raise ValueError("At least one method is required")
    return methods


class AmortizedScoreRuntime:
    """Frozen Stage-1 score model evaluated at arbitrary candidate anchors."""

    def __init__(
        self,
        run_dir: str | Path,
        method: str,
        device: str = "auto",
        *,
        checkpoint_path: str | Path | None = None,
        require_fixed_final_ema: bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.method = str(method)
        if self.method not in {"linear", "shared_radial"}:
            raise ValueError("method must be 'linear' or 'shared_radial'")
        self.device = resolve_device(device)

        config_path = self.run_dir / "config.json"
        stats_path = self.run_dir / "feature_stats.npz"
        self.checkpoint_is_override = checkpoint_path is not None
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else self.run_dir / f"model_{self.method}.pt"
        )
        for path in (config_path, stats_path, self.checkpoint_path):
            if not path.exists():
                raise FileNotFoundError(path)

        self.config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        # Older checkpoints recorded the only supported mode explicitly;
        # current clean runs omit the redundant field.
        if self.config.get("feature_pi_mode", "anchor") != "anchor":
            raise ValueError("The clean runtime requires feature_pi_mode='anchor'")
        self.stats = {key: np.asarray(value) for key, value in np.load(stats_path).items()}
        self.tau = float(self.config["tau"])
        self.n_blocks = int(self.config["n_blocks"])
        self.block_size = int(self.config["block_size"])
        self.sigma_q = float(self.config["sigma_q"])
        self.pi_min = float(self.config["anchor_pi_min"])
        self.pi_max = float(self.config["anchor_pi_max"])
        self.u_min = float(stage1.logit_np(self.pi_min))
        self.u_max = float(stage1.logit_np(self.pi_max))
        self.anchor_mean = float(np.asarray(self.stats["anchor_u_mean"]))
        self.anchor_sd = float(np.asarray(self.stats["anchor_u_sd"]))
        if self.anchor_sd <= 0:
            raise ValueError("anchor_u_sd must be positive")

        payload = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("method") != self.method:
            raise ValueError(
                f"Checkpoint method mismatch: expected {self.method!r}, got {payload.get('method')!r}"
            )
        checkpoint_config = payload.get("config", {})
        if self.checkpoint_is_override:
            if not isinstance(checkpoint_config, dict):
                raise ValueError("Explicit checkpoint is missing a mapping-valued config")
            run_config_json = json.dumps(
                self.config,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            checkpoint_config_json = json.dumps(
                checkpoint_config,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            if checkpoint_config_json != run_config_json:
                differing_keys = sorted(
                    key
                    for key in set(self.config) | set(checkpoint_config)
                    if self.config.get(key) != checkpoint_config.get(key)
                )
                raise ValueError(
                    "Explicit checkpoint config does not exactly match config.json; "
                    f"differing keys: {differing_keys[:12]}"
                )
        for key in (
            "n_blocks",
            "block_size",
            "tau",
            "sigma_q",
            "feature_pi_mode",
            "hidden",
            "depth",
        ):
            if key in checkpoint_config and checkpoint_config[key] != self.config.get(key):
                raise ValueError(f"Checkpoint/config mismatch for {key}")

        if self.method == "linear":
            model: torch.nn.Module = stage1.LinearCSBetaDeepSets(
                int(self.config["hidden"]),
                int(self.config["depth"]),
            )
        else:
            if not bool(self.config.get("gate_condition_on_anchor", 0)):
                raise ValueError(
                    "Shared-radial runtime requires an anchor-conditioned gate"
                )
            model = stage1.SharedRawRadialCSBetaDeepSets(
                int(self.config["hidden"]),
                int(self.config["depth"]),
                int(self.config["gate_hidden"]),
                gate_condition_on_anchor=True,
                s1_mean=self.stats["s1_mean"],
                s1_sd=self.stats["s1_sd"],
                s2_mean=self.stats["s2_mean"],
                s2_sd=self.stats["s2_sd"],
                block_mean=self.stats["block_mean"],
                block_sd=self.stats["block_sd"],
            )
        model.load_state_dict(payload["state_dict"], strict=True)
        self.model = model.to(self.device).eval()
        self.label = str(payload.get("label", self.method))
        self.training_info = dict(payload.get("training_info", {}))
        self.selection = str(payload.get("selection", "validation_best"))
        self.checkpoint_step = payload.get(
            "checkpoint_step", self.training_info.get("best_step")
        )
        self.checkpoint_source = str(
            payload.get(
                "checkpoint_source",
                self.training_info.get("best_source", "unknown"),
            )
        )
        if self.checkpoint_source not in {"raw", "ema"}:
            raise ValueError(
                f"Unexpected checkpoint source={self.checkpoint_source!r}"
            )
        if require_fixed_final_ema:
            if not self.checkpoint_is_override:
                raise ValueError(
                    "require_fixed_final_ema=True requires an explicit checkpoint_path"
                )
            if self.selection != "fixed_milestone":
                raise ValueError(
                    "Locked Stage-2 checkpoint must have selection='fixed_milestone'; "
                    f"got {self.selection!r}"
                )
            if self.checkpoint_source != "ema":
                raise ValueError(
                    "Locked Stage-2 checkpoint must have checkpoint_source='ema'; "
                    f"got {self.checkpoint_source!r}"
                )
            if "iters" not in self.config:
                raise ValueError(
                    "config.json is missing 'iters', so fixed-final status cannot be verified"
                )
            if self.checkpoint_step is None or int(self.checkpoint_step) != int(
                self.config["iters"]
            ):
                raise ValueError(
                    "Locked Stage-2 checkpoint must be the final training milestone: "
                    f"checkpoint_step={self.checkpoint_step!r}, iters={self.config['iters']!r}"
                )
        self.checkpoint_sha256 = hashlib.sha256(
            self.checkpoint_path.read_bytes()
        ).hexdigest()
        # Backward-compatible alias used by existing downstream code.
        self.best_source = self.checkpoint_source

        self._s1_mean = torch.as_tensor(
            float(np.asarray(self.stats["s1_mean"])), dtype=torch.float64, device=self.device
        )
        self._s1_sd = torch.as_tensor(
            float(np.asarray(self.stats["s1_sd"])), dtype=torch.float64, device=self.device
        )
        self._s2_mean = torch.as_tensor(
            float(np.asarray(self.stats["s2_mean"])), dtype=torch.float64, device=self.device
        )
        self._s2_sd = torch.as_tensor(
            float(np.asarray(self.stats["s2_sd"])), dtype=torch.float64, device=self.device
        )
        self._block_mean = torch.as_tensor(
            self.stats["block_mean"], dtype=torch.float32, device=self.device
        )
        self._block_sd = torch.as_tensor(
            self.stats["block_sd"], dtype=torch.float32, device=self.device
        )

    def metadata(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_is_override": self.checkpoint_is_override,
            "method": self.method,
            "label": self.label,
            "device": self.device,
            "selection": self.selection,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_step": self.checkpoint_step,
            "best_source": self.training_info.get("best_source"),
            "best_step": self.training_info.get("best_step"),
            "best_val_loss": self.training_info.get("best_val_loss"),
            "pi_min": self.pi_min,
            "pi_max": self.pi_max,
            "sigma_q": self.sigma_q,
            "n_blocks": self.n_blocks,
            "block_size": self.block_size,
            "tau": self.tau,
        }

    def validate_y(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        expected = (self.n_blocks, self.block_size)
        if y_arr.ndim != 3 or tuple(y_arr.shape[1:]) != expected:
            raise ValueError(f"Expected y shape (n, {expected[0]}, {expected[1]}), got {y_arr.shape}")
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y contains non-finite values")
        return y_arr

    def precompute_log_ratios(self, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        y_arr = self.validate_y(y)
        lr1 = stage1.marginal_log_ratio(y_arr, self.tau)
        lr2 = stage1.pairwise_log_ratio(y_arr, self.tau)
        return (
            torch.as_tensor(lr1, dtype=torch.float64, device=self.device),
            torch.as_tensor(lr2, dtype=torch.float64, device=self.device),
        )

    def _coerce_u(self, u: float | np.ndarray | torch.Tensor, n: int, require_grad: bool) -> torch.Tensor:
        if isinstance(u, torch.Tensor):
            value = u.to(device=self.device, dtype=torch.float64)
        else:
            value = torch.as_tensor(u, dtype=torch.float64, device=self.device)
        if value.ndim == 0:
            value = value.repeat(n)
        elif value.ndim == 1 and value.shape[0] == 1 and n != 1:
            value = value.repeat(n)
        elif value.ndim != 1 or value.shape[0] != n:
            raise ValueError(f"u must be scalar or length {n}, got shape {tuple(value.shape)}")
        if torch.any(value < self.u_min - 1e-12) or torch.any(value > self.u_max + 1e-12):
            raise ValueError(
                f"u is outside trained anchor support [{self.u_min:.6g}, {self.u_max:.6g}]"
            )
        if require_grad and not value.requires_grad:
            value = value.detach().clone().requires_grad_(True)
        return value

    def score_from_log_ratios_tensor(
        self,
        lr1: torch.Tensor,
        lr2: torch.Tensor,
        u: float | np.ndarray | torch.Tensor,
        *,
        require_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n = int(lr1.shape[0])
        if lr1.ndim != 3 or lr2.ndim != 3 or int(lr2.shape[0]) != n:
            raise ValueError("log-ratio tensors must have shapes (n, K, m) and (n, K, n_pairs)")
        u_tensor = self._coerce_u(u, n, require_grad=require_grad)
        u_b = u_tensor.reshape(n, 1, 1)
        pi_b = torch.sigmoid(u_b)
        s1 = torch.sigmoid(u_b + lr1) - pi_b
        s2 = torch.sigmoid(u_b + lr2) - pi_b
        s1_z = ((s1 - self._s1_mean) / self._s1_sd).to(torch.float32)
        s2_z = ((s2 - self._s2_mean) / self._s2_sd).to(torch.float32)
        anchor_z = ((u_tensor - self.anchor_mean) / self.anchor_sd).to(torch.float32)

        if self.method == "linear":
            block_raw = torch.stack([s1_z.mean(dim=2), s2_z.mean(dim=2)], dim=-1)
            block = (block_raw - self._block_mean) / self._block_sd
            score = self.model(block, anchor_z)
        else:
            score = self.model(s1_z, s2_z, anchor_z)
        return score, u_tensor

    def score(
        self,
        y: np.ndarray,
        u: float | np.ndarray,
        *,
        batch_size: int = 256,
    ) -> np.ndarray:
        y_arr = self.validate_y(y)
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.ndim == 0:
            u_arr = np.full(y_arr.shape[0], float(u_arr), dtype=np.float64)
        if u_arr.shape != (y_arr.shape[0],):
            raise ValueError(f"u must be scalar or shape {(y_arr.shape[0],)}, got {u_arr.shape}")
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, y_arr.shape[0], int(batch_size)):
                stop = min(start + int(batch_size), y_arr.shape[0])
                lr1, lr2 = self.precompute_log_ratios(y_arr[start:stop])
                score, _ = self.score_from_log_ratios_tensor(lr1, lr2, u_arr[start:stop])
                out.append(score.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(out, axis=0)

    def score_grid(
        self,
        y: np.ndarray,
        u_grid: np.ndarray,
        *,
        batch_size: int = 256,
        grid_chunk_size: int = 1,
    ) -> np.ndarray:
        y_arr = self.validate_y(y)
        u_values = np.asarray(u_grid, dtype=np.float64).reshape(-1)
        if np.any(u_values < self.u_min - 1e-12) or np.any(u_values > self.u_max + 1e-12):
            raise ValueError("u_grid contains values outside trained anchor support")
        if int(grid_chunk_size) < 1:
            raise ValueError("grid_chunk_size must be positive")
        out = np.empty((y_arr.shape[0], u_values.size), dtype=np.float64)
        for start in range(0, y_arr.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), y_arr.shape[0])
            lr1, lr2 = self.precompute_log_ratios(y_arr[start:stop])
            with torch.no_grad():
                for grid_start in range(0, u_values.size, int(grid_chunk_size)):
                    grid_stop = min(grid_start + int(grid_chunk_size), u_values.size)
                    u_chunk = u_values[grid_start:grid_stop]
                    n_data = int(lr1.shape[0])
                    n_u = int(u_chunk.size)
                    # Evaluate multiple parameter locations in one model call.
                    # This is especially useful for one-dimensional grid diagnostics.
                    lr1_chunk = (
                        lr1[:, None, :, :]
                        .expand(-1, n_u, -1, -1)
                        .reshape(n_data * n_u, lr1.shape[1], lr1.shape[2])
                    )
                    lr2_chunk = (
                        lr2[:, None, :, :]
                        .expand(-1, n_u, -1, -1)
                        .reshape(n_data * n_u, lr2.shape[1], lr2.shape[2])
                    )
                    u_repeated = np.tile(u_chunk, n_data)
                    score, _ = self.score_from_log_ratios_tensor(
                        lr1_chunk,
                        lr2_chunk,
                        u_repeated,
                    )
                    out[start:stop, grid_start:grid_stop] = (
                        score.detach().cpu().numpy().astype(np.float64).reshape(n_data, n_u)
                    )
        return out

    def score_and_derivative(
        self,
        y: np.ndarray,
        u: float | np.ndarray,
        *,
        batch_size: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_arr = self.validate_y(y)
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.ndim == 0:
            u_arr = np.full(y_arr.shape[0], float(u_arr), dtype=np.float64)
        if u_arr.shape != (y_arr.shape[0],):
            raise ValueError(f"u must be scalar or shape {(y_arr.shape[0],)}, got {u_arr.shape}")
        scores: list[np.ndarray] = []
        derivatives: list[np.ndarray] = []
        for start in range(0, y_arr.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), y_arr.shape[0])
            lr1, lr2 = self.precompute_log_ratios(y_arr[start:stop])
            score, u_tensor = self.score_from_log_ratios_tensor(
                lr1,
                lr2,
                u_arr[start:stop],
                require_grad=True,
            )
            derivative = torch.autograd.grad(score.sum(), u_tensor, create_graph=False)[0]
            scores.append(score.detach().cpu().numpy().astype(np.float64))
            derivatives.append(derivative.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(scores), np.concatenate(derivatives)

    def legacy_score(self, y: np.ndarray, u: float | np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Evaluate through the original NumPy Stage-1 feature path."""
        y_arr = self.validate_y(y)
        u_arr = np.asarray(u, dtype=np.float64)
        if u_arr.ndim == 0:
            u_arr = np.full(y_arr.shape[0], float(u_arr), dtype=np.float64)
        if u_arr.shape != (y_arr.shape[0],):
            raise ValueError(f"u must be scalar or shape {(y_arr.shape[0],)}, got {u_arr.shape}")
        pi = stage1.sigmoid_np(u_arr)
        feat = stage1.featurize(y_arr, self.tau, pi, self.stats)
        feat["anchor_z"] = ((u_arr - self.anchor_mean) / self.anchor_sd).astype(np.float32)
        feat["anchor_pi"] = pi.astype(np.float64)
        return stage1.predict(self.model, feat, self.method, self.device, int(batch_size)).astype(np.float64)

    def run_sanity_checks(self, seed: int = 20260709) -> dict[str, object]:
        rng = np.random.default_rng(int(seed))
        pi_values = np.asarray([0.07, 0.30, 0.68], dtype=np.float64)
        u_values = stage1.logit_np(pi_values)
        y = stage1.simulate_common_factor(
            rng,
            u_values,
            n_blocks=self.n_blocks,
            block_size=self.block_size,
            tau=self.tau,
        )
        runtime = self.score(y, u_values, batch_size=3)
        legacy = self.legacy_score(y, u_values, batch_size=3)
        forward_max_abs = float(np.max(np.abs(runtime - legacy)))

        score, derivative = self.score_and_derivative(y, u_values, batch_size=3)
        # The network body is float32.  A very small h is dominated by float32
        # cancellation, so use a moderate step for this derivative sanity check.
        h = 3e-3
        plus = self.score(y, u_values + h, batch_size=3)
        minus = self.score(y, u_values - h, batch_size=3)
        fd = (plus - minus) / (2.0 * h)
        derivative_max_abs = float(np.max(np.abs(derivative - fd)))
        derivative_rel = float(
            np.max(np.abs(derivative - fd) / np.maximum(1e-6, np.abs(fd)))
        )

        block_perm = self.score(y[:, ::-1, :].copy(), u_values, batch_size=3)
        within_perm = self.score(y[:, :, ::-1].copy(), u_values, batch_size=3)
        block_perm_max_abs = float(np.max(np.abs(score - block_perm)))
        within_perm_max_abs = float(np.max(np.abs(score - within_perm)))

        checks = {
            **self.metadata(),
            "forward_max_abs": forward_max_abs,
            "derivative_fd_max_abs": derivative_max_abs,
            "derivative_fd_max_rel": derivative_rel,
            "block_permutation_max_abs": block_perm_max_abs,
            "within_block_permutation_max_abs": within_perm_max_abs,
            "all_scores_finite": bool(np.all(np.isfinite(score))),
            "all_derivatives_finite": bool(np.all(np.isfinite(derivative))),
        }
        checks["passed"] = bool(
            forward_max_abs < 2e-5
            and derivative_max_abs < 2e-3
            and derivative_rel < 2e-3
            and block_perm_max_abs < 2e-5
            and within_perm_max_abs < 2e-5
            and checks["all_scores_finite"]
            and checks["all_derivatives_finite"]
        )
        return checks


def logit_grid(pi_min: float, pi_max: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 < pi_min < pi_max < 1.0):
        raise ValueError("pi bounds must lie in (0, 1)")
    if size < 2:
        raise ValueError("grid size must be at least 2")
    pi = np.linspace(float(pi_min), float(pi_max), int(size), dtype=np.float64)
    return pi, stage1.logit_np(pi)


def normal_relative_density(z: float) -> float:
    """N(0,1) density at z relative to its density at zero."""
    return math.exp(-0.5 * float(z) ** 2)
