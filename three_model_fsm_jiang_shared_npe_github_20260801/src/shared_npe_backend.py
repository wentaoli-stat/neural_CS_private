"""Shared one-parameter NPE utilities for the focused Model 1 bundle."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def import_sbi():
    """Import sbi with the compatibility shims used by the formal run."""
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_  # type: ignore[attr-defined]
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_  # type: ignore[attr-defined]
    try:
        import importlib
        import matplotlib.style as mpl_style

        if not hasattr(mpl_style, "core"):
            mpl_style.core = importlib.import_module("matplotlib.style.core")  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        from sbi.inference import NPE
        from sbi.neural_nets import posterior_nn
        from sbi.utils import BoxUniform
    except ImportError as exc:
        raise ImportError("Stage 2 requires the sbi package") from exc
    return NPE, posterior_nn, BoxUniform


def p_to_unit(p: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32).reshape(-1, 1)
    unit = (p - float(args.p_prior_min)) / (
        float(args.p_prior_max) - float(args.p_prior_min)
    )
    return np.clip(unit, 1e-5, 1.0 - 1e-5).astype(np.float32)


def unit_to_p(unit: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    unit = np.asarray(unit, dtype=np.float64).reshape(-1)
    return float(args.p_prior_min) + (
        float(args.p_prior_max) - float(args.p_prior_min)
    ) * unit


def train_npe(
    NPE,
    posterior_nn,
    BoxUniform,
    method: str,
    x_np: np.ndarray,
    p_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: str,
    data_device: str,
) -> dict[str, object]:
    """Train the scalar NPE exactly as in the frozen formal Stage-2 run."""
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    theta = torch.as_tensor(p_to_unit(p_train, args), dtype=torch.float32)
    x = torch.as_tensor(np.asarray(x_np, dtype=np.float32), dtype=torch.float32)
    density_estimator = posterior_nn(
        model=args.sbi_model,
        z_score_theta="independent",
        z_score_x="independent",
        hidden_features=int(args.sbi_hidden_features),
        num_transforms=int(args.sbi_num_transforms),
        num_bins=int(args.sbi_num_bins),
        num_components=int(args.sbi_num_components),
    )
    prior = BoxUniform(
        low=torch.zeros(1, dtype=torch.float32, device=torch.device(device)),
        high=torch.ones(1, dtype=torch.float32, device=torch.device(device)),
    )
    inference = NPE(
        prior=prior,
        density_estimator=density_estimator,
        device=device,
        show_progress_bars=False,
    )
    print(f"training NPE: {method}, theta={tuple(theta.shape)}, x={tuple(x.shape)}")
    start = time.time()
    estimator = inference.append_simulations(theta, x, data_device=data_device).train(
        training_batch_size=int(args.sbi_batch_size),
        learning_rate=float(args.sbi_lr),
        max_num_epochs=int(args.max_epochs),
        validation_fraction=float(args.validation_fraction),
        stop_after_epochs=int(args.stop_after_epochs),
    )
    posterior = inference.build_posterior(estimator)
    print(f"finished {method} in {time.time() - start:.1f}s")
    return {"method": method, "posterior": posterior, "device": device, "seed": seed}


def posterior_samples(
    result: dict[str, object],
    x_obs: np.ndarray,
    n: int,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    if str(result["device"]).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    x = torch.as_tensor(
        np.asarray(x_obs, dtype=np.float32),
        dtype=torch.float32,
        device=torch.device(str(result["device"])),
    )
    samples = result["posterior"].sample((int(n),), x=x)
    return unit_to_p(samples.detach().cpu().numpy(), args)


def exact_w1(samples: np.ndarray, axis: np.ndarray, cdf: np.ndarray) -> float:
    samples = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    probs = (np.arange(samples.size, dtype=np.float64) + 0.5) / samples.size
    quantiles = np.interp(probs, cdf, axis)
    return float(np.mean(np.abs(samples - quantiles)))
