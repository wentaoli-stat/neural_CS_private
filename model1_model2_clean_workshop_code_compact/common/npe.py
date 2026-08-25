"""Small, self-contained SBI-NPE utilities used by both cleaned models."""

from __future__ import annotations

import importlib
import math
import time
from typing import Any

import numpy as np
import torch


def import_sbi():
    """Import SBI lazily so Stage-1 remains usable without the optional dependency."""
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_  # type: ignore[attr-defined]
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_  # type: ignore[attr-defined]
    try:
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
        raise ImportError("Stage-2 requires `sbi`; install requirements.txt first.") from exc
    return NPE, posterior_nn, BoxUniform


def resolve_device(value: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value


def pi_to_unit(pi: np.ndarray, pi_min: float, pi_max: float, eps: float = 1e-5) -> np.ndarray:
    unit = (np.asarray(pi, dtype=np.float32) - float(pi_min)) / (float(pi_max) - float(pi_min))
    return np.clip(unit, eps, 1.0 - eps).reshape(-1, 1).astype(np.float32)


def unit_to_pi(unit: np.ndarray, pi_min: float, pi_max: float) -> np.ndarray:
    unit = np.asarray(unit, dtype=np.float32).reshape(-1)
    return (float(pi_min) + (float(pi_max) - float(pi_min)) * unit).astype(np.float32)


def make_prior(BoxUniform: Any, device: str):
    torch_device = torch.device(device)
    return BoxUniform(
        low=torch.zeros(1, dtype=torch.float32, device=torch_device),
        high=torch.ones(1, dtype=torch.float32, device=torch_device),
    )


def train_sbi_npe_method(
    NPE: Any,
    posterior_nn: Any,
    BoxUniform: Any,
    method: str,
    x_np: np.ndarray,
    pi_train: np.ndarray,
    seed: int,
    args: Any,
    device: str,
    data_device: str,
) -> dict[str, object]:
    """Train one conditional density estimator under the shared Stage-2 recipe."""
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    theta = torch.as_tensor(
        pi_to_unit(pi_train, float(args.pi_prior_min), float(args.pi_prior_max)),
        dtype=torch.float32,
    )
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
    inference = NPE(
        prior=make_prior(BoxUniform, device),
        density_estimator=density_estimator,
        device=device,
        show_progress_bars=True,
    )
    print(f"\nTraining sbi-NPE for {method} (seed={seed})")
    print("theta unit", tuple(theta.shape), "x", tuple(x.shape), "model", args.sbi_model)
    started = time.time()
    estimator = inference.append_simulations(theta, x, data_device=data_device).train(
        training_batch_size=int(args.sbi_batch_size),
        learning_rate=float(args.sbi_lr),
        max_num_epochs=int(args.max_epochs),
        validation_fraction=float(args.validation_fraction),
        stop_after_epochs=int(args.stop_after_epochs),
    )
    posterior = inference.build_posterior(estimator)
    print(f"{method} sbi-NPE seconds:", round(time.time() - started, 2))
    return {"method": method, "posterior": posterior, "device": str(device), "seed": int(seed)}


def sample_sbi_posterior(
    result: dict[str, object],
    x_obs: np.ndarray,
    posterior_n: int,
    seed: int,
    args: Any,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    if str(result["device"]).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    x = torch.as_tensor(
        np.asarray(x_obs, dtype=np.float32),
        dtype=torch.float32,
        device=torch.device(str(result["device"])),
    )
    samples_unit = result["posterior"].sample((int(posterior_n),), x=x)  # type: ignore[union-attr]
    return unit_to_pi(
        samples_unit.detach().cpu().numpy(),
        float(args.pi_prior_min),
        float(args.pi_prior_max),
    )


def exact_sample_w1(samples: np.ndarray, grid: np.ndarray, cdf: np.ndarray) -> float:
    samples = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    if samples.size == 0:
        return float("nan")
    probs = (np.arange(samples.size, dtype=np.float64) + 0.5) / samples.size
    return float(np.mean(np.abs(samples - np.interp(probs, cdf, grid))))


def exact_cdf_at(x: float, grid: np.ndarray, cdf: np.ndarray) -> float:
    return float(np.interp(float(x), grid, cdf, left=0.0, right=1.0))


def _standard_error(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0


def summarize_full_posterior(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate exact-posterior fidelity by true pi and method."""
    grouped: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in rows:
        if "w1_to_exact" in row:
            grouped.setdefault((float(row["pi_true"]), str(row["method"])), []).append(row)
    output: list[dict[str, object]] = []
    for (pi_true, method), group in sorted(grouped.items()):
        mean_sq = [float(row["mean_sq_err_to_exact"]) for row in group]
        output.append(
            {
                "pi_true": pi_true,
                "method": method,
                "n_seeds": len(group),
                "mean_abs_err_to_exact": float(np.mean([float(row["mean_abs_err_to_exact"]) for row in group])),
                "rmse_mean_to_exact": float(math.sqrt(np.mean(mean_sq))),
                "w1_to_exact": float(np.mean([float(row["w1_to_exact"]) for row in group])),
                "q05_abs_err_to_exact": float(np.mean([float(row["q05_abs_err_to_exact"]) for row in group])),
                "q50_abs_err_to_exact": float(np.mean([float(row["q50_abs_err_to_exact"]) for row in group])),
                "q95_abs_err_to_exact": float(np.mean([float(row["q95_abs_err_to_exact"]) for row in group])),
                "sd_abs_err_to_exact": float(np.mean([float(row["sd_abs_err_to_exact"]) for row in group])),
                "cdf_at_true_mean": float(np.mean([float(row["cdf_at_true"]) for row in group])),
                "cdf_at_true_sd": float(np.std([float(row["cdf_at_true"]) for row in group])),
                "coverage90": float(np.mean([float(row["pi_coverage90"]) for row in group])),
                "w1_se": _standard_error([float(row["w1_to_exact"]) for row in group]),
            }
        )
    return output

