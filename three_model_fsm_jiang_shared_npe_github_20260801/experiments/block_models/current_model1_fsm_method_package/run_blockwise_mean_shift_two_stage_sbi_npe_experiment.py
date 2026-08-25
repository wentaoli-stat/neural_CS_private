#!/usr/bin/env python3
"""Two-stage sbi-NPE for the blockwise shared mean-shift mixture.

Model 1:
    Z_k iid Bernoulli(pi), k=1,...,K
    Y_ki | Z_k=0 ~ N(0, 1)
    Y_ki | Z_k=1 ~ N(tau, 1)

Stage 1 trains local likelihood-free FSM summaries at a fixed center pi0.
Stage 2 freezes those summaries, simulates pi from a prior interval, trains one
sbi-NPE posterior per summary, and evaluates posterior mean MSE on fixed test
seeds.  The exact model likelihood is used only as a reference posterior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import run_blockwise_mean_shift_fsm_experiment as fsm


METHOD_ORDER = [
    "exact likelihood grid",
    "raw data",
    "raw ridge FSM score",
    "linear block CS",
    "linear CS ridge FSM score",
    "linear CS DeepSets FSM score",
    "MLP radial-gate CS linear FSM score",
    "MLP radial-gate CS DeepSets FSM score",
]


def import_sbi():
    # Some older arviz/xarray builds imported by sbi still reference NumPy 1.x
    # scalar aliases.  Adding aliases here keeps the experiment usable in the
    # mixed local/Colab/AutoDL environments we have been using.
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
        raise ImportError(
            "This script requires `sbi`. Install it with `pip install sbi`, then rerun."
        ) from exc
    return NPE, posterior_nn, BoxUniform


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def standard_error(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def pi_to_unit(pi: np.ndarray, pi_min: float, pi_max: float, eps: float = 1e-5) -> np.ndarray:
    out = (np.asarray(pi, dtype=np.float32) - float(pi_min)) / (float(pi_max) - float(pi_min))
    return np.clip(out, eps, 1.0 - eps).reshape(-1, 1).astype(np.float32)


def unit_to_pi(unit: np.ndarray, pi_min: float, pi_max: float) -> np.ndarray:
    unit = np.asarray(unit, dtype=np.float32).reshape(-1)
    return (float(pi_min) + (float(pi_max) - float(pi_min)) * unit).astype(np.float32)


def make_prior(BoxUniform, device: str):
    torch_device = torch.device(device)
    return BoxUniform(
        low=torch.zeros(1, dtype=torch.float32, device=torch_device),
        high=torch.ones(1, dtype=torch.float32, device=torch_device),
    )


def sample_pi_prior(rng: np.random.Generator, n: int, pi_min: float, pi_max: float) -> np.ndarray:
    return rng.uniform(float(pi_min), float(pi_max), size=int(n)).astype(np.float64)


def simulate_from_pi(
    rng: np.random.Generator,
    pi: np.ndarray,
    n_blocks: int,
    block_size: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    pi = np.asarray(pi, dtype=np.float64).reshape(-1)
    u = np.array([fsm.logit(float(p)) for p in pi], dtype=np.float64)
    y, log_r = fsm.simulate_from_u(rng, u, int(n_blocks), int(block_size), float(tau))
    return y, log_r


def zscore_fit(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x, dtype=np.float64).mean(axis=0, keepdims=True)
    sd = np.asarray(x, dtype=np.float64).std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return mean, sd


def zscore_apply(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean) / sd


def train_stage1(args: argparse.Namespace, device: str) -> dict[str, object]:
    rng = np.random.default_rng(int(args.stage1_seed))
    torch.manual_seed(int(args.stage1_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.stage1_seed))

    pi0 = float(args.center_pi)
    u0 = fsm.logit(pi0)
    print("\n==== Stage 1: FSM summaries ====")
    print(f"center pi0={pi0}, u0={u0:.4f}, tau={args.tau}")
    print(f"sigma_q={args.sigma_q}, K={args.n_blocks}, block_size={args.block_size}")

    y_train, log_r_train, target_train = fsm.make_fsm_data(
        rng,
        int(args.stage1_n_train),
        u0,
        float(args.sigma_q),
        int(args.n_blocks),
        int(args.block_size),
        float(args.tau),
    )
    y_val, log_r_val, target_val = fsm.make_fsm_data(
        rng,
        int(args.stage1_n_val),
        u0,
        float(args.sigma_q),
        int(args.n_blocks),
        int(args.block_size),
        float(args.tau),
    )

    s_train = fsm.score_u_from_log_ratio(log_r_train, pi0)
    s_val = fsm.score_u_from_log_ratio(log_r_val, pi0)

    raw_train_flat = y_train.reshape(int(args.stage1_n_train), -1)
    raw_val_flat = y_val.reshape(int(args.stage1_n_val), -1)
    raw_mean, raw_sd = zscore_fit(raw_train_flat)
    raw_train = zscore_apply(raw_train_flat, raw_mean, raw_sd)
    raw_val = zscore_apply(raw_val_flat, raw_mean, raw_sd)

    subscore_mean = float(s_train.mean())
    subscore_sd = float(s_train.std())
    if subscore_sd < 1e-8:
        subscore_sd = 1.0
    struct_train = (s_train - subscore_mean) / subscore_sd
    struct_val = (s_val - subscore_mean) / subscore_sd
    block_train = struct_train.sum(axis=2)
    block_val = struct_val.sum(axis=2)

    raw_coef = fsm.fit_linear_ridge(raw_train, target_train, ridge=float(args.ridge))
    block_coef = fsm.fit_linear_ridge(block_train, target_train, ridge=float(args.ridge))

    models: dict[str, torch.nn.Module] = {}
    traces: list[dict[str, object]] = []

    if "linear_deepsets" in args.stage1_methods:
        linear_deepsets, trace = fsm.train_torch_model(
            fsm.RhoOnlyMarginalBlockSums(int(args.hidden)),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=int(args.stage1_iters),
            batch_size=int(args.stage1_batch_size),
            lr=float(args.stage1_lr),
            weight_decay=float(args.stage1_weight_decay),
            patience=int(args.stage1_patience),
            print_every=int(args.stage1_print_every),
            device=device,
            name="linear CS DeepSets FSM",
        )
        models["linear_deepsets"] = linear_deepsets
        traces.extend({"method": "linear CS DeepSets FSM score", **row} for row in trace)

    gate_label = "polynomial" if args.clean_gate_kind == "log_polynomial" else "MLP"
    if "radial_linear" in args.stage1_methods:
        radial_linear, trace = fsm.train_torch_model(
            fsm.RadialGateMarginalCS(
                int(args.n_blocks),
                int(args.gate_hidden),
                block_coef,
                str(args.clean_gate_kind),
                int(args.poly_degree),
                str(args.gate_input),
            ),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=int(args.stage1_iters),
            batch_size=int(args.stage1_batch_size),
            lr=float(args.stage1_lr),
            weight_decay=float(args.stage1_weight_decay),
            patience=int(args.stage1_patience),
            print_every=int(args.stage1_print_every),
            device=device,
            name=f"{gate_label} radial-gate CS linear FSM",
        )
        models["radial_linear"] = radial_linear
        traces.extend({"method": f"{gate_label} radial-gate CS linear FSM score", **row} for row in trace)

    if "radial_deepsets" in args.stage1_methods:
        radial_deepsets, trace = fsm.train_torch_model(
            fsm.RadialGateMarginalDeepSets(
                int(args.hidden),
                int(args.gate_hidden),
                str(args.clean_gate_kind),
                int(args.poly_degree),
                str(args.gate_input),
            ),
            struct_train,
            target_train,
            struct_val,
            target_val,
            iters=int(args.stage1_iters),
            batch_size=int(args.stage1_batch_size),
            lr=float(args.stage1_lr),
            weight_decay=float(args.stage1_weight_decay),
            patience=int(args.stage1_patience),
            print_every=int(args.stage1_print_every),
            device=device,
            name=f"{gate_label} radial-gate CS DeepSets FSM",
        )
        models["radial_deepsets"] = radial_deepsets
        traces.extend({"method": f"{gate_label} radial-gate CS DeepSets FSM score", **row} for row in trace)

    return {
        "pi0": pi0,
        "u0": u0,
        "raw_mean": raw_mean,
        "raw_sd": raw_sd,
        "subscore_mean": subscore_mean,
        "subscore_sd": subscore_sd,
        "raw_coef": raw_coef,
        "block_coef": block_coef,
        "models": models,
        "trace_rows": traces,
        "device": device,
        "gate_label": gate_label,
    }


def compute_features(
    stage1: dict[str, object],
    y: np.ndarray,
    log_r: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    pi0 = float(stage1["pi0"])
    s = fsm.score_u_from_log_ratio(log_r, pi0)
    struct = (s - float(stage1["subscore_mean"])) / float(stage1["subscore_sd"])
    block = struct.sum(axis=2)

    raw_flat = y.reshape(y.shape[0], -1)
    raw_z = zscore_apply(raw_flat, stage1["raw_mean"], stage1["raw_sd"])

    features: dict[str, np.ndarray] = {
        "raw data": raw_flat.astype(np.float32),
        "raw ridge FSM score": fsm.predict_linear(
            np.asarray(stage1["raw_coef"], dtype=np.float64), raw_z
        )
        .reshape(-1, 1)
        .astype(np.float32),
        "linear block CS": block.astype(np.float32),
        "linear CS ridge FSM score": fsm.predict_linear(
            np.asarray(stage1["block_coef"], dtype=np.float64), block
        )
        .reshape(-1, 1)
        .astype(np.float32),
    }

    models = stage1["models"]
    device = str(stage1["device"])
    gate_label = str(stage1.get("gate_label", "MLP"))
    if "linear_deepsets" in models:
        features["linear CS DeepSets FSM score"] = fsm.predict_torch(
            models["linear_deepsets"], struct, device
        ).reshape(-1, 1).astype(np.float32)
    if "radial_linear" in models:
        features[f"{gate_label} radial-gate CS linear FSM score"] = fsm.predict_torch(
            models["radial_linear"], struct, device
        ).reshape(-1, 1).astype(np.float32)
    if "radial_deepsets" in models:
        features[f"{gate_label} radial-gate CS DeepSets FSM score"] = fsm.predict_torch(
            models["radial_deepsets"], struct, device
        ).reshape(-1, 1).astype(np.float32)
    return features


def stage1_score_diagnostics(
    stage1: dict[str, object],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate frozen scalar Stage-1 scores against exact/smoothed targets.

    This is diagnostic only.  Training remains likelihood-free.
    """
    if int(args.stage1_diagnostic_n_test) <= 0:
        return [], []

    rng = np.random.default_rng(int(args.stage1_diagnostic_seed))
    pi0 = float(stage1["pi0"])
    y_eval, log_r_eval = fsm.simulate_center(
        rng,
        int(args.stage1_diagnostic_n_test),
        pi0,
        int(args.n_blocks),
        int(args.block_size),
        float(args.tau),
    )
    exact_score = fsm.score_u_from_log_ratio(log_r_eval.sum(axis=2), pi0).sum(axis=1)
    features = compute_features(stage1, y_eval, log_r_eval, args)

    predictions: dict[str, np.ndarray] = {}
    for method, values in features.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 2 and values.shape[1] == 1:
            predictions[method] = values.reshape(-1)

    summary_rows: list[dict[str, object]] = []
    for method, pred in predictions.items():
        row: dict[str, object] = {
            "method": method,
            "n_test": int(args.stage1_diagnostic_n_test),
            "center_pi": pi0,
            "tau": float(args.tau),
            "n_blocks": int(args.n_blocks),
            "block_size": int(args.block_size),
            "sigma_q": float(args.sigma_q),
        }
        row.update(fsm.metrics(exact_score, pred))
        summary_rows.append(row)
    summary_rows = sorted(summary_rows, key=lambda row: float(row["std_mse"]))

    decomposition_rows: list[dict[str, object]] = []
    if bool(args.compute_stage1_smoothed_target_diagnostics):
        smoothed_target = fsm.smoothed_fsm_target_from_log_ratio(
            log_r_eval,
            float(stage1["u0"]),
            float(args.sigma_q),
            grid_size=int(args.stage1_smooth_grid_size),
            grid_width=float(args.stage1_smooth_grid_width),
            chunk_size=int(args.stage1_smooth_chunk_size),
        )
        decomposition_rows = fsm.target_decomposition_rows(
            predictions,
            exact_score,
            smoothed_target,
        )

    return summary_rows, decomposition_rows


def exact_loglik_grid(y: np.ndarray, pi_grid: np.ndarray, tau: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    pi_grid = np.asarray(pi_grid, dtype=np.float64)
    log_r_block = (float(tau) * y - 0.5 * float(tau) ** 2).sum(axis=1)
    out = []
    for pi in pi_grid:
        log_mix = np.logaddexp(np.log1p(-pi), np.log(pi) + log_r_block)
        out.append(float(np.sum(log_mix)))
    return np.asarray(out, dtype=np.float64)


def exact_posterior_grid_arrays(
    y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.linspace(float(args.pi_prior_min), float(args.pi_prior_max), int(args.grid_size))
    logw = exact_loglik_grid(y, grid, float(args.tau))
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= np.sum(w)
    cdf = np.cumsum(w)
    return grid, w, cdf


def exact_posterior_grid(y: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    grid, w, cdf = exact_posterior_grid_arrays(y, args)
    mean = float(np.sum(w * grid))
    var = float(np.sum(w * (grid - mean) ** 2))
    q05 = float(np.interp(0.05, cdf, grid))
    q50 = float(np.interp(0.50, cdf, grid))
    q95 = float(np.interp(0.95, cdf, grid))
    return {"mean": mean, "std": math.sqrt(var), "q05": q05, "q50": q50, "q95": q95}


def exact_sample_w1(samples: np.ndarray, grid: np.ndarray, cdf: np.ndarray) -> float:
    samples = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    if samples.size == 0:
        return float("nan")
    probs = (np.arange(samples.size, dtype=np.float64) + 0.5) / samples.size
    exact_quantiles = np.interp(probs, cdf, grid)
    return float(np.mean(np.abs(samples - exact_quantiles)))


def exact_cdf_at(x: float, grid: np.ndarray, cdf: np.ndarray) -> float:
    return float(np.interp(float(x), grid, cdf, left=0.0, right=1.0))


def train_sbi_npe_method(
    NPE,
    posterior_nn,
    BoxUniform,
    method: str,
    x_np: np.ndarray,
    pi_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: str,
    data_device: str,
) -> dict[str, object]:
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    theta_unit = pi_to_unit(pi_train, float(args.pi_prior_min), float(args.pi_prior_max))
    theta = torch.as_tensor(theta_unit, dtype=torch.float32)
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
    t0 = time.time()
    estimator = inference.append_simulations(theta, x, data_device=data_device).train(
        training_batch_size=int(args.sbi_batch_size),
        learning_rate=float(args.sbi_lr),
        max_num_epochs=int(args.max_epochs),
        validation_fraction=float(args.validation_fraction),
        stop_after_epochs=int(args.stop_after_epochs),
    )
    posterior = inference.build_posterior(estimator)
    print(f"{method} sbi-NPE seconds:", round(time.time() - t0, 2))
    return {"method": method, "posterior": posterior, "device": str(device), "seed": int(seed)}


def sample_sbi_posterior(
    result: dict[str, object],
    x_obs: np.ndarray,
    posterior_n: int,
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
    samples_unit = result["posterior"].sample((int(posterior_n),), x=x)
    return unit_to_pi(
        samples_unit.detach().cpu().numpy(),
        float(args.pi_prior_min),
        float(args.pi_prior_max),
    )


def posterior_seed(method: str, obs_seed: int) -> int:
    offsets = {method: 1000 + 137 * idx for idx, method in enumerate(METHOD_ORDER)}
    if method in offsets:
        return offsets[method] + int(obs_seed)
    digest = hashlib.blake2b(method.encode("utf-8"), digest_size=4).hexdigest()
    return 10000 + (int(digest, 16) % 1_000_000) + int(obs_seed)


def sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("pi_true", 0.0)),
            order.get(str(row["method"]), len(order)),
            int(row.get("seed", 0)),
        ),
    )


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in rows:
        by_key.setdefault((float(row["pi_true"]), str(row["method"])), []).append(row)
    out = []
    for (pi_true, method), group in by_key.items():
        mse = [float(row["pi_sq_err"]) for row in group]
        out.append(
            {
                "pi_true": float(pi_true),
                "method": method,
                "n_seeds": len(group),
                "pi_avg_mse": float(np.mean(mse)),
                "pi_mse_se": standard_error(mse),
                "pi_avg_post_std": float(np.mean([float(row["pi_post_std"]) for row in group])),
                "pi_coverage90": float(np.mean([float(row["pi_coverage90"]) for row in group])),
            }
        )
    return sort_rows(out)


def summarize_full_posterior(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_key: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in rows:
        if "w1_to_exact" not in row:
            continue
        by_key.setdefault((float(row["pi_true"]), str(row["method"])), []).append(row)
    out = []
    for (pi_true, method), group in by_key.items():
        out.append(
            {
                "pi_true": float(pi_true),
                "method": method,
                "n_seeds": len(group),
                "mean_abs_err_to_exact": float(np.mean([float(row["mean_abs_err_to_exact"]) for row in group])),
                "rmse_mean_to_exact": float(math.sqrt(np.mean([float(row["mean_sq_err_to_exact"]) for row in group]))),
                "w1_to_exact": float(np.mean([float(row["w1_to_exact"]) for row in group])),
                "q05_abs_err_to_exact": float(np.mean([float(row["q05_abs_err_to_exact"]) for row in group])),
                "q50_abs_err_to_exact": float(np.mean([float(row["q50_abs_err_to_exact"]) for row in group])),
                "q95_abs_err_to_exact": float(np.mean([float(row["q95_abs_err_to_exact"]) for row in group])),
                "sd_abs_err_to_exact": float(np.mean([float(row["sd_abs_err_to_exact"]) for row in group])),
                "cdf_at_true_mean": float(np.mean([float(row["cdf_at_true"]) for row in group])),
                "cdf_at_true_sd": float(np.std([float(row["cdf_at_true"]) for row in group])),
                "coverage90": float(np.mean([float(row["pi_coverage90"]) for row in group])),
            }
        )
    return sort_rows(out)


def ks_uniform(values: list[float]) -> float:
    vals = np.sort(np.asarray(values, dtype=np.float64))
    if vals.size == 0:
        return float("nan")
    n = vals.size
    upper = np.arange(1, n + 1, dtype=np.float64) / n
    lower = np.arange(0, n, dtype=np.float64) / n
    return float(np.max(np.maximum(np.abs(upper - vals), np.abs(vals - lower))))


def summarize_full_posterior_pooled(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if "w1_to_exact" not in row or str(row["method"]) == "exact likelihood grid":
            continue
        by_method.setdefault(str(row["method"]), []).append(row)
    out = []
    for method, group in by_method.items():
        cdf_values = [float(row["cdf_at_true"]) for row in group]
        out.append(
            {
                "method": method,
                "n": len(group),
                "mean_abs_err_to_exact": float(np.mean([float(row["mean_abs_err_to_exact"]) for row in group])),
                "rmse_mean_to_exact": float(math.sqrt(np.mean([float(row["mean_sq_err_to_exact"]) for row in group]))),
                "w1_to_exact": float(np.mean([float(row["w1_to_exact"]) for row in group])),
                "q05_abs_err_to_exact": float(np.mean([float(row["q05_abs_err_to_exact"]) for row in group])),
                "q50_abs_err_to_exact": float(np.mean([float(row["q50_abs_err_to_exact"]) for row in group])),
                "q95_abs_err_to_exact": float(np.mean([float(row["q95_abs_err_to_exact"]) for row in group])),
                "sd_abs_err_to_exact": float(np.mean([float(row["sd_abs_err_to_exact"]) for row in group])),
                "cdf_at_true_mean": float(np.mean(cdf_values)),
                "cdf_at_true_sd": float(np.std(cdf_values)),
                "cdf_at_true_ks_uniform": ks_uniform(cdf_values),
                "coverage90": float(np.mean([float(row["pi_coverage90"]) for row in group])),
            }
        )
    return sorted(out, key=lambda row: float(row["w1_to_exact"]))


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    data_device = resolve_device(args.data_device)

    NPE, posterior_nn, BoxUniform = import_sbi()
    print("torch:", torch.__version__)
    print("sbi device:", device)
    print("sbi data_device:", data_device)
    print("Output directory:", out_dir)

    stage1 = train_stage1(args, device)
    stage1_score_rows, stage1_decomposition_rows = stage1_score_diagnostics(stage1, args)

    rng_train = np.random.default_rng(int(args.sbi_train_seed))
    pi_train = sample_pi_prior(
        rng_train,
        int(args.n_sbi_train),
        float(args.pi_prior_min),
        float(args.pi_prior_max),
    )
    y_train, log_r_train = simulate_from_pi(
        rng_train,
        pi_train,
        int(args.n_blocks),
        int(args.block_size),
        float(args.tau),
    )
    feature_dict = compute_features(stage1, y_train, log_r_train, args)

    methods = [m.strip() for m in str(args.sbi_methods).split(",") if m.strip()]
    missing = [m for m in methods if m not in feature_dict]
    if missing:
        raise ValueError(f"Unknown or unavailable SBI methods: {missing}. Available: {sorted(feature_dict)}")

    results = []
    for idx, method in enumerate(methods):
        results.append(
            train_sbi_npe_method(
                NPE,
                posterior_nn,
                BoxUniform,
                method,
                feature_dict[method],
                pi_train,
                int(args.sbi_seed) + 173 * idx,
                args,
                device,
                data_device,
            )
        )

    rows: list[dict[str, object]] = []
    test_pi_values = (
        [float(x.strip()) for x in str(args.test_pi_values).split(",") if x.strip()]
        if args.test_pi_values
        else [float(args.test_pi)]
    )
    for pi_true in test_pi_values:
        if not (args.pi_prior_min <= pi_true <= args.pi_prior_max):
            raise ValueError(f"test pi={pi_true} must lie inside the SBI prior interval")
        for obs_seed in parse_int_list(args.test_seeds):
            # Use an independent generator per seed and pi for exact reproducibility.
            rng = np.random.default_rng(int(obs_seed))
            y_obs, log_r_obs = simulate_from_pi(
                rng,
                np.asarray([pi_true]),
                int(args.n_blocks),
                int(args.block_size),
                float(args.tau),
            )
            obs_features = compute_features(stage1, y_obs, log_r_obs, args)

            exact_grid, exact_w, exact_cdf = exact_posterior_grid_arrays(y_obs[0], args)
            exact = exact_posterior_grid(y_obs[0], args)
            exact_sq_err = (exact["mean"] - pi_true) ** 2
            exact_cdf_true = exact_cdf_at(pi_true, exact_grid, exact_cdf)
            rows.append(
                {
                    "method": "exact likelihood grid",
                    "seed": int(obs_seed),
                    "pi_true": pi_true,
                    "pi_post_mean": exact["mean"],
                    "pi_post_std": exact["std"],
                    "pi_q05": exact["q05"],
                    "pi_q50": exact["q50"],
                    "pi_q95": exact["q95"],
                    "pi_sq_err": float(exact_sq_err),
                    "pi_coverage90": float(exact["q05"] <= pi_true <= exact["q95"]),
                    "exact_pi_post_mean": exact["mean"],
                    "exact_pi_post_std": exact["std"],
                    "exact_pi_q05": exact["q05"],
                    "exact_pi_q50": exact["q50"],
                    "exact_pi_q95": exact["q95"],
                    "mean_abs_err_to_exact": 0.0,
                    "mean_sq_err_to_exact": 0.0,
                    "sd_abs_err_to_exact": 0.0,
                    "q05_abs_err_to_exact": 0.0,
                    "q50_abs_err_to_exact": 0.0,
                    "q95_abs_err_to_exact": 0.0,
                    "w1_to_exact": 0.0,
                    "cdf_at_true": exact_cdf_true,
                    "exact_cdf_at_true": exact_cdf_true,
                    "cdf_at_true_abs_err_to_exact": 0.0,
                }
            )

            for result in results:
                method = str(result["method"])
                samples = sample_sbi_posterior(
                    result,
                    obs_features[method][0],
                    int(args.posterior_n),
                    posterior_seed(method, int(obs_seed)),
                    args,
                )
                mean = float(np.mean(samples))
                std = float(np.std(samples))
                q05 = float(np.quantile(samples, 0.05))
                q50 = float(np.quantile(samples, 0.50))
                q95 = float(np.quantile(samples, 0.95))
                cdf_at_true = float(np.mean(np.asarray(samples).reshape(-1) <= pi_true))
                rows.append(
                    {
                        "method": method,
                        "seed": int(obs_seed),
                        "pi_true": pi_true,
                        "pi_post_mean": mean,
                        "pi_post_std": std,
                        "pi_q05": q05,
                        "pi_q50": q50,
                        "pi_q95": q95,
                        "pi_sq_err": float((mean - pi_true) ** 2),
                        "pi_coverage90": float(q05 <= pi_true <= q95),
                        "exact_pi_post_mean": exact["mean"],
                        "exact_pi_post_std": exact["std"],
                        "exact_pi_q05": exact["q05"],
                        "exact_pi_q50": exact["q50"],
                        "exact_pi_q95": exact["q95"],
                        "mean_abs_err_to_exact": abs(mean - exact["mean"]),
                        "mean_sq_err_to_exact": float((mean - exact["mean"]) ** 2),
                        "sd_abs_err_to_exact": abs(std - exact["std"]),
                        "q05_abs_err_to_exact": abs(q05 - exact["q05"]),
                        "q50_abs_err_to_exact": abs(q50 - exact["q50"]),
                        "q95_abs_err_to_exact": abs(q95 - exact["q95"]),
                        "w1_to_exact": exact_sample_w1(samples, exact_grid, exact_cdf),
                        "cdf_at_true": cdf_at_true,
                        "exact_cdf_at_true": exact_cdf_true,
                        "cdf_at_true_abs_err_to_exact": abs(cdf_at_true - exact_cdf_true),
                    }
                )

    summary = summarize(rows)
    full_by_pi = summarize_full_posterior(rows)
    full_pooled = summarize_full_posterior_pooled(rows)
    write_csv(out_dir / "posterior_by_seed.csv", sort_rows(rows))
    write_csv(out_dir / "posterior_summary.csv", summary)
    write_csv(out_dir / "posterior_full_metrics_by_pi.csv", full_by_pi)
    write_csv(out_dir / "posterior_full_metrics_pooled.csv", full_pooled)
    write_csv(out_dir / "stage1_training_trace.csv", stage1["trace_rows"])
    write_csv(out_dir / "stage1_score_summary_direct.csv", stage1_score_rows)
    write_csv(out_dir / "stage1_score_target_decomposition.csv", stage1_decomposition_rows)
    with (out_dir / "config.json").open("w") as file:
        json.dump(jsonable_args(args), file, indent=2)

    print("\n==== Posterior summary ====")
    print(f"{'pi_true':>8}  {'method':<34}{'pi_mse':>12}{'se':>10}{'post_sd':>10}{'cov90':>10}")
    print("-" * 86)
    for row in summary:
        print(
            f"{float(row['pi_true']):>8.3g}  "
            f"{row['method']:<34}"
            f"{float(row['pi_avg_mse']):>12.4g}"
            f"{float(row['pi_mse_se']):>10.3g}"
            f"{float(row['pi_avg_post_std']):>10.4g}"
            f"{float(row['pi_coverage90']):>10.3g}"
        )
    print("\nSaved:")
    print(out_dir / "posterior_by_seed.csv")
    print(out_dir / "posterior_summary.csv")
    print(out_dir / "posterior_full_metrics_by_pi.csv")
    print(out_dir / "posterior_full_metrics_pooled.csv")
    print(out_dir / "stage1_training_trace.csv")
    if stage1_score_rows:
        print(out_dir / "stage1_score_summary_direct.csv")
    if stage1_decomposition_rows:
        print(out_dir / "stage1_score_target_decomposition.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-blocks", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--tau", type=float, default=0.2)

    parser.add_argument("--center-pi", type=float, default=0.3)
    parser.add_argument("--sigma-q", type=float, default=0.5)
    parser.add_argument("--stage1-n-train", type=int, default=20_000)
    parser.add_argument("--stage1-n-val", type=int, default=2_000)
    parser.add_argument("--stage1-seed", type=int, default=20260626)
    parser.add_argument("--stage1-methods", type=str, default="linear_deepsets,radial_linear,radial_deepsets")
    parser.add_argument("--stage1-iters", type=int, default=2_000)
    parser.add_argument("--stage1-batch-size", type=int, default=512)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-patience", type=int, default=10)
    parser.add_argument("--stage1-print-every", type=int, default=100)
    parser.add_argument(
        "--stage1-diagnostic-n-test",
        type=int,
        default=0,
        help="If positive, evaluate frozen scalar Stage-1 scores against the exact local score on this many center samples.",
    )
    parser.add_argument("--stage1-diagnostic-seed", type=int, default=20260704)
    parser.add_argument(
        "--compute-stage1-smoothed-target-diagnostics",
        action="store_true",
        help="Also compute the exact one-dimensional FSM smoothed target and decompose Stage-1 score error.",
    )
    parser.add_argument("--stage1-smooth-grid-size", type=int, default=2001)
    parser.add_argument("--stage1-smooth-grid-width", type=float, default=8.0)
    parser.add_argument("--stage1-smooth-chunk-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument(
        "--clean-gate-kind",
        choices=["monotone_mlp", "positive_mlp", "log_polynomial"],
        default="monotone_mlp",
    )
    parser.add_argument(
        "--gate-input",
        choices=["abs", "signed"],
        default="abs",
        help="Input to radial multiplier: abs keeps the old odd map; signed allows asymmetric positive multiplier.",
    )
    parser.add_argument("--ridge", type=float, default=1e-6)

    parser.add_argument("--pi-prior-min", type=float, default=0.05)
    parser.add_argument("--pi-prior-max", type=float, default=0.7)
    parser.add_argument("--n-sbi-train", type=int, default=20_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260701)
    parser.add_argument(
        "--sbi-methods",
        type=str,
        default=(
            "raw data,raw ridge FSM score,linear block CS,linear CS ridge FSM score,"
            "linear CS DeepSets FSM score,"
            "MLP radial-gate CS linear FSM score,MLP radial-gate CS DeepSets FSM score"
        ),
    )
    parser.add_argument("--sbi-model", choices=("mdn", "maf", "nsf", "made"), default="mdn")
    parser.add_argument("--sbi-hidden-features", type=int, default=64)
    parser.add_argument("--sbi-num-components", type=int, default=5)
    parser.add_argument("--sbi-num-transforms", type=int, default=5)
    parser.add_argument("--sbi-num-bins", type=int, default=8)
    parser.add_argument("--sbi-batch-size", type=int, default=256)
    parser.add_argument("--sbi-lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--sbi-seed", type=int, default=52000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-device", type=str, default="auto")

    parser.add_argument("--test-pi", type=float, default=0.3)
    parser.add_argument(
        "--test-pi-values",
        type=str,
        default="",
        help="Optional comma-separated test pi values. Overrides --test-pi when nonempty.",
    )
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--test-seed-base", type=int, default=100)
    parser.add_argument("--posterior-n", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Blockwise Gaussian Mixture/runs/mean_shift_two_stage_sbi_npe"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.stage1_methods = [m.strip() for m in str(args.stage1_methods).split(",") if m.strip()]
    allowed = {"linear_deepsets", "radial_linear", "radial_deepsets"}
    bad = [m for m in args.stage1_methods if m not in allowed]
    if bad:
        raise ValueError(f"Unknown stage1 methods {bad}; allowed={sorted(allowed)}")
    if not (0.0 < args.center_pi < 1.0):
        raise ValueError("--center-pi must be in (0, 1)")
    if not (0.0 < args.pi_prior_min < args.pi_prior_max < 1.0):
        raise ValueError("Require 0 < --pi-prior-min < --pi-prior-max < 1")
    test_pi_values = (
        [float(x.strip()) for x in str(args.test_pi_values).split(",") if x.strip()]
        if args.test_pi_values
        else [float(args.test_pi)]
    )
    for pi_true in test_pi_values:
        if not (args.pi_prior_min <= pi_true <= args.pi_prior_max):
            raise ValueError("all test pi values must lie inside the SBI prior interval")
    if args.sigma_q <= 0:
        raise ValueError("--sigma-q must be positive")
    run(args)


if __name__ == "__main__":
    main()
