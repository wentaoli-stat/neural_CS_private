#!/usr/bin/env python3
"""Two-stage sbi-NPE for the HMM common-factor variance mixture.

Model 3:
    B_t in {0, 1}
    P(B_t=1 | B_{t-1}=0) = p01
    P(B_t=1 | B_{t-1}=1) = p11
    Y_t | B_t=0 ~ N(0, I_m)
    Y_t | B_t=1 ~ N(0, I_m + tau^2 11')

Stage 1 trains likelihood-free FSM summaries at a fixed center (p01, p11).
Stage 2 freezes those summaries, trains one sbi-NPE posterior per summary, and
compares posterior mean MSE against an exact HMM likelihood grid reference.
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

import run_hmm_common_factor_fsm_experiment as fsm


PARAM_NAMES = ("p01", "p11")
METHOD_ORDER = [
    "exact likelihood grid",
    "raw data",
    "raw ridge FSM score",
    "raw GRU FSM score",
    "time-indexed linear CS",
    "linear CS ridge FSM score",
    "linear CS time-deepsets FSM score",
    "linear CS GRU FSM score",
    "nested structured sequential CS GRU FSM score",
    "structured sequential CS GRU FSM score",
    "positive-MLP radial-gate CS linear FSM score",
    "positive-MLP radial-gate CS time-deepsets FSM score",
    "MLP radial-gate CS linear FSM score",
    "MLP radial-gate CS time-deepsets FSM score",
    "polynomial radial-gate CS linear FSM score",
    "polynomial radial-gate CS time-deepsets FSM score",
]


def import_sbi():
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
        raise ImportError("This script requires `sbi`. Install it with `pip install sbi`.") from exc
    return NPE, posterior_nn, BoxUniform


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_pair(text: str) -> np.ndarray:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(values) != 2:
        raise ValueError("Expected exactly two comma-separated values.")
    return np.asarray(values, dtype=np.float64)


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
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def params_to_unit(theta: np.ndarray, args: argparse.Namespace, eps: float = 1e-5) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float32).reshape(-1, 2)
    mins = np.asarray([args.p01_prior_min, args.p11_prior_min], dtype=np.float32)
    maxs = np.asarray([args.p01_prior_max, args.p11_prior_max], dtype=np.float32)
    out = (theta - mins[None, :]) / (maxs - mins)[None, :]
    return np.clip(out, eps, 1.0 - eps).astype(np.float32)


def unit_to_params(unit: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    unit = np.asarray(unit, dtype=np.float32).reshape(-1, 2)
    mins = np.asarray([args.p01_prior_min, args.p11_prior_min], dtype=np.float32)
    maxs = np.asarray([args.p01_prior_max, args.p11_prior_max], dtype=np.float32)
    return (mins[None, :] + (maxs - mins)[None, :] * unit).astype(np.float32)


def make_prior(BoxUniform, device: str):
    torch_device = torch.device(device)
    return BoxUniform(
        low=torch.zeros(2, dtype=torch.float32, device=torch_device),
        high=torch.ones(2, dtype=torch.float32, device=torch_device),
    )


def sample_param_prior(rng: np.random.Generator, n: int, args: argparse.Namespace) -> np.ndarray:
    p01 = rng.uniform(float(args.p01_prior_min), float(args.p01_prior_max), size=int(n))
    p11 = rng.uniform(float(args.p11_prior_min), float(args.p11_prior_max), size=int(n))
    return np.stack([p01, p11], axis=1).astype(np.float64)


def params_to_u(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1, 2)
    return np.stack([np.vectorize(fsm.logit)(theta[:, 0]), np.vectorize(fsm.logit)(theta[:, 1])], axis=1)


def simulate_from_params(
    rng: np.random.Generator,
    theta: np.ndarray,
    length: int,
    block_size: int,
    tau: float,
    init_prob: float,
) -> np.ndarray:
    y, _ = fsm.simulate_hmm_common_factor(
        rng,
        params_to_u(theta),
        int(length),
        int(block_size),
        float(tau),
        float(init_prob),
    )
    return y


def zscore_fit(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x, dtype=np.float64).mean(axis=0, keepdims=True)
    sd = np.asarray(x, dtype=np.float64).std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return mean, sd


def zscore_apply(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - mean) / sd


def standardize_scalar_fit(x: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    mean = float(np.asarray(x, dtype=np.float64).mean())
    sd = float(np.asarray(x, dtype=np.float64).std())
    if sd < eps:
        sd = 1.0
    return mean, sd


def make_stage1_arrays(
    y: np.ndarray,
    pi_ref: float,
    tau: float,
    s1_mean: float,
    s1_sd: float,
    s2_mean: float,
    s2_sd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s1 = fsm.local_score_u_from_log_ratio(fsm.marginal_log_ratio(y, tau), pi_ref)
    s2 = fsm.local_score_u_from_log_ratio(fsm.pairwise_log_ratio(y, tau), pi_ref)
    s1_std = ((s1 - s1_mean) / s1_sd).astype(np.float32)
    s2_std = ((s2 - s2_mean) / s2_sd).astype(np.float32)
    time_cs = np.stack([s1_std.sum(axis=2), s2_std.sum(axis=2)], axis=2).astype(np.float32)
    global_cs = time_cs.sum(axis=1).astype(np.float32)
    time_flat = time_cs.reshape(y.shape[0], -1).astype(np.float32)
    struct = np.concatenate([s1_std, s2_std], axis=2).astype(np.float32)
    return struct, time_cs, time_flat, global_cs


def init_probability(args: argparse.Namespace) -> float:
    if float(args.init_prob) >= 0.0:
        return float(args.init_prob)
    return fsm.fixed_stationary_prob(float(args.p01), float(args.p11))


def train_stage1(args: argparse.Namespace, device: str) -> dict[str, object]:
    rng = np.random.default_rng(int(args.stage1_seed))
    torch.manual_seed(int(args.stage1_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.stage1_seed))

    u0 = np.asarray([fsm.logit(float(args.p01)), fsm.logit(float(args.p11))], dtype=np.float64)
    sigma_q = parse_float_pair(args.sigma_q_values) if args.sigma_q_values else np.asarray([args.sigma_q, args.sigma_q], dtype=np.float64)
    init_prob = init_probability(args)

    print("\n==== Stage 1: HMM FSM summaries ====")
    print(f"center p01={args.p01}, p11={args.p11}, u0={u0.tolist()}, tau={args.tau}")
    print(f"sigma_q={sigma_q.tolist()}, T={args.length}, block_size={args.block_size}, init_prob={init_prob:.4f}")

    y_train, target_train = fsm.make_fsm_data(
        rng,
        int(args.stage1_n_train),
        u0,
        sigma_q,
        int(args.length),
        int(args.block_size),
        float(args.tau),
        init_prob,
    )
    y_val, target_val = fsm.make_fsm_data(
        rng,
        int(args.stage1_n_val),
        u0,
        sigma_q,
        int(args.length),
        int(args.block_size),
        float(args.tau),
        init_prob,
    )

    s1_train = fsm.local_score_u_from_log_ratio(fsm.marginal_log_ratio(y_train, args.tau), float(args.pi_ref))
    s2_train = fsm.local_score_u_from_log_ratio(fsm.pairwise_log_ratio(y_train, args.tau), float(args.pi_ref))
    s1_mean, s1_sd = standardize_scalar_fit(s1_train)
    s2_mean, s2_sd = standardize_scalar_fit(s2_train)

    struct_train, time_train, time_flat_train, global_train = make_stage1_arrays(
        y_train, float(args.pi_ref), float(args.tau), s1_mean, s1_sd, s2_mean, s2_sd
    )
    struct_val, time_val, time_flat_val, global_val = make_stage1_arrays(
        y_val, float(args.pi_ref), float(args.tau), s1_mean, s1_sd, s2_mean, s2_sd
    )

    raw_train_flat = y_train.reshape(int(args.stage1_n_train), -1)
    raw_val_flat = y_val.reshape(int(args.stage1_n_val), -1)
    raw_mean, raw_sd = zscore_fit(raw_train_flat)
    raw_train = zscore_apply(raw_train_flat, raw_mean, raw_sd).astype(np.float32)
    raw_val = zscore_apply(raw_val_flat, raw_mean, raw_sd).astype(np.float32)
    raw_seq_train = raw_train.reshape(int(args.stage1_n_train), int(args.length), int(args.block_size))
    raw_seq_val = raw_val.reshape(int(args.stage1_n_val), int(args.length), int(args.block_size))

    raw_coef = fsm.fit_linear_ridge(raw_train, target_train, ridge=float(args.ridge))
    global_coef = fsm.fit_linear_ridge(global_train, target_train, ridge=float(args.ridge))
    time_coef = fsm.fit_linear_ridge(time_flat_train, target_train, ridge=float(args.ridge))

    models: dict[str, torch.nn.Module] = {}
    traces: list[dict[str, object]] = []

    if "linear_deepsets" in args.stage1_methods:
        model, trace = fsm.train_torch_model(
            fsm.TimeDeepSets(2, int(args.hidden), int(args.depth)),
            time_train,
            target_train,
            time_val,
            target_val,
            iters=int(args.stage1_iters),
            batch_size=int(args.stage1_batch_size),
            lr=float(args.stage1_lr),
            weight_decay=float(args.stage1_weight_decay),
            patience=int(args.stage1_patience),
            print_every=int(args.stage1_print_every),
            device=device,
            name="linear CS time-deepsets FSM",
        )
        models["linear_deepsets"] = model
        traces.extend({"method": "linear CS time-deepsets FSM score", **row} for row in trace)

    if "raw_gru" in args.stage1_methods:
        model, trace = fsm.train_torch_model(
            fsm.RawSequenceGRU(int(args.block_size), int(args.gru_hidden)),
            raw_seq_train,
            target_train,
            raw_seq_val,
            target_val,
            iters=int(args.stage1_iters),
            batch_size=int(args.stage1_batch_size),
            lr=float(args.stage1_lr),
            weight_decay=float(args.stage1_weight_decay),
            patience=int(args.stage1_patience),
            print_every=int(args.stage1_print_every),
            device=device,
            name="raw GRU FSM",
        )
        models["raw_gru"] = model
        traces.extend({"method": "raw GRU FSM score", **row} for row in trace)

    if "linear_gru" in args.stage1_methods:
        model, trace = fsm.train_torch_model(
            fsm.LinearCSGRU(2, int(args.gru_hidden)),
            time_train,
            target_train,
            time_val,
            target_val,
            iters=int(args.stage1_iters),
            batch_size=int(args.stage1_batch_size),
            lr=float(args.stage1_lr),
            weight_decay=float(args.stage1_weight_decay),
            patience=int(args.stage1_patience),
            print_every=int(args.stage1_print_every),
            device=device,
            name="linear CS GRU FSM",
        )
        models["linear_gru"] = model
        traces.extend({"method": "linear CS GRU FSM score", **row} for row in trace)

    gate_label = fsm.gate_label_for(str(args.clean_gate_kind))
    if "radial_linear" in args.stage1_methods:
        model, trace = fsm.train_torch_model(
            fsm.RadialGateEmissionLinearHead(
                int(args.length),
                int(args.block_size),
                int(args.gate_hidden),
                time_coef,
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
        models["radial_linear"] = model
        traces.extend({"method": f"{gate_label} radial-gate CS linear FSM score", **row} for row in trace)

    if "radial_deepsets" in args.stage1_methods:
        model, trace = fsm.train_torch_model(
            fsm.RadialGateEmissionDeepSets(
                int(args.block_size),
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
            name=f"{gate_label} radial-gate CS time-deepsets FSM",
        )
        models["radial_deepsets"] = model
        traces.extend({"method": f"{gate_label} radial-gate CS time-deepsets FSM score", **row} for row in trace)

    if "nested_structured_gru" in args.stage1_methods:
        model, trace = fsm.train_torch_model(
            fsm.NestedStructuredEmissionGRU(
                int(args.length),
                int(args.block_size),
                int(args.embed_dim),
                int(args.gru_hidden),
                time_coef,
                residual_scale_init=0.0,
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
            name="nested structured sequential CS GRU FSM",
        )
        models["nested_structured_gru"] = model
        traces.extend({"method": "nested structured sequential CS GRU FSM score", **row} for row in trace)

    if "structured_gru" in args.stage1_methods:
        model, trace = fsm.train_torch_model(
            fsm.StructuredEmissionGRU(int(args.block_size), int(args.embed_dim), int(args.gru_hidden)),
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
            name="structured sequential CS GRU FSM",
        )
        models["structured_gru"] = model
        traces.extend({"method": "structured sequential CS GRU FSM score", **row} for row in trace)

    return {
        "init_prob": init_prob,
        "raw_mean": raw_mean,
        "raw_sd": raw_sd,
        "s1_mean": s1_mean,
        "s1_sd": s1_sd,
        "s2_mean": s2_mean,
        "s2_sd": s2_sd,
        "raw_coef": raw_coef,
        "global_coef": global_coef,
        "time_coef": time_coef,
        "models": models,
        "trace_rows": traces,
        "device": device,
        "gate_label": gate_label,
    }


def compute_features(stage1: dict[str, object], y: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    struct, time_cs, time_flat, global_cs = make_stage1_arrays(
        y,
        float(args.pi_ref),
        float(args.tau),
        float(stage1["s1_mean"]),
        float(stage1["s1_sd"]),
        float(stage1["s2_mean"]),
        float(stage1["s2_sd"]),
    )
    raw_flat = y.reshape(y.shape[0], -1)
    raw_z = zscore_apply(raw_flat, stage1["raw_mean"], stage1["raw_sd"]).astype(np.float32)
    raw_seq = raw_z.reshape(y.shape[0], int(args.length), int(args.block_size))

    features: dict[str, np.ndarray] = {
        "raw data": raw_flat.astype(np.float32),
        "raw ridge FSM score": fsm.predict_linear(stage1["raw_coef"], raw_z).astype(np.float32),
        "time-indexed linear CS": time_flat.astype(np.float32),
        "linear CS ridge FSM score": fsm.predict_linear(stage1["time_coef"], time_flat).astype(np.float32),
    }

    models = stage1["models"]
    device = str(stage1["device"])
    gate_label = str(stage1.get("gate_label", "MLP"))
    if "linear_deepsets" in models:
        features["linear CS time-deepsets FSM score"] = fsm.predict_torch(
            models["linear_deepsets"], time_cs, device, int(args.stage1_batch_size)
        ).astype(np.float32)
    if "raw_gru" in models:
        features["raw GRU FSM score"] = fsm.predict_torch(
            models["raw_gru"], raw_seq, device, int(args.stage1_batch_size)
        ).astype(np.float32)
    if "linear_gru" in models:
        features["linear CS GRU FSM score"] = fsm.predict_torch(
            models["linear_gru"], time_cs, device, int(args.stage1_batch_size)
        ).astype(np.float32)
    if "radial_linear" in models:
        features[f"{gate_label} radial-gate CS linear FSM score"] = fsm.predict_torch(
            models["radial_linear"], struct, device, int(args.stage1_batch_size)
        ).astype(np.float32)
    if "radial_deepsets" in models:
        features[f"{gate_label} radial-gate CS time-deepsets FSM score"] = fsm.predict_torch(
            models["radial_deepsets"], struct, device, int(args.stage1_batch_size)
        ).astype(np.float32)
    if "nested_structured_gru" in models:
        features["nested structured sequential CS GRU FSM score"] = fsm.predict_torch(
            models["nested_structured_gru"], struct, device, int(args.stage1_batch_size)
        ).astype(np.float32)
    if "structured_gru" in models:
        features["structured sequential CS GRU FSM score"] = fsm.predict_torch(
            models["structured_gru"], struct, device, int(args.stage1_batch_size)
        ).astype(np.float32)
    return features


def stage1_score_diagnostics(stage1: dict[str, object], args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate frozen scalar Stage-1 scores against the exact center HMM score.

    This is diagnostic only. FSM training remains likelihood-free.
    """
    if int(args.stage1_diagnostic_n_test) <= 0:
        return [], []
    rng = np.random.default_rng(int(args.stage1_diagnostic_seed))
    u0 = np.asarray([fsm.logit(float(args.p01)), fsm.logit(float(args.p11))], dtype=np.float64)
    y_eval, _ = fsm.simulate_center(
        rng,
        int(args.stage1_diagnostic_n_test),
        u0,
        int(args.length),
        int(args.block_size),
        float(args.tau),
        float(stage1["init_prob"]),
    )
    exact_score = fsm.hmm_transition_score_u(
        fsm.full_emission_log_ratio(y_eval, float(args.tau)),
        float(args.p01),
        float(args.p11),
        float(stage1["init_prob"]),
    )
    features = compute_features(stage1, y_eval, args)

    summary_rows: list[dict[str, object]] = []
    by_param_rows: list[dict[str, object]] = []
    for method, values in features.items():
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(PARAM_NAMES):
            continue
        row: dict[str, object] = {
            "method": method,
            "n_test": int(args.stage1_diagnostic_n_test),
            "p01": float(args.p01),
            "p11": float(args.p11),
            "tau": float(args.tau),
            "length": int(args.length),
            "block_size": int(args.block_size),
            "sigma_q": str(args.sigma_q_values or args.sigma_q),
        }
        row.update(fsm.summary_metrics(exact_score, values))
        summary_rows.append(row)
        by_param_rows.extend(fsm.by_param_metrics(exact_score, values, method))
    return sorted(summary_rows, key=lambda row: float(row["std_mse"])), by_param_rows


def hmm_loglik_grid(y: np.ndarray, p01_grid: np.ndarray, p11_grid: np.ndarray, tau: float, init_prob: float) -> np.ndarray:
    e1 = fsm.full_emission_log_ratio(y[None, :, :], float(tau))[0]
    grid_shape = p01_grid.shape
    p01 = p01_grid.reshape(-1)
    p11 = p11_grid.reshape(-1)

    log_p00 = np.log1p(-p01)
    log_p01 = np.log(p01)
    log_p10 = np.log1p(-p11)
    log_p11 = np.log(p11)
    alpha0 = np.full_like(p01, math.log1p(-init_prob), dtype=np.float64)
    alpha1 = np.full_like(p01, math.log(init_prob) + e1[0], dtype=np.float64)
    for t in range(1, e1.shape[0]):
        new0 = np.logaddexp(alpha0 + log_p00, alpha1 + log_p10)
        new1 = e1[t] + np.logaddexp(alpha0 + log_p01, alpha1 + log_p11)
        alpha0, alpha1 = new0, new1
    return np.logaddexp(alpha0, alpha1).reshape(grid_shape)


def exact_posterior_grid_arrays(
    y: np.ndarray,
    args: argparse.Namespace,
    init_prob: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p01_axis = np.linspace(float(args.p01_prior_min), float(args.p01_prior_max), int(args.grid_size))
    p11_axis = np.linspace(float(args.p11_prior_min), float(args.p11_prior_max), int(args.grid_size))
    p01_grid, p11_grid = np.meshgrid(p01_axis, p11_axis, indexing="ij")
    logw = hmm_loglik_grid(y, p01_grid, p11_grid, float(args.tau), float(init_prob))
    logw -= np.max(logw)
    w = np.exp(logw)
    w /= np.sum(w)
    cdf_p01 = np.cumsum(w.sum(axis=1))
    cdf_p11 = np.cumsum(w.sum(axis=0))
    return p01_axis, p11_axis, w, cdf_p01, cdf_p11, logw


def exact_posterior_grid(y: np.ndarray, args: argparse.Namespace, init_prob: float) -> dict[str, float]:
    p01_axis, p11_axis, w, cdf_p01, cdf_p11, _ = exact_posterior_grid_arrays(y, args, init_prob)
    p01_grid, p11_grid = np.meshgrid(p01_axis, p11_axis, indexing="ij")
    mean_p01 = float(np.sum(w * p01_grid))
    mean_p11 = float(np.sum(w * p11_grid))
    std_p01 = float(np.sqrt(np.sum(w * (p01_grid - mean_p01) ** 2)))
    std_p11 = float(np.sqrt(np.sum(w * (p11_grid - mean_p11) ** 2)))
    return {
        "p01_mean": mean_p01,
        "p11_mean": mean_p11,
        "p01_std": std_p01,
        "p11_std": std_p11,
        "p01_q05": float(np.interp(0.05, cdf_p01, p01_axis)),
        "p01_q50": float(np.interp(0.50, cdf_p01, p01_axis)),
        "p01_q95": float(np.interp(0.95, cdf_p01, p01_axis)),
        "p11_q05": float(np.interp(0.05, cdf_p11, p11_axis)),
        "p11_q50": float(np.interp(0.50, cdf_p11, p11_axis)),
        "p11_q95": float(np.interp(0.95, cdf_p11, p11_axis)),
    }


def exact_marginal_w1(samples: np.ndarray, axis: np.ndarray, cdf: np.ndarray) -> float:
    samples = np.sort(np.asarray(samples, dtype=np.float64).reshape(-1))
    if samples.size == 0:
        return float("nan")
    probs = (np.arange(samples.size, dtype=np.float64) + 0.5) / samples.size
    exact_quantiles = np.interp(probs, cdf, axis)
    return float(np.mean(np.abs(samples - exact_quantiles)))


def exact_cdf_at(x: float, axis: np.ndarray, cdf: np.ndarray) -> float:
    return float(np.interp(float(x), axis, cdf, left=0.0, right=1.0))


def train_sbi_npe_method(
    NPE,
    posterior_nn,
    BoxUniform,
    method: str,
    x_np: np.ndarray,
    theta_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: str,
    data_device: str,
) -> dict[str, object]:
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    theta = torch.as_tensor(params_to_unit(theta_train, args), dtype=torch.float32)
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


def sample_sbi_posterior(result: dict[str, object], x_obs: np.ndarray, posterior_n: int, seed: int, args: argparse.Namespace) -> np.ndarray:
    torch.manual_seed(int(seed))
    if str(result["device"]).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    x = torch.as_tensor(np.asarray(x_obs, dtype=np.float32), dtype=torch.float32, device=torch.device(str(result["device"])))
    samples_unit = result["posterior"].sample((int(posterior_n),), x=x)
    return unit_to_params(samples_unit.detach().cpu().numpy(), args)


def posterior_seed(method: str, obs_seed: int) -> int:
    offsets = {method: 1000 + 137 * idx for idx, method in enumerate(METHOD_ORDER)}
    if method in offsets:
        return offsets[method] + int(obs_seed)
    digest = hashlib.blake2b(method.encode("utf-8"), digest_size=4).hexdigest()
    return 10_000 + (int(digest, 16) % 1_000_000) + int(obs_seed)


def sort_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    order = {method: idx for idx, method in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: (order.get(str(row["method"]), len(order)), int(row.get("seed", 0))))


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    out = []
    for method, group in by_method.items():
        p01_mse = np.asarray([float(row["p01_sq_err"]) for row in group])
        p11_mse = np.asarray([float(row["p11_sq_err"]) for row in group])
        avg_mse = (p01_mse + p11_mse) / 2.0
        out.append(
            {
                "method": method,
                "n_seeds": len(group),
                "p01_avg_mse": float(np.mean(p01_mse)),
                "p11_avg_mse": float(np.mean(p11_mse)),
                "avg_mse": float(np.mean(avg_mse)),
                "avg_mse_se": standard_error(avg_mse),
                "p01_avg_post_std": float(np.mean([float(row["p01_post_std"]) for row in group])),
                "p11_avg_post_std": float(np.mean([float(row["p11_post_std"]) for row in group])),
                "joint_marginal_coverage90": float(np.mean([float(row["joint_marginal_coverage90"]) for row in group])),
            }
        )
    return sorted(out, key=lambda row: METHOD_ORDER.index(str(row["method"])) if str(row["method"]) in METHOD_ORDER else len(METHOD_ORDER))


def ks_uniform(values: list[float]) -> float:
    vals = np.sort(np.asarray(values, dtype=np.float64))
    if vals.size == 0:
        return float("nan")
    n = vals.size
    upper = np.arange(1, n + 1, dtype=np.float64) / n
    lower = np.arange(0, n, dtype=np.float64) / n
    return float(np.max(np.maximum(np.abs(upper - vals), np.abs(vals - lower))))


def summarize_full_posterior(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if "w1_to_exact" not in row or str(row["method"]) == "exact likelihood grid":
            continue
        by_method.setdefault(str(row["method"]), []).append(row)
    out = []
    for method, group in by_method.items():
        cdf_p01 = [float(row["p01_cdf_at_true"]) for row in group]
        cdf_p11 = [float(row["p11_cdf_at_true"]) for row in group]
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
                "cdf_at_true_mean": float(np.mean(cdf_p01 + cdf_p11)),
                "cdf_at_true_sd": float(np.std(cdf_p01 + cdf_p11)),
                "cdf_at_true_ks_uniform": ks_uniform(cdf_p01 + cdf_p11),
                "joint_marginal_coverage90": float(np.mean([float(row["joint_marginal_coverage90"]) for row in group])),
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
    stage1_score_rows, stage1_by_param_rows = stage1_score_diagnostics(stage1, args)
    init_prob = float(stage1["init_prob"])

    rng_train = np.random.default_rng(int(args.sbi_train_seed))
    theta_train = sample_param_prior(rng_train, int(args.n_sbi_train), args)
    y_train = simulate_from_params(
        rng_train, theta_train, int(args.length), int(args.block_size), float(args.tau), init_prob
    )
    feature_dict = compute_features(stage1, y_train, args)

    methods = [m.strip() for m in str(args.sbi_methods).split(",") if m.strip()]
    missing = [m for m in methods if m not in feature_dict]
    if missing:
        raise ValueError(f"Unknown or unavailable SBI methods: {missing}. Available: {sorted(feature_dict)}")

    results = []
    for idx, method in enumerate(methods):
        results.append(
            train_sbi_npe_method(
                NPE, posterior_nn, BoxUniform, method, feature_dict[method], theta_train,
                int(args.sbi_seed) + 173 * idx, args, device, data_device,
            )
        )

    rows: list[dict[str, object]] = []
    theta_true = np.asarray([[float(args.test_p01), float(args.test_p11)]], dtype=np.float64)
    for obs_seed in parse_int_list(args.test_seeds):
        rng = np.random.default_rng(int(obs_seed))
        y_obs = simulate_from_params(rng, theta_true, int(args.length), int(args.block_size), float(args.tau), init_prob)
        obs_features = compute_features(stage1, y_obs, args)

        p01_axis, p11_axis, exact_w, exact_cdf_p01, exact_cdf_p11, _ = exact_posterior_grid_arrays(
            y_obs[0], args, init_prob
        )
        exact = exact_posterior_grid(y_obs[0], args, init_prob)
        exact_p01_cdf_true = exact_cdf_at(float(args.test_p01), p01_axis, exact_cdf_p01)
        exact_p11_cdf_true = exact_cdf_at(float(args.test_p11), p11_axis, exact_cdf_p11)
        rows.append(
            {
                "method": "exact likelihood grid",
                "seed": int(obs_seed),
                "p01_true": float(args.test_p01),
                "p11_true": float(args.test_p11),
                "p01_post_mean": exact["p01_mean"],
                "p11_post_mean": exact["p11_mean"],
                "p01_post_std": exact["p01_std"],
                "p11_post_std": exact["p11_std"],
                "p01_q05": exact["p01_q05"],
                "p01_q50": exact["p01_q50"],
                "p01_q95": exact["p01_q95"],
                "p11_q05": exact["p11_q05"],
                "p11_q50": exact["p11_q50"],
                "p11_q95": exact["p11_q95"],
                "p01_sq_err": float((exact["p01_mean"] - float(args.test_p01)) ** 2),
                "p11_sq_err": float((exact["p11_mean"] - float(args.test_p11)) ** 2),
                "joint_marginal_coverage90": float(
                    exact["p01_q05"] <= float(args.test_p01) <= exact["p01_q95"]
                    and exact["p11_q05"] <= float(args.test_p11) <= exact["p11_q95"]
                ),
                "exact_p01_post_mean": exact["p01_mean"],
                "exact_p11_post_mean": exact["p11_mean"],
                "exact_p01_post_std": exact["p01_std"],
                "exact_p11_post_std": exact["p11_std"],
                "exact_p01_q05": exact["p01_q05"],
                "exact_p01_q50": exact["p01_q50"],
                "exact_p01_q95": exact["p01_q95"],
                "exact_p11_q05": exact["p11_q05"],
                "exact_p11_q50": exact["p11_q50"],
                "exact_p11_q95": exact["p11_q95"],
                "mean_abs_err_to_exact": 0.0,
                "mean_sq_err_to_exact": 0.0,
                "sd_abs_err_to_exact": 0.0,
                "q05_abs_err_to_exact": 0.0,
                "q50_abs_err_to_exact": 0.0,
                "q95_abs_err_to_exact": 0.0,
                "w1_to_exact": 0.0,
                "p01_cdf_at_true": exact_p01_cdf_true,
                "p11_cdf_at_true": exact_p11_cdf_true,
                "exact_p01_cdf_at_true": exact_p01_cdf_true,
                "exact_p11_cdf_at_true": exact_p11_cdf_true,
            }
        )

        for result in results:
            method = str(result["method"])
            samples = sample_sbi_posterior(result, obs_features[method][0], int(args.posterior_n), posterior_seed(method, int(obs_seed)), args)
            means = samples.mean(axis=0)
            stds = samples.std(axis=0)
            q05 = np.quantile(samples, 0.05, axis=0)
            q50 = np.quantile(samples, 0.50, axis=0)
            q95 = np.quantile(samples, 0.95, axis=0)
            p01_cdf_true = float(np.mean(samples[:, 0] <= float(args.test_p01)))
            p11_cdf_true = float(np.mean(samples[:, 1] <= float(args.test_p11)))
            rows.append(
                {
                    "method": method,
                    "seed": int(obs_seed),
                    "p01_true": float(args.test_p01),
                    "p11_true": float(args.test_p11),
                    "p01_post_mean": float(means[0]),
                    "p11_post_mean": float(means[1]),
                    "p01_post_std": float(stds[0]),
                    "p11_post_std": float(stds[1]),
                    "p01_q05": float(q05[0]),
                    "p01_q50": float(q50[0]),
                    "p01_q95": float(q95[0]),
                    "p11_q05": float(q05[1]),
                    "p11_q50": float(q50[1]),
                    "p11_q95": float(q95[1]),
                    "p01_sq_err": float((means[0] - float(args.test_p01)) ** 2),
                    "p11_sq_err": float((means[1] - float(args.test_p11)) ** 2),
                    "joint_marginal_coverage90": float(
                        q05[0] <= float(args.test_p01) <= q95[0]
                        and q05[1] <= float(args.test_p11) <= q95[1]
                    ),
                    "exact_p01_post_mean": exact["p01_mean"],
                    "exact_p11_post_mean": exact["p11_mean"],
                    "exact_p01_post_std": exact["p01_std"],
                    "exact_p11_post_std": exact["p11_std"],
                    "exact_p01_q05": exact["p01_q05"],
                    "exact_p01_q50": exact["p01_q50"],
                    "exact_p01_q95": exact["p01_q95"],
                    "exact_p11_q05": exact["p11_q05"],
                    "exact_p11_q50": exact["p11_q50"],
                    "exact_p11_q95": exact["p11_q95"],
                    "mean_abs_err_to_exact": float(np.mean(np.abs(means - np.asarray([exact["p01_mean"], exact["p11_mean"]])))),
                    "mean_sq_err_to_exact": float(np.mean((means - np.asarray([exact["p01_mean"], exact["p11_mean"]])) ** 2)),
                    "sd_abs_err_to_exact": float(np.mean(np.abs(stds - np.asarray([exact["p01_std"], exact["p11_std"]])))),
                    "q05_abs_err_to_exact": float(np.mean(np.abs(q05 - np.asarray([exact["p01_q05"], exact["p11_q05"]])))),
                    "q50_abs_err_to_exact": float(np.mean(np.abs(q50 - np.asarray([exact["p01_q50"], exact["p11_q50"]])))),
                    "q95_abs_err_to_exact": float(np.mean(np.abs(q95 - np.asarray([exact["p01_q95"], exact["p11_q95"]])))),
                    "w1_to_exact": float(
                        np.mean(
                            [
                                exact_marginal_w1(samples[:, 0], p01_axis, exact_cdf_p01),
                                exact_marginal_w1(samples[:, 1], p11_axis, exact_cdf_p11),
                            ]
                        )
                    ),
                    "p01_cdf_at_true": p01_cdf_true,
                    "p11_cdf_at_true": p11_cdf_true,
                    "exact_p01_cdf_at_true": exact_p01_cdf_true,
                    "exact_p11_cdf_at_true": exact_p11_cdf_true,
                }
            )

    summary = summarize(rows)
    full_summary = summarize_full_posterior(rows)
    write_csv(out_dir / "posterior_by_seed.csv", sort_rows(rows))
    write_csv(out_dir / "posterior_summary.csv", summary)
    write_csv(out_dir / "posterior_full_metrics_pooled.csv", full_summary)
    write_csv(out_dir / "stage1_training_trace.csv", stage1["trace_rows"])
    write_csv(out_dir / "stage1_score_summary_direct.csv", stage1_score_rows)
    write_csv(out_dir / "stage1_score_by_param_direct.csv", stage1_by_param_rows)
    with (out_dir / "config.json").open("w") as file:
        json.dump(jsonable_args(args), file, indent=2)

    print("\n==== Posterior summary ====")
    print(f"{'method':<42}{'p01_mse':>11}{'p11_mse':>11}{'avg_mse':>11}{'se':>10}{'cov90':>9}")
    print("-" * 94)
    for row in summary:
        print(
            f"{row['method']:<42}"
            f"{float(row['p01_avg_mse']):>11.4g}"
            f"{float(row['p11_avg_mse']):>11.4g}"
            f"{float(row['avg_mse']):>11.4g}"
            f"{float(row['avg_mse_se']):>10.3g}"
            f"{float(row['joint_marginal_coverage90']):>9.3g}"
        )
    print("\nSaved:")
    print(out_dir / "posterior_by_seed.csv")
    print(out_dir / "posterior_summary.csv")
    print(out_dir / "posterior_full_metrics_pooled.csv")
    print(out_dir / "stage1_training_trace.csv")
    if stage1_score_rows:
        print(out_dir / "stage1_score_summary_direct.csv")
        print(out_dir / "stage1_score_by_param_direct.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--p01", type=float, default=0.06)
    parser.add_argument("--p11", type=float, default=0.94)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--pi-ref", type=float, default=0.3)
    parser.add_argument("--init-prob", type=float, default=-1.0, help="Negative means fixed stationary prob at center.")
    parser.add_argument("--sigma-q", type=float, default=0.4)
    parser.add_argument("--sigma-q-values", type=str, default="0.4,0.4")
    parser.add_argument("--stage1-n-train", type=int, default=20_000)
    parser.add_argument("--stage1-n-val", type=int, default=4_000)
    parser.add_argument("--stage1-seed", type=int, default=20260702)
    parser.add_argument("--stage1-methods", type=str, default="linear_deepsets,radial_linear,radial_deepsets")
    parser.add_argument("--stage1-iters", type=int, default=3_000)
    parser.add_argument("--stage1-batch-size", type=int, default=512)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-patience", type=int, default=30)
    parser.add_argument("--stage1-print-every", type=int, default=100)
    parser.add_argument(
        "--stage1-diagnostic-n-test",
        type=int,
        default=0,
        help="If positive, evaluate frozen scalar Stage-1 scores against the exact center HMM score.",
    )
    parser.add_argument("--stage1-diagnostic-seed", type=int, default=20260704)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument(
        "--clean-gate-kind",
        choices=["monotone_mlp", "positive_mlp", "log_polynomial"],
        default="positive_mlp",
    )
    parser.add_argument(
        "--gate-input",
        choices=["abs", "signed"],
        default="signed",
        help="Input to radial multiplier: abs keeps the old odd map; signed allows asymmetric positive multipliers.",
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--p01-prior-min", type=float, default=0.01)
    parser.add_argument("--p01-prior-max", type=float, default=0.20)
    parser.add_argument("--p11-prior-min", type=float, default=0.80)
    parser.add_argument("--p11-prior-max", type=float, default=0.99)
    parser.add_argument("--n-sbi-train", type=int, default=20_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260703)
    parser.add_argument(
        "--sbi-methods",
        type=str,
        default=(
            "raw data,raw ridge FSM score,"
            "time-indexed linear CS,linear CS ridge FSM score,linear CS time-deepsets FSM score,"
            "positive-MLP radial-gate CS linear FSM score,"
            "positive-MLP radial-gate CS time-deepsets FSM score"
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
    parser.add_argument("--sbi-seed", type=int, default=54000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--data-device", type=str, default="auto")
    parser.add_argument("--test-p01", type=float, default=0.06)
    parser.add_argument("--test-p11", type=float, default=0.94)
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--posterior-n", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--output-dir", type=Path, default=Path("Blockwise Gaussian Mixture/runs/hmm_common_factor_two_stage_sbi_npe"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.stage1_methods = [m.strip() for m in str(args.stage1_methods).split(",") if m.strip()]
    allowed = {
        "linear_deepsets",
        "raw_gru",
        "linear_gru",
        "radial_linear",
        "radial_deepsets",
        "nested_structured_gru",
        "structured_gru",
    }
    bad = [m for m in args.stage1_methods if m not in allowed]
    if bad:
        raise ValueError(f"Unknown stage1 methods {bad}; allowed={sorted(allowed)}")
    if not (0.0 < args.p01 < 1.0 and 0.0 < args.p11 < 1.0):
        raise ValueError("--p01 and --p11 must be in (0, 1)")
    if not (0.0 < args.p01_prior_min < args.p01_prior_max < 1.0):
        raise ValueError("Require 0 < p01 prior min < max < 1")
    if not (0.0 < args.p11_prior_min < args.p11_prior_max < 1.0):
        raise ValueError("Require 0 < p11 prior min < max < 1")
    if not (args.p01_prior_min <= args.test_p01 <= args.p01_prior_max):
        raise ValueError("--test-p01 must lie inside the SBI prior interval")
    if not (args.p11_prior_min <= args.test_p11 <= args.p11_prior_max):
        raise ValueError("--test-p11 must lie inside the SBI prior interval")
    gate_label = fsm.gate_label_for(str(args.clean_gate_kind))
    if gate_label != "positive-MLP":
        args.sbi_methods = str(args.sbi_methods).replace("positive-MLP radial-gate", f"{gate_label} radial-gate")
    run(args)


if __name__ == "__main__":
    main()
