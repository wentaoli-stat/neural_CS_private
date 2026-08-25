#!/usr/bin/env python3
"""Score-consuming downstream diagnostics for Model 2.

This script uses the same Stage-1 FSM summaries as
run_blockwise_common_factor_two_stage_sbi_npe_experiment.py, but it does not
train an SBI posterior.  Instead it asks whether the frozen scalar scores can be
used directly in tasks that consume score scale:

1. response curves and Godambe relative efficiency;
2. one-step Bartlett and sensitivity-corrected estimators;
3. Wald coverage based on the information implied by the scalar summary.

The exact score and the proposal-smoothed oracle score are diagnostics only.
FSM training remains likelihood-free.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_blockwise_common_factor_fsm_experiment as fsm
import run_blockwise_common_factor_information_precheck as precheck
import run_blockwise_common_factor_two_stage_sbi_npe_experiment as npe


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


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
        elif isinstance(value, (list, tuple)):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def pi_to_u(pi: np.ndarray | float) -> np.ndarray:
    pi_arr = np.asarray(pi, dtype=np.float64)
    return np.log(pi_arr) - np.log1p(-pi_arr)


def u_to_pi(u: np.ndarray | float) -> np.ndarray:
    return fsm.pi_from_u(np.asarray(u, dtype=np.float64))


def exact_center_score(y: np.ndarray, pi0: float, tau: float) -> np.ndarray:
    return fsm.score_u_from_log_ratio(fsm.full_log_ratio(y, float(tau)), float(pi0)).sum(axis=1)


def smoothed_oracle_score(
    y: np.ndarray,
    pi0: float,
    tau: float,
    sigma_q: float,
    grid_size: int,
    grid_radius: float,
    chunk_size: int,
) -> np.ndarray:
    return precheck.smoothed_fsm_score_from_full_lr(
        full_lr=fsm.full_log_ratio(y, float(tau)),
        u0=fsm.logit(float(pi0)),
        sigma_q=float(sigma_q),
        grid_size=int(grid_size),
        grid_radius=float(grid_radius),
        chunk_size=int(chunk_size),
    )


def scalar_feature_dict(
    stage1: dict[str, Any],
    y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    features = npe.compute_features(stage1, y, args)
    out: dict[str, np.ndarray] = {}
    for method, value in features.items():
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            out[method] = arr.reshape(-1)
    out["oracle exact center score"] = exact_center_score(y, float(args.center_pi), float(args.tau))
    out["oracle smoothed FSM target"] = smoothed_oracle_score(
        y,
        pi0=float(args.center_pi),
        tau=float(args.tau),
        sigma_q=float(args.sigma_q),
        grid_size=int(args.smooth_grid_size),
        grid_radius=float(args.smooth_grid_radius),
        chunk_size=int(args.smooth_chunk_size),
    )
    return out


def default_methods_for_gate(gate_label: str) -> list[str]:
    return [
        "raw ridge FSM score",
        "linear marginal CS ridge FSM score",
        "linear marginal CS DeepSets FSM score",
        f"{gate_label} radial-gate marginal-only CS DeepSets FSM score",
        "linear marginal+pairwise CS ridge FSM score",
        "linear marginal+pairwise CS DeepSets FSM score",
        f"{gate_label} radial-gate marginal+pairwise CS linear FSM score",
        f"{gate_label} radial-gate marginal+pairwise CS DeepSets FSM score",
        "oracle exact center score",
        "oracle smoothed FSM target",
    ]


def save_stage1_checkpoint(path: Path, stage1: dict[str, Any], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage1": stage1,
            "args": jsonable_args(args),
            "note": "Pickled internal checkpoint for Model 2 score-consuming diagnostics.",
        },
        path,
    )


def load_stage1_checkpoint(path: Path, device: str) -> dict[str, Any]:
    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    stage1 = payload["stage1"]
    stage1["device"] = device
    models = stage1.get("models", {})
    for model in models.values():
        if isinstance(model, torch.nn.Module):
            model.to(device)
    return stage1


def prepare_stage1(args: argparse.Namespace, device: str, out_dir: Path) -> dict[str, Any]:
    if args.stage1_checkpoint and Path(args.stage1_checkpoint).exists():
        print("Loading Stage-1 checkpoint:", args.stage1_checkpoint)
        return load_stage1_checkpoint(Path(args.stage1_checkpoint), device)

    stage1 = npe.train_stage1(args, device)
    write_csv(out_dir / "stage1_training_trace.csv", stage1["trace_rows"])
    if args.save_stage1_checkpoint:
        save_stage1_checkpoint(Path(args.save_stage1_checkpoint), stage1, args)
    else:
        save_stage1_checkpoint(out_dir / "stage1_checkpoint.pt", stage1, args)
    return stage1


def estimate_local_slope(u: np.ndarray, mu: np.ndarray, u0: float, radius: float, min_points: int = 5) -> float:
    u = np.asarray(u, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mask = np.abs(u - float(u0)) <= float(radius)
    if np.sum(mask) < min_points:
        order = np.argsort(np.abs(u - float(u0)))[: min(min_points, u.size)]
        mask = np.zeros_like(u, dtype=bool)
        mask[order] = True
    x = u[mask] - float(u0)
    y = mu[mask]
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[1])


def affine_r2(u: np.ndarray, mu: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    design = np.column_stack([np.ones_like(u), u])
    coef, *_ = np.linalg.lstsq(design, mu, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((mu - pred) ** 2))
    ss_tot = float(np.sum((mu - np.mean(mu)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def compute_response_and_method_stats(
    stage1: dict[str, Any],
    methods: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(int(args.response_seed))
    pi_values = parse_float_list(str(args.response_pi_values))
    if not pi_values:
        pi_values = np.linspace(float(args.pi_prior_min), float(args.pi_prior_max), int(args.response_grid_size)).tolist()
    u_values = pi_to_u(np.asarray(pi_values, dtype=np.float64))
    u0 = fsm.logit(float(args.center_pi))

    response_rows: list[dict[str, object]] = []
    values_by_method: dict[str, list[tuple[float, float, float, float]]] = {m: [] for m in methods}
    center_values: dict[str, np.ndarray] = {}
    center_exact_score: np.ndarray | None = None

    for pi in pi_values:
        y = fsm.simulate_from_u(
            rng,
            np.full(int(args.response_n), fsm.logit(float(pi)), dtype=np.float64),
            n_blocks=int(args.n_blocks),
            block_size=int(args.block_size),
            tau=float(args.tau),
        )
        feature_values = scalar_feature_dict(stage1, y, args)
        if abs(float(pi) - float(args.center_pi)) < 1e-12:
            center_exact_score = exact_center_score(y, float(args.center_pi), float(args.tau))
        for method in methods:
            if method not in feature_values:
                raise ValueError(f"Method {method!r} is unavailable. Available: {sorted(feature_values)}")
            vals = np.asarray(feature_values[method], dtype=np.float64).reshape(-1)
            mean = float(np.mean(vals))
            var = float(np.var(vals))
            second = float(np.mean(vals**2))
            values_by_method[method].append((float(pi), float(pi_to_u(pi)), mean, var))
            if abs(float(pi) - float(args.center_pi)) < 1e-12:
                center_values[method] = vals
            response_rows.append(
                {
                    "method": method,
                    "pi": float(pi),
                    "u": float(pi_to_u(pi)),
                    "n": int(args.response_n),
                    "mean": mean,
                    "var": var,
                    "second_moment": second,
                    "sd": math.sqrt(max(var, 0.0)),
                }
            )

    if center_exact_score is None:
        # Ensure exact I is estimated at the center even if center_pi is not in response grid.
        y_center = fsm.simulate_from_u(
            rng,
            np.full(int(args.response_n), u0, dtype=np.float64),
            n_blocks=int(args.n_blocks),
            block_size=int(args.block_size),
            tau=float(args.tau),
        )
        center_exact_score = exact_center_score(y_center, float(args.center_pi), float(args.tau))
        center_features = scalar_feature_dict(stage1, y_center, args)
        for method in methods:
            center_values[method] = np.asarray(center_features[method], dtype=np.float64).reshape(-1)

    fisher_i = float(np.var(center_exact_score))
    method_stats: dict[str, dict[str, float]] = {}
    stat_rows: list[dict[str, object]] = []
    for method in methods:
        rows = values_by_method[method]
        pi_arr = np.asarray([r[0] for r in rows], dtype=np.float64)
        u_arr = np.asarray([r[1] for r in rows], dtype=np.float64)
        mu_arr = np.asarray([r[2] for r in rows], dtype=np.float64)
        h = estimate_local_slope(u_arr, mu_arr, u0, float(args.response_slope_radius))
        center_vals = np.asarray(center_values[method], dtype=np.float64)
        mu0 = float(np.mean(center_vals))
        var0 = float(np.var(center_vals))
        second0 = float(np.mean(center_vals**2))
        godambe = float(h**2 / var0) if var0 > 1e-12 else float("nan")
        re_det = float(godambe / fisher_i) if fisher_i > 1e-12 else float("nan")
        r2 = affine_r2(u_arr, mu_arr)
        stats = {
            "mu0": mu0,
            "var0": var0,
            "second0": second0,
            "H": h,
            "G": godambe,
            "I": fisher_i,
            "RE": re_det,
            "affine_r2": r2,
            "nonlinearity_index": float(1.0 - r2) if np.isfinite(r2) else float("nan"),
            "bartlett_ratio": float(second0 / godambe) if np.isfinite(godambe) and abs(godambe) > 1e-12 else float("nan"),
            "center_corr_exact": float(np.corrcoef(center_vals, center_exact_score)[0, 1])
            if np.std(center_vals) > 1e-12
            else float("nan"),
        }
        method_stats[method] = stats
        stat_rows.append(
            {
                "method": method,
                "center_pi": float(args.center_pi),
                "n_center": int(center_vals.size),
                "mu0": stats["mu0"],
                "var0": stats["var0"],
                "second0": stats["second0"],
                "H": stats["H"],
                "G": stats["G"],
                "I_exact": stats["I"],
                "RE": stats["RE"],
                "affine_r2": stats["affine_r2"],
                "nonlinearity_index": stats["nonlinearity_index"],
                "bartlett_ratio_second_over_G": stats["bartlett_ratio"],
                "center_corr_exact": stats["center_corr_exact"],
            }
        )
    return response_rows, stat_rows, method_stats


def summarize_estimator_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["estimator"]), float(row["pi_true"])), []).append(row)
    out: list[dict[str, object]] = []
    for (method, estimator, pi_true), group in grouped.items():
        err = np.asarray([float(r["pi_error"]) for r in group], dtype=np.float64)
        sq = err**2
        out.append(
            {
                "method": method,
                "estimator": estimator,
                "pi_true": pi_true,
                "n": len(group),
                "pi_mse": float(np.mean(sq)),
                "pi_rmse": float(np.sqrt(np.mean(sq))),
                "pi_mae": float(np.mean(np.abs(err))),
                "coverage90": float(np.mean([float(r["coverage90"]) for r in group])),
                "mean_ci_width_pi": float(np.mean([float(r["ci_width_pi"]) for r in group])),
            }
        )
    return sorted(out, key=lambda r: (float(r["pi_true"]), str(r["method"]), str(r["estimator"])))


def run_one_step_tasks(
    stage1: dict[str, Any],
    methods: list[str],
    method_stats: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    z90 = 1.6448536269514722
    u0 = fsm.logit(float(args.center_pi))
    rows: list[dict[str, object]] = []
    for pi_true in parse_float_list(str(args.test_pi_values)):
        for seed in parse_int_list(str(args.test_seeds)):
            rng = np.random.default_rng(int(seed))
            y = fsm.simulate_from_u(
                rng,
                np.asarray([fsm.logit(float(pi_true))], dtype=np.float64),
                n_blocks=int(args.n_blocks),
                block_size=int(args.block_size),
                tau=float(args.tau),
            )
            feature_values = scalar_feature_dict(stage1, y, args)
            for method in methods:
                t_value = float(np.asarray(feature_values[method], dtype=np.float64).reshape(-1)[0])
                stats = method_stats[method]
                bartlett_info = float(stats["second0"])
                godambe_info = float(stats["G"])
                variants = [
                    (
                        "bartlett",
                        bartlett_info,
                        t_value / bartlett_info if bartlett_info > 1e-12 else float("nan"),
                    ),
                    (
                        "godambe",
                        godambe_info,
                        (t_value - stats["mu0"]) / stats["H"] if abs(stats["H"]) > 1e-12 else float("nan"),
                    ),
                ]
                for estimator, info, u_increment in variants:
                    if not np.isfinite(info) or info <= 1e-12 or not np.isfinite(u_increment):
                        u_hat = float("nan")
                        pi_hat = float("nan")
                        lo_pi = float("nan")
                        hi_pi = float("nan")
                    else:
                        u_hat = u0 + float(u_increment)
                        se_u = 1.0 / math.sqrt(float(info))
                        lo_u = u_hat - z90 * se_u
                        hi_u = u_hat + z90 * se_u
                        pi_hat = float(u_to_pi(u_hat))
                        lo_pi = float(u_to_pi(lo_u))
                        hi_pi = float(u_to_pi(hi_u))
                    rows.append(
                        {
                            "method": method,
                            "estimator": estimator,
                            "seed": int(seed),
                            "pi_true": float(pi_true),
                            "summary_value": t_value,
                            "info_used": float(info),
                            "u_hat": u_hat,
                            "pi_hat": pi_hat,
                            "pi_error": float(pi_hat - pi_true) if np.isfinite(pi_hat) else float("nan"),
                            "ci90_lo_pi": lo_pi,
                            "ci90_hi_pi": hi_pi,
                            "ci_width_pi": float(hi_pi - lo_pi) if np.isfinite(lo_pi) and np.isfinite(hi_pi) else float("nan"),
                            "coverage90": float(lo_pi <= pi_true <= hi_pi)
                            if np.isfinite(lo_pi) and np.isfinite(hi_pi)
                            else float("nan"),
                        }
                    )
    return rows, summarize_estimator_rows(rows)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = npe.resolve_device(str(args.device))
    print("torch:", torch.__version__)
    print("device:", device)
    print("Output directory:", out_dir)
    print(
        "setting:",
        {
            "tau": args.tau,
            "n_blocks": args.n_blocks,
            "block_size": args.block_size,
            "center_pi": args.center_pi,
            "sigma_q": args.sigma_q,
        },
    )

    stage1 = prepare_stage1(args, device, out_dir)
    local_checkpoint = out_dir / "stage1_checkpoint.pt"
    if not local_checkpoint.exists():
        save_stage1_checkpoint(local_checkpoint, stage1, args)
    gate_label = str(stage1.get("gate_label", fsm.gate_label_for(str(args.clean_gate_kind))))
    if str(args.methods).strip().lower() == "default":
        methods = default_methods_for_gate(gate_label)
    else:
        methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    response_rows, method_stat_rows, method_stats = compute_response_and_method_stats(stage1, methods, args)
    one_step_rows, one_step_summary = run_one_step_tasks(stage1, methods, method_stats, args)

    write_csv(out_dir / "response_curve.csv", response_rows)
    write_csv(out_dir / "method_information_summary.csv", method_stat_rows)
    write_csv(out_dir / "one_step_by_seed.csv", one_step_rows)
    write_csv(out_dir / "one_step_summary_by_pi.csv", one_step_summary)
    write_csv(out_dir / "one_step_summary_pooled.csv", summarize_pooled(one_step_rows))
    with (out_dir / "config.json").open("w") as file:
        json.dump(jsonable_args(args), file, indent=2)

    print("\n==== Method information summary ====")
    print(f"{'method':<62}{'RE':>10}{'G':>12}{'H':>12}{'Bartlett':>12}{'nonlin':>10}")
    print("-" * 118)
    for row in sorted(method_stat_rows, key=lambda r: float(r["RE"]), reverse=True):
        print(
            f"{str(row['method']):<62}"
            f"{float(row['RE']):>10.4g}"
            f"{float(row['G']):>12.4g}"
            f"{float(row['H']):>12.4g}"
            f"{float(row['bartlett_ratio_second_over_G']):>12.4g}"
            f"{float(row['nonlinearity_index']):>10.4g}"
        )

    print("\n==== One-step pooled summary ====")
    pooled = summarize_pooled(one_step_rows)
    print(f"{'method':<62}{'estimator':<14}{'rmse':>10}{'mae':>10}{'cov90':>10}{'width':>10}")
    print("-" * 116)
    for row in pooled:
        print(
            f"{str(row['method']):<62}"
            f"{str(row['estimator']):<14}"
            f"{float(row['pi_rmse']):>10.4g}"
            f"{float(row['pi_mae']):>10.4g}"
            f"{float(row['coverage90']):>10.3g}"
            f"{float(row['mean_ci_width_pi']):>10.4g}"
        )

    print("\nSaved:")
    print(out_dir / "response_curve.csv")
    print(out_dir / "method_information_summary.csv")
    print(out_dir / "one_step_by_seed.csv")
    print(out_dir / "one_step_summary_by_pi.csv")
    print(out_dir / "one_step_summary_pooled.csv")
    print(out_dir / "stage1_checkpoint.pt")


def summarize_pooled(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["estimator"])), []).append(row)
    out: list[dict[str, object]] = []
    for (method, estimator), group in grouped.items():
        err = np.asarray([float(r["pi_error"]) for r in group], dtype=np.float64)
        finite = np.isfinite(err)
        if not np.any(finite):
            continue
        group_finite = [g for g, ok in zip(group, finite) if ok]
        err = err[finite]
        sq = err**2
        out.append(
            {
                "method": method,
                "estimator": estimator,
                "n": len(group_finite),
                "pi_mse": float(np.mean(sq)),
                "pi_rmse": float(np.sqrt(np.mean(sq))),
                "pi_mae": float(np.mean(np.abs(err))),
                "coverage90": float(np.mean([float(r["coverage90"]) for r in group_finite])),
                "mean_ci_width_pi": float(np.mean([float(r["ci_width_pi"]) for r in group_finite])),
            }
        )
    return sorted(out, key=lambda r: (str(r["estimator"]), float(r["pi_rmse"])))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-blocks", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--center-pi", type=float, default=0.3)
    parser.add_argument("--sigma-q", type=float, default=0.15)
    parser.add_argument("--pi-prior-min", type=float, default=0.05)
    parser.add_argument("--pi-prior-max", type=float, default=0.7)

    parser.add_argument("--stage1-n-train", type=int, default=20_000)
    parser.add_argument("--stage1-n-val", type=int, default=5_000)
    parser.add_argument("--stage1-seed", type=int, default=20260705)
    parser.add_argument(
        "--stage1-methods",
        type=str,
        default="marginal_linear_deepset,marginal_radial_deepset,linear_deepset,radial_linear,radial_deepset",
    )
    parser.add_argument("--stage1-iters", type=int, default=3_000)
    parser.add_argument("--stage1-batch-size", type=int, default=512)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    parser.add_argument("--stage1-weight-decay", type=float, default=1e-3)
    parser.add_argument("--stage1-patience", type=int, default=30)
    parser.add_argument("--stage1-print-every", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gate-hidden", type=int, default=16)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument(
        "--clean-gate-kind",
        choices=["monotone_mlp", "positive_mlp", "log_polynomial"],
        default="positive_mlp",
    )
    parser.add_argument("--gate-input", choices=["abs", "signed"], default="signed")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--stage1-checkpoint", type=Path, default=None)
    parser.add_argument("--save-stage1-checkpoint", type=Path, default=None)

    parser.add_argument(
        "--methods",
        type=str,
        default="default",
        help="Comma-separated scalar methods, or 'default'.",
    )
    parser.add_argument("--response-pi-values", type=str, default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65")
    parser.add_argument("--response-grid-size", type=int, default=25)
    parser.add_argument("--response-n", type=int, default=2_000)
    parser.add_argument("--response-seed", type=int, default=202607051)
    parser.add_argument("--response-slope-radius", type=float, default=0.35)
    parser.add_argument("--smooth-grid-size", type=int, default=301)
    parser.add_argument("--smooth-grid-radius", type=float, default=5.0)
    parser.add_argument("--smooth-chunk-size", type=int, default=256)

    parser.add_argument("--test-pi-values", type=str, default="0.15,0.2,0.3,0.4,0.5")
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--output-dir", type=Path, default=Path("Blockwise Gaussian Mixture/runs/model2_score_consuming_tasks"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.stage1_methods = [m.strip() for m in str(args.stage1_methods).split(",") if m.strip()]
    allowed = {
        "marginal_linear_deepset",
        "marginal_radial_deepset",
        "linear_deepset",
        "radial_linear",
        "radial_deepset",
        "free_phi_deepset",
    }
    bad = [m for m in args.stage1_methods if m not in allowed]
    if bad:
        raise ValueError(f"Unknown stage1 methods {bad}; allowed={sorted(allowed)}")
    if not (0.0 < args.center_pi < 1.0):
        raise ValueError("--center-pi must be in (0, 1)")
    if args.sigma_q <= 0:
        raise ValueError("--sigma-q must be positive")
    run(args)


if __name__ == "__main__":
    main()
