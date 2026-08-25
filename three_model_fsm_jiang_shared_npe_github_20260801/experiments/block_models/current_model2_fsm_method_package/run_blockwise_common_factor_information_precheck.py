#!/usr/bin/env python3
"""Information-ceiling precheck for Model 2.

Model 2 is the blockwise common-factor variance mixture:

    B_k ~ Bernoulli(pi)
    Y_k | B_k=0 ~ N(0, I_m)
    Y_k | B_k=1 ~ N(0, I_m + tau^2 11')

Only pi is unknown.  Before training summaries, this script estimates whether a
candidate tau creates a useful gap between marginal-only evidence and the full
block evidence.  The marginal-only ceiling is estimated as

    Var(E[g_full(Y_k) | A_k]) / Var(g_full(Y_k)),

where A_k is the sum of marginal log-likelihood-ratio evidence in one block.
This is an oracle ceiling for any method that only uses the marginal channel.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

import run_blockwise_common_factor_fsm_experiment as fsm


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def conditional_variance_ceiling(score: np.ndarray, stat: np.ndarray, n_bins: int) -> tuple[float, float]:
    """Estimate Var(E[score | stat]) by equal-count binning."""
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    stat = np.asarray(stat, dtype=np.float64).reshape(-1)
    order = np.argsort(stat)
    score_sorted = score[order]
    bins = np.array_split(score_sorted, int(n_bins))
    weights = np.asarray([len(b) for b in bins], dtype=np.float64)
    means = np.asarray([float(np.mean(b)) for b in bins], dtype=np.float64)
    weights = weights / np.sum(weights)
    score_mean = float(np.mean(score))
    info = float(np.sum(weights * (means - score_mean) ** 2))
    mean_abs_bin_bias = float(np.sum(weights * np.abs(means - score_mean)))
    return info, mean_abs_bin_bias


def logsumexp_np(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def smoothed_fsm_score_from_full_lr(
    full_lr: np.ndarray,
    u0: float,
    sigma_q: float,
    grid_size: int,
    grid_radius: float,
    chunk_size: int,
) -> np.ndarray:
    """Oracle FSM smoothing target E[(u-u0)/sigma_q^2 | Y].

    This is the Bayes-optimal target of local FSM regression under the Gaussian
    proposal q(u|u0). It is computed by one-dimensional quadrature over u.  The
    input full_lr has shape (n_obs, n_blocks) and contains blockwise log f1/f0.
    """
    if sigma_q <= 0:
        raise ValueError("sigma_q must be positive")
    full_lr = np.asarray(full_lr, dtype=np.float64)
    if full_lr.ndim != 2:
        raise ValueError("full_lr must have shape (n_obs, n_blocks)")

    u_grid = np.linspace(
        float(u0) - float(grid_radius) * float(sigma_q),
        float(u0) + float(grid_radius) * float(sigma_q),
        int(grid_size),
        dtype=np.float64,
    )
    pi_grid = fsm.pi_from_u(u_grid)
    log_pi = np.log(np.clip(pi_grid, 1e-300, 1.0))
    log_one_minus_pi = np.log(np.clip(1.0 - pi_grid, 1e-300, 1.0))
    log_q_kernel = -0.5 * ((u_grid - float(u0)) / float(sigma_q)) ** 2
    proposal_score_grid = (u_grid - float(u0)) / (float(sigma_q) ** 2)

    out = np.empty(full_lr.shape[0], dtype=np.float64)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, full_lr.shape[0], chunk_size):
        stop = min(start + chunk_size, full_lr.shape[0])
        lr = full_lr[start:stop]
        # shape: (batch, grid, blocks)
        block_log_mix = np.logaddexp(
            log_one_minus_pi[None, :, None],
            log_pi[None, :, None] + lr[:, None, :],
        )
        log_weight = log_q_kernel[None, :] + np.sum(block_log_mix, axis=2)
        log_norm = logsumexp_np(log_weight, axis=1)
        weights = np.exp(log_weight - log_norm[:, None])
        out[start:stop] = np.sum(weights * proposal_score_grid[None, :], axis=1)
    return out


def evaluate_smoothing_floor(
    rng: np.random.Generator,
    tau: float,
    sigma_q: float,
    pi0: float,
    block_size: int,
    n_blocks: int,
    n_mc: int,
    grid_size: int,
    grid_radius: float,
    chunk_size: int,
) -> dict[str, object]:
    u0 = fsm.logit(float(pi0))
    y = fsm.simulate_from_u(
        rng,
        np.full(int(n_mc), u0, dtype=np.float64),
        n_blocks=int(n_blocks),
        block_size=int(block_size),
        tau=float(tau),
    )
    full_lr = fsm.full_log_ratio(y, float(tau))
    exact_score = np.sum(fsm.score_u_from_log_ratio(full_lr, float(pi0)), axis=1)
    smooth_score = smoothed_fsm_score_from_full_lr(
        full_lr=full_lr,
        u0=u0,
        sigma_q=float(sigma_q),
        grid_size=int(grid_size),
        grid_radius=float(grid_radius),
        chunk_size=int(chunk_size),
    )

    exact_var = float(np.var(exact_score))
    mse = float(np.mean((smooth_score - exact_score) ** 2))
    std_mse = mse / exact_var if exact_var > 0 else float("nan")
    corr = float(np.corrcoef(smooth_score, exact_score)[0, 1])
    likelihood_width_u = 1.0 / math.sqrt(exact_var) if exact_var > 0 else float("nan")
    return {
        "pi0": float(pi0),
        "tau": float(tau),
        "sigma_q": float(sigma_q),
        "block_size": int(block_size),
        "n_blocks": int(n_blocks),
        "n_mc": int(n_mc),
        "grid_size": int(grid_size),
        "grid_radius": float(grid_radius),
        "exact_score_var": exact_var,
        "likelihood_width_u": likelihood_width_u,
        "sigma_over_width": float(sigma_q) / likelihood_width_u if likelihood_width_u > 0 else float("nan"),
        "smoothing_floor_mse": mse,
        "smoothing_floor_std_mse": std_mse,
        "smoothing_floor_rmse": math.sqrt(mse),
        "smooth_exact_corr": corr,
        "exact_score_sd": float(np.std(exact_score)),
        "smooth_score_sd": float(np.std(smooth_score)),
    }


def evaluate_tau(
    rng: np.random.Generator,
    tau: float,
    pi0: float,
    block_size: int,
    n_mc: int,
    n_bins: int,
    n_blocks_for_sd: int,
) -> dict[str, object]:
    u0 = fsm.logit(float(pi0))
    y = fsm.simulate_from_u(
        rng,
        np.full(int(n_mc), u0, dtype=np.float64),
        n_blocks=1,
        block_size=int(block_size),
        tau=float(tau),
    )

    full_lr = fsm.full_log_ratio(y, float(tau)).reshape(-1)
    full_score = fsm.score_u_from_log_ratio(full_lr, float(pi0)).reshape(-1)

    marginal_lr_sum = fsm.marginal_log_ratio(y, float(tau)).sum(axis=2).reshape(-1)
    pairwise_lr_sum = fsm.pairwise_log_ratio(y, float(tau)).sum(axis=2).reshape(-1)
    recovered_lr = fsm.recover_block_log_ratio_from_marginal_pairwise_evidence(
        marginal_lr_sum.reshape(-1, 1),
        pairwise_lr_sum.reshape(-1, 1),
        float(tau),
        int(block_size),
    ).reshape(-1)

    full_info = float(np.mean((full_score - np.mean(full_score)) ** 2))
    marginal_info, marginal_bin_bias = conditional_variance_ceiling(
        full_score, marginal_lr_sum, int(n_bins)
    )
    marginal_ratio = marginal_info / full_info if full_info > 0 else float("nan")
    total_info = float(n_blocks_for_sd) * full_info
    marginal_total_info = float(n_blocks_for_sd) * marginal_info

    return {
        "pi0": float(pi0),
        "tau": float(tau),
        "block_size": int(block_size),
        "n_mc": int(n_mc),
        "n_bins": int(n_bins),
        "n_blocks_for_sd": int(n_blocks_for_sd),
        "full_info_per_block": full_info,
        "marginal_ceiling_info_per_block": marginal_info,
        "marginal_ceiling_ratio": marginal_ratio,
        "full_total_info": total_info,
        "marginal_total_info": marginal_total_info,
        "full_effective_sd_u": 1.0 / math.sqrt(total_info) if total_info > 0 else float("nan"),
        "marginal_effective_sd_u": (
            1.0 / math.sqrt(marginal_total_info) if marginal_total_info > 0 else float("nan")
        ),
        "marginal_mean_abs_bin_score": marginal_bin_bias,
        "recovered_log_ratio_rmse": float(np.sqrt(np.mean((recovered_lr - full_lr) ** 2))),
        "score_mean_abs": float(abs(np.mean(full_score))),
    }


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    rows = [
        evaluate_tau(
            rng,
            tau,
            pi0=float(args.pi),
            block_size=int(args.block_size),
            n_mc=int(args.n_mc),
            n_bins=int(args.n_bins),
            n_blocks_for_sd=int(args.n_blocks_for_sd),
        )
        for tau in parse_float_list(args.tau_values)
    ]

    write_csv(out_dir / "model2_information_precheck.csv", rows)

    sigma_values = parse_float_list(args.sigma_q_values)
    smoothing_rows: list[dict[str, object]] = []
    if sigma_values:
        for tau in parse_float_list(args.tau_values):
            for sigma_q in sigma_values:
                smoothing_rows.append(
                    evaluate_smoothing_floor(
                        rng,
                        tau=float(tau),
                        sigma_q=float(sigma_q),
                        pi0=float(args.pi),
                        block_size=int(args.block_size),
                        n_blocks=int(args.n_blocks_for_sd),
                        n_mc=int(args.n_floor_mc),
                        grid_size=int(args.floor_grid_size),
                        grid_radius=float(args.floor_grid_radius),
                        chunk_size=int(args.floor_chunk_size),
                    )
                )
        write_csv(out_dir / "model2_smoothing_floor_precheck.csv", smoothing_rows)

    md_lines = [
        "# Model 2 Information Precheck",
        "",
        f"`pi0={args.pi}`, `block_size={args.block_size}`, `n_mc={args.n_mc}`, "
        f"`n_bins={args.n_bins}`, `n_blocks_for_sd={args.n_blocks_for_sd}`.",
        "",
        "| tau | marginal/full info | full total info | full effective SD in u | recovered LR RMSE |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            "| "
            f"{float(row['tau']):.4g} | "
            f"{float(row['marginal_ceiling_ratio']):.4g} | "
            f"{float(row['full_total_info']):.4g} | "
            f"{float(row['full_effective_sd_u']):.4g} | "
            f"{float(row['recovered_log_ratio_rmse']):.3e} |"
        )
    md_lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A small marginal/full ratio means marginal-only summaries have a real information ceiling.",
            "- A very small full total information means the posterior will be prior-dominated.",
            "- A useful Model 2 setting should have both a visible marginal/full gap and enough total information.",
            "- `recovered LR RMSE` should be near numerical zero; it checks that marginal+pairwise evidence can recover the full block likelihood ratio.",
        ]
    )
    if smoothing_rows:
        md_lines.extend(
            [
                "",
                "## FSM smoothing-floor precheck",
                "",
                "The floor is the standardized MSE between the exact local score at `u0` "
                "and the oracle FSM-smoothed score induced by `sigma_q`.  It is an "
                "unavoidable bias term: if this is large, no Stage-1 architecture can "
                "closely approximate the exact local score under that proposal width.",
                "",
                "| tau | sigma_q | likelihood width in u | sigma/width | smoothing floor std_mse | corr | decision |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
            ]
        )
        for row in smoothing_rows:
            floor = float(row["smoothing_floor_std_mse"])
            decision = "OK" if floor <= float(args.floor_threshold) else "too smoothed"
            md_lines.append(
                "| "
                f"{float(row['tau']):.4g} | "
                f"{float(row['sigma_q']):.4g} | "
                f"{float(row['likelihood_width_u']):.4g} | "
                f"{float(row['sigma_over_width']):.4g} | "
                f"{floor:.4g} | "
                f"{float(row['smooth_exact_corr']):.4g} | "
                f"{decision} |"
            )
    (out_dir / "model2_information_precheck.md").write_text("\n".join(md_lines) + "\n")

    print("\n==== Model 2 information precheck ====")
    print(f"{'tau':>8}{'marg/full':>14}{'full_info*K':>14}{'sd_u':>12}{'recover_rmse':>16}")
    print("-" * 68)
    for row in rows:
        print(
            f"{float(row['tau']):>8.3g}"
            f"{float(row['marginal_ceiling_ratio']):>14.4g}"
            f"{float(row['full_total_info']):>14.4g}"
            f"{float(row['full_effective_sd_u']):>12.4g}"
            f"{float(row['recovered_log_ratio_rmse']):>16.3e}"
        )
    print("\nSaved:")
    print(out_dir / "model2_information_precheck.csv")
    if smoothing_rows:
        print("\n==== Model 2 FSM smoothing-floor precheck ====")
        print(f"{'tau':>8}{'sigma':>10}{'width_u':>12}{'sig/width':>12}{'floor':>12}{'corr':>10}{'decision':>16}")
        print("-" * 82)
        for row in smoothing_rows:
            floor = float(row["smoothing_floor_std_mse"])
            decision = "OK" if floor <= float(args.floor_threshold) else "too smoothed"
            print(
                f"{float(row['tau']):>8.3g}"
                f"{float(row['sigma_q']):>10.3g}"
                f"{float(row['likelihood_width_u']):>12.4g}"
                f"{float(row['sigma_over_width']):>12.4g}"
                f"{floor:>12.4g}"
                f"{float(row['smooth_exact_corr']):>10.4g}"
                f"{decision:>16}"
            )
        print(out_dir / "model2_smoothing_floor_precheck.csv")
    print(out_dir / "model2_information_precheck.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi", type=float, default=0.3)
    parser.add_argument("--tau-values", type=str, default="0.2,0.3,0.5,0.75,1.0,1.5")
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--n-mc", type=int, default=200_000)
    parser.add_argument("--n-bins", type=int, default=200)
    parser.add_argument("--n-blocks-for-sd", type=int, default=40)
    parser.add_argument(
        "--sigma-q-values",
        type=str,
        default="",
        help="Optional comma-separated sigma_q values for the oracle FSM smoothing-floor precheck.",
    )
    parser.add_argument("--n-floor-mc", type=int, default=20_000)
    parser.add_argument("--floor-grid-size", type=int, default=801)
    parser.add_argument("--floor-grid-radius", type=float, default=6.0)
    parser.add_argument("--floor-chunk-size", type=int, default=256)
    parser.add_argument("--floor-threshold", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--output-dir", type=str, default="runs/model2_information_precheck")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0.0 < args.pi < 1.0):
        raise ValueError("--pi must be in (0, 1)")
    if args.block_size < 2:
        raise ValueError("--block-size must be at least 2")
    if args.n_mc < args.n_bins:
        raise ValueError("--n-mc should be at least --n-bins")
    run(args)


if __name__ == "__main__":
    main()
