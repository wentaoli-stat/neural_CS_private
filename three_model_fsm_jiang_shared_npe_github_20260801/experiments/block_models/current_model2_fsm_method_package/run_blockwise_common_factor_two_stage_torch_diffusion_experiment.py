#!/usr/bin/env python3
"""Two-stage PyTorch score-based diffusion posterior inference for Model 2.

This script keeps the same frozen Stage-1 FSM summaries as the sbi-NPE script
and replaces the Stage-2 posterior estimator with a conditional DDPM-style
score/noise model.  It is deliberately torch-only so it runs on AutoDL images
where the JAX CUDA plugin stack is brittle.
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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import run_blockwise_common_factor_fsm_experiment as fsm

from run_blockwise_common_factor_two_stage_sbi_npe_experiment import (
    METHOD_ORDER,
    compute_features,
    exact_posterior_grid,
    exact_posterior_grid_arrays,
    exact_sample_w1,
    exact_cdf_at,
    jsonable_args,
    parse_int_list,
    sample_pi_prior,
    simulate_from_pi,
    sort_rows,
    stage1_score_diagnostics,
    standard_error,
    summarize_full_posterior,
    summarize_full_posterior_pooled,
    train_stage1,
    write_csv,
)


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def unit_from_pi(pi: np.ndarray, pi_min: float, pi_max: float, eps: float = 1e-6) -> np.ndarray:
    unit = (np.asarray(pi, dtype=np.float64).reshape(-1) - float(pi_min)) / (
        float(pi_max) - float(pi_min)
    )
    return np.clip(unit, eps, 1.0 - eps)


def latent_from_pi(pi: np.ndarray, pi_min: float, pi_max: float) -> np.ndarray:
    unit = unit_from_pi(pi, pi_min, pi_max)
    return (np.log(unit) - np.log1p(-unit)).reshape(-1, 1).astype(np.float32)


def pi_from_latent(z: np.ndarray, pi_min: float, pi_max: float) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    unit = 1.0 / (1.0 + np.exp(-z))
    return (float(pi_min) + (float(pi_max) - float(pi_min)) * unit).astype(np.float64)


def fit_standardizer(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return mean.astype(np.float32), sd.astype(np.float32)


def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ConditionalEpsilonNet(nn.Module):
    def __init__(self, x_dim: int, hidden: int, depth: int, t_dim: int):
        super().__init__()
        self.t_dim = int(t_dim)
        x_layers: list[nn.Module] = [nn.Linear(x_dim, hidden), nn.SiLU()]
        for _ in range(max(depth - 1, 0)):
            x_layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.x_net = nn.Sequential(*x_layers)

        in_dim = 1 + hidden + t_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.main = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_emb = self.x_net(x)
        t_emb = sinusoidal_timestep_embedding(t, self.t_dim)
        inp = torch.cat([z_t, x_emb, t_emb], dim=-1)
        return self.main(inp)


class TorchDiffusionPosterior:
    def __init__(
        self,
        model: ConditionalEpsilonNet,
        *,
        x_mean: np.ndarray,
        x_sd: np.ndarray,
        z_mean: np.ndarray,
        z_sd: np.ndarray,
        betas: torch.Tensor,
        device: str,
        pi_min: float,
        pi_max: float,
    ):
        self.model = model.eval()
        self.x_mean = np.asarray(x_mean, dtype=np.float32)
        self.x_sd = np.asarray(x_sd, dtype=np.float32)
        self.z_mean = np.asarray(z_mean, dtype=np.float32)
        self.z_sd = np.asarray(z_sd, dtype=np.float32)
        self.device = torch.device(device)
        self.pi_min = float(pi_min)
        self.pi_max = float(pi_max)

        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(self, x_obs: np.ndarray, n: int, seed: int) -> np.ndarray:
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        x = (np.asarray(x_obs, dtype=np.float32).reshape(1, -1) - self.x_mean) / self.x_sd
        x_t = torch.as_tensor(x, dtype=torch.float32, device=self.device).repeat(int(n), 1)
        z = torch.randn((int(n), 1), dtype=torch.float32, device=self.device)

        num_steps = int(self.betas.numel())
        for step in reversed(range(num_steps)):
            t = torch.full((int(n),), step, dtype=torch.long, device=self.device)
            eps = self.model(z, x_t, t)
            beta = self.betas[step]
            alpha = self.alphas[step]
            abar = self.alpha_bar[step]
            mean = (z - beta / torch.sqrt(1.0 - abar) * eps) / torch.sqrt(alpha)
            if step > 0:
                z = mean + torch.sqrt(beta) * torch.randn_like(z)
            else:
                z = mean
        z_np = z.detach().cpu().numpy() * self.z_sd + self.z_mean
        return pi_from_latent(z_np, self.pi_min, self.pi_max)


def make_beta_schedule(n_steps: int, beta_min: float, beta_max: float, device: str) -> torch.Tensor:
    return torch.linspace(float(beta_min), float(beta_max), int(n_steps), dtype=torch.float32, device=device)


def train_diffusion_method(
    method: str,
    x_np: np.ndarray,
    pi_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: str,
) -> dict[str, object]:
    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    x_np = np.asarray(x_np, dtype=np.float32)
    z_np = latent_from_pi(pi_train, float(args.pi_prior_min), float(args.pi_prior_max))
    x_mean, x_sd = fit_standardizer(x_np)
    z_mean, z_sd = fit_standardizer(z_np)
    x_std = (x_np - x_mean) / x_sd
    z_std = (z_np - z_mean) / z_sd

    x_t = torch.as_tensor(x_std, dtype=torch.float32)
    z_t = torch.as_tensor(z_std, dtype=torch.float32)
    dataset = TensorDataset(x_t, z_t)
    n_val = max(1, int(len(dataset) * float(args.validation_fraction)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(seed) + 99),
    )
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    # Keep validation deterministic. Resampling diffusion times and noise at
    # every check makes early stopping compare different objectives.
    val_indices = torch.as_tensor(val_ds.indices, dtype=torch.long)
    x_val = x_t[val_indices]
    z_val = z_t[val_indices]
    val_generator = torch.Generator().manual_seed(int(seed) + 10_007)
    val_t = torch.randint(
        0,
        int(args.diffusion_steps),
        (n_val,),
        generator=val_generator,
        dtype=torch.long,
    )
    val_eps = torch.randn(z_val.shape, generator=val_generator, dtype=z_val.dtype)
    val_loader = DataLoader(
        TensorDataset(x_val, z_val, val_t, val_eps),
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
    )

    model = ConditionalEpsilonNet(
        x_dim=x_np.shape[1],
        hidden=int(args.diffusion_hidden),
        depth=int(args.diffusion_depth),
        t_dim=int(args.diffusion_t_embed_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    betas = make_beta_schedule(int(args.diffusion_steps), float(args.beta_min), float(args.beta_max), device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    def batch_loss(
        xb: torch.Tensor,
        zb: torch.Tensor,
        t: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = xb.shape[0]
        if t is None:
            t = torch.randint(0, int(args.diffusion_steps), (b,), device=device)
        if eps is None:
            eps = torch.randn_like(zb)
        abar = alpha_bar[t].unsqueeze(-1)
        z_noisy = torch.sqrt(abar) * zb + torch.sqrt(1.0 - abar) * eps
        eps_pred = model(z_noisy, xb, t)
        return torch.mean((eps_pred - eps) ** 2)

    best_loss = float("inf")
    best_state = None
    best_step = 0
    bad = 0
    trace = []
    t0 = time.time()
    print(f"\nTraining torch diffusion for {method} (seed={seed})")
    print("z", z_np.shape, "x", x_np.shape, "device", device)

    loader_iter = iter(train_loader)
    for step in range(1, int(args.train_steps) + 1):
        model.train()
        try:
            xb, zb = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            xb, zb = next(loader_iter)
        xb = xb.to(device)
        zb = zb.to(device)
        opt.zero_grad(set_to_none=True)
        loss = batch_loss(xb, zb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        opt.step()

        if step == 1 or step % int(args.print_every) == 0 or step == int(args.train_steps):
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for xvb, zvb, tvb, epsvb in val_loader:
                    batch_n = int(xvb.shape[0])
                    fixed_loss = batch_loss(
                        xvb.to(device),
                        zvb.to(device),
                        tvb.to(device),
                        epsvb.to(device),
                    )
                    val_loss_sum += float(fixed_loss.item()) * batch_n
                    val_count += batch_n
            val_loss = val_loss_sum / max(val_count, 1)
            improved = val_loss < best_loss - float(args.min_delta)
            if improved:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_step = step
                bad = 0
            else:
                bad += 1
            trace.append(
                {
                    "method": method,
                    "step": step,
                    "train_loss": float(loss.item()),
                    "val_loss": val_loss,
                    "is_best": int(improved),
                    "best_step_so_far": int(best_step),
                    "best_val_loss_so_far": float(best_loss),
                }
            )
            print(
                f"{method} step {step}/{args.train_steps}: train={loss.item():.4g}, "
                f"val={val_loss:.4g}, best={best_loss:.4g}@{best_step}"
            )
            if not improved:
                if int(args.patience) > 0 and bad >= int(args.patience):
                    print(f"{method}: early stopping at step {step}, best_val={best_loss:.4g}")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    sec = time.time() - t0
    print(f"{method} torch diffusion seconds:", round(sec, 2))

    return {
        "method": method,
        "posterior": TorchDiffusionPosterior(
            model,
            x_mean=x_mean,
            x_sd=x_sd,
            z_mean=z_mean,
            z_sd=z_sd,
            betas=betas.detach().cpu(),
            device=device,
            pi_min=float(args.pi_prior_min),
            pi_max=float(args.pi_prior_max),
        ),
        "trace": trace,
        "seconds": float(sec),
        "dim_data": int(x_np.shape[1]),
        "best_step": int(best_step),
        "best_val_loss": float(best_loss),
        "validation_noise": "fixed",
    }


def posterior_seed(method: str, obs_seed: int) -> int:
    offsets = {method: 1000 + 137 * idx for idx, method in enumerate(METHOD_ORDER)}
    if method in offsets:
        return offsets[method] + int(obs_seed)
    digest = hashlib.blake2b(method.encode("utf-8"), digest_size=4).hexdigest()
    return 10000 + (int(digest, 16) % 1_000_000) + int(obs_seed)


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


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "device:", device)
    print("Output directory:", out_dir)

    stage1 = train_stage1(args, device)
    stage1_score_rows = stage1_score_diagnostics(stage1, args)

    rng_train = np.random.default_rng(int(args.diffusion_train_seed))
    pi_train = sample_pi_prior(rng_train, int(args.n_diffusion_train), float(args.pi_prior_min), float(args.pi_prior_max))
    y_train = simulate_from_pi(rng_train, pi_train, int(args.n_blocks), int(args.block_size), float(args.tau))
    feature_dict = compute_features(stage1, y_train, args)

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    missing = [m for m in methods if m not in feature_dict]
    if missing:
        raise ValueError(f"Unknown methods: {missing}. Available: {sorted(feature_dict)}")

    results = []
    trace_rows = []
    for idx, method in enumerate(methods):
        res = train_diffusion_method(
            method,
            feature_dict[method],
            pi_train,
            int(args.diffusion_seed) + 173 * idx,
            args,
            device,
        )
        results.append(res)
        trace_rows.extend(res["trace"])

    rows: list[dict[str, object]] = []
    samples_dir = out_dir / "posterior_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    test_pi_values = (
        [float(x.strip()) for x in str(args.test_pi_values).split(",") if x.strip()]
        if args.test_pi_values
        else [float(args.test_pi)]
    )
    for pi_true in test_pi_values:
        if not (args.pi_prior_min <= pi_true <= args.pi_prior_max):
            raise ValueError(f"test pi={pi_true} must lie inside the posterior prior interval")
        for obs_seed in parse_int_list(args.test_seeds):
            rng = np.random.default_rng(int(obs_seed))
            y_obs = simulate_from_pi(
                rng,
                np.asarray([pi_true]),
                int(args.n_blocks),
                int(args.block_size),
                float(args.tau),
            )
            obs_features = compute_features(stage1, y_obs, args)

            exact_grid, exact_w, exact_cdf = exact_posterior_grid_arrays(y_obs[0], args)
            exact = exact_posterior_grid(y_obs[0], args)
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
                    "pi_sq_err": float((exact["mean"] - pi_true) ** 2),
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
                samples = result["posterior"].sample(
                    obs_features[method][0],
                    int(args.posterior_n),
                    posterior_seed(method, int(obs_seed)),
                )
                np.save(
                    samples_dir
                    / f"{method.replace(' ', '_').replace('/', '_')}_pi{pi_true:g}_seed{obs_seed}.npy",
                    samples,
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
    write_csv(out_dir / "diffusion_training_trace.csv", trace_rows)
    with (out_dir / "config.json").open("w") as file:
        json.dump(jsonable_args(args), file, indent=2)

    print("\n==== Torch diffusion posterior summary ====")
    print(f"{'pi_true':>8}  {'method':<42}{'pi_mse':>12}{'se':>10}{'post_sd':>10}{'cov90':>10}")
    print("-" * 94)
    for row in summary:
        print(
            f"{float(row['pi_true']):>8.3g}  "
            f"{row['method']:<42}"
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
    print(out_dir / "diffusion_training_trace.csv")
    if stage1_score_rows:
        print(out_dir / "stage1_score_summary_direct.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-blocks", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--tau", type=float, default=1.0)

    parser.add_argument("--center-pi", type=float, default=0.3)
    parser.add_argument("--sigma-q", type=float, default=0.15)
    parser.add_argument("--stage1-n-train", type=int, default=20_000)
    parser.add_argument("--stage1-n-val", type=int, default=4_000)
    parser.add_argument("--stage1-seed", type=int, default=20260702)
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
    parser.add_argument(
        "--stage1-diagnostic-n-test",
        type=int,
        default=0,
        help="If positive, evaluate frozen scalar Stage-1 scores against the exact local score on this many center samples.",
    )
    parser.add_argument("--stage1-diagnostic-seed", type=int, default=20260704)
    parser.add_argument("--hidden", type=int, default=64)
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
        help="Input to radial multiplier: abs keeps the old odd map; signed allows asymmetric positive multiplier.",
    )
    parser.add_argument("--ridge", type=float, default=1e-6)

    parser.add_argument("--pi-prior-min", type=float, default=0.05)
    parser.add_argument("--pi-prior-max", type=float, default=0.7)
    parser.add_argument("--n-diffusion-train", type=int, default=20_000)
    parser.add_argument("--diffusion-train-seed", type=int, default=20260703)
    parser.add_argument("--methods", type=str, default=(
        "raw data,raw ridge FSM score,"
        "linear marginal block CS,linear marginal CS ridge FSM score,linear marginal CS DeepSets FSM score,"
        "MLP radial-gate marginal-only CS DeepSets FSM score,"
        "linear marginal+pairwise block CS,linear marginal+pairwise CS ridge FSM score,"
        "linear marginal+pairwise CS DeepSets FSM score,"
        "MLP radial-gate marginal+pairwise CS linear FSM score,"
        "MLP radial-gate marginal+pairwise CS DeepSets FSM score"
    ))

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--diffusion-seed", type=int, default=20260704)
    parser.add_argument("--diffusion-steps", type=int, default=200)
    parser.add_argument("--beta-min", type=float, default=1e-4)
    parser.add_argument("--beta-max", type=float, default=0.02)
    parser.add_argument("--train-steps", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--diffusion-hidden", type=int, default=128)
    parser.add_argument("--diffusion-depth", type=int, default=3)
    parser.add_argument("--diffusion-t-embed-dim", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--print-every", type=int, default=200)

    parser.add_argument("--posterior-n", type=int, default=1000)
    parser.add_argument("--test-pi", type=float, default=0.3)
    parser.add_argument(
        "--test-pi-values",
        type=str,
        default="0.1,0.3,0.5,0.65",
        help="Optional comma-separated test pi values. Overrides --test-pi when nonempty.",
    )
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--grid-size", type=int, default=5000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Blockwise Gaussian Mixture/runs/common_factor_two_stage_torch_diffusion"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
    if not (0.0 < args.pi_prior_min < args.pi_prior_max < 1.0):
        raise ValueError("Require 0 < --pi-prior-min < --pi-prior-max < 1")
    if args.sigma_q <= 0:
        raise ValueError("--sigma-q must be positive")
    gate_label = fsm.gate_label_for(str(args.clean_gate_kind))
    if gate_label != "MLP":
        method_names = [m.strip() for m in str(args.methods).split(",") if m.strip()]
        args.methods = ",".join(
            f"{gate_label} radial-gate{m[len('MLP radial-gate'):]}"
            if m.startswith("MLP radial-gate")
            else m
            for m in method_names
        )
    test_pi_values = (
        [float(x.strip()) for x in str(args.test_pi_values).split(",") if x.strip()]
        if args.test_pi_values
        else [float(args.test_pi)]
    )
    for pi_true in test_pi_values:
        if not (args.pi_prior_min <= pi_true <= args.pi_prior_max):
            raise ValueError("all test pi values must lie inside the posterior prior interval")
    run(args)


if __name__ == "__main__":
    main()
