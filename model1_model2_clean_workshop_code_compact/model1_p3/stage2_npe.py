#!/usr/bin/env python3
"""Deployable pilot+score NPE for the p=3 Model 1 FSM field.

For every Stage-2 simulation,

    beta ~ Uniform(anchor box),  Y ~ p(. | beta),
    beta_hat = composite_marginal_pilot(Y),
    context  = (beta_hat, S_frozen(Y, beta_hat)) in R^6,

with beta = (logit pi, tau, log sigma). The prior is uniform on the Stage-1
anchor box in beta coordinates (at p=1 the packaged prior is uniform in pi).
The simulated parameter is only the NPE target.

Evaluation against the exact posterior uses a two-pass grid: a coarse grid on
the whole box locates the posterior, then a fine grid (default 80^3) covers
posterior mean +/- 8 SD (plus two coarse cells), clipped to the box. The exact
posterior is evaluation-only. Reported distances:

* marginal W1 per coordinate, in natural beta units;
* sliced W1: the mean over fixed random unit directions of the 1-d W1 between
  projections, in box-width-standardized coordinates. It stands in for the
  multivariate W1, which is not computed.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from common.npe import exact_sample_w1, import_sbi
from model1_p3 import scores as sc
from model1_p3 import stage1


METHOD_LABELS = {
    "pilot": "pilot-only NPE (p=3)",
    "linear": "linear FSM pilot+score NPE (p=3)",
    "gate": "gate FSM pilot+score NPE (p=3)",
    "stacked": "stacked FSM pilot+score NPE (p=3)",
}
METHOD_ORDER = ("pilot", "linear", "gate", "stacked")
PAIRED_CONTRASTS = (
    ("stacked", "gate"), ("gate", "linear"), ("stacked", "linear"),
    ("linear", "pilot"), ("gate", "pilot"), ("stacked", "pilot"),
)
COORDS = sc.PARAM_NAMES
CONTEXT_DEFINITION = "(beta_pilot(Y), S_frozen(Y, beta_pilot(Y)))"
EXACT = "exact likelihood grid"


def parse_list(text: str, cast):
    return [cast(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_betas(text: str) -> np.ndarray:
    """'pi:tau:sigma;...' -> (n, 3) beta array."""
    rows = []
    for item in str(text).split(";"):
        if item.strip():
            pi, tau, sigma = (float(v) for v in item.split(":"))
            rows.append([float(sc.logit_np(pi)), tau, math.log(sigma)])
    return np.asarray(rows, dtype=np.float64)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def standard_error(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")


# --------------------------------------------------------------------------- frozen score
class FrozenScore:
    """Frozen p=3 Stage-1 checkpoint evaluated at arbitrary (Y, beta)."""

    def __init__(self, run_dir: Path, method: str, device: str = "cpu"):
        self.run_dir, self.method, self.device = Path(run_dir), method, device
        self.config = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        self.stats = {k: np.asarray(v) for k, v in np.load(self.run_dir / "feature_stats.npz").items()}
        self.sigma_q = np.asarray(self.stats["sigma_q"], dtype=np.float64)
        payload = torch.load(self.run_dir / f"model_{method}.pt", map_location="cpu", weights_only=False)
        if payload.get("method") != method:
            raise ValueError(f"checkpoint method mismatch: {payload.get('method')} != {method}")
        model = stage1.build_model(method, self.config, self.stats)
        model.load_state_dict(payload["state_dict"], strict=True)
        self.model = model.to(device).eval()

    def score(self, y: np.ndarray, beta: np.ndarray, batch_size: int = 256, chunk: int = 2048) -> np.ndarray:
        out = []
        for start in range(0, y.shape[0], chunk):
            stop = min(start + chunk, y.shape[0])
            data = stage1.featurize(y[start:stop], beta[start:stop], self.stats)
            pred = stage1.predict(self.model, data, self.method, self.device, batch_size)
            out.append(np.asarray(pred, dtype=np.float64) / self.sigma_q)
        return np.concatenate(out, axis=0)


def box_from_config(config: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(config["anchor_low"], dtype=np.float64), np.asarray(config["anchor_high"], dtype=np.float64)


# --------------------------------------------------------------------------- data-only pilot
def _composite_loglik(y: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Marginal composite log-likelihood; y (n, K, m) or (G, n, K, m) broadcast with beta (..., 3)."""
    u, tau, lam = beta[..., 0], beta[..., 1], beta[..., 2]
    shape = u.shape + (1, 1)
    u, tau, lam = u.reshape(shape), tau.reshape(shape), lam.reshape(shape)
    inv_sigma = torch.exp(-lam)
    log_norm = -lam - 0.5 * math.log(2.0 * math.pi)
    log0 = torch.nn.functional.logsigmoid(-u) + log_norm - 0.5 * (y * inv_sigma) ** 2
    log1 = torch.nn.functional.logsigmoid(u) + log_norm - 0.5 * ((y - tau) * inv_sigma) ** 2
    return torch.logaddexp(log0, log1).sum(dim=(-2, -1))


def data_only_pilot(
    y: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Box-constrained maximizer of the marginal composite likelihood of Y alone.

    A (u, tau) grid with sigma profiled from the moment identity
    Var(Y) = sigma^2 + pi(1-pi)tau^2 gives the start; Adam on box-standardized
    coordinates, clamped to the box, refines it. Deterministic.
    """
    y_all = np.asarray(y, dtype=np.float64)
    width = high - low
    n_grid = int(args.pilot_grid_points)
    u_axis = np.linspace(low[0], high[0], n_grid)
    tau_axis = np.linspace(low[1], high[1], n_grid)
    grid_u, grid_tau = (a.reshape(-1) for a in np.meshgrid(u_axis, tau_axis, indexing="ij"))
    out = np.empty((y_all.shape[0], 3), dtype=np.float64)
    low_t = torch.as_tensor(low, dtype=torch.float32)
    width_t = torch.as_tensor(width, dtype=torch.float32)
    chunk = int(args.pilot_batch_size)
    for start in range(0, y_all.shape[0], chunk):
        stop = min(start + chunk, y_all.shape[0])
        y_np = y_all[start:stop]
        var_y = y_np.reshape(y_np.shape[0], -1).var(axis=1)
        pi_grid = sc.sigmoid_np(grid_u)
        sigma2 = var_y[None, :] - (pi_grid * (1.0 - pi_grid) * grid_tau**2)[:, None]
        lam = 0.5 * np.log(np.clip(sigma2, math.exp(2.0 * low[2]), math.exp(2.0 * high[2])))
        cand = np.stack([np.broadcast_to(grid_u[:, None], lam.shape),
                         np.broadcast_to(grid_tau[:, None], lam.shape), lam], axis=-1)
        y_t = torch.as_tensor(y_np, dtype=torch.float32)
        with torch.no_grad():
            ll = torch.stack([
                _composite_loglik(y_t, torch.as_tensor(cand[g], dtype=torch.float32))
                for g in range(cand.shape[0])
            ])
        best = ll.argmax(dim=0).numpy()
        init = cand[best, np.arange(y_np.shape[0])]
        z = torch.as_tensor((init - low) / width, dtype=torch.float32).clamp(0, 1).requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=float(args.pilot_lr))
        scale = 1.0 / float(y_np.shape[1] * y_np.shape[2])
        for _ in range(int(args.pilot_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss = -_composite_loglik(y_t, low_t + width_t * z).sum() * scale
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                z.clamp_(0.0, 1.0)
        out[start:stop] = (low_t + width_t * z.detach()).numpy().astype(np.float64)
    tol = 1e-4 * width
    boundary = np.any((out <= low + tol) | (out >= high - tol), axis=1)
    status = np.where(boundary, "boundary", "interior").astype("U16")
    return out, status


# --------------------------------------------------------------------------- exact posterior
def _grid_loglik(block_sum: np.ndarray, block_sq: np.ndarray, m: int,
                 axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    u_axis, tau_axis, lam_axis = axes
    out = np.empty((u_axis.size, tau_axis.size, lam_axis.size), dtype=np.float64)
    K = block_sum.size
    tau = tau_axis[:, None, None]
    lam = lam_axis[None, :, None]
    inv_s2 = np.exp(-2.0 * lam)
    base = -K * m * lam[..., 0] - 0.5 * K * m * math.log(2.0 * math.pi) - 0.5 * block_sq.sum() * inv_s2[..., 0]
    log_ratio = (tau * block_sum[None, None, :] - 0.5 * m * tau**2) * inv_s2  # (T, L, K)
    for i, u in enumerate(u_axis):
        log_pi = -np.logaddexp(0.0, -u)
        log_1mpi = -np.logaddexp(0.0, u)
        out[i] = base + np.logaddexp(log_1mpi, log_pi + log_ratio).sum(axis=-1)
    return out


def _normalize(loglik: np.ndarray) -> np.ndarray:
    w = np.exp(loglik - loglik.max())
    return w / w.sum()


def _marginal_moments(axes, weights):
    means, sds = [], []
    for axis_index, axis in enumerate(axes):
        other = tuple(i for i in range(3) if i != axis_index)
        pmf = weights.sum(axis=other)
        mean = float(np.sum(pmf * axis))
        means.append(mean)
        sds.append(float(np.sqrt(max(np.sum(pmf * (axis - mean) ** 2), 0.0))))
    return np.asarray(means), np.asarray(sds)


def exact_posterior(y: np.ndarray, low: np.ndarray, high: np.ndarray, coarse: int, fine: int) -> dict:
    """Uniform-prior posterior on the box by a coarse-then-zoomed grid (evaluation only)."""
    y = np.asarray(y, dtype=np.float64)
    m = y.shape[1]
    block_sum, block_sq = y.sum(axis=1), (y**2).sum(axis=1)
    coarse_axes = tuple(np.linspace(low[i], high[i], coarse) for i in range(3))
    coarse_w = _normalize(_grid_loglik(block_sum, block_sq, m, coarse_axes))
    mean0, sd0 = _marginal_moments(coarse_axes, coarse_w)
    spacing = (high - low) / (coarse - 1)
    lo = np.maximum(low, mean0 - 8.0 * sd0 - 2.0 * spacing)
    hi = np.minimum(high, mean0 + 8.0 * sd0 + 2.0 * spacing)
    axes = tuple(np.linspace(lo[i], hi[i], fine) for i in range(3))
    weights = _normalize(_grid_loglik(block_sum, block_sq, m, axes))
    mean, sd = _marginal_moments(axes, weights)
    marginals = []
    edge_mass = 0.0
    for axis_index, axis in enumerate(axes):
        other = tuple(i for i in range(3) if i != axis_index)
        pmf = weights.sum(axis=other)
        cdf = np.cumsum(pmf)
        marginals.append({"grid": axis, "cdf": cdf,
                          "q05": float(np.interp(0.05, cdf, axis)), "q95": float(np.interp(0.95, cdf, axis))})
        if lo[axis_index] > low[axis_index]:
            edge_mass = max(edge_mass, float(pmf[0]))
        if hi[axis_index] < high[axis_index]:
            edge_mass = max(edge_mass, float(pmf[-1]))
    return {"axes": axes, "weights": weights, "mean": mean, "sd": sd, "marginals": marginals,
            "window_edge_mass": edge_mass}


def fixed_directions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def sliced_w1(exact: dict, samples_by_method: dict[str, np.ndarray], low: np.ndarray, high: np.ndarray,
              directions: np.ndarray, bins: int = 4000) -> dict[str, float]:
    """Mean over directions of W1 between projected exact grid mass and projected samples."""
    width = high - low
    mesh = np.meshgrid(*exact["axes"], indexing="ij")
    weights = exact["weights"].reshape(-1)
    keep = weights > 1e-13 * weights.max()
    coords = np.stack([(g.reshape(-1)[keep] - low[i]) / width[i] for i, g in enumerate(mesh)], axis=1)
    w = weights[keep] / weights[keep].sum()
    scaled = {k: (v - low) / width for k, v in samples_by_method.items()}
    totals = {k: 0.0 for k in samples_by_method}
    for direction in directions:
        proj = coords @ direction
        sample_proj = {k: v @ direction for k, v in scaled.items()}
        lo = min(proj.min(), *(p.min() for p in sample_proj.values()))
        hi = max(proj.max(), *(p.max() for p in sample_proj.values()))
        edges = np.linspace(lo, hi, bins + 1)
        cdf_exact = np.cumsum(np.histogram(proj, bins=edges, weights=w)[0])
        step = edges[1] - edges[0]
        for key, p in sample_proj.items():
            cdf_s = np.cumsum(np.histogram(p, bins=edges)[0]) / p.size
            totals[key] += float(np.abs(cdf_s - cdf_exact).sum() * step)
    return {k: v / directions.shape[0] for k, v in totals.items()}


# --------------------------------------------------------------------------- NPE
def train_npe(label: str, x: np.ndarray, theta_unit: np.ndarray, seed: int, args: argparse.Namespace):
    NPE, posterior_nn, BoxUniform = import_sbi()
    torch.manual_seed(int(seed))
    density = posterior_nn(
        model=args.sbi_model, z_score_theta="independent", z_score_x="independent",
        hidden_features=int(args.sbi_hidden_features), num_transforms=int(args.sbi_num_transforms),
        num_bins=int(args.sbi_num_bins), num_components=int(args.sbi_num_components),
    )
    prior = BoxUniform(low=torch.zeros(3), high=torch.ones(3))
    inference = NPE(prior=prior, density_estimator=density, device=args.device, show_progress_bars=False)
    started = time.time()
    estimator = inference.append_simulations(
        torch.as_tensor(theta_unit, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32),
        data_device=args.data_device,
    ).train(
        training_batch_size=int(args.sbi_batch_size), learning_rate=float(args.sbi_lr),
        max_num_epochs=int(args.max_epochs), validation_fraction=float(args.validation_fraction),
        stop_after_epochs=int(args.stop_after_epochs),
    )
    summary = inference._summary
    info = {"method": label, "seconds": round(time.time() - started, 2),
            "epochs_trained": int(summary["epochs_trained"][-1]),
            "best_validation_loss": float(summary["best_validation_loss"][-1])}
    print("NPE", info, flush=True)
    return inference.build_posterior(estimator), estimator, info


def sample_posterior(posterior, x_obs: np.ndarray, n: int, seed: int, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    torch.manual_seed(int(seed))
    unit = posterior.sample((int(n),), x=torch.as_tensor(x_obs, dtype=torch.float32), show_progress_bars=False)
    return low + (high - low) * unit.detach().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- summaries
def summarize(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys) + (row["method"],), []).append(row)
    out = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        record = {k: v for k, v in zip(keys + ("method",), key)}
        record["n"] = len(group)
        for c in COORDS:
            record[f"mse_{c}"] = float(np.mean([float(r[f"sq_err_{c}"]) for r in group]))
            record[f"rmse_to_exact_{c}"] = float(math.sqrt(np.mean([float(r[f"mean_sq_err_to_exact_{c}"]) for r in group])))
            record[f"w1_{c}"] = float(np.mean([float(r[f"w1_{c}"]) for r in group]))
            record[f"post_sd_{c}"] = float(np.mean([float(r[f"post_sd_{c}"]) for r in group]))
            record[f"exact_sd_{c}"] = float(np.mean([float(r[f"exact_sd_{c}"]) for r in group]))
            record[f"sd_abs_err_{c}"] = float(np.mean([float(r[f"sd_abs_err_{c}"]) for r in group]))
            record[f"cov90_{c}"] = float(np.mean([float(r[f"cov90_{c}"]) for r in group]))
        sw = [float(r["sliced_w1"]) for r in group]
        record["sliced_w1"] = float(np.mean(sw))
        record["sliced_w1_se"] = standard_error(sw)
        out.append(record)
    return out


def paired_sliced_w1(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = {(r["beta_index"], r["seed"], r["method"]): float(r["sliced_w1"]) for r in rows if r["method"] != EXACT}
    present = {m for _, _, m in values}
    out = []
    for a, b in PAIRED_CONTRASTS:
        la, lb = METHOD_LABELS[a], METHOD_LABELS[b]
        if la not in present or lb not in present:
            continue
        diff = np.asarray([v - values[(i, s, lb)] for (i, s, m), v in values.items()
                           if m == la and (i, s, lb) in values])
        base = np.mean([v for (i, s, m), v in values.items() if m == lb])
        out.append({"reference": la, "competitor": lb, "n_pairs": int(diff.size),
                    "mean_paired_difference": float(diff.mean()), "se": standard_error(diff),
                    "relative_percent": float(100.0 * diff.mean() / base),
                    "reference_wins": int(np.sum(diff < 0))})
    return out


# --------------------------------------------------------------------------- run
def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m for m in METHOD_ORDER if m in parse_list(args.methods, str)]
    if set(parse_list(args.methods, str)) - set(METHOD_ORDER):
        raise ValueError(f"--methods must use {METHOD_ORDER}")
    score_methods = [m for m in methods if m != "pilot"]
    stage1_dir = Path(args.stage1_run_dir)
    config = json.loads((stage1_dir / "config.json").read_text(encoding="utf-8"))
    low, high = box_from_config(config)
    n_blocks, block_size = int(config["n_blocks"]), int(config["block_size"])
    frozen = {m: FrozenScore(stage1_dir, m, args.device) for m in score_methods}
    forbidden = {"beta", "beta_true", "theta", "pi", "pi_true"}
    if forbidden.intersection(inspect.signature(data_only_pilot).parameters):
        raise AssertionError("data_only_pilot exposes an oracle parameter")
    print("Stage-1 run:", stage1_dir, "box:", low.round(4).tolist(), high.round(4).tolist(), flush=True)

    # Pilot sanity: determinism and correlation with the truth on a small bank.
    rng = np.random.default_rng(int(args.sanity_seed))
    beta_check = rng.uniform(low, high, size=(int(args.pilot_sanity_n), 3))
    y_check = sc.simulate(rng, beta_check, n_blocks, block_size)
    first, _ = data_only_pilot(y_check, low, high, args)
    second, _ = data_only_pilot(y_check, low, high, args)
    if not np.array_equal(first, second):
        raise AssertionError("data-only pilot is not deterministic")
    corr = [float(np.corrcoef(first[:, i], beta_check[:, i])[0, 1]) for i in range(3)]
    if min(corr) < float(args.min_pilot_correlation):
        raise AssertionError(f"pilot correlation too low: {corr}")
    write_csv(out_dir / "pilot_sanity.csv", [{"coord": c, "corr_pilot_true": v} for c, v in zip(COORDS, corr)])
    print("pilot sanity corr:", dict(zip(COORDS, np.round(corr, 4))), flush=True)

    rng = np.random.default_rng(int(args.sbi_train_seed))
    beta_train = rng.uniform(low, high, size=(int(args.n_sbi_train), 3))
    y_train = sc.simulate(rng, beta_train, n_blocks, block_size)
    started = time.time()
    pilot_train, status_train = data_only_pilot(y_train, low, high, args)
    print("training pilot seconds:", round(time.time() - started, 1), flush=True)
    contexts = {m: np.concatenate([pilot_train, frozen[m].score(y_train, pilot_train)], axis=1).astype(np.float32)
                for m in score_methods}
    if "pilot" in methods:
        contexts["pilot"] = pilot_train.astype(np.float32)
    del y_train
    err = pilot_train - beta_train
    write_csv(out_dir / "training_context_summary.csv", [{
        "coord": c, "pilot_rmse": float(np.sqrt(np.mean(err[:, i] ** 2))), "pilot_bias": float(err[:, i].mean()),
        "boundary_rate": float(np.mean(status_train == "boundary")),
        **{f"score_sd_{m}": float(contexts[m][:, 3 + i].std()) for m in score_methods},
    } for i, c in enumerate(COORDS)])
    np.savez_compressed(out_dir / "training_contexts.npz", beta_train=beta_train, pilot=pilot_train,
                        pilot_status=status_train, **{f"context_{m}": v for m, v in contexts.items()})

    theta_unit = ((beta_train - low) / (high - low)).clip(1e-5, 1 - 1e-5)
    posteriors, npe_rows = {}, []
    for m in methods:
        posterior, estimator, info = train_npe(METHOD_LABELS[m], contexts[m], theta_unit, int(args.sbi_seed), args)
        posteriors[m] = posterior
        npe_rows.append(info)
        torch.save({"method": m, "state_dict": estimator.state_dict(), "config": vars(args) | {"output_dir": str(out_dir)},
                    "anchor_low": low.tolist(), "anchor_high": high.tolist(),
                    "context": "beta_pilot(Y)" if m == "pilot" else CONTEXT_DEFINITION},
                   out_dir / f"npe_{m}_state.pt")
    write_csv(out_dir / "npe_training_summary.csv", npe_rows)

    test_betas = parse_betas(args.test_betas)
    if np.any(test_betas < low) or np.any(test_betas > high):
        raise ValueError("a test beta lies outside the prior box")
    directions = fixed_directions(int(args.sliced_directions), int(args.sliced_seed))
    rows: list[dict[str, object]] = []
    started = time.time()
    for beta_index, beta_true in enumerate(test_betas):
        for obs_seed in parse_list(args.test_seeds, int):
            rng = np.random.default_rng(int(obs_seed) * 1000 + beta_index)
            y_obs = sc.simulate(rng, beta_true[None, :], n_blocks, block_size)
            pilot_obs, status_obs = data_only_pilot(y_obs, low, high, args)
            obs_contexts = {m: np.concatenate([pilot_obs, frozen[m].score(y_obs, pilot_obs)], axis=1)[0]
                            for m in score_methods}
            obs_contexts["pilot"] = pilot_obs[0]
            exact = exact_posterior(y_obs[0], low, high, int(args.coarse_grid_size), int(args.grid_size))
            samples = {m: sample_posterior(posteriors[m], obs_contexts[m], int(args.posterior_n),
                                           int(args.posterior_seed) + int(obs_seed) * 100 + beta_index, low, high)
                       for m in methods}
            sliced = sliced_w1(exact, samples, low, high, directions)
            pi_true, tau_true, sigma_true = float(sc.sigmoid_np(beta_true[0])), float(beta_true[1]), float(math.exp(beta_true[2]))
            common = {"beta_index": beta_index, "seed": int(obs_seed), "pi_true": pi_true, "tau_true": tau_true,
                      "sigma_true": sigma_true, "pilot_status": str(status_obs[0]),
                      "window_edge_mass": exact["window_edge_mass"],
                      **{f"pilot_{c}": float(pilot_obs[0, i]) for i, c in enumerate(COORDS)}}
            exact_row = {"method": EXACT, **common}
            for i, c in enumerate(COORDS):
                marg = exact["marginals"][i]
                exact_row.update({
                    f"post_mean_{c}": exact["mean"][i], f"post_sd_{c}": exact["sd"][i],
                    f"exact_mean_{c}": exact["mean"][i], f"exact_sd_{c}": exact["sd"][i],
                    f"sq_err_{c}": float((exact["mean"][i] - beta_true[i]) ** 2),
                    f"mean_sq_err_to_exact_{c}": 0.0, f"sd_abs_err_{c}": 0.0,
                    f"cov90_{c}": float(marg["q05"] <= beta_true[i] <= marg["q95"]), f"w1_{c}": 0.0,
                })
            exact_row["sliced_w1"] = 0.0
            rows.append(exact_row)
            for m in methods:
                s = samples[m]
                row = {"method": METHOD_LABELS[m], **common}
                for i, c in enumerate(COORDS):
                    marg = exact["marginals"][i]
                    mean, sd = float(s[:, i].mean()), float(s[:, i].std())
                    q05, q95 = np.quantile(s[:, i], [0.05, 0.95])
                    row.update({
                        f"post_mean_{c}": mean, f"post_sd_{c}": sd,
                        f"exact_mean_{c}": exact["mean"][i], f"exact_sd_{c}": exact["sd"][i],
                        f"sq_err_{c}": float((mean - beta_true[i]) ** 2),
                        f"mean_sq_err_to_exact_{c}": float((mean - exact["mean"][i]) ** 2),
                        f"sd_abs_err_{c}": abs(sd - exact["sd"][i]),
                        f"cov90_{c}": float(q05 <= beta_true[i] <= q95),
                        f"w1_{c}": exact_sample_w1(s[:, i], marg["grid"], marg["cdf"]),
                    })
                row["sliced_w1"] = sliced[m]
                rows.append(row)
        print(f"test beta {beta_index + 1}/{len(test_betas)} done; {time.time() - started:.0f}s", flush=True)

    write_csv(out_dir / "posterior_by_seed.csv", rows)
    write_csv(out_dir / "posterior_summary_by_beta.csv", summarize(rows, ("beta_index", "pi_true", "tau_true", "sigma_true")))
    pooled = summarize([{**r, "pooled": "all"} for r in rows], ("pooled",))
    write_csv(out_dir / "posterior_summary_pooled.csv", pooled)
    write_csv(out_dir / "paired_sliced_w1.csv", paired_sliced_w1(rows))
    (out_dir / "config.json").write_text(json.dumps(
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
         "stage1_run_dir": str(stage1_dir), "anchor_low": low.tolist(), "anchor_high": high.tolist(),
         "prior": "uniform on the anchor box in (logit pi, tau, log sigma)",
         "context_definition": CONTEXT_DEFINITION,
         "true_parameter_role": "NPE target only; never passed to the pilot or context builder",
         "multivariate_distance": "sliced W1 in box-standardized coordinates"}, indent=2), encoding="utf-8")
    print("\n==== p=3 pooled summary ====")
    for row in pooled:
        print(f"{row['method']:<38} sliced W1 {row['sliced_w1']:.5f}  "
              + "  ".join(f"W1_{c} {row['w1_' + c]:.5f}" for c in COORDS), flush=True)
    for row in paired_sliced_w1(rows):
        print(row, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-run-dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="pilot,linear,gate,stacked")
    parser.add_argument("--n-sbi-train", type=int, default=50_000)
    parser.add_argument("--sbi-train-seed", type=int, default=20260723)
    parser.add_argument("--sbi-model", choices=("mdn", "maf", "nsf"), default="mdn")
    parser.add_argument("--sbi-hidden-features", type=int, default=64)
    parser.add_argument("--sbi-num-components", type=int, default=8)
    parser.add_argument("--sbi-num-transforms", type=int, default=5)
    parser.add_argument("--sbi-num-bins", type=int, default=8)
    parser.add_argument("--sbi-batch-size", type=int, default=256)
    parser.add_argument("--sbi-lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--sbi-seed", type=int, default=54_000)
    parser.add_argument("--test-betas", type=str, default=";".join(
        f"{p}:{t}:{s}" for p in (0.10, 0.30, 0.65) for t in (0.8, 1.5) for s in (0.85, 1.2)))
    parser.add_argument("--test-seeds", type=str, default="100,101,102,103,104,105,106,107,108,109")
    parser.add_argument("--posterior-n", type=int, default=5_000)
    parser.add_argument("--posterior-seed", type=int, default=87_000)
    parser.add_argument("--coarse-grid-size", type=int, default=40)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--sliced-directions", type=int, default=100)
    parser.add_argument("--sliced-seed", type=int, default=20260913)
    parser.add_argument("--pilot-grid-points", type=int, default=9)
    parser.add_argument("--pilot-steps", type=int, default=300)
    parser.add_argument("--pilot-lr", type=float, default=0.01)
    parser.add_argument("--pilot-batch-size", type=int, default=1024)
    parser.add_argument("--pilot-sanity-n", type=int, default=128)
    parser.add_argument("--min-pilot-correlation", type=float, default=0.5)
    parser.add_argument("--sanity-seed", type=int, default=20260710)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/model1_p3_stage2"))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
