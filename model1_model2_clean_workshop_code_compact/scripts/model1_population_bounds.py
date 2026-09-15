#!/usr/bin/env python3
"""Population bounds for the Model 1 setting comparison, computed without training.

For each data-generating setting (K blocks of m observations, shift tau), tube
width sigma_q and generating value pi, with a = logit(pi):

  I          exact Fisher information in u = logit(pi), by 1-D quadrature
  eff_lin    Godambe efficiency G_lin / I of the best linear composite score
             (the equally weighted sum of local scores), by quadrature
  D          fraction of exact-score variance no readout of ILSA's pooled local
             scores can recover (Monte Carlo over blocks, binning, held out)
  F          standardized error of the tube-smoothed FSM target s_sigma against
             the exact score g (Monte Carlo over datasets, quadrature in u)
  kappa      E[s_sigma g] / E[g^2], the tube's shrinkage of the score
  bound_nlsa best standardized exact-score MSE with unlimited data and capacity:
             F, since local scores are invertible and NLSA can recover L
  bound_ilsa F + D * (1 - (1 - kappa)^2)
  gain_max   1 - bound_nlsa / bound_ilsa, the largest possible Stage-1 gain
  w1_ilsa    mean W1 (pi units) between the exact posterior and the posterior
             given only the per-block pooled scores evaluated at the generating
             value. A reference, not a bound on Stage 2: the Stage-2 context also
             holds the pilot, which uses pooled scores at 201 evaluation points.
             The NLSA counterpart is 0.

The two bounds treat observed information as constant across datasets, which is
accurate for K >= 20; they approximate the projection onto the block-additive
class rather than compute it exactly. No checkpoint, NPE or exact posterior from
a training run is used; observed values from runs/setting are only listed beside.

    python scripts/model1_population_bounds.py
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETTING_RUNS = ROOT / "runs" / "setting"
SETTINGS = {  # name: (K, m, tau, run tag, queued sigma_q strings)
    "current": (20, 20, 0.5, "base_k20m20t0.5", ["0.20"]),
    "A": (20, 20, 0.7, "k20m20t0.7", ["0.20", "0.168"]),
    "B": (80, 5, 1.1, "k80m5t1.1", ["0.20", "0.095"]),
}
PI_VALUES = (0.07, 0.10, 0.30, 0.50, 0.65)
GRID_SIGMAS = (0.05, 0.10, 0.15, 0.20, 0.30)
GH_X, GH_W = np.polynomial.hermite.hermgauss(160)


def expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softplus(x):
    return np.logaddexp(0.0, x)


def logit(p: float) -> float:
    return math.log(p) - math.log1p(-p)


def normal_expect(f, mean: float, sd: float) -> float:
    return float(np.sum(GH_W * f(mean + math.sqrt(2.0) * sd * GH_X)) / math.sqrt(math.pi))


def information(pi: float, K: int, m: int, tau: float) -> tuple[float, float]:
    """Exact Fisher information and Godambe information of the summed local scores."""
    a, v = logit(pi), tau * tau
    h = lambda l: expit(a + l) - pi  # noqa: E731
    info_block = 0.0
    es, es2, esb = {}, {}, {}
    for active, prob in ((1, pi), (0, 1.0 - pi)):
        sign = 1.0 if active else -1.0
        info_block += prob * normal_expect(lambda L: h(L) ** 2, sign * m * v / 2, math.sqrt(m * v))
        es[active] = normal_expect(h, sign * v / 2, tau)
        es2[active] = normal_expect(lambda l: h(l) ** 2, sign * v / 2, tau)
        if m == 1:
            esb[active] = es2[active]
        else:
            l = sign * v / 2 + math.sqrt(2.0) * tau * GH_X
            r = sign * (m - 1) * v / 2 + math.sqrt(2.0 * (m - 1) * v) * GH_X
            esb[active] = float(GH_W @ (h(l)[:, None] * h(l[:, None] + r[None, :])) @ GH_W) / math.pi
    sens = K * m * (pi * esb[1] + (1 - pi) * esb[0])
    var = K * (m * (pi * es2[1] + (1 - pi) * es2[0]) + m * (m - 1) * (pi * es[1] ** 2 + (1 - pi) * es[0] ** 2))
    return K * info_block, sens * sens / var


def pooling(pi: float, m: int, tau: float, n: int, bins: int, rng) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """Pooling loss D, and the block log-likelihood ratio of the pooled score A."""
    a = logit(pi)
    A, B, Z = np.empty(n), np.empty(n), np.empty(n)
    for start in range(0, n, 200_000):
        size = min(200_000, n - start)
        active = rng.random(size) < pi
        l = tau * (rng.standard_normal((size, m)) + tau * active[:, None]) - tau * tau / 2
        A[start:start + size] = (expit(a + l) - pi).mean(axis=1)
        B[start:start + size] = expit(a + l.sum(axis=1)) - pi
        Z[start:start + size] = active
    half = n // 2
    edges = np.quantile(A[:half], np.linspace(0, 1, bins + 1)[1:-1])
    fit = np.searchsorted(edges, A[:half])
    count = np.maximum(np.bincount(fit, minlength=bins), 1)
    mean_b = np.bincount(fit, weights=B[:half], minlength=bins) / count
    D = float(np.mean((B[half:] - mean_b[np.searchsorted(edges, A[half:])]) ** 2) / B[half:].var())
    # P(active | A) on the full sample gives the likelihood ratio of A by Bayes' rule.
    idx = np.searchsorted(edges, A)
    count = np.maximum(np.bincount(idx, minlength=bins), 1)
    centre = np.bincount(idx, weights=A, minlength=bins) / count
    p1 = np.clip(np.bincount(idx, weights=Z, minlength=bins) / count, 0.5 / count, 1 - 0.5 / count)
    log_ratio = np.maximum.accumulate(np.log(p1) - np.log1p(-p1) - a)
    return D, (centre, log_ratio)


def smoothing(pi: float, K: int, m: int, tau: float, sigmas, n_ds: int, rng) -> dict[float, tuple[float, float]]:
    a, v = logit(pi), tau * tau
    active = rng.random((n_ds, K)) < pi
    L = tau * math.sqrt(m) * rng.standard_normal((n_ds, K)) + m * v * active - m * v / 2
    g0 = (expit(a + L) - pi).sum(axis=1)
    z = np.linspace(-7.0, 7.0, 201)
    out = {}
    for sigma in sigmas:
        u = a + sigma * z
        s = np.empty(n_ds)
        for start in range(0, n_ds, 100):
            Lc = L[start:start + 100][:, None, :] + u[None, :, None]
            loglik = softplus(Lc).sum(axis=2) - K * softplus(u)[None, :] - z * z / 2
            w = np.exp(loglik - loglik.max(axis=1, keepdims=True))
            g = (expit(Lc) - expit(u)[None, :, None]).sum(axis=2)
            s[start:start + 100] = (w * g).sum(axis=1) / w.sum(axis=1)
        second = float(np.mean(g0 ** 2))
        out[sigma] = (float(np.mean((s - g0) ** 2)) / second, float(np.mean(s * g0)) / second)
    return out


def posterior_gap(pi: float, K: int, m: int, tau: float, ratio, n_post: int, rng) -> float:
    a = logit(pi)
    grid = np.linspace(0.05, 0.70, 1500)
    ug, step = np.log(grid) - np.log1p(-grid), grid[1] - grid[0]
    centre, log_ratio = ratio
    w1 = []
    for start in range(0, n_post, 50):
        size = min(50, n_post - start)
        active = rng.random((size, K)) < pi
        l = tau * (rng.standard_normal((size, K, m)) + tau * active[..., None]) - tau * tau / 2
        exact = l.sum(axis=2)
        pooled = np.interp((expit(a + l) - pi).mean(axis=2), centre, log_ratio)
        cdfs = []
        for r in (exact, pooled):
            ll = softplus(ug[None, :, None] + r[:, None, :]).sum(axis=2) - K * softplus(ug)[None, :]
            w = np.exp(ll - ll.max(axis=1, keepdims=True))
            cdfs.append(np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1))
        w1.extend(np.sum(np.abs(cdfs[0] - cdfs[1]), axis=1) * step)
    return float(np.mean(w1))


def observed(tag: str, sq: str) -> dict[tuple[str, float], tuple[float, int]]:
    """Mean over finished seeds of Stage-1 std MSE ('s1') and Stage-2 W1 ('s2') per arm and pi."""
    acc: dict[tuple[str, str, float], list[float]] = {}
    for run in SETTING_RUNS.glob(f"s1_{tag}_sq{sq}_*"):
        path = run / "score_summary_by_pi.csv"
        if path.exists():
            for r in csv.DictReader(path.open()):
                arm = r["method"].split()[0]
                acc.setdefault(("s1", arm, round(float(r["pi"]), 4)), []).append(float(r["std_mse"]))
    for run in SETTING_RUNS.glob(f"s2_{tag}_sq{sq}_*"):
        path = run / "posterior_by_seed.csv"
        if not path.exists():
            continue
        per: dict[tuple[str, float], list[float]] = {}
        for r in csv.DictReader(path.open()):
            arm = r["method"].split()[0]
            if arm in ("linear", "radial", "stacked"):
                per.setdefault((arm, round(float(r["pi_true"]), 4)), []).append(float(r["w1_to_exact"]))
        for (arm, pi), vals in per.items():
            acc.setdefault(("s2", arm, pi), []).append(float(np.mean(vals)))
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


def fmt(x, digits=4):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-blocks-mc", type=int, default=1_000_000)
    p.add_argument("--bins", type=int, default=1000)
    p.add_argument("--n-datasets", type=int, default=2000)
    p.add_argument("--n-posterior", type=int, default=400)
    p.add_argument("--seed", type=int, default=20260916)
    p.add_argument("--output", default=str(ROOT / "new_results" / "MODEL1_POPULATION_BOUNDS_TABLES.md"))
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    rows, grid_rows = [], []
    for name, (K, m, tau, tag, queued) in SETTINGS.items():
        sigmas = sorted({float(s) for s in queued} | set(GRID_SIGMAS))
        for pi in PI_VALUES:
            I, G = information(pi, K, m, tau)
            D, ratio = pooling(pi, m, tau, args.n_blocks_mc, args.bins, rng)
            smooth = smoothing(pi, K, m, tau, sigmas, args.n_datasets, rng)
            w1 = posterior_gap(pi, K, m, tau, ratio, args.n_posterior, rng)
            for sigma, (F, kappa) in smooth.items():
                b_nlsa, b_ilsa = F, F + D * (1 - (1 - kappa) ** 2)
                row = {"setting": name, "K": K, "m": m, "tau": tau, "sigma_q": sigma, "pi": pi,
                       "I": I, "sigma_q_sqrtI": sigma * math.sqrt(I), "eff_lin": G / I, "D": D,
                       "F": F, "kappa": kappa, "bound_ilsa": b_ilsa, "bound_nlsa": b_nlsa,
                       "gain_max": 1 - b_nlsa / b_ilsa, "w1_ilsa": w1}
                grid_rows.append(row)
                match = [s for s in queued if abs(float(s) - sigma) < 1e-9]
                if match:
                    obs = observed(tag, match[0])
                    for stage in ("s1", "s2"):
                        for arm in ("linear", "radial", "stacked"):
                            val = obs.get((stage, arm, round(pi, 4)))
                            row[f"obs_{stage}_{arm}"] = val[0] if val else math.nan
                            row[f"n_{stage}"] = max(row.get(f"n_{stage}", 0), val[1] if val else 0)
                    rows.append(row)
            print(f"{name} pi={pi}: I={I:.3f} eff_lin={G / I:.3f} D={D:.4f} w1_ilsa={w1:.5f}", flush=True)

    csv_path = SETTING_RUNS / "population_bounds.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(grid_rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(grid_rows)

    L = ["# Model 1 population bounds by setting", "",
         "Generated by `scripts/model1_population_bounds.py`; do not edit by hand. Bounds assume unlimited "
         "data and network capacity; observed values are means over the finished seeds in `runs/setting`.", "",
         "## Stage 1: best reachable standardized exact-score MSE at the queued tube widths", "",
         "| setting | σ_q | π | σ_q√I | D | F | bound ILSA | bound NLSA | max gain | observed ILSA | observed gate | observed stacked | seeds |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['setting']} | {r['sigma_q']} | {r['pi']} | {fmt(r['sigma_q_sqrtI'], 3)} | {fmt(r['D'])} | "
                 f"{fmt(r['F'])} | {fmt(r['bound_ilsa'])} | {fmt(r['bound_nlsa'])} | {100 * r['gain_max']:.0f}% | "
                 f"{fmt(r['obs_s1_linear'])} | {fmt(r['obs_s1_radial'])} | {fmt(r['obs_s1_stacked'])} | {r['n_s1']} |")
    L += ["", "### Pooled over π (ratio of means, as in the setting-comparison tables)", "",
          "| setting | σ_q | bound ILSA | bound NLSA | max gain | observed gate gain | observed stacked gain | seeds |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in SETTINGS:
        for sigma in sorted({r["sigma_q"] for r in rows if r["setting"] == name}):
            sub = [r for r in rows if r["setting"] == name and r["sigma_q"] == sigma]
            bi, bn = np.mean([r["bound_ilsa"] for r in sub]), np.mean([r["bound_nlsa"] for r in sub])
            ol = np.mean([r["obs_s1_linear"] for r in sub])
            og = np.mean([r["obs_s1_radial"] for r in sub])
            os_ = np.mean([r["obs_s1_stacked"] for r in sub])
            L.append(f"| {name} | {sigma} | {bi:.4f} | {bn:.4f} | {100 * (1 - bn / bi):.0f}% | "
                     f"{'—' if np.isnan(ol) else f'{100 * (og / ol - 1):+.0f}%'} | "
                     f"{'—' if np.isnan(ol) else f'{100 * (os_ / ol - 1):+.0f}%'} | {sub[0]['n_s1']} |")
    L += ["", "## Tube width: largest possible Stage-1 gain, pooled over π", "",
          "| setting | " + " | ".join(f"σ_q {s}" for s in GRID_SIGMAS) + " |",
          "|---|" + "---:|" * len(GRID_SIGMAS)]
    for name in SETTINGS:
        cells = []
        for s in GRID_SIGMAS:
            sub = [r for r in grid_rows if r["setting"] == name and abs(r["sigma_q"] - s) < 1e-9]
            bi, bn = np.mean([r["bound_ilsa"] for r in sub]), np.mean([r["bound_nlsa"] for r in sub])
            cells.append(f"{100 * (1 - bn / bi):.0f}% (NLSA {bn:.3f})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "## Information and the posterior gap (independent of σ_q)", "",
          "W1 pooled is the W1 between the exact posterior and the posterior given only each block's pooled "
          "local score at the generating value; NLSA's counterpart is 0. It is a reference, not a bound: the "
          "Stage-2 context also holds the pilot, computed from pooled scores at 201 evaluation points. "
          "Observed W1 is from Stage 2 at the first queued tube width.", "",
          "| setting | π | I | linear efficiency | D | W1 pooled | observed W1 ILSA | observed W1 gate | observed gap | seeds |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, (_, _, _, _, queued) in SETTINGS.items():
        for r in [r for r in rows if r["setting"] == name and abs(r["sigma_q"] - float(queued[0])) < 1e-9]:
            gap = r["obs_s2_linear"] - r["obs_s2_radial"]
            L.append(f"| {name} | {r['pi']} | {r['I']:.3f} | {r['eff_lin']:.3f} | {r['D']:.4f} | {r['w1_ilsa']:.5f} | "
                     f"{fmt(r['obs_s2_linear'], 5)} | {fmt(r['obs_s2_radial'], 5)} | {fmt(gap, 5)} | {r['n_s2']} |")
    Path(args.output).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("wrote", args.output, "and", csv_path)


if __name__ == "__main__":
    main()
