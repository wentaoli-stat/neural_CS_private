#!/usr/bin/env python3
"""Stage-2 reference for Model 1: the pilot+score NPE fed ideal scores instead of trained networks.

Everything matches scripts/run_model1_stage2_npe_stacked.sh: prior, the 50,000-simulation bank and
its seed, the data-only pilot, the MDN recipe and NPE seed, the 60 test datasets, posterior seeds,
the exact grid posterior and W1. Only the score in the context (u_pilot, score) changes:

  ideal_nlsa  score = g(Y, u_pilot), the exact full score at the pilot. NLSA can represent it,
              since local scores are invertible and recover every block likelihood ratio.
  ideal_ilsa  score = sum_k m(A_k; u_pilot), where A_k is block k's pooled local score at the
              pilot and m(A; a) = E[h_a(L) | A] under data generated at a: the population ILSA
              score as the tube width goes to zero.

The tube's shrinkage of the score is a function of the pilot alone, so it does not change the
information in the pair; the reference applies at every sigma_q and every Stage-1 seed. Against
observed Stage-2 runs on the same streams:
  observed - ideal          the Stage-1 learning error that reaches the posterior
  ideal_ilsa - ideal_nlsa   the posterior gap this pipeline can reach
The NPE keeps its finite simulations and capacity, so ideal_nlsa's W1 is not zero.

The generating parameter only simulates data and is the NPE target, as in Stage 2. m(A; a) is
built from simulations on a grid of evaluation points spanning the prior, never at a test value.

    python scripts/model1_stage2_reference.py --settings current,A,B
    python scripts/model1_stage2_reference.py --summary-only
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model1 import npe_utils, stage1  # noqa: E402
from model1 import stage2_npe as s2  # noqa: E402

SETTING_RUNS = ROOT / "runs" / "setting"
OUT_ROOT = SETTING_RUNS / "stage2_reference"
SETTINGS = {  # name: (K, m, tau, Stage-1/2 run tag)
    "current": (20, 20, 0.5, "base_k20m20t0.5"),
    "A": (20, 20, 0.7, "k20m20t0.7"),
    "B": (80, 5, 1.1, "k80m5t1.1"),
}
LABELS = {"ideal_nlsa": "ideal NLSA (exact score at pilot)", "ideal_ilsa": "ideal ILSA (pooled scores at pilot)"}
LAUNCHER_ARGS = [
    "--stage1-run-dir", "unused", "--pi-prior-min", "0.05", "--pi-prior-max", "0.70",
    "--n-sbi-train", "50000", "--sbi-train-seed", "20260723", "--sbi-model", "mdn",
    "--sbi-hidden-features", "64", "--sbi-num-components", "8", "--sbi-batch-size", "256",
    "--sbi-lr", "5e-4", "--max-epochs", "300", "--validation-fraction", "0.10",
    "--stop-after-epochs", "20", "--sbi-seed", "54000", "--shared-npe-seed", "1",
    "--test-pi-values", "0.07,0.10,0.30,0.50,0.65,0.68",
    "--test-seeds", "100,101,102,103,104,105,106,107,108,109",
    "--posterior-n", "5000", "--posterior-seed", "87000", "--shared-posterior-seed", "1",
    "--grid-size", "5000", "--context-score-batch-size", "256", "--pilot-grid-size", "201",
    "--pilot-batch-size", "512", "--pilot-grid-chunk-size", "16", "--pilot-progress-every", "10",
    "--device", "cpu", "--data-device", "cpu",
]


def expit(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def ilsa_lookup(m: int, tau: float, u_min: float, u_max: float, n_grid: int, n_blocks: int,
                bins: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E[h_a(L) | A] on equal-count bins of A, for evaluation points a on a grid over the prior."""
    a_grid = np.linspace(u_min, u_max, n_grid)
    per_bin = n_blocks // bins
    n = per_bin * bins
    centres, means = np.empty((n_grid, bins)), np.empty((n_grid, bins))
    for i, a in enumerate(a_grid):
        pi = float(expit(a))
        A, B = np.empty(n), np.empty(n)
        for start in range(0, n, 200_000):
            size = min(200_000, n - start)
            active = rng.random(size) < pi
            l = tau * (rng.standard_normal((size, m)) + tau * active[:, None]) - tau * tau / 2
            A[start:start + size] = (expit(a + l) - pi).mean(axis=1)
            B[start:start + size] = expit(a + l.sum(axis=1)) - pi
        order = np.argsort(A)
        centres[i] = A[order].reshape(bins, per_bin).mean(axis=1)
        means[i] = B[order].reshape(bins, per_bin).mean(axis=1)
    return a_grid, centres, means


def ideal_contexts(y: np.ndarray, pilot_u: np.ndarray, tau: float, lookup) -> dict[str, np.ndarray]:
    a_grid, centres, means = lookup
    exact = np.empty(y.shape[0])
    pooled = np.empty(y.shape[0])
    for start in range(0, y.shape[0], 2000):
        u = pilot_u[start:start + 2000]
        l = tau * y[start:start + 2000] - tau * tau / 2
        p = expit(u)
        exact[start:start + 2000] = (expit(u[:, None] + l.sum(axis=2)) - p[:, None]).sum(axis=1)
        A = (expit(u[:, None, None] + l) - p[:, None, None]).mean(axis=2)
        cell = np.clip(np.searchsorted(a_grid, u) - 1, 0, a_grid.size - 2)
        w = (u - a_grid[cell]) / (a_grid[cell + 1] - a_grid[cell])
        out = np.empty(u.size)
        for c in np.unique(cell):
            idx = np.flatnonzero(cell == c)
            lo = np.interp(A[idx], centres[c], means[c])
            hi = np.interp(A[idx], centres[c + 1], means[c + 1])
            out[idx] = ((1 - w[idx])[:, None] * lo + w[idx][:, None] * hi).sum(axis=1)
        pooled[start:start + 2000] = out
    return {"ideal_nlsa": np.column_stack([pilot_u, exact]).astype(np.float32),
            "ideal_ilsa": np.column_stack([pilot_u, pooled]).astype(np.float32)}


def replicate_args(rep: int) -> list[str]:
    """Replicate REP >= 1 shifts every Stage-2 stream as run_model1_stage2_npe_stacked_rep.sh does."""
    argv = list(LAUNCHER_ARGS)
    if rep:
        shifts = {"--sbi-train-seed": str(20260723 + rep), "--sbi-seed": str(54000 + 1000 * rep),
                  "--posterior-seed": str(87000 + 1000 * rep),
                  "--test-seeds": ",".join(str(100 + 10 * rep + i) for i in range(10))}
        for flag, value in shifts.items():
            argv[argv.index(flag) + 1] = value
    return argv


def run_setting(name: str, rep: int, lookup_blocks: int, lookup_grid: int, bins: int) -> None:
    K, m, tau, tag = SETTINGS[name]
    out_dir = OUT_ROOT / (name if rep == 0 else f"{name}_rep{rep}")
    if (out_dir / "posterior_by_seed.csv").exists():
        print(f"{out_dir.name}: already done")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    args = s2.build_parser().parse_args([*replicate_args(rep), "--output-dir", str(out_dir)])
    args.device = npe_utils.resolve_device(str(args.device))
    args.tau = tau
    metadata = {"n_blocks": K, "block_size": m, "tau": tau, "pi_min": 0.05, "pi_max": 0.70}
    started = time.time()

    rng = np.random.default_rng(int(args.sbi_train_seed))
    pi_train = rng.uniform(args.pi_prior_min, args.pi_prior_max, size=int(args.n_sbi_train))
    y_train = s2.simulate(rng, pi_train, metadata)
    # Observed Stage-2 runs cached pilots only for the replicate-0 bank.
    cache = None if rep else next(iter(sorted(SETTING_RUNS.glob(f"s2_{tag}_sq*_*/pilot_cache.npz"))), None)
    if cache is not None:
        with np.load(cache) as c:
            if not np.allclose(c["pi_train"], pi_train, rtol=0, atol=2e-7) or not np.isclose(float(c["tau"]), tau):
                raise ValueError(f"pilot cache does not match this bank: {cache}")
            pilot_train = np.asarray(c["pilot_u"], dtype=np.float64)
        print(f"{name}: pilot cache {cache}", flush=True)
    else:
        pilot_train, _ = s2.data_only_pilot(y_train, metadata, args)

    u_min, u_max = stage1.logit_np(args.pi_prior_min), stage1.logit_np(args.pi_prior_max)
    lookup = ilsa_lookup(m, tau, float(u_min), float(u_max), lookup_grid, lookup_blocks, bins,
                         np.random.default_rng(20260917))
    contexts = ideal_contexts(y_train, pilot_train, tau, lookup)
    del y_train
    print(f"{name}: contexts ready after {time.time() - started:.0f} s", flush=True)

    NPE, posterior_nn, BoxUniform = npe_utils.import_sbi()
    results = {method: npe_utils.train_sbi_npe_method(
        NPE, posterior_nn, BoxUniform, LABELS[method], contexts[method], pi_train,
        int(args.sbi_seed), args, args.device, npe_utils.resolve_device("cpu"))
        for method in LABELS}
    print(f"{name}: NPE trained after {time.time() - started:.0f} s", flush=True)

    rows = []
    for pi_true in s2.parse_list(args.test_pi_values, float):
        for obs_seed in s2.parse_list(args.test_seeds, int):
            y_obs = s2.simulate(np.random.default_rng(obs_seed), np.asarray([pi_true]), metadata)
            pilot_u, status = s2.data_only_pilot(y_obs, metadata, args)
            obs = ideal_contexts(y_obs, pilot_u, tau, lookup)
            grid, _, cdf, exact = s2.exact_posterior(y_obs[0], args)
            for method in LABELS:
                samples = npe_utils.sample_sbi_posterior(
                    results[method], obs[method][0], int(args.posterior_n),
                    int(args.posterior_seed) + obs_seed, args)
                mean, sd = float(np.mean(samples)), float(np.std(samples))
                q05, q95 = np.quantile(samples, [0.05, 0.95])
                rows.append({
                    "setting": name, "method": method, "seed": obs_seed, "pi_true": pi_true,
                    "pilot_u_context": float(pilot_u[0]), "pilot_status": str(status[0]),
                    "score_context": float(obs[method][0, 1]),
                    "pi_post_mean": mean, "pi_post_std": sd,
                    "pi_coverage90": float(q05 <= pi_true <= q95),
                    "exact_pi_post_mean": exact["mean"], "exact_pi_post_std": exact["sd"],
                    "mean_sq_err_to_exact": (mean - exact["mean"]) ** 2,
                    "sd_abs_err_to_exact": abs(sd - exact["sd"]),
                    "w1_to_exact": npe_utils.exact_sample_w1(samples, grid, cdf),
                })
    s2.write_csv(out_dir / "posterior_by_seed.csv", rows)
    print(f"{name}: done after {time.time() - started:.0f} s", flush=True)


def observed_w1() -> dict[tuple[str, str], dict[tuple[str, float], list[float]]]:
    """Per (setting, sigma_q): per (arm, pi) list of seed means of W1 from finished Stage-2 runs."""
    out: dict[tuple[str, str], dict[tuple[str, float], list[float]]] = {}
    for name, (_, _, _, tag) in SETTINGS.items():
        for run in sorted(SETTING_RUNS.glob(f"s2_{tag}_sq*_*")):
            path = run / "posterior_by_seed.csv"
            if not path.exists():
                continue
            sq = run.name[len(f"s2_{tag}_sq"):].rsplit("_", 1)[0]
            per: dict[tuple[str, float], list[float]] = {}
            for r in csv.DictReader(path.open()):
                arm = r["method"].split()[0]
                arm = "pilot" if arm == "pilot-only" else arm
                if arm in ("pilot", "linear", "radial", "stacked"):
                    per.setdefault((arm, round(float(r["pi_true"]), 4)), []).append(float(r["w1_to_exact"]))
            bucket = out.setdefault((name, sq), {})
            for key, vals in per.items():
                bucket.setdefault(key, []).append(float(np.mean(vals)))
    return out


def summarize() -> None:
    obs = observed_w1()
    L = ["# Model 1 Stage-2 reference: ideal scores through the same pipeline", "",
         "Generated by `scripts/model1_stage2_reference.py`; do not edit by hand. W1 to the exact posterior, "
         "π units. Ideal rows use one NPE fit on the same bank, NPE seed and 60 test datasets as every observed "
         "run, so they pair with each observed seed. Observed values are means over finished seeds.", ""]
    for name in SETTINGS:
        runs = [d for d in (OUT_ROOT / name, *sorted(OUT_ROOT.glob(f"{name}_rep*")))
                if (d / "posterior_by_seed.csv").exists()]
        if not runs:
            continue
        # (method, pi) -> one mean over the 60 test datasets per replicate.
        ideal: dict[tuple[str, float], list[float]] = {}
        for run in runs:
            per: dict[tuple[str, float], list[float]] = {}
            for r in csv.DictReader((run / "posterior_by_seed.csv").open()):
                per.setdefault((r["method"], round(float(r["pi_true"]), 4)), []).append(float(r["w1_to_exact"]))
            for key, vals in per.items():
                ideal.setdefault(key, []).append(float(np.mean(vals)))
        sqs = sorted(sq for (n, sq) in obs if n == name)
        K, m, tau, _ = SETTINGS[name]
        L += [f"## {name}: K={K}, m={m}, τ={tau}", "",
              f"Ideal columns: mean over {len(runs)} NPE replicate(s); gap SE is over replicates. "
              "Observed runs use the replicate-0 streams.", ""]
        head = "| π | ideal NLSA | ideal ILSA | ideal gap | gap SE |"
        sep = "|---:|---:|---:|---:|---:|"
        for sq in sqs:
            n = max((len(v) for v in obs[(name, sq)].values()), default=0)
            head += f" pilot only σ{sq} | ILSA σ{sq} | gate σ{sq} | stacked σ{sq} | ILSA−gate σ{sq} (n={n}) |"
            sep += "---:|---:|---:|---:|---:|"
        L += [head, sep]
        pis = sorted({pi for (_, pi) in ideal})
        for pi in pis + ["pooled"]:
            use = pis if pi == "pooled" else [pi]
            mean = lambda d, key: float(np.mean([np.mean(d[(key, p)]) for p in use])) if all((key, p) in d for p in use) else float("nan")  # noqa: E731
            nl, il = mean(ideal, "ideal_nlsa"), mean(ideal, "ideal_ilsa")
            rep_gaps = [np.mean([ideal[("ideal_ilsa", p)][i] - ideal[("ideal_nlsa", p)][i] for p in use])
                        for i in range(len(runs))]
            se = f"{np.std(rep_gaps, ddof=1) / np.sqrt(len(rep_gaps)):.5f}" if len(rep_gaps) > 1 else "—"
            cells = f"| {pi} | {nl:.5f} | {il:.5f} | {il - nl:+.5f} | {se} |"
            for sq in sqs:
                o = obs[(name, sq)]
                vals = [mean(o, arm) for arm in ("pilot", "linear", "radial", "stacked")]
                cells += " " + " | ".join("—" if np.isnan(v) else f"{v:.5f}" for v in vals)
                gap = vals[1] - vals[2]
                cells += f" | {'—' if np.isnan(gap) else f'{gap:+.5f}'} |"
            L.append(cells)
        L.append("")
    out = ROOT / "new_results" / "MODEL1_STAGE2_REFERENCE_TABLES.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("wrote", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--settings", default="current,A,B")
    p.add_argument("--lookup-blocks", type=int, default=400_000)
    p.add_argument("--lookup-grid", type=int, default=61)
    p.add_argument("--bins", type=int, default=400)
    p.add_argument("--threads", type=int, default=3)
    p.add_argument("--reps", default="0", help="Comma list of replicate indices; 0 is the observed runs' streams.")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--no-summary", action="store_true", help="Skip the table, e.g. for parallel queue jobs.")
    a = p.parse_args()
    # Windows consoles default to a legacy code page that cannot print the tables' symbols.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.set_num_threads(a.threads)
    if not a.summary_only:
        for name in [s.strip() for s in a.settings.split(",") if s.strip()]:
            for rep in [int(r) for r in a.reps.split(",") if r.strip()]:
                run_setting(name, rep, a.lookup_blocks, a.lookup_grid, a.bins)
    if not a.no_summary:
        summarize()


if __name__ == "__main__":
    main()
