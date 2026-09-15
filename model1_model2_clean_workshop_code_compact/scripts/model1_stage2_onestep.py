#!/usr/bin/env python3
"""Model 1 Stage 2 with the one-step evaluation point instead of the pilot.

Same streams as scripts/run_model1_stage2_npe_stacked.sh: prior, 50,000-simulation bank, data-only
pilot, MDN recipe and NPE seed, 60 test datasets, posterior seeds and exact grid. For each arm with a
frozen Stage-1 score S the context is (beta_1, S(Y, beta_1)), where beta_1 is one Newton step from
the pilot using simulation estimates of S's centering and sensitivity (model1/onestep.py). The step
uses only data and simulations, never the generating parameter.

Rows mirror posterior_by_seed.csv, so W1 compares directly with the pilot-evaluated Stage-2 run of
the same Stage-1 directory (runs/setting/s2_<same name>).

    python scripts/model1_stage2_onestep.py --stage1-run-dir runs/setting/s1_k80m5t1.1_sq0.20_20260709
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
sys.path.insert(0, str(ROOT / "scripts"))

from model1 import npe_utils, onestep, stage1  # noqa: E402
from model1 import runtime as score_runtime  # noqa: E402
from model1 import stage2_npe as s2  # noqa: E402
from model1_stage2_reference import LAUNCHER_ARGS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1-run-dir", type=Path, required=True)
    p.add_argument("--methods", default="linear,radial,stacked")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--sensitivity-n", type=int, default=2000, help="Simulated datasets per grid point.")
    p.add_argument("--sensitivity-grid", type=int, default=41)
    p.add_argument("--sensitivity-delta", type=float, default=0.1)
    p.add_argument("--max-step", type=float, default=3.0)
    p.add_argument("--threads", type=int, default=3)
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    torch.set_num_threads(a.threads)
    s1_dir = a.stage1_run_dir
    out_dir = a.output_dir or ROOT / "runs" / "setting" / "onestep" / s1_dir.name.replace("s1_", "s2onestep_", 1)
    if (out_dir / "posterior_by_seed.csv").exists():
        print(f"{out_dir}: already done")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    methods = [x.strip() for x in a.methods.split(",") if x.strip()]
    runtimes = {m: score_runtime.AmortizedScoreRuntime(s1_dir, m, device="cpu") for m in methods}
    metadata = s2.validate_matching_runtimes(runtimes)
    args = s2.build_parser().parse_args([*LAUNCHER_ARGS, "--output-dir", str(out_dir)])
    args.device = npe_utils.resolve_device("cpu")
    args.tau = float(metadata["tau"])
    K, m, tau = int(metadata["n_blocks"]), int(metadata["block_size"]), float(metadata["tau"])
    score_fns = {name: (lambda y, u, rt=rt: rt.score(y, u, batch_size=512).reshape(-1)) for name, rt in runtimes.items()}

    rng = np.random.default_rng(int(args.sbi_train_seed))
    pi_train = rng.uniform(args.pi_prior_min, args.pi_prior_max, size=int(args.n_sbi_train))
    y_train = s2.simulate(rng, pi_train, metadata)
    s2_dir = s1_dir.parent / s1_dir.name.replace("s1_", "s2_", 1)
    cache = s2_dir / "pilot_cache.npz"
    if cache.exists():
        with np.load(cache) as c:
            if not np.allclose(c["pi_train"], pi_train, rtol=0, atol=2e-7):
                raise ValueError(f"pilot cache does not match this bank: {cache}")
            pilot_train = np.asarray(c["pilot_u"], dtype=np.float64)
    else:
        pilot_train, _ = s2.data_only_pilot(y_train, metadata, args)

    u_lo, u_hi = float(stage1.logit_np(args.pi_prior_min)), float(stage1.logit_np(args.pi_prior_max))
    u_grid = np.linspace(u_lo, u_hi, a.sensitivity_grid)
    sens, contexts, diag_rows = {}, {}, []
    for name, fn in score_fns.items():
        b, H = onestep.sensitivity_grid(fn, K, m, tau, u_grid, a.sensitivity_n, a.sensitivity_delta, seed=20260918)
        sens[name] = (b, H)
        beta1, diag = onestep.one_step(fn, y_train, pilot_train, u_grid, b, H, u_lo, u_hi, a.max_step)
        contexts[name] = np.column_stack([beta1, fn(y_train, beta1)]).astype(np.float32)
        true_u = stage1.logit_np(pi_train)
        diag_rows.append({"method": name, "split": "training bank",
                          "pilot_rmse_to_true_u": float(np.sqrt(np.mean((pilot_train - true_u) ** 2))),
                          "beta1_rmse_to_true_u": float(np.sqrt(np.mean((beta1 - true_u) ** 2))), **diag})
        print(f"{name}: beta_1 ready after {time.time() - started:.0f} s; {diag_rows[-1]}", flush=True)
    del y_train
    s2.write_csv(out_dir / "sensitivity_grid.csv", [
        {"method": name, "u": float(u), "b": float(b[i]), "H": float(H[i])}
        for name, (b, H) in sens.items() for i, u in enumerate(u_grid)])

    NPE, posterior_nn, BoxUniform = npe_utils.import_sbi()
    results = {name: npe_utils.train_sbi_npe_method(
        NPE, posterior_nn, BoxUniform, f"{name} one-step", contexts[name], pi_train,
        int(args.sbi_seed), args, args.device, npe_utils.resolve_device("cpu")) for name in methods}
    print(f"NPE trained after {time.time() - started:.0f} s", flush=True)

    rows = []
    for pi_true in s2.parse_list(args.test_pi_values, float):
        for obs_seed in s2.parse_list(args.test_seeds, int):
            y_obs = s2.simulate(np.random.default_rng(obs_seed), np.asarray([pi_true]), metadata)
            pilot_u, status = s2.data_only_pilot(y_obs, metadata, args)
            grid, _, cdf, exact = s2.exact_posterior(y_obs[0], args)
            for name, fn in score_fns.items():
                b, H = sens[name]
                beta1, _ = onestep.one_step(fn, y_obs, pilot_u, u_grid, b, H, u_lo, u_hi, a.max_step)
                context = np.array([beta1[0], fn(y_obs, beta1)[0]], dtype=np.float32)
                samples = npe_utils.sample_sbi_posterior(
                    results[name], context, int(args.posterior_n), int(args.posterior_seed) + obs_seed, args)
                mean, sd = float(np.mean(samples)), float(np.std(samples))
                q05, q95 = np.quantile(samples, [0.05, 0.95])
                rows.append({
                    "method": f"{name} one-step pilot+score NPE", "seed": obs_seed, "pi_true": pi_true,
                    "pilot_u_context": float(pilot_u[0]), "pilot_status": str(status[0]),
                    "beta1_context": float(beta1[0]), "score_context": float(context[1]),
                    "pi_post_mean": mean, "pi_post_std": sd, "pi_coverage90": float(q05 <= pi_true <= q95),
                    "exact_pi_post_mean": exact["mean"], "exact_pi_post_std": exact["sd"],
                    "mean_sq_err_to_exact": (mean - exact["mean"]) ** 2,
                    "sd_abs_err_to_exact": abs(sd - exact["sd"]),
                    "w1_to_exact": npe_utils.exact_sample_w1(samples, grid, cdf),
                })
    s2.write_csv(out_dir / "posterior_by_seed.csv", rows)
    s2.write_csv(out_dir / "onestep_diagnostics.csv", diag_rows)

    pilot_rows = s2_dir / "posterior_by_seed.csv"
    if pilot_rows.exists():
        at_pilot: dict[str, list[float]] = {}
        for r in csv.DictReader(pilot_rows.open()):
            at_pilot.setdefault(r["method"].split()[0], []).append(float(r["w1_to_exact"]))
        print("\npooled W1: evaluated at the pilot -> at the one-step point")
        for name in methods:
            one = [r["w1_to_exact"] for r in rows if r["method"].startswith(name + " ")]
            print(f"  {name:<8} {np.mean(at_pilot[name]):.5f} -> {np.mean(one):.5f}")
    print(f"done after {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
