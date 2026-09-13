#!/usr/bin/env python3
"""Held-out NPE log loss for Model 1 (p=1) Stage-2 runs: a data-only selection score.

For one fresh simulation bank (pi ~ U(prior), Y ~ p(. | pi); seed independent of
every Stage-2 bank), each saved NPE state is scored by

    NLL = -mean_i log q(pi_i | context_i),   context_i = (u_pilot(Y_i), S_frozen(Y_i, u_pilot)),

the expected log score of the true parameter, a proper scoring rule. The pilot and
the frozen Stage-1 score are recomputed exactly as in model1.stage2_npe, so a
bandwidth or method can be chosen from simulations alone. The exact posterior is
never evaluated here.

    python scripts/heldout_npe_nll_model1.py runs/s2_sq0.20_20260709 runs/s2_sq1.052_20260709 ...

Results are appended to runs/heldout_nll/heldout_nll.csv; directories already
scored with the same bank are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.npe import import_sbi, pi_to_unit  # noqa: E402
from model1 import runtime as score_runtime  # noqa: E402
from model1 import stage2_npe  # noqa: E402

FIELDS = ["stage2_dir", "stage1_dir", "sigma_q", "iters", "method", "bank_seed", "n",
          "nll_mean", "nll_se"]


def load_estimator(state_path: Path, training_contexts: np.lib.npyio.NpzFile):
    _, posterior_nn, _ = import_sbi()
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    method = payload["method"]
    builder = posterior_nn(
        model=config["sbi_model"],
        z_score_theta="independent",
        z_score_x="independent",
        hidden_features=int(config["sbi_hidden_features"]),
        num_transforms=int(config["sbi_num_transforms"]),
        num_bins=int(config["sbi_num_bins"]),
        num_components=int(config["sbi_num_components"]),
    )
    # The batch only fixes shapes; z-score buffers are overwritten by the state dict.
    theta = torch.as_tensor(pi_to_unit(training_contexts["pi_train"][:512],
                                       float(config["pi_prior_min"]), float(config["pi_prior_max"])))
    x = torch.as_tensor(np.asarray(training_contexts[f"pilot_score_{method}"][:512], dtype=np.float32))
    estimator = builder(theta, x)
    estimator.load_state_dict(payload["state_dict"], strict=True)
    return estimator.eval(), config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage2_dirs", nargs="+", type=Path)
    parser.add_argument("--n-bank", type=int, default=20_000)
    parser.add_argument("--bank-seed", type=int, default=20260914)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/heldout_nll/heldout_nll.csv")
    parser.add_argument("--pilot-cache", type=Path, default=None,
                        help="Defaults to <output dir>/bank_<seed>_<n>.npz")
    args = parser.parse_args()

    done: set[tuple[str, int, int]] = set()
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            done = {(r["stage2_dir"], int(r["bank_seed"]), int(r["n"])) for r in csv.DictReader(handle)}
    todo = [d for d in args.stage2_dirs
            if (d.as_posix(), args.bank_seed, args.n_bank) not in done
            and (d / "posterior_by_seed.csv").exists()]
    if not todo:
        print("nothing to score")
        return

    first = json.loads((todo[0] / "config.json").read_text(encoding="utf-8"))
    metadata = first["stage1_metadata"]
    pilot_args = argparse.Namespace(
        pi_prior_min=float(first["pi_prior_min"]), pi_prior_max=float(first["pi_prior_max"]),
        pilot_grid_size=int(first["pilot_grid_size"]), device="cpu",
        pilot_batch_size=int(first["pilot_batch_size"]),
        pilot_grid_chunk_size=int(first["pilot_grid_chunk_size"]), pilot_progress_every=0,
    )
    cache = args.pilot_cache or args.output.parent / f"bank_{args.bank_seed}_{args.n_bank}.npz"
    rng = np.random.default_rng(args.bank_seed)
    pi_bank = rng.uniform(pilot_args.pi_prior_min, pilot_args.pi_prior_max, size=args.n_bank)
    y_bank = stage2_npe.simulate(rng, pi_bank, metadata)
    if cache.exists():
        with np.load(cache) as stored:
            if not np.allclose(stored["pi_bank"], pi_bank, rtol=0, atol=1e-12):
                raise ValueError(f"bank cache does not match seed/size: {cache}")
            pilot_u = stored["pilot_u"]
    else:
        pilot_u, _ = stage2_npe.data_only_pilot(y_bank, metadata, pilot_args)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, pi_bank=pi_bank, pilot_u=pilot_u)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for stage2_dir in todo:
            config = json.loads((stage2_dir / "config.json").read_text(encoding="utf-8"))
            model_keys = ("n_blocks", "block_size", "tau")
            if any(config["stage1_metadata"][k] != metadata[k] for k in model_keys) \
                    or float(config["pi_prior_min"]) != pilot_args.pi_prior_min \
                    or float(config["pi_prior_max"]) != pilot_args.pi_prior_max:
                raise ValueError(f"{stage2_dir} uses a different model or prior")
            stage1_dir = Path(config["stage1_run_dir"])
            stage1_config = json.loads((stage1_dir / "config.json").read_text(encoding="utf-8"))
            methods = stage2_npe.parse_list(config["methods"], str)
            runtimes = {m: score_runtime.AmortizedScoreRuntime(stage1_dir, m, device="cpu")
                        for m in methods if m != "pilot"}
            contexts = stage2_npe.score_contexts(runtimes, y_bank, pilot_u, 256)
            contexts["pilot"] = pilot_u[:, None].astype(np.float32)
            with np.load(stage2_dir / "training_contexts.npz") as training_contexts:
                for method in methods:
                    estimator, npe_config = load_estimator(stage2_dir / f"npe_{method}_state.pt",
                                                           training_contexts)
                    theta = torch.as_tensor(pi_to_unit(pi_bank, float(npe_config["pi_prior_min"]),
                                                       float(npe_config["pi_prior_max"])))
                    x = torch.as_tensor(contexts[method])
                    with torch.no_grad():
                        log_q = np.concatenate([
                            estimator.log_prob(theta[None, i:i + 4096], condition=x[i:i + 4096])
                            .reshape(-1).numpy()
                            for i in range(0, theta.shape[0], 4096)
                        ]).astype(np.float64)
                    # Report in pi units: log q(pi) = log q(unit) - log(prior width).
                    nll = -(log_q - math.log(pilot_args.pi_prior_max - pilot_args.pi_prior_min))
                    row = {
                        "stage2_dir": stage2_dir.as_posix(), "stage1_dir": stage1_dir.as_posix(),
                        "sigma_q": float(stage1_config["sigma_q"]), "iters": int(stage1_config["iters"]),
                        "method": method, "bank_seed": args.bank_seed, "n": args.n_bank,
                        "nll_mean": float(nll.mean()), "nll_se": float(nll.std(ddof=1) / math.sqrt(nll.size)),
                    }
                    writer.writerow(row)
                    handle.flush()
                    print(row, flush=True)


if __name__ == "__main__":
    main()
