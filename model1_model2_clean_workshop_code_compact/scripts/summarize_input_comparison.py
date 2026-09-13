#!/usr/bin/env python3
"""Tables for the Model 1 (p=1) Stage-2 input comparison under runs/ic/.

Reads each arm's posterior_by_seed.csv, checks the arms are evaluated on the
same 60 test datasets against the same exact posterior, and prints Markdown.
Score arms are replicated over Stage-1 seeds (NPE seed fixed at 54000). The two
direct-NPE arms do not read Stage 1, so they are replicated over NPE seeds
instead, and their paired differences are taken cell by cell against the
reference arm's Stage-1-seed-averaged W1. The two SEs therefore measure
different sources of variation, and the tables say which.

    python scripts/summarize_input_comparison.py [--sigma-q 1.052] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IC = ROOT / "runs" / "ic"
SEEDS = [20260709, 20260710, 20260711, 20260712, 20260713]
EXACT = "exact likelihood grid"
# (key, display, source kind, method label inside posterior_by_seed.csv)
ARMS = [
    ("pilot", "pilot only", "nlsa", "pilot-only NPE"),
    ("npe_raw", "NPE on raw Y", "direct", "raw data NPE"),
    ("npe_sub", "NPE on all subscores", "direct", "all-subscores NPE"),
    ("fs_raw", "raw-input Fisher score", "raw", "direct amortized raw-FSM pilot+score NPE"),
    ("fs_sub", "all-subscores Fisher score", "sub", "all-subscores flat MLP FSM pilot+score NPE"),
    ("linear", "linear ILSA score", "nlsa", "linear FSM pilot+score NPE"),
    ("gate", "NLSA gate score (paper)", "nlsa", "radial FSM pilot+score NPE"),
    ("stacked", "NLSA stacked m_dim=8 score", "nlsa", "stacked FSM pilot+score NPE"),
]
REFERENCE = "gate"


DIRECT_DIR = {"npe_raw": "npe_raw", "npe_sub": "npe_subscores_at_pilot"}


def direct_dirs(key: str) -> list[Path]:
    base = DIRECT_DIR[key]
    dirs = [IC / base] + sorted(IC.glob(f"{base}_sbi*"))
    return [d for d in dirs if (d / "posterior_by_seed.csv").exists()]


def load(path: Path, label: str) -> dict[tuple[float, int], dict[str, float]]:
    out = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if r["method"] != label:
                continue
            key = (round(float(r["pi_true"]), 6), int(r["seed"]))
            out[key] = {
                "w1": float(r["w1_to_exact"]),
                "sq_truth": float(r["pi_sq_err"]),
                "sq_exact": float(r["mean_sq_err_to_exact"]),
                "sd": float(r.get("pi_post_std") or r.get("pi_post_sd")),
                "exact_sd": float(r.get("exact_pi_post_std") or r.get("exact_pi_post_sd")),
                "cov": float(r.get("pi_coverage90") or r.get("coverage90")),
            }
    return out


def exact_means(path: Path) -> dict[tuple[float, int], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(round(float(r["pi_true"]), 6), int(r["seed"])): float(r["pi_post_mean"])
                for r in csv.DictReader(handle) if r["method"] == EXACT}


def arm_cells(key: str, kind: str, label: str, sq: str) -> dict[int | None, dict]:
    """Cells per replicate: Stage-1 seed for score arms, NPE-seed directory for direct arms."""
    if kind == "direct":
        return {d.name: load(d / "posterior_by_seed.csv", label) for d in direct_dirs(key)}
    cells = {}
    for seed in SEEDS:
        path = IC / f"s2_{kind}_sq{sq}_{seed}" / "posterior_by_seed.csv"
        if path.exists():
            got = load(path, label)
            if got:
                cells[seed] = got
    return cells


def summarize(cells: dict[tuple, dict]) -> dict[str, float]:
    vals = list(cells.values())
    return {
        "w1": float(np.mean([v["w1"] for v in vals])),
        "mse_truth": float(np.mean([v["sq_truth"] for v in vals])),
        "rmse_exact": float(np.sqrt(np.mean([v["sq_exact"] for v in vals]))),
        "sd_err": float(np.mean([abs(v["sd"] - v["exact_sd"]) for v in vals])),
        "cov": float(np.mean([v["cov"] for v in vals])),
    }


def fmt(x, d=5):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |",
                      "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-q", default="1.052")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sq = args.sigma_q

    data = {key: arm_cells(key, kind, label, sq) for key, _, kind, label in ARMS}
    names = {key: disp for key, disp, _, _ in ARMS}
    kinds = {key: kind for key, _, kind, _ in ARMS}

    # Pairing check: every file must hold the same exact posterior for each test dataset.
    exact_ref, checked = None, 0
    for key, _, kind, _ in ARMS:
        paths = ([d / "posterior_by_seed.csv" for d in direct_dirs(key)]
                 if kind == "direct" else
                 [IC / f"s2_{kind}_sq{sq}_{s}" / "posterior_by_seed.csv" for s in SEEDS])
        for path in paths:
            if not path.exists():
                continue
            means = exact_means(path)
            if exact_ref is None:
                exact_ref = means
            if set(means) != set(exact_ref) or max(abs(means[k] - exact_ref[k]) for k in means) > 1e-9:
                raise SystemExit(f"unpaired evaluation: {path} disagrees with the exact posterior bank")
            checked += 1

    lines = [f"Stage-2 input comparison, Model 1 p=1, σ_q={sq}. {checked} posterior files checked: "
             f"all share one exact posterior on {len(exact_ref or {})} test datasets.\n"]

    rows = []
    for key, disp, kind, _ in ARMS:
        per = data[key]
        if not per:
            rows.append([disp, "0", "—", "—", "—", "—", "—", "—"])
            continue
        stats = [summarize(c) for c in per.values()]
        w1 = [s["w1"] for s in stats]
        se = float(np.std(w1, ddof=1) / np.sqrt(len(w1))) if len(w1) > 1 else float("nan")
        unit = "NPE seeds" if kind == "direct" else "Stage-1 seeds"
        rows.append([disp, f"{len(per)} {unit}",
                     fmt(float(np.mean(w1))), fmt(se),
                     fmt(float(np.mean([s["mse_truth"] for s in stats])), 6),
                     fmt(float(np.mean([s["rmse_exact"] for s in stats]))),
                     fmt(float(np.mean([s["sd_err"] for s in stats]))),
                     fmt(float(np.mean([s["cov"] for s in stats])), 3)])
    lines.append("### Posterior quality, pooled over 6 true π × 10 test datasets\n")
    lines.append(table(["input to NPE", "replicates", "W1 to exact", "SE",
                        "post-mean MSE vs truth", "RMSE vs exact mean", "|SD − exact SD|", "cov90"],
                       rows) + "\n")

    pis = sorted({k[0] for c in (data[REFERENCE] or {None: {}}).values() for k in c}) or \
          sorted({k[0] for per in data.values() for c in per.values() for k in c})
    prow = []
    for key, disp, _, _ in ARMS:
        per = data[key]
        if not per:
            continue
        vals = []
        for pi in pis:
            vals.append(fmt(float(np.mean([np.mean([v["w1"] for k, v in c.items() if k[0] == pi])
                                            for c in per.values()]))))
        prow.append([disp] + vals)
    lines.append("### W1 to exact, per true π\n")
    lines.append(table(["input to NPE"] + [f"π={p:g}" for p in pis], prow) + "\n")

    ref = data[REFERENCE]
    crow = []
    for key, disp, kind, _ in ARMS:
        if key == REFERENCE or not data[key] or not ref:
            continue
        diffs = []
        if kinds[key] == "direct":
            ref_bar = {k: float(np.mean([ref[s][k]["w1"] for s in ref])) for k in next(iter(ref.values()))}
            for comp in data[key].values():
                common = sorted(set(comp) & set(ref_bar))
                diffs.append(float(np.mean([comp[k]["w1"] - ref_bar[k] for k in common])))
        else:
            for seed, ref_cells in ref.items():
                comp = data[key].get(seed)
                if comp is None:
                    continue
                common = sorted(set(comp) & set(ref_cells))
                diffs.append(float(np.mean([comp[k]["w1"] - ref_cells[k]["w1"] for k in common])))
        if not diffs:
            continue
        ref_mean = float(np.mean([summarize(ref[s])["w1"] for s in ref]))
        se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
        unit = "NPE seeds" if kinds[key] == "direct" else "Stage-1 seeds"
        crow.append([f"{disp} − {names[REFERENCE]}", f"{len(diffs)} {unit}", f"{np.mean(diffs):+.5f}", fmt(se),
                     f"{sum(d > 0 for d in diffs)}/{len(diffs)}", f"{100 * np.mean(diffs) / ref_mean:+.1f}%"])
    lines.append(f"### Paired W1 against {names[REFERENCE]} (positive = worse than the paper's method)\n")
    lines.append(table(["comparison", "replicates", "mean ΔW1", "SE", "paper method better", "Δ / W1(paper)"],
                       crow) + "\n")

    srow = []
    for tag, disp in (("raw", "raw-input Fisher score"), ("sub", "all-subscores Fisher score")):
        for seed in SEEDS:
            chosen = None
            for lr in ("1e-4", "1e-3"):
                d = IC / f"s1_{tag}_lr{lr}_sq{sq}_{seed}"
                if (d / "training_info.json").exists():
                    info = json.loads((d / "training_info.json").read_text())
                    if chosen is None or info["best_val_loss"] < chosen[1]:
                        chosen = (lr, info["best_val_loss"], d, info["best_step"])
            if chosen and (chosen[2] / "score_summary_by_pi.csv").exists():
                with (chosen[2] / "score_summary_by_pi.csv").open() as h:
                    mse = np.mean([float(r["std_mse"]) for r in csv.DictReader(h)])
                srow.append([disp, str(seed), chosen[0], str(chosen[3]), fmt(float(mse))])
    for seed in SEEDS:
        f = IC / f"s1_nlsa_sq{sq}_{seed}" / "score_summary_by_pi.csv"
        if f.exists():
            with f.open() as h:
                rows_ = list(csv.DictReader(h))
            for pre, disp in (("linear", "linear ILSA score"), ("radial", "NLSA gate score (paper)"),
                              ("stacked", "NLSA stacked m_dim=8 score")):
                srow.append([disp, str(seed), "1e-3",
                             "—", fmt(float(np.mean([float(r["std_mse"]) for r in rows_
                                                     if r["method"].startswith(pre)])))])
    lines.append("### Stage-1 Fisher-score accuracy of each score arm (frozen exact-score standardized MSE)\n")
    lines.append(table(["score", "seed", "lr (by validation)", "best step", "std MSE"], srow) + "\n")

    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
