#!/usr/bin/env python3
"""Narrow versus wide tube bandwidth for every Fisher-score arm of the Model 1
Stage-2 input comparison (runs/ic/).

Within each method the two bandwidths share the Stage-1 seed, the NPE bank, the
NPE seed, the pilot and the test datasets, so differences are paired by seed.

    python scripts/compare_input_bandwidths.py [--narrow 0.20] [--wide 1.052] [--out FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_input_comparison as sic  # noqa: E402

METHODS = [a for a in sic.ARMS if a[2] != "direct"]


def stage1_mse(key: str, kind: str, sq: str, seed: int) -> float | None:
    if kind in ("raw", "sub"):
        best = None
        for lr in ("1e-4", "1e-3"):
            d = sic.IC / f"s1_{kind}_lr{lr}_sq{sq}_{seed}"
            if (d / "training_info.json").exists():
                loss = json.loads((d / "training_info.json").read_text())["best_val_loss"]
                if best is None or loss < best[0]:
                    best = (loss, d)
        path = best[1] / "score_summary_by_pi.csv" if best else None
        prefix = None
    else:
        if key == "pilot":
            return None
        path = sic.IC / f"s1_nlsa_sq{sq}_{seed}" / "score_summary_by_pi.csv"
        prefix = {"linear": "linear", "gate": "radial", "stacked": "stacked"}[key]
    if path is None or not path.exists():
        return None
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if prefix is None or r["method"].startswith(prefix)]
    return float(np.mean([float(r["std_mse"]) for r in rows])) if rows else None


def mean_se(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return None, None
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrow", default="0.20")
    ap.add_argument("--wide", default="1.052")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    cells = {sq: {key: sic.arm_cells(key, kind, label, sq) for key, _, kind, label in METHODS}
             for sq in (a.narrow, a.wide)}

    # Both bandwidths must be scored on the same exact posteriors.
    ref = None
    for sq in (a.narrow, a.wide):
        for key, _, kind, _ in METHODS:
            for seed in cells[sq][key]:
                path = sic.IC / f"s2_{kind}_sq{sq}_{seed}" / "posterior_by_seed.csv"
                means = sic.exact_means(path)
                ref = ref or means
                if set(means) != set(ref) or max(abs(means[k] - ref[k]) for k in means) > 1e-9:
                    raise SystemExit(f"unpaired evaluation: {path}")

    out = [f"Tube bandwidth σ_q = {a.narrow} (narrow) vs {a.wide} (wide, draft rule), Model 1 p=1, "
           "Stage 2. Paired within method by Stage-1 seed.\n"]

    rows = []
    for key, disp, kind, _ in METHODS:
        row = [disp]
        for sq in (a.narrow, a.wide):
            per = cells[sq][key]
            m, s = mean_se([sic.summarize(c)["w1"] for c in per.values()])
            row += [str(len(per)), sic.fmt(m), sic.fmt(s)]
        for sq in (a.narrow, a.wide):
            m, _ = mean_se([x for x in (stage1_mse(key, kind, sq, s) for s in sic.SEEDS) if x is not None])
            row.append(sic.fmt(m, 4))
        rows.append(row)
    out.append("### Posterior W1 to exact and Stage-1 score accuracy, by bandwidth\n")
    out.append(sic.table(["score", "seeds (narrow)", f"W1 σ_q={a.narrow}", "SE",
                          "seeds (wide)", f"W1 σ_q={a.wide}", "SE",
                          f"Stage-1 std MSE σ_q={a.narrow}", f"Stage-1 std MSE σ_q={a.wide}"], rows) + "\n")

    prow = []
    for key, disp, _, _ in METHODS:
        n_cells, w_cells = cells[a.narrow][key], cells[a.wide][key]
        seeds = sorted(set(n_cells) & set(w_cells))
        if not seeds:
            continue
        diffs = [float(np.mean([w_cells[s][k]["w1"] - n_cells[s][k]["w1"]
                                for k in set(w_cells[s]) & set(n_cells[s])])) for s in seeds]
        m, se = mean_se(diffs)
        base = float(np.mean([sic.summarize(n_cells[s])["w1"] for s in seeds]))
        prow.append([disp, str(len(seeds)), f"{m:+.5f}", sic.fmt(se),
                     f"{sum(d < 0 for d in diffs)}/{len(diffs)}", f"{100 * m / base:+.1f}%"])
    out.append("### Paired W1, wide − narrow (negative favours the draft's wide bandwidth)\n")
    out.append(sic.table(["score", "seeds", "mean ΔW1", "SE", "wide better", "Δ / W1(narrow)"], prow) + "\n")

    for sq in (a.narrow, a.wide):
        ref_cells = cells[sq][sic.REFERENCE]
        crow = []
        for key, disp, _, _ in METHODS:
            if key == sic.REFERENCE or not cells[sq][key] or not ref_cells:
                continue
            seeds = sorted(set(cells[sq][key]) & set(ref_cells))
            diffs = [float(np.mean([cells[sq][key][s][k]["w1"] - ref_cells[s][k]["w1"]
                                    for k in set(cells[sq][key][s]) & set(ref_cells[s])])) for s in seeds]
            m, se = mean_se(diffs)
            base = float(np.mean([sic.summarize(ref_cells[s])["w1"] for s in seeds]))
            crow.append([f"{disp} − gate", str(len(seeds)), f"{m:+.5f}", sic.fmt(se),
                         f"{sum(d > 0 for d in diffs)}/{len(diffs)}", f"{100 * m / base:+.1f}%"])
        out.append(f"### Paired W1 against the NLSA gate at σ_q = {sq} (positive = gate better)\n")
        out.append(sic.table(["comparison", "seeds", "mean ΔW1", "SE", "gate better", "Δ / W1(gate)"], crow) + "\n")

    text = "\n".join(out)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
