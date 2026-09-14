#!/usr/bin/env python3
"""Regenerate the tables in FOLLOWUP_FINDINGS_20260913.md from runs/.

Covers the four next steps of STAGE2_FINDINGS_20260913.md:

  F1  Model 1 p=1, sigma_q=1.052 trained for 60k steps vs the 20k-step runs.
  F2  Bandwidth grid sigma_q in {0.20, 0.5, 0.977, 1.052, 2.0}: held-out NPE log
      loss (simulation-only selection, runs/heldout_nll/heldout_nll.csv) against
      posterior W1 to the exact posterior.
  F3  Stage-2 replicates (new NPE bank, NPE seed, posterior seed and test bank):
      method and bandwidth contrasts with seed-, replicate- and cell-level SEs.
  F4  Model 1 p=3 Stage 2.
  F5  Model 2 learning-rate audit (Stage 1 only).

    python scripts/summarize_followup_findings.py            # rewrite tables
    python scripts/summarize_followup_findings.py --stdout   # print only

Missing runs are skipped, and every table states its n.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "FOLLOWUP_FINDINGS_20260913.md"
SEEDS = (20260709, 20260710, 20260711, 20260712, 20260713)
SIGMA_GRID = ("0.20", "0.5", "0.977", "1.052", "2.0")
P1_METHODS = ("pilot", "linear", "radial", "stacked")
P1_LABELS = {
    "pilot": "pilot-only NPE",
    "linear": "linear FSM pilot+score NPE",
    "radial": "radial FSM pilot+score NPE",
    "stacked": "stacked FSM pilot+score NPE",
}
NAMES = {"pilot": "pilot only", "linear": "linear (ILSA)", "radial": "gate", "gate": "gate",
         "stacked": "stacked m_dim=8", "shared_radial": "gate"}
P3_METHODS = ("pilot", "linear", "gate", "stacked")
P3_LABELS = {
    "pilot": "pilot-only NPE (p=3)",
    "linear": "linear FSM pilot+score NPE (p=3)",
    "gate": "gate FSM pilot+score NPE (p=3)",
    "stacked": "stacked FSM pilot+score NPE (p=3)",
}
P3_COORDS = ("u", "tau", "log_sigma")
BEGIN, END = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
NL = chr(10)


# --------------------------------------------------------------------------- helpers
def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def se(values) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")


def fmt(value, digits: int = 5) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def signed(value, digits: int = 5) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:+.{digits}f}"


def table(header: list[str], rows: list[list[str]], align: str) -> str:
    if not rows:
        return "_no runs available yet_"
    lines = ["| " + " | ".join(header) + " |", "|" + align + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return NL.join(lines)


def p1_tag(sigma: str, iters: int = 20000) -> str:
    return f"sq{sigma}" + ("" if iters == 20000 else f"_it{iters}")


def p1_stage2_dir(sigma: str, seed: int, iters: int = 20000, rep: int = 0) -> Path:
    prefix = "s2" if rep == 0 else f"s2rep{rep}"
    return RUNS / f"{prefix}_{p1_tag(sigma, iters)}_{seed}"


def p1_cells(directory: Path) -> dict[str, dict[tuple, dict[str, str]]] | None:
    path = directory / "posterior_by_seed.csv"
    if not path.exists():
        return None
    inverse = {label: key for key, label in P1_LABELS.items()}
    out: dict[str, dict[tuple, dict[str, str]]] = {}
    for row in read_csv(path):
        key = inverse.get(row["method"], "exact")
        out.setdefault(key, {})[(float(row["pi_true"]), int(row["seed"]))] = row
    return out


def stage1_std_mse(sigma: str, seed: int, method: str, iters: int = 20000) -> float | None:
    path = RUNS / f"s1_{p1_tag(sigma, iters)}_{seed}" / "score_summary_by_pi.csv"
    if not path.exists():
        return None
    rows = [r for r in read_csv(path) if r["method"].startswith(method)]
    return float(np.mean([float(r["std_mse"]) for r in rows])) if rows else None


def mean_metric(cells: dict[tuple, dict[str, str]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in cells.values()]))


def unit_contrast(units: list[tuple[dict, dict]], key: str = "w1_to_exact") -> dict[str, float] | None:
    """units: (cells A, cells B) per unit; returns seed-level mean/SE/wins and cell SE."""
    unit_diffs, cell_diffs, base = [], [], []
    for a, b in units:
        common = sorted(set(a) & set(b))
        if not common:
            continue
        d = [float(a[k][key]) - float(b[k][key]) for k in common]
        unit_diffs.append(float(np.mean(d)))
        cell_diffs.extend(d)
        base.append(float(np.mean([float(b[k][key]) for k in common])))
    if not unit_diffs:
        return None
    return {"n": len(unit_diffs), "mean": float(np.mean(unit_diffs)), "se": se(unit_diffs),
            "wins": sum(d < 0 for d in unit_diffs), "rel": 100.0 * float(np.mean(unit_diffs)) / float(np.mean(base)),
            "n_cells": len(cell_diffs), "se_cells": se(cell_diffs)}


CONTRAST_HEADER = ["comparison (A − B)", "n units", "mean Δ", "SE (units)", "A better", "Δ / B", "n cells", "SE (cells)"]
CONTRAST_ALIGN = "---|---:|---:|---:|---:|---:|---:|---:"


def contrast_row(label: str, result: dict[str, float] | None, digits: int = 5) -> list[str] | None:
    if result is None:
        return None
    return [label, str(result["n"]), signed(result["mean"], digits), fmt(result["se"], digits),
            f"{result['wins']}/{result['n']}", f"{result['rel']:+.1f}%", str(result["n_cells"]),
            fmt(result["se_cells"], digits)]


# --------------------------------------------------------------------------- F1
def f1_tables() -> str:
    parts = []
    rows = []
    for method in P1_METHODS[1:]:
        long = {s: stage1_std_mse("1.052", s, method, 60000) for s in SEEDS}
        short = {s: stage1_std_mse("1.052", s, method, 20000) for s in SEEDS}
        seeds = [s for s in SEEDS if long[s] is not None and short[s] is not None]
        if not seeds:
            continue
        diffs = [long[s] - short[s] for s in seeds]
        best = []
        for s in seeds:
            info = json.loads((RUNS / f"s1_{p1_tag('1.052', 60000)}_{s}" / "training_info.json").read_text())
            best.append(str(info[method]["best_step"]))
        rows.append([NAMES[method], str(len(seeds)), fmt(np.mean([short[s] for s in seeds])),
                     fmt(np.mean([long[s] for s in seeds])), signed(float(np.mean(diffs))), fmt(se(diffs)),
                     f"{sum(d < 0 for d in diffs)}/{len(seeds)}", ", ".join(best)])
    parts += ["### F1a. Stage-1 standardized MSE at σ_q=1.052: 20k vs 60k steps (negative Δ favours 60k)",
              table(["method", "n seeds", "20k", "60k", "Δ (60k − 20k)", "SE", "60k better", "60k best steps"],
                    rows, "---|---:|---:|---:|---:|---:|---:|---"), ""]

    rows = []
    for method in P1_METHODS:
        long_units, bw_units, w1_long, w1_short = [], [], [], []
        for s in SEEDS:
            long_c = p1_cells(p1_stage2_dir("1.052", s, 60000))
            short_c = p1_cells(p1_stage2_dir("1.052", s))
            narrow_c = p1_cells(p1_stage2_dir("0.20", s))
            if long_c and short_c:
                long_units.append((long_c[method], short_c[method]))
                w1_long.append(mean_metric(long_c[method], "w1_to_exact"))
                w1_short.append(mean_metric(short_c[method], "w1_to_exact"))
            if long_c and narrow_c:
                bw_units.append((long_c[method], narrow_c[method]))
        row = contrast_row(f"{NAMES[method]}: σ_q=1.052 60k − 1.052 20k", unit_contrast(long_units))
        if row:
            rows.append(row)
        row = contrast_row(f"{NAMES[method]}: σ_q=1.052 60k − 0.20 20k", unit_contrast(bw_units))
        if row:
            rows.append(row)
    parts += ["### F1b. Stage-2 W1 to the exact posterior: paired differences (unit = Stage-1 seed)",
              table(CONTRAST_HEADER, rows, CONTRAST_ALIGN), ""]

    rows = []
    for a, b in (("radial", "linear"), ("stacked", "linear"), ("stacked", "radial")):
        units = []
        for s in SEEDS:
            c = p1_cells(p1_stage2_dir("1.052", s, 60000))
            if c:
                units.append((c[a], c[b]))
        row = contrast_row(f"σ_q=1.052 60k: {NAMES[a]} − {NAMES[b]}", unit_contrast(units))
        if row:
            rows.append(row)
    parts += ["### F1c. Method contrasts after 60k-step Stage 1 (W1)", table(CONTRAST_HEADER, rows, CONTRAST_ALIGN)]
    return NL.join(parts)


# --------------------------------------------------------------------------- F2
def heldout_rows() -> dict[tuple[str, int, str], float]:
    path = RUNS / "heldout_nll" / "heldout_nll.csv"
    if not path.exists():
        return {}
    out = {}
    for row in read_csv(path):
        name = Path(row["stage2_dir"]).name
        if not name.startswith("s2_sq") or "_it" in name:
            continue
        sigma = name[len("s2_sq"):].rsplit("_", 1)[0]
        seed = int(name.rsplit("_", 1)[1])
        out[(sigma, seed, row["method"])] = float(row["nll_mean"])
    return out


def f2_tables() -> str:
    nll = heldout_rows()
    rows = []
    selection_rows = []
    for method in P1_METHODS:
        per_sigma = {}
        for sigma in SIGMA_GRID:
            seeds = [s for s in SEEDS if p1_cells(p1_stage2_dir(sigma, s))]
            if not seeds:
                continue
            w1 = [mean_metric(p1_cells(p1_stage2_dir(sigma, s))[method], "w1_to_exact") for s in seeds]
            cov = [mean_metric(p1_cells(p1_stage2_dir(sigma, s))[method], "pi_coverage90") for s in seeds]
            nl = [nll[(sigma, s, method)] for s in seeds if (sigma, s, method) in nll]
            s1 = [v for v in (stage1_std_mse(sigma, s, method) for s in seeds) if v is not None] if method != "pilot" else []
            per_sigma[sigma] = {"seeds": seeds, "w1": dict(zip(seeds, w1)),
                                "nll": {s: nll[(sigma, s, method)] for s in seeds if (sigma, s, method) in nll}}
            rows.append([NAMES[method], sigma, str(len(seeds)), fmt(np.mean(s1)) if s1 else "—",
                         fmt(float(np.mean(w1))), fmt(se(w1)), fmt(float(np.mean(cov)), 3),
                         fmt(float(np.mean(nl)), 4) if nl else "—", fmt(se(nl), 4) if len(nl) > 1 else "—",
                         str(len(nl))])
        if method == "pilot" or not per_sigma:
            continue
        complete = [s for s in SEEDS if all(s in per_sigma.get(sig, {}).get("nll", {}) for sig in SIGMA_GRID)
                    and all(s in per_sigma.get(sig, {}).get("w1", {}) for sig in SIGMA_GRID)]
        if not complete:
            continue
        nll_mean = {sig: float(np.mean([per_sigma[sig]["nll"][s] for s in complete])) for sig in SIGMA_GRID}
        w1_mean = {sig: float(np.mean([per_sigma[sig]["w1"][s] for s in complete])) for sig in SIGMA_GRID}
        chosen = min(nll_mean, key=nll_mean.get)
        oracle = min(w1_mean, key=w1_mean.get)
        per_seed_match = sum(
            min(SIGMA_GRID, key=lambda sig: per_sigma[sig]["nll"][s]) == min(SIGMA_GRID, key=lambda sig: per_sigma[sig]["w1"][s])
            for s in complete)
        rank_nll = sorted(SIGMA_GRID, key=nll_mean.get)
        rank_w1 = sorted(SIGMA_GRID, key=w1_mean.get)
        spearman = float(np.corrcoef([rank_nll.index(g) for g in SIGMA_GRID], [rank_w1.index(g) for g in SIGMA_GRID])[0, 1])
        selection_rows.append([NAMES[method], str(len(complete)), chosen, fmt(w1_mean[chosen]), oracle,
                               fmt(w1_mean[oracle]), fmt(w1_mean["0.20"]),
                               f"{100.0 * (w1_mean[chosen] - w1_mean['0.20']) / w1_mean['0.20']:+.1f}%",
                               f"{per_seed_match}/{len(complete)}", f"{spearman:+.2f}"])
    parts = [
        "### F2a. Bandwidth grid: Stage-1 score MSE, posterior W1 and held-out NPE log loss (lower is better)",
        "W1 and log loss are means over Stage-1 seeds; the log loss is −mean log q(π | context) on one fresh 20k bank.",
        "",
        table(["method", "σ_q", "n seeds", "Stage-1 std MSE", "W1", "W1 SE", "cov90", "held-out NLL", "NLL SE", "n NLL"],
              rows, "---|---:|---:|---:|---:|---:|---:|---:|---:|---:"),
        "",
        "### F2b. Simulation-only bandwidth selection vs the exact-posterior ranking",
        "‘selected’ minimizes pooled held-out NLL; ‘W1-best’ is the oracle choice. Rank correlation is over the 5 bandwidths.",
        "",
        table(["method", "n seeds", "selected σ_q", "W1 at selected", "W1-best σ_q", "best W1", "W1 at 0.20",
               "selected vs 0.20", "per-seed agreement", "rank corr (NLL, W1)"],
              selection_rows, "---|---:|---:|---:|---:|---:|---:|---:|---:|---:"),
        "",
    ]
    paired_rows = []
    for method in P1_METHODS[1:]:
        for sigma in SIGMA_GRID[1:]:
            w1_units, nll_diffs = [], []
            for s in SEEDS:
                wide, narrow = p1_cells(p1_stage2_dir(sigma, s)), p1_cells(p1_stage2_dir("0.20", s))
                if wide and narrow:
                    w1_units.append((wide[method], narrow[method]))
                if (sigma, s, method) in nll and ("0.20", s, method) in nll:
                    nll_diffs.append(nll[(sigma, s, method)] - nll[("0.20", s, method)])
            w1 = unit_contrast(w1_units)
            if w1 is None or not nll_diffs:
                continue
            paired_rows.append([NAMES[method], sigma, str(w1["n"]), signed(w1["mean"]), fmt(w1["se"]),
                                f"{w1['wins']}/{w1['n']}", f"{w1['rel']:+.1f}%",
                                signed(float(np.mean(nll_diffs)), 5), fmt(se(nll_diffs), 5),
                                f"{sum(x < 0 for x in nll_diffs)}/{len(nll_diffs)}"])
    parts += [
        "### F2c. Each bandwidth against σ_q=0.20, paired by Stage-1 seed (negative favours the wider tube)",
        "The NLL difference is a log-density difference on the same 20k held-out bank; exp(−ΔNLL) is the density ratio.",
        "",
        table(["method", "σ_q", "n seeds", "ΔW1", "SE", "wider better", "ΔW1 / W1(0.20)", "ΔNLL", "SE", "wider better (NLL)"],
              paired_rows, "---|---:|---:|---:|---:|---:|---:|---:|---:|---:"),
    ]
    return NL.join(parts)


# --------------------------------------------------------------------------- F3
def f3_tables() -> str:
    reps = (0, 1, 2)
    rows = []
    specs = [(f"σ_q={sig}: {NAMES[a]} − {NAMES[b]}", (sig, a), (sig, b))
             for sig in ("0.20", "1.052") for a, b in (("radial", "linear"), ("stacked", "linear"), ("stacked", "radial"))]
    specs += [(f"{NAMES[m]}: σ_q=1.052 − σ_q=0.20", ("1.052", m), ("0.20", m)) for m in P1_METHODS[1:]]
    for label, (sig_a, m_a), (sig_b, m_b) in specs:
        grid: dict[tuple[int, int], float] = {}
        base: list[float] = []
        cell_diffs = []
        for s in SEEDS:
            for r in reps:
                ca, cb = p1_cells(p1_stage2_dir(sig_a, s, rep=r)), p1_cells(p1_stage2_dir(sig_b, s, rep=r))
                if not ca or not cb:
                    continue
                common = sorted(set(ca[m_a]) & set(cb[m_b]))
                d = [float(ca[m_a][k]["w1_to_exact"]) - float(cb[m_b][k]["w1_to_exact"]) for k in common]
                grid[(s, r)] = float(np.mean(d))
                base.append(float(np.mean([float(cb[m_b][k]["w1_to_exact"]) for k in common])))
                cell_diffs.extend(d)
        full_seeds = [s for s in SEEDS if all((s, r) in grid for r in reps)]
        if not full_seeds:
            continue
        mat = np.asarray([[grid[(s, r)] for r in reps] for s in full_seeds])
        seed_means = mat.mean(axis=1)
        rep_means = mat.mean(axis=0)
        # Two-way random-effects SE of the grand mean: var(seed)/S + var(rep)/R + var(resid)/(S R).
        S, R = mat.shape
        grand = mat.mean()
        ms_seed = R * np.sum((seed_means - grand) ** 2) / max(S - 1, 1)
        ms_rep = S * np.sum((rep_means - grand) ** 2) / max(R - 1, 1)
        resid = mat - seed_means[:, None] - rep_means[None, :] + grand
        ms_res = np.sum(resid**2) / max((S - 1) * (R - 1), 1)
        var_seed = max((ms_seed - ms_res) / R, 0.0)
        var_rep = max((ms_rep - ms_res) / S, 0.0)
        se_two_way = math.sqrt(var_seed / S + var_rep / R + ms_res / (S * R))
        rel = 100.0 * float(grand) / float(np.mean(base))
        rows.append([label, f"{S}×{R}", signed(float(grand)), f"{rel:+.1f}%", fmt(se(seed_means)), fmt(se(rep_means)),
                     fmt(se_two_way), f"{int(np.sum(mat < 0))}/{mat.size}",
                     ", ".join(signed(float(v)) for v in rep_means), fmt(math.sqrt(var_seed)), fmt(math.sqrt(var_rep)),
                     fmt(math.sqrt(ms_res))])
    return NL.join([
        "### F3. W1 contrasts over Stage-1 seeds × Stage-2 replicates (negative favours A)",
        "Replicate 0 is the original run; replicates 1–2 change the NPE bank, NPE seed, posterior seed and test bank.",
        "Each unit is one (seed, replicate) pair averaged over its 60 test datasets; Δ / B divides by B's mean W1.",
        "SE (two-way) treats seeds and replicates as random effects; SD columns are the variance-component estimates.",
        "",
        table(["comparison (A − B)", "seeds × reps", "mean ΔW1", "Δ / B", "SE over seeds", "SE over reps", "SE (two-way)",
               "A better (seed×rep units)", "mean Δ by replicate", "SD seed", "SD rep", "SD resid"],
              rows, "---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:"),
    ])


# --------------------------------------------------------------------------- F4
def p3_cells(seed: int):
    path = RUNS / f"p3_s2_{seed}" / "posterior_by_seed.csv"
    if not path.exists():
        return None
    inverse = {label: key for key, label in P3_LABELS.items()}
    out: dict[str, dict[tuple, dict[str, str]]] = {}
    for row in read_csv(path):
        key = inverse.get(row["method"], "exact")
        out.setdefault(key, {})[(int(row["beta_index"]), int(row["seed"]))] = row
    return out


def f4_tables() -> str:
    seeds = [s for s in SEEDS if p3_cells(s)]
    parts = []
    rows = []
    for method in P3_METHODS[1:]:
        vals = []
        for s in SEEDS:
            path = RUNS / f"p3_s1_{s}" / "score_summary_by_beta.csv"
            if path.exists():
                r = [float(x["std_mse_mean"]) for x in read_csv(path) if x["method"] == method]
                if r:
                    vals.append(float(np.mean(r)))
        if vals:
            rows.append([NAMES[method], str(len(vals)), fmt(float(np.mean(vals))), fmt(se(vals))])
    parts += ["### F4a. p=3 Stage-1 standardized exact-score MSE (mean over coordinates and evaluation points)",
              table(["method", "n seeds", "std MSE", "SE"], rows, "---|---:|---:|---:"), ""]
    p3_s1: dict[str, dict[int, float]] = {}
    for method in P3_METHODS[1:]:
        for s in SEEDS:
            path = RUNS / f"p3_s1_{s}" / "score_summary_by_beta.csv"
            if path.exists():
                r = [float(x["std_mse_mean"]) for x in read_csv(path) if x["method"] == method]
                if r:
                    p3_s1.setdefault(method, {})[s] = float(np.mean(r))
    rows = []
    for a, b in (("stacked", "gate"), ("gate", "linear"), ("stacked", "linear")):
        common = sorted(set(p3_s1.get(a, {})) & set(p3_s1.get(b, {})))
        if not common:
            continue
        d = [p3_s1[a][s] - p3_s1[b][s] for s in common]
        rows.append([f"{NAMES[a]} − {NAMES[b]}", str(len(common)), signed(float(np.mean(d))), fmt(se(d)),
                     f"{sum(x < 0 for x in d)}/{len(d)}",
                     f"{100.0 * float(np.mean(d)) / float(np.mean([p3_s1[b][s] for s in common])):+.1f}%"])
    parts += ["#### F4a′. p=3 Stage-1 paired differences (readouts matched at lr 1e-3; negative favours A)",
              table(["comparison (A − B)", "n seeds", "mean Δ", "SE", "A better", "Δ / B"], rows,
                    "---|---:|---:|---:|---:|---:"), ""]
    if not seeds:
        return NL.join(parts + ["_no p=3 Stage-2 runs available yet_"])
    loaded = {s: p3_cells(s) for s in seeds}
    rows = []
    for method in ("exact",) + P3_METHODS:
        per_seed = [loaded[s][method] for s in seeds]
        sliced = [mean_metric(c, "sliced_w1") for c in per_seed]
        row = [("exact posterior" if method == "exact" else NAMES[method]), str(len(seeds)),
               fmt(float(np.mean(sliced))), fmt(se(sliced)) if method != "exact" else "—"]
        for c in P3_COORDS:
            row.append(fmt(float(np.mean([mean_metric(x, f"w1_{c}") for x in per_seed]))))
        for c in P3_COORDS:
            row.append(fmt(float(np.mean([mean_metric(x, f"sq_err_{c}") for x in per_seed])), 6))
        for c in P3_COORDS:
            row.append(fmt(float(np.mean([mean_metric(x, f"sd_abs_err_{c}") for x in per_seed]))))
        for c in P3_COORDS:
            row.append(fmt(float(np.mean([mean_metric(x, f"cov90_{c}") for x in per_seed])), 3))
        rows.append(row)
    header = (["method", "n seeds", "sliced W1", "SE"] + [f"W1 {c}" for c in P3_COORDS]
              + [f"MSE {c}" for c in P3_COORDS] + [f"|SD err| {c}" for c in P3_COORDS]
              + [f"cov90 {c}" for c in P3_COORDS])
    parts += ["### F4b. p=3 Stage-2 posterior metrics, pooled over 12 true β × 10 test datasets",
              "Sliced W1 is in box-width-standardized coordinates; marginal W1 and MSE are in (logit π, τ, log σ) units.",
              "",
              table(header, rows, "---|---:" + "|---:" * (len(header) - 2)), ""]
    rows = []
    for a, b in (("stacked", "gate"), ("gate", "linear"), ("stacked", "linear"),
                 ("linear", "pilot"), ("gate", "pilot"), ("stacked", "pilot")):
        for key, label in (("sliced_w1", "sliced W1"),) + tuple((f"w1_{c}", f"W1 {c}") for c in P3_COORDS):
            row = contrast_row(f"{NAMES[a]} − {NAMES[b]} ({label})",
                               unit_contrast([(loaded[s][a], loaded[s][b]) for s in seeds], key))
            if row:
                rows.append(row)
    parts += ["### F4c. p=3 paired contrasts (unit = Stage-1 seed; negative favours A)",
              table(CONTRAST_HEADER, rows, CONTRAST_ALIGN), ""]
    edge = max(float(r["window_edge_mass"]) for s in seeds for r in loaded[s]["exact"].values())
    boundary = np.mean([r["pilot_status"] == "boundary" for s in seeds for r in loaded[s]["exact"].values()])
    parts.append(f"Largest exact-grid window-edge marginal mass: {edge:.2e}. Test-dataset pilot boundary rate: {boundary:.3f}.")
    return NL.join(parts)


# --------------------------------------------------------------------------- F5
def f5_tables() -> str:
    rows = []
    values: dict[tuple[str, str], dict[int, float]] = {}
    for lr in ("1e-4", "1e-3"):
        for s in SEEDS:
            path = RUNS / f"m2audit_{lr}_{s}" / "score_summary_by_pi.csv"
            if not path.exists():
                continue
            data = read_csv(path)
            for method, prefix in (("linear", "linear"), ("shared_radial", "shared")):
                r = [float(x["std_mse"]) for x in data if x["method"].startswith(prefix)
                     or x.get("method_key", "") == method]
                if r:
                    values.setdefault((method, lr), {})[s] = float(np.mean(r))
    for (method, lr), per_seed in sorted(values.items()):
        rows.append([NAMES[method], lr, str(len(per_seed)), fmt(float(np.mean(list(per_seed.values())))),
                     fmt(se(per_seed.values())), ", ".join(fmt(v) for _, v in sorted(per_seed.items()))])
    parts = ["### F5a. Model 2 Stage-1 standardized MSE by learning rate",
             table(["method", "lr (readout and gate)", "n seeds", "std MSE", "SE", "per seed"], rows,
                   "---|---:|---:|---:|---:|---"), ""]
    rows = []
    for label, a, b in (("linear: lr 1e-3 − lr 1e-4", ("linear", "1e-3"), ("linear", "1e-4")),
                        ("gate: lr 1e-3 − lr 1e-4", ("shared_radial", "1e-3"), ("shared_radial", "1e-4")),
                        ("gate − linear, both lr 1e-4", ("shared_radial", "1e-4"), ("linear", "1e-4")),
                        ("gate − linear, both lr 1e-3", ("shared_radial", "1e-3"), ("linear", "1e-3"))):
        va, vb = values.get(a, {}), values.get(b, {})
        common = sorted(set(va) & set(vb))
        if not common:
            continue
        d = [va[s] - vb[s] for s in common]
        rows.append([label, str(len(common)), signed(float(np.mean(d))), fmt(se(d)), f"{sum(x < 0 for x in d)}/{len(d)}",
                     f"{100.0 * float(np.mean(d)) / float(np.mean([vb[s] for s in common])):+.1f}%"])
    parts += ["### F5b. Model 2 paired differences (negative favours A)",
              table(["comparison (A − B)", "n seeds", "mean Δ", "SE", "A better", "Δ / B"], rows, "---|---:|---:|---:|---:|---:")]
    return NL.join(parts)


def build() -> str:
    return NL.join([f1_tables(), "", f2_tables(), "", f3_tables(), "", f4_tables(), "", f5_tables()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    tables = build()
    if args.stdout or not OUT.exists():
        print(tables)
        return
    text = OUT.read_text(encoding="utf-8")
    start, stop = text.index(BEGIN) + len(BEGIN), text.index(END)
    OUT.write_text(text[:start] + NL + tables + NL + text[stop:], encoding="utf-8")
    print(f"Updated tables in {OUT}")


if __name__ == "__main__":
    main()
