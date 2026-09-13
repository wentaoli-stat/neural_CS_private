#!/usr/bin/env python3
"""Regenerate the tables in STAGE2_FINDINGS_20260913.md from runs/.

Expected layout (created by the 2026-09-13 handoff protocol):

    runs/s1_sq{SIGMA_Q}_{SEED}/   Stage 1 (config.json, training_info.json,
                                  score_summary_by_pi.csv)
    runs/s2_sq{SIGMA_Q}_{SEED}/   Stage 2 (posterior_by_seed.csv)

Every number is recomputed from those files. The tables are written between
the TABLES markers of the findings file, so the narrative is preserved:

    python scripts/summarize_stage2_findings.py            # rewrite tables
    python scripts/summarize_stage2_findings.py --stdout   # print only

Paired differences use two units. The primary unit is the Stage-1 training
seed: each seed's W1 is averaged over its 60 test cells (6 true pi x 10 test
datasets), and the SE and win count are over seeds. The cell-level version
pairs individual (seed, pi, test dataset) cells; the NPE training seed and
test bank are fixed across runs, so it measures test-bank variability only.
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
OUT = ROOT / "STAGE2_FINDINGS_20260913.md"
SIGMAS = ("0.20", "1.052")
SEEDS = (20260709, 20260710, 20260711, 20260712, 20260713)
METHODS = ("pilot", "linear", "radial", "stacked")
LABELS = {
    "pilot": "pilot-only NPE",
    "linear": "linear FSM pilot+score NPE",
    "radial": "radial FSM pilot+score NPE",
    "stacked": "stacked FSM pilot+score NPE",
}
NAMES = {"pilot": "pilot only", "linear": "linear (ILSA)", "radial": "gate", "stacked": "stacked m_dim=8"}
EXACT = "exact likelihood grid"
BEGIN, END = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
NL = chr(10)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stage1_dir(sigma: str, seed: int) -> Path:
    return RUNS / f"s1_sq{sigma}_{seed}"


def stage2_dir(sigma: str, seed: int) -> Path:
    return RUNS / f"s2_sq{sigma}_{seed}"


def available_seeds(sigma: str) -> list[int]:
    return [s for s in SEEDS if (stage2_dir(sigma, s) / "posterior_by_seed.csv").exists()]


def cells(sigma: str, seed: int) -> dict[str, dict[tuple[float, int], dict[str, str]]]:
    """method -> (pi_true, test seed) -> row, plus the exact reference."""
    out: dict[str, dict[tuple[float, int], dict[str, str]]] = {}
    inverse = {label: key for key, label in LABELS.items()}
    inverse[EXACT] = "exact"
    for row in read_csv(stage2_dir(sigma, seed) / "posterior_by_seed.csv"):
        key = inverse[row["method"]]
        out.setdefault(key, {})[(float(row["pi_true"]), int(row["seed"]))] = row
    return out


def fmt(value: float | None, digits: int = 5) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def se(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else float("nan")


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("---" + "|---:" * (len(header) - 1))
    lines = ["| " + " | ".join(header) + " |", "|" + align + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return NL.join(lines)


# ---------------------------------------------------------------- Stage 1 checks
def stage1_provenance() -> str:
    rows = []
    for sigma in SIGMAS:
        for seed in SEEDS:
            directory = stage1_dir(sigma, seed)
            if not (directory / "training_info.json").exists():
                continue
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            info = json.loads((directory / "training_info.json").read_text(encoding="utf-8"))
            shas = {item["minibatch_sha256"] for item in info.values()}
            nesting = [float(item.get("nested_initialization_max_abs_diff", 0.0))
                       for item in info.values()]
            best = ", ".join(f"{m}:{info[m]['best_step']}" for m in ("linear", "radial", "stacked")
                             if m in info)
            rows.append([
                sigma, str(seed),
                f"{config['lr']:g} / {config['gate_lr']:g} / {config['joint_rho_lr']:g}",
                "yes" if len(shas) == 1 else f"NO ({len(shas)})",
                f"{max(nesting):.1e}",
                best,
            ])
    return table(["σ_q", "seed", "lr / gate_lr / joint_rho_lr", "one minibatch SHA", "max nesting err",
                  "best step"], rows, "---|---:|---|---|---:|---")


def stage1_mse(sigma: str, seed: int, method: str) -> float | None:
    path = stage1_dir(sigma, seed) / "score_summary_by_pi.csv"
    if not path.exists():
        return None
    rows = [r for r in read_csv(path) if r["method"].startswith(method)]
    return float(np.mean([float(r["std_mse"]) for r in rows])) if rows else None


def stage1_table() -> str:
    rows = []
    for sigma in SIGMAS:
        for method in METHODS[1:]:
            values = {s: stage1_mse(sigma, s, method) for s in SEEDS}
            values = {s: v for s, v in values.items() if v is not None}
            if not values:
                continue
            arr = list(values.values())
            rows.append([sigma, NAMES[method], str(len(arr)), fmt(float(np.mean(arr))),
                         fmt(se(arr)) if len(arr) > 1 else "—"])
    return table(["σ_q", "method", "n seeds", "std MSE (mean)", "SE"], rows, "---|---|---:|---:|---:")


def stage1_paired_table() -> str:
    rows = []
    for sigma in SIGMAS:
        for a, b in (("stacked", "radial"), ("radial", "linear"), ("stacked", "linear")):
            diffs, base = [], []
            for seed in SEEDS:
                va, vb = stage1_mse(sigma, seed, a), stage1_mse(sigma, seed, b)
                if va is not None and vb is not None:
                    diffs.append(va - vb)
                    base.append(vb)
            if not diffs:
                continue
            rows.append([f"σ_q={sigma}: {NAMES[a]} − {NAMES[b]}", str(len(diffs)),
                         f"{np.mean(diffs):+.5f}", fmt(se(diffs)) if len(diffs) > 1 else "—",
                         f"{sum(d < 0 for d in diffs)}/{len(diffs)}",
                         f"{100.0 * np.mean(diffs) / np.mean(base):+.1f}%"])
    return table(["comparison (A − B)", "n seeds", "mean Δ std MSE", "SE", "A better", "Δ / B"], rows,
                 "---|---:|---:|---:|---:|---:")


# ---------------------------------------------------------------- Stage 2 metrics
def metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    return {
        "mse": float(np.mean([float(r["pi_sq_err"]) for r in rows])),
        "rmse_exact": float(math.sqrt(np.mean([float(r["mean_sq_err_to_exact"]) for r in rows]))),
        "w1": float(np.mean([float(r["w1_to_exact"]) for r in rows])),
        "sd": float(np.mean([float(r["pi_post_std"]) for r in rows])),
        "exact_sd": float(np.mean([float(r["exact_pi_post_std"]) for r in rows])),
        "sd_abs_err": float(np.mean([float(r["sd_abs_err_to_exact"]) for r in rows])),
        "cov90": float(np.mean([float(r["pi_coverage90"]) for r in rows])),
    }


def seed_metrics(sigma: str, method: str, pi: float | None) -> list[dict[str, float]]:
    """One metrics dict per available Stage-1 seed."""
    out = []
    for seed in available_seeds(sigma):
        table_cells = cells(sigma, seed).get(method, {})
        rows = [row for (p, _), row in table_cells.items() if pi is None or p == pi]
        if rows:
            out.append(metrics(rows))
    return out


def metric_rows(pi: float | None) -> list[list[str]]:
    rows = []
    for sigma in SIGMAS:
        seeds = available_seeds(sigma)
        if not seeds:
            continue
        exact = [metrics([row for (p, _), row in cells(sigma, s)["exact"].items() if pi is None or p == pi])
                 for s in seeds]
        rows.append([sigma, "exact posterior", str(len(seeds)),
                     fmt(np.mean([m["mse"] for m in exact]), 6), "0", "0", "—",
                     fmt(np.mean([m["exact_sd"] for m in exact]), 4), "0",
                     fmt(np.mean([m["cov90"] for m in exact]), 3)])
        for method in METHODS:
            per_seed = seed_metrics(sigma, method, pi)
            if not per_seed:
                continue
            w1 = [m["w1"] for m in per_seed]
            rows.append([
                sigma, NAMES[method], str(len(per_seed)),
                fmt(np.mean([m["mse"] for m in per_seed]), 6),
                fmt(np.mean([m["rmse_exact"] for m in per_seed]), 5),
                fmt(np.mean(w1), 5),
                fmt(se(w1), 5) if len(w1) > 1 else "—",
                fmt(np.mean([m["sd"] for m in per_seed]), 4),
                fmt(np.mean([m["sd_abs_err"] for m in per_seed]), 5),
                fmt(np.mean([m["cov90"] for m in per_seed]), 3),
            ])
    return rows


METRIC_HEADER = ["σ_q", "method", "n seeds", "post-mean MSE vs truth", "RMSE vs exact mean",
                 "W1 to exact", "W1 SE (seeds)", "post SD", "|SD − exact SD|", "cov90"]
METRIC_ALIGN = "---|---|---:|---:|---:|---:|---:|---:|---:|---:"


def per_pi_w1_table() -> str:
    pis = sorted({p for sigma in SIGMAS for s in available_seeds(sigma)
                  for (p, _) in cells(sigma, s)["exact"]})
    rows = []
    for sigma in SIGMAS:
        for method in METHODS:
            values = [seed_metrics(sigma, method, p) for p in pis]
            if not values or not values[0]:
                continue
            rows.append([sigma, NAMES[method], str(len(values[0]))]
                        + [fmt(float(np.mean([m["w1"] for m in v])), 5) for v in values])
    return table(["σ_q", "method", "n seeds"] + [f"π={p:g}" for p in pis], rows,
                 "---|---|---:" + "|---:" * len(pis))


# ---------------------------------------------------------------- paired differences
def paired(pairs: list[tuple[dict, dict]]) -> list[str]:
    """pairs: per Stage-1 seed, (reference cells, competitor cells) keyed by (pi, test seed)."""
    seed_diffs, cell_diffs = [], []
    for ref, comp in pairs:
        keys = sorted(set(ref) & set(comp))
        diff = [float(ref[k]["w1_to_exact"]) - float(comp[k]["w1_to_exact"]) for k in keys]
        if diff:
            seed_diffs.append(float(np.mean(diff)))
            cell_diffs.extend(diff)
    if not seed_diffs:
        return []
    comp_mean = float(np.mean([np.mean([float(r["w1_to_exact"]) for r in c.values()]) for _, c in pairs]))
    return [
        str(len(seed_diffs)),
        f"{np.mean(seed_diffs):+.5f}",
        fmt(se(seed_diffs), 5) if len(seed_diffs) > 1 else "—",
        f"{sum(d < 0 for d in seed_diffs)}/{len(seed_diffs)}",
        f"{100.0 * np.mean(seed_diffs) / comp_mean:+.1f}%",
        str(len(cell_diffs)),
        fmt(se(cell_diffs), 5),
        f"{sum(d < 0 for d in cell_diffs)}/{len(cell_diffs)}",
    ]


PAIRED_HEADER = ["comparison (A − B)", "n seeds", "mean ΔW1", "SE (seeds)", "A better (seeds)",
                 "Δ / W1(B)", "n cells", "SE (cells)", "A better (cells)"]
PAIRED_ALIGN = "---|---:|---:|---:|---:|---:|---:|---:|---:"


def method_contrasts() -> str:
    rows = []
    for sigma in SIGMAS:
        seeds = available_seeds(sigma)
        loaded = {s: cells(sigma, s) for s in seeds}
        for a, b in (("stacked", "radial"), ("radial", "linear"), ("stacked", "linear"),
                     ("linear", "pilot"), ("radial", "pilot"), ("stacked", "pilot")):
            pairs = [(loaded[s][a], loaded[s][b]) for s in seeds if a in loaded[s] and b in loaded[s]]
            result = paired(pairs)
            if result:
                rows.append([f"σ_q={sigma}: {NAMES[a]} − {NAMES[b]}"] + result)
    return table(PAIRED_HEADER, rows, PAIRED_ALIGN)


def bandwidth_contrasts() -> str:
    rows = []
    wide, narrow = SIGMAS[1], SIGMAS[0]
    seeds = [s for s in SEEDS if s in available_seeds(wide) and s in available_seeds(narrow)]
    loaded = {(sig, s): cells(sig, s) for sig in SIGMAS for s in seeds}
    for method in METHODS[1:]:
        pairs = [(loaded[(wide, s)][method], loaded[(narrow, s)][method]) for s in seeds]
        result = paired(pairs)
        if result:
            rows.append([f"{NAMES[method]}: σ_q={wide} − σ_q={narrow}"] + result)
    # Cross-bandwidth best-versus-best: each method at its better pooled bandwidth is
    # not computed here, to avoid selecting on the evaluation metric.
    return table(PAIRED_HEADER, rows, PAIRED_ALIGN)


def pilot_identity() -> str:
    """The pilot-only NPE does not read Stage 1, so its rows should agree across runs."""
    reference = None
    max_diff = 0.0
    n_runs = 0
    for sigma in SIGMAS:
        for seed in available_seeds(sigma):
            w1 = {k: float(r["w1_to_exact"]) for k, r in cells(sigma, seed)["pilot"].items()}
            n_runs += 1
            if reference is None:
                reference = w1
            else:
                max_diff = max(max_diff, max(abs(w1[k] - reference[k]) for k in reference))
    return f"Pilot-only W1 max abs difference across {n_runs} Stage-2 runs: {max_diff:.2e}."


def build() -> str:
    parts = [
        "### T1. Stage-1 provenance",
        stage1_provenance(),
        "",
        "### T2. Stage-1 frozen exact-score standardized MSE (pooled over π ∈ {0.10, 0.30, 0.50, 0.65})",
        stage1_table(),
        "",
        "#### T2b. Stage-1 paired differences in standardized MSE (negative favours A; SE over seeds)",
        stage1_paired_table(),
        "",
        "### T3. Stage-2 posterior metrics, pooled over 6 true π × 10 test datasets",
        "Means over Stage-1 seeds of each seed's 60-cell average; RMSE is per seed, then averaged.",
        "",
        table(METRIC_HEADER, metric_rows(None), METRIC_ALIGN),
        "",
        "### T4. Stage-2 W1 to the exact posterior, per true π",
        per_pi_w1_table(),
        "",
    ]
    pis = sorted({p for sigma in SIGMAS for s in available_seeds(sigma) for (p, _) in cells(sigma, s)["exact"]})
    for p in pis:
        parts += [f"#### T5 (π = {p:g}). All posterior metrics", table(METRIC_HEADER, metric_rows(p), METRIC_ALIGN), ""]
    parts += [
        "### T6. Paired W1 differences between methods (negative favours A)",
        method_contrasts(),
        "",
        "### T7. Paired W1 differences between bandwidths, within method (negative favours σ_q=1.052)",
        bandwidth_contrasts(),
        "",
        pilot_identity(),
    ]
    return NL.join(parts)


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
