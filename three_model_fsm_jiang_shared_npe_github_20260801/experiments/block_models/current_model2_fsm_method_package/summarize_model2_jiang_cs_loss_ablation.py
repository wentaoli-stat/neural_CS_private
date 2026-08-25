#!/usr/bin/env python3
"""Audit and summarize paired Model-2 Jiang loss-ablation runs."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METHOD_ORDER = [
    "linear_cs_plain_base",
    "structured_cs_plain",
    "linear_cs",
    "structured_cs",
]
METHOD_LABELS = {
    "linear_cs_plain_base": "M2 J-linear-CS-plain-base",
    "structured_cs_plain": "M2 J-structured-CS-plain",
    "linear_cs": "M2 J-linear-CS-full",
    "structured_cs": "M2 J-structured-CS-full",
}
COMPARISONS = [
    ("plain_structured_vs_plain_linear", "structured_cs_plain", "linear_cs_plain_base"),
    ("full_structured_vs_full_linear", "structured_cs", "linear_cs"),
    ("full_structured_vs_plain_structured", "structured_cs", "structured_cs_plain"),
    ("full_linear_vs_plain_linear", "linear_cs", "linear_cs_plain_base"),
]


def se(values: np.ndarray) -> float:
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / math.sqrt(values.size))


def load_and_audit(pattern: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    run_dirs = [
        path
        for item in sorted(glob.glob(pattern))
        if (path := Path(item)).is_dir()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matched {pattern!r}")
    for run_dir in run_dirs:
        summary_path = run_dir / "score_summary_by_pi.csv"
        sanity_path = run_dir / "sanity_checks.json"
        config_path = run_dir / "config.json"
        trace_path = run_dir / "training_trace.csv"
        for path in (summary_path, sanity_path, config_path, trace_path):
            if not path.exists():
                raise FileNotFoundError(path)
        frame = pd.read_csv(summary_path)
        if set(frame["method_id"]) != set(METHOD_ORDER):
            raise ValueError(
                f"{run_dir} has methods {sorted(set(frame['method_id']))}, "
                f"expected {sorted(METHOD_ORDER)}"
            )
        frame["run_dir"] = str(run_dir)
        frames.append(frame)

        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        trace = pd.read_csv(trace_path)
        plain_trace = trace[trace["method"].isin(
            ["linear_cs_plain_base", "structured_cs_plain"]
        )]
        plain_penalty = pd.to_numeric(
            plain_trace.get("train_fisher_penalty", pd.Series(dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        max_plain_penalty = float(plain_penalty.abs().max()) if len(plain_penalty) else 0.0
        contracts = config.get("loss_contracts", {})
        plain_contract_ok = all(
            contracts.get(method, {}).get("fisher_curvature_penalty") is False
            and contracts.get(method, {}).get("global_bias_subtraction") is False
            and contracts.get(method, {}).get("conditional_debias_regression") is False
            for method in ("linear_cs_plain_base", "structured_cs_plain")
        )
        full_contract_ok = all(
            contracts.get(method, {}).get("fisher_curvature_penalty") is True
            and contracts.get(method, {}).get("global_bias_subtraction") is True
            and contracts.get(method, {}).get("conditional_debias_regression") is True
            for method in ("linear_cs", "structured_cs")
        )
        audit = {
            "run_dir": str(run_dir),
            "seed": int(frame["seed"].iloc[0]),
            "sanity_passed": bool(sanity.get("passed")),
            "plain_strict_nesting": float(
                sanity.get("plain", {}).get("strict_nesting_max_abs", np.nan)
            ),
            "full_strict_nesting": float(
                sanity.get("full", {}).get("strict_nesting_max_abs", np.nan)
            ),
            "gate_initial_difference": float(
                sanity.get("plain_full_gate_initial_max_abs", np.nan)
            ),
            "max_plain_fisher_penalty": max_plain_penalty,
            "plain_contract_ok": plain_contract_ok,
            "full_contract_ok": full_contract_ok,
        }
        audit["passed"] = bool(
            audit["sanity_passed"]
            and audit["plain_strict_nesting"] < 2e-6
            and audit["full_strict_nesting"] < 2e-6
            and audit["gate_initial_difference"] == 0.0
            and max_plain_penalty == 0.0
            and plain_contract_ok
            and full_contract_ok
        )
        if not audit["passed"]:
            raise RuntimeError(f"Run audit failed: {audit}")
        audits.append(audit)
    data = pd.concat(frames, ignore_index=True)
    duplicates = data.duplicated(["seed", "pi", "method_id"], keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate paired cells:\n"
            + data.loc[
                duplicates, ["seed", "pi", "method_id", "run_dir"]
            ].to_string(index=False)
        )
    expected = set(METHOD_ORDER)
    for (seed, pi), group in data.groupby(["seed", "pi"]):
        if set(group["method_id"]) != expected:
            raise ValueError(f"Incomplete cell seed={seed}, pi={pi}")
    return data, audits


def method_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (pi, method), group in data.groupby(["pi", "method_id"], sort=True):
        mse = group["mse"].to_numpy(dtype=float)
        rows.append(
            {
                "pi": float(pi),
                "method_id": method,
                "method": METHOD_LABELS[method],
                "n_seeds": int(group["seed"].nunique()),
                "mse_mean": float(np.mean(mse)),
                "mse_se": se(mse),
                "std_mse_mean": float(group["std_mse"].mean()),
                "corr_mean": float(group["corr"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    result["order"] = result["method_id"].map(order)
    return result.sort_values(["pi", "order"]).drop(columns="order")


def paired(data: pd.DataFrame, candidate: str, baseline: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    pi_values: list[float | str] = sorted(data["pi"].unique().tolist()) + ["pooled"]
    for pi in pi_values:
        subset = data if pi == "pooled" else data[data["pi"] == pi]
        wide = (
            subset.groupby(["seed", "method_id"], as_index=False)["mse"]
            .mean()
            .pivot(index="seed", columns="method_id", values="mse")
            .dropna(subset=[candidate, baseline])
        )
        candidate_values = wide[candidate].to_numpy(dtype=float)
        baseline_values = wide[baseline].to_numpy(dtype=float)
        difference = candidate_values - baseline_values
        relative = 100.0 * (
            baseline_values - candidate_values
        ) / np.maximum(baseline_values, 1e-12)
        if candidate_values.size >= 2:
            test = stats.ttest_rel(candidate_values, baseline_values)
            t_stat, p_value = float(test.statistic), float(test.pvalue)
        else:
            t_stat = p_value = float("nan")
        rows.append(
            {
                "pi": pi,
                "candidate_id": candidate,
                "candidate": METHOD_LABELS[candidate],
                "baseline_id": baseline,
                "baseline": METHOD_LABELS[baseline],
                "n_seeds": int(candidate_values.size),
                "candidate_mse_mean": float(np.mean(candidate_values)),
                "baseline_mse_mean": float(np.mean(baseline_values)),
                "paired_mse_difference_mean": float(np.mean(difference)),
                "paired_mse_difference_se": se(difference),
                "relative_improvement_pct_mean": float(np.mean(relative)),
                "relative_improvement_pct_se": se(relative),
                "candidate_win_rate": float(np.mean(candidate_values < baseline_values)),
                "paired_t": t_stat,
                "paired_p_two_sided": p_value,
            }
        )
    return pd.DataFrame(rows)


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data, audits = load_and_audit(args.run_glob)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = method_summary(data)
    comparisons = {
        name: paired(data, candidate, baseline)
        for name, candidate, baseline in COMPARISONS
    }
    data.to_csv(output / "all_paired_score_rows.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "run_audit.csv", index=False)
    summary.to_csv(output / "method_summary_by_pi.csv", index=False)
    for name, frame in comparisons.items():
        frame.to_csv(output / f"paired_{name}.csv", index=False)

    lines = [
        "# Model 2 Jiang CS Loss Ablation",
        "",
        "All four arms share D^S/D^R caches, gate initialization, and paired test draws within seed.",
        "",
        "- Plain arms: direct score matching only.",
        "- Full arms: direct SM, Fisher curvature penalty, global bias subtraction, and conditional debias regression.",
        "- Linear/structured arms use the identical frozen marginal/pairwise CS pipeline; the structured arm adds only the signed positive gates.",
        "",
        "## MSE by test pi",
        "",
        "| pi | method | seeds | MSE mean | MSE SE | std MSE | corr |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.pi:g} | {row.method} | {row.n_seeds} | {fmt(row.mse_mean)} | "
            f"{fmt(row.mse_se)} | {fmt(row.std_mse_mean)} | {fmt(row.corr_mean)} |"
        )
    for name, frame in comparisons.items():
        lines.extend(
            [
                "",
                f"## {name.replace('_', ' ')}",
                "",
                "Positive improvement means the candidate has lower MSE.",
                "",
                "| pi | improvement | SE | wins | paired p |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in frame.itertuples(index=False):
            pi = row.pi if isinstance(row.pi, str) else f"{float(row.pi):g}"
            lines.append(
                f"| {pi} | {fmt(row.relative_improvement_pct_mean)}% | "
                f"{fmt(row.relative_improvement_pct_se)} pp | "
                f"{fmt(row.candidate_win_rate)} | {fmt(row.paired_p_two_sided)} |"
            )
    (output / "MODEL2_JIANG_CS_LOSS_ABLATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("Saved audited Model-2 summary to", output)


if __name__ == "__main__":
    main()
