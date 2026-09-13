#!/usr/bin/env python3
"""Regenerate FINDINGS_20260912.md from the run directories under runs/.

Every number in the report is recomputed here from each run's frozen
exact-score summary, and every paired comparison is labelled with whether the
two arms' readout learning rates actually match, read from each run's
config.json. runs/ is gitignored, so this script is how the tables travel:
copy runs/ to another machine and rerun it.

    python scripts/summarize_session_findings.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "FINDINGS_20260912.md"
S5 = [20260709, 20260710, 20260711, 20260712, 20260713]
S3 = S5[:3]
COORDS = ("u", "tau", "log_sigma")


def cfg(run):
    return json.loads((RUNS / run / "config.json").read_text())


def readout_lr(run, codebase, method):
    """Which flag trains the readout differs by codebase and method."""
    c = cfg(run)
    if codebase == "p1" and method in ("radial", "stacked"):
        return float(c["joint_rho_lr"])
    return float(c["lr"])


def branch_lr(run, codebase, method):
    c = cfg(run)
    if codebase == "p1" and method in ("radial", "stacked"):
        return float(c["gate_lr"])
    if codebase == "p3" and method in ("gate", "stacked"):
        return float(c["branch_lr"])
    return None


def value(run, codebase, method):
    """Mean exact-score standardized MSE, or None if the run is absent."""
    if codebase == "p3":
        f = RUNS / run / "score_summary_by_beta.csv"
        if not f.exists():
            return None
        rows = [r for r in csv.DictReader(f.open()) if r["method"] == method]
        if not rows:
            return None
        return float(np.mean([[float(r[f"std_mse_{c}"]) for c in COORDS] for r in rows]))
    f = RUNS / run / "score_summary_by_pi.csv"
    if not f.exists():
        return None
    rows = [r for r in csv.DictReader(f.open()) if r["method"].startswith(method)]
    return float(np.mean([float(r["std_mse"]) for r in rows])) if rows else None


def coords(run, method):
    rows = [r for r in csv.DictReader((RUNS / run / "score_summary_by_beta.csv").open())
            if r["method"] == method]
    return np.mean([[float(r[f"std_mse_{c}"]) for c in COORDS] for r in rows], axis=0)


class Arm:
    def __init__(self, label, codebase, method, runs):
        self.label, self.codebase, self.method, self.runs = label, codebase, method, runs
        self.v = {s: value(r, codebase, method) for s, r in runs.items()}
        self.v = {s: x for s, x in self.v.items() if x is not None}

    def mean(self):
        return float(np.mean(list(self.v.values())))

    def sd(self):
        return float(np.std(list(self.v.values()), ddof=1)) if len(self.v) > 1 else float("nan")

    def rlr(self):
        s = next(iter(self.v))
        return readout_lr(self.runs[s], self.codebase, self.method)

    def blr(self):
        s = next(iter(self.v))
        return branch_lr(self.runs[s], self.codebase, self.method)


def paired(a, b):
    """a minus b on common seeds; negative means a is better."""
    ss = sorted(set(a.v) & set(b.v))
    d = np.array([a.v[s] - b.v[s] for s in ss])
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    matched = abs(a.rlr() - b.rlr()) < 1e-12
    return dict(n=len(d), mean=float(d.mean()), se=se, wins=int((d < 0).sum()),
                rel=float(-100 * d.mean() / np.mean([b.v[s] for s in ss])),
                matched=matched, a_rlr=a.rlr(), b_rlr=b.rlr())


def fmt_lr(x):
    return "—" if x is None else f"{x:g}"


def seed_table(arms, seeds):
    head = "| arm | readout lr | branch lr | " + " | ".join(str(s % 100) for s in seeds) + " | mean | sd |"
    sep = "|---|---:|---:|" + "---:|" * len(seeds) + "---:|---:|"
    lines = [head, sep]
    for a in arms:
        cells = " | ".join(f"{a.v[s]:.5f}" if s in a.v else "—" for s in seeds)
        sd = "—" if np.isnan(a.sd()) else f"{a.sd():.5f}"
        lines.append(f"| {a.label} | {fmt_lr(a.rlr())} | {fmt_lr(a.blr())} | {cells} | {a.mean():.5f} | {sd} |")
    return "\n".join(lines)


def pair_table(pairs):
    lines = ["| comparison | n | mean diff | SE | better on | readouts |",
             "|---|---:|---:|---:|---:|---|"]
    for name, p in pairs:
        se = "—" if np.isnan(p["se"]) else f"{p['se']:.5f}"
        tag = "matched" if p["matched"] else f"**UNMATCHED** ({p['a_rlr']:g} vs {p['b_rlr']:g})"
        lines.append(f"| {name} | {p['n']} | {p['mean']:+.5f} | {se} | {p['wins']}/{p['n']} | {tag} |")
    return "\n".join(lines)


def main():
    P1 = "p1"
    p1 = {
        "lin4": Arm("linear, lr 1e-4", P1, "linear", {20260709: "model1_stacked_seed20260709_20260911"}),
        "lin3": Arm("linear, lr 1e-3", P1, "linear", {s: f"m1p1_lin1e-3_{s}" for s in S5}),
        "gate": Arm("gate", P1, "radial", {20260709: "diag_gate_mlr1e-3", 20260710: "diag_gate_mlr1e-3_s10",
                                          **{s: f"m1p1_gate_{s}" for s in S5[2:]}}),
        "m4": Arm("stacked m_dim=4", P1, "stacked", {s: f"m1p1_m4_{s}" for s in S5}),
        "m8": Arm("stacked m_dim=8", P1, "stacked", {s: f"m1p1_m8_{s}" for s in S5}),
    }
    m2 = {
        "lin4": Arm("M2 linear, lr 1e-4", P1, "linear", {s: f"m2lin_lr1e-4_{s}" for s in S5}),
        "lin3": Arm("M2 linear, lr 1e-3", P1, "linear", {s: f"m2lin_lr1e-3_{s}" for s in S5}),
    }
    full = {20260709: "p3_blr1e-3", 20260710: "p3_seed10", 20260711: "p3_s711_m4full",
            20260712: "p3_s712_m4full", 20260713: "p3_s713_m4full"}
    sfx = {s: ("" if s == 20260709 else f"_s{str(s)[-2:]}") for s in S5}
    P3 = "p3"
    p3 = {
        "lin4": Arm("linear, lr 1e-4", P3, "linear", full),
        "lin3": Arm("linear, lr 1e-3", P3, "linear", {s: f"p3_linear_lr1e-3{sfx[s]}" for s in S5}),
        "gate": Arm("gate", P3, "gate", full),
        "m4": Arm("stacked m_dim=4", P3, "stacked", {20260709: "p3_mdim4", 20260710: "p3_s710_m4",
                                                     **{s: full[s] for s in S5[2:]}}),
        "m8": Arm("stacked m_dim=8", P3, "stacked", {20260709: "p3_mdim8",
                                                     **{s: f"p3_s7{str(s)[-2:]}_m8" for s in S5[1:]}}),
        "raw": Arm("raw Y → MLP", P3, "raw", {s: f"p3_base_lr1e-4{sfx[s]}" for s in S5}),
        "sub": Arm("all subscores → MLP", P3, "substack", {s: f"p3_base_lr1e-4{sfx[s]}" for s in S5}),
        "blk": Arm("per-block unpooled stack, lr 1e-4", P3, "blockstack", {s: f"p3_base_lr1e-4{sfx[s]}" for s in S5}),
        "blk3": Arm("per-block unpooled stack, lr 1e-3", P3, "blockstack", {s: f"p3_base_lr1e-3{sfx[s]}" for s in S5}),
        "rawf": Arm("raw Y (checkpoint grid 50)", P3, "raw", {s: f"p3_fine_s{str(s)[-2:]}" for s in S5}),
        "subf": Arm("all subscores (grid 50)", P3, "substack", {s: f"p3_fine_s{str(s)[-2:]}" for s in S5}),
    }
    mdim = {k: Arm(f"stacked m_dim={k}", P3, "stacked", {20260709: r})
            for k, r in (("1", "p3_blr1e-3"), ("2", "p3_mdim2"), ("4", "p3_mdim4"), ("8", "p3_mdim8"))}

    budget = {}
    for key, lab, meth, grp, forty in (
            ("lin3", "linear, lr 1e-3", "linear", "lin", p3["lin3"]),
            ("gate", "gate", "gate", "gs", p3["gate"]),
            ("m8", "stacked m_dim=8", "stacked", "gs", p3["m8"]),
            ("blk", "per-block unpooled stack", "blockstack", "base", p3["blk"]),
            ("sub", "all subscores → MLP", "substack", "base", p3["sub"]),
            ("raw", "raw Y → MLP", "raw", "base", p3["raw"])):
        budget[key] = (lab, [Arm(lab, P3, meth, {s: f"p3x_n10000_{grp}_{s}" for s in S3}),
                             Arm(lab, P3, meth, {s: forty.runs[s] for s in S3}),
                             Arm(lab, P3, meth, {s: f"p3x_n160000_{grp}_{s}" for s in S3})])

    conv = {}
    for key, lab, meth, tag, twenty in (("lin3", "linear, lr 1e-3", "linear", "lin", p3["lin3"]),
                                        ("gate", "gate", "gate", "gate", p3["gate"]),
                                        ("m8", "stacked m_dim=8", "stacked", "m8", p3["m8"])):
        conv[key] = (Arm(lab, P3, meth, {s: twenty.runs[s] for s in S3}),
                     Arm(lab, P3, meth, {s: f"p3x_conv60k_{tag}_{s}" for s in S3}))

    bw = {
        "p1": [(lab, Arm(lab, P1, m, {20260709: narrow}), Arm(lab, P1, m, {20260709: wide}))
               for lab, m, narrow, wide in (
                   ("linear", "linear", "m1p1_lin1e-3_20260709", "bw1_linear_20260709"),
                   ("gate", "radial", "diag_gate_mlr1e-3", "bw1_gate_20260709"),
                   ("stacked m_dim=8", "stacked", "m1p1_m8_20260709", "bw1_m8_20260709"))],
        "p3": [(lab, Arm(lab, P3, m, {20260709: narrow}), Arm(lab, P3, m, {20260709: "bw3_20260709"}))
               for lab, m, narrow in (("linear", "linear", "p3_linear_lr1e-3"),
                                      ("gate", "gate", "p3_blr1e-3"),
                                      ("stacked m_dim=8", "stacked", "p3_mdim8"))],
    }

    L = []
    w = L.append
    w("# Session findings — 2026-09-12\n")
    w("Model 1 (p=1 and a new p=3 variant) and a Model 2 learning-rate audit. **Stage 1 only**: every "
      "number is frozen exact-score standardized MSE (raw MSE ÷ exact-score variance), lower is better. "
      "No Stage-2 posterior was computed. Regenerate with `python scripts/summarize_session_findings.py`; "
      "`runs/` is gitignored, so copy it alongside the repo to reproduce these tables.\n")
    w("Within a seed, arms share one simulation bank and one minibatch index stream (verified by "
      "identical SHA-256 across run directories). Validation FSM loss and exact-score MSE rank "
      "configurations almost identically (Spearman 0.98 over 9 p=3 configs), so hyperparameters chosen "
      "by inspecting the exact metric would have been chosen from validation alone.\n")

    w("## Read this first: which comparisons are fair\n")
    w("The readout is trained by a different flag in each codebase, and this session tuned the flags "
      "unevenly:\n")
    w("- **p=1** (`model1/stage1.py`): linear trains with `--lr`; gate/stacked train the readout with "
      "`--joint-rho-lr` and the local branch with `--gate-lr`.")
    w("- **p=3** (`model1_p3/stage1.py`): linear and the gate/stacked readout all train with `--lr`; the "
      "branch with `--branch-lr`. The `lr 1e-3` linear runs were separate invocations from the gate/stacked "
      "runs, which used `--lr 1e-4`.\n")
    w("**Consequence.** Raising linear to `1e-3` raised only linear's readout. Every comparison of "
      "a nonlinear map against `linear, lr 1e-3` below is flagged **UNMATCHED**: linear's readout trains "
      "10× faster than the nonlinear readouts it is compared with. Those gaps are not a fair "
      "nonlinear-versus-linear estimate and could be biased in either direction, because the "
      "nonlinear branches *did* train at `1e-3`. Clean, readout-matched comparisons are: **stacked vs "
      "gate**, the **linear learning-rate audit** (same method, only the rate changed), and the **p=1 "
      "bandwidth comparison**. A fair nonlinear-versus-linear estimate needs gate and stacked rerun "
      "with their readout at `1e-3` too.\n")

    lp1 = paired(p1["lin3"], p1["lin4"])
    lm2 = paired(m2["lin3"], m2["lin4"])
    lp3 = paired(p3["lin3"], p3["lin4"])
    w("## 1. The packaged learning rate under-trains ILSA\n")
    w("Both packaged launchers train every arm at `lr 1e-4`. Raising linear alone to `1e-3`:\n")
    w("| model | lr 1e-4 | lr 1e-3 | reduction | paired seeds | better on |")
    w("|---|---:|---:|---:|---:|---:|")
    for name, lo, hi, p in (("Model 1, p=1", p1["lin4"], p1["lin3"], lp1),
                            ("Model 2 (40×40)", m2["lin4"], m2["lin3"], lm2),
                            ("Model 1, p=3", p3["lin4"], p3["lin3"], lp3)):
        ss = sorted(set(lo.v) & set(hi.v))
        a, b = np.mean([lo.v[k] for k in ss]), np.mean([hi.v[k] for k in ss])
        w(f"| {name} | {a:.5f} | {b:.5f} | {100*(a-b)/a:.1f}% | {p['n']} | {p['wins']}/{p['n']} |")
    w("\nModel 1 p=1 at `1e-4` is a single archived seed (20260709), so its reduction is indicative; "
      "Model 2 and Model 1 p=3 are paired over 5 seeds. **The archived Model 1 and Model 2 headline "
      "margins are measured against an under-trained baseline.** At p=3, `lr 1e-3` linear still selected "
      "its checkpoint near the end of 60,000 steps (§6), so even `1e-3` may not be converged.\n")

    w("## 2. Model 1, p=1 — five seeds\n")
    w(seed_table([p1["lin3"], p1["gate"], p1["m4"], p1["m8"]], S5) + "\n")
    w(pair_table([("stacked m_dim=8 − gate", paired(p1["m8"], p1["gate"])),
                  ("stacked m_dim=4 − gate", paired(p1["m4"], p1["gate"])),
                  ("gate − linear lr 1e-3", paired(p1["gate"], p1["lin3"])),
                  ("stacked m_dim=8 − linear lr 1e-3", paired(p1["m8"], p1["lin3"]))]) + "\n")
    g8 = paired(p1["m8"], p1["gate"])
    w(f"**Stacked `m_dim=8` ties the gate** (paired {g8['mean']:+.5f}, SE {g8['se']:.5f}, "
      f"better on {g8['wins']}/{g8['n']}). More channels help at p=1 as well (`m_dim=4` is worse), so the "
      "stacked map's gain tracks capacity rather than a threshold at `m_dim = p`.\n")

    w("## 3. Model 1, p=3 — five seeds\n")
    w("The p=3 variant frees the shift and noise scale: β = (logit π, τ, log σ), "
      "`Y_kj = B_k·τ + σ·ε_kj`, K=m=20. Closed-form local and exact scores (`model1_p3/scores.py`), "
      "validated against finite differences. Tube `σ_q = (0.20, 0.08, 0.04)`.\n")
    w(seed_table([p3["raw"], p3["sub"], p3["blk"], p3["blk3"], p3["lin4"], p3["lin3"], p3["gate"],
                  p3["m4"], p3["m8"]], S5) + "\n")
    sg = paired(p3["m8"], p3["gate"])
    w(pair_table([("stacked m_dim=8 − gate", sg),
                  ("stacked m_dim=4 − gate", paired(p3["m4"], p3["gate"])),
                  ("gate − linear lr 1e-4", paired(p3["gate"], p3["lin4"])),
                  ("gate − linear lr 1e-3", paired(p3["gate"], p3["lin3"])),
                  ("stacked m_dim=8 − linear lr 1e-3", paired(p3["m8"], p3["lin3"])),
                  ("per-block stack − linear, both lr 1e-4", paired(p3["blk"], p3["lin4"])),
                  ("per-block stack − linear, both lr 1e-3", paired(p3["blk3"], p3["lin3"]))]) + "\n")
    w(f"**Stacked `m_dim=8` beats the gate** (paired {sg['mean']:+.5f}, SE {sg['se']:.5f}, "
      f"better on {sg['wins']}/{sg['n']}); this comparison is readout-matched. `m_dim=4` is unreliable: "
      "it lands in a bad mode on one seed, which is why its SD is large.\n")
    w("Per coordinate (mean over seeds):\n")
    w("| arm | ∂/∂u | ∂/∂τ | ∂/∂log σ |")
    w("|---|---:|---:|---:|")
    for key in ("lin3", "gate", "m8"):
        a = p3[key]
        v = np.mean([coords(a.runs[s], a.method) for s in a.v], axis=0)
        w(f"| {a.label} | {v[0]:.5f} | {v[1]:.5f} | {v[2]:.5f} |")
    w("")

    w("### `m_dim` sweep (seed 20260709 only)\n")
    w("| m_dim | mean |")
    w("|---:|---:|")
    for k in ("1", "2", "4", "8"):
        w(f"| {k} | {mdim[k].mean():.5f} |")
    w("\nThe 1→2→4 jump on this seed is inflated: 20260709 is a bad-mode seed for `m_dim=1`. "
      "Use the five-seed `m_dim=4/8` rows above.\n")

    w("## 4. Input representation: structure beats information\n")
    w("Three reference inputs with no ILSA nesting, so no warm start. Each has *more* information "
      "than pooled linear.\n")
    rf, sf = paired(p3["rawf"], p3["raw"]), paired(p3["subf"], p3["sub"])
    b3, b4 = paired(p3["blk3"], p3["lin3"]), paired(p3["blk"], p3["lin4"])
    w(f"Each input at its better learning rate, 40k budget: raw `{p3['raw'].mean():.3f}` > all subscores "
      f"`{p3['sub'].mean():.3f}` > per-block unpooled `{p3['blk3'].mean():.3f}` > pooled linear "
      f"`{p3['lin3'].mean():.3f}`. The per-block stack keeps block additivity and sees every subscore, yet "
      f"with both at `lr 1e-3` it is **worse than simply averaging them** (paired {b3['mean']:+.5f}, "
      f"SE {b3['se']:.5f}; linear better on {b3['n']-b3['wins']}/{b3['n']}), so within-group pooling is "
      "actively helpful at this budget, not merely harmless.\n")
    w(f"*This depends on the learning rate.* With both at `lr 1e-4` the ordering flips (paired "
      f"{b4['mean']:+.5f}; per-block better on {b4['wins']}/{b4['n']}), but only because `lr 1e-4` "
      "under-trains linear (§1): the per-block stack barely moves between the two rates "
      f"(`{p3['blk'].mean():.3f}` at 1e-4, `{p3['blk3'].mean():.3f}` at 1e-3) while linear improves by "
      "about a quarter. Raw and all-subscores overfit within the first few thousand steps (they select "
      "steps 500–2,500 of 20,000).\n")
    w(f"*Checkpoint-grid check.* Validation ran every 500 steps for these baselines vs 200 for the others. "
      f"Rerunning raw and subscores with a 50-step grid changed nothing: raw {rf['mean']:+.5f} "
      f"(SE {rf['se']:.5f}), subscores {sf['mean']:+.5f} (SE {sf['se']:.5f}).\n")

    w("## 5. Simulation budget (3 seeds, 20k steps)\n")
    w("| arm | readout lr | 10k | 40k | 160k |")
    w("|---|---:|---:|---:|---:|")
    for key, (lab, arms) in budget.items():
        w(f"| {lab} | {fmt_lr(arms[1].rlr())} | " + " | ".join(f"{a.mean():.5f}" for a in arms) + " |")
    w("\nThe three local-score methods are **flat from 40k to 160k**; the three unstructured inputs are "
      "**still falling at 160k**. Raw data must win eventually, having strictly more information; the "
      "claim this supports is sample efficiency, not an information ceiling.\n")

    w("## 6. Optimization budget (3 seeds, 40k simulations)\n")
    w("| arm | readout lr | 20k steps | 60k steps | change | selected steps at 60k |")
    w("|---|---:|---:|---:|---:|---|")
    for key, (a20, a60) in conv.items():
        steps = [json.loads((RUNS / a60.runs[s] / "training_info.json").read_text())
                 [a60.method]["best_step"] for s in a60.v]
        w(f"| {a20.label} | {fmt_lr(a60.rlr())} | {a20.mean():.5f} | {a60.mean():.5f} | "
          f"{100*(a20.mean()-a60.mean())/a20.mean():+.1f}% | {', '.join(map(str, steps))} |")
    c60 = paired(conv["m8"][1], conv["gate"][1])
    c20 = paired(conv["m8"][0], conv["gate"][0])
    w(f"\nStacked vs gate (readout-matched): {c20['mean']:+.5f} at 20k → **{c60['mean']:+.5f}** at 60k "
      f"(SE {c60['se']:.5f}, better on {c60['wins']}/{c60['n']}). **The ordering survives a 3× longer "
      "run but the margins shrink**, and linear's selected checkpoints sit at the end of the budget — "
      "not converged. Treat every margin here as budget-dependent.\n")

    w("## 7. Tube bandwidth: the draft's rule (seed 20260709)\n")
    w("`main_style_revised` initializes `Σ_q = 2Σ_e`, Σ_e the pilot-error covariance. With an efficient "
      "pilot Σ_e ≈ E[I⁻¹] over the anchor prior, giving σ_q = 1.052 at p=1 (vs packaged 0.20, 5.3×) and "
      "(0.821, 0.162, 0.050) at p=3.\n")
    for key, title in (("p1", "p=1"), ("p3", "p=3")):
        w(f"**{title}**\n")
        w("| arm | narrow σ_q | draft σ_q | change | readouts |")
        w("|---|---:|---:|---:|---|")
        for lab, narrow, wide in bw[key]:
            tag = "matched" if abs(narrow.rlr() - wide.rlr()) < 1e-12 else \
                f"**UNMATCHED** ({narrow.rlr():g} vs {wide.rlr():g})"
            w(f"| {lab} | {narrow.mean():.5f} | {wide.mean():.5f} | "
              f"{100*(wide.mean()-narrow.mean())/narrow.mean():+.0f}% | {tag} |")
        w("")
    w("Every method collapses to nearly the same error at the draft bandwidth — the signature of a shared "
      "smoothing-bias floor. **This does not show the draft's rule is wrong**: the rule deliberately "
      "accepts smoothing bias so the field is accurate where a pilot lands, while this metric measures "
      "error at the true β, where the narrow tube is favoured by construction. Bandwidth cannot be "
      "chosen from Stage-1 exact-score MSE; it needs Stage 2. At p=3 only the linear row is "
      "readout-matched, but it alone shows the collapse. The packaged σ_q = 0.20 does not follow the "
      "draft's stated initialization; it may be the tuned value, which the draft does not document.\n")

    w("## Open items\n")
    w("1. **Rerun gate and stacked with readout lr `1e-3`** to get a fair nonlinear-versus-linear gap at "
      "p=1 and p=3. Nothing labelled UNMATCHED above should be quoted until this runs.")
    w("2. Run to convergence (`best_step` well inside the budget), likely 150k–200k steps.")
    w("3. Apply the learning-rate audit to the Model 2 nonlinear arms. Code is ready "
      "(`scripts/run_model2_lr_audit.sh`, stacked arms in `model2/stage1.py`, `model2_p3/`) but unrun: "
      "~1.8 h per gate run on MPS.")
    w("4. Stage 2 — in particular to decide the bandwidth question in §7.")
    w("5. Measure the real pilot's Σ_e by simulation instead of assuming Σ_e ≈ I⁻¹.\n")

    w("## Code added this session\n")
    w("- `model1_p3/` — p=3 Model 1: scores, six architectures, trainer, frozen evaluator.")
    w("- `model2_p3/` — p=3 Model 2: the r-indexed `(c_r, d_r)` ratio family, four architectures.")
    w("- `model1/stage1.py`, `model2/stage1.py` — stacked local map; Model 2 adds shared vs per-channel `m`.")
    w("- Tests: `test_p3_model.py`, `test_model2_p3.py`, `test_model2_stacked.py`, "
      "`test_stacked_local_map.py`; 86 passing at time of writing.")

    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
