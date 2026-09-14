# Follow-up findings — 2026-09-13/14

This report carries out the four "next steps" of `STAGE2_FINDINGS_20260913.md` for Model 1 (p=1 and p=3), plus the linear-only Model 2 learning-rate audit (option A). **Every figure is over 5 Stage-1 training seeds (20260709–20260713) unless stated otherwise.** The tables at the end are regenerated from `runs/` by `python scripts/summarize_followup_findings.py` (set `PYTHONIOENCODING=utf-8` on a Windows console when using `--stdout`). Percentages are paired mean differences divided by the comparison arm's mean. W1 is always the Wasserstein-1 distance of an NPE posterior to the exact posterior, averaged over 60 test datasets (p=1) or 120 (p=3).

## Read this first: the earlier SEs were too small

**Step 3 shows that the Stage-2 random streams — NPE simulation bank, NPE seed, posterior seed and test bank — contribute more variance to method contrasts than Stage-1 training does.** `STAGE2_FINDINGS_20260913.md` held all of these fixed, so its SEs covered Stage-1 variability only, and its point estimates came from one draw of the Stage-2 streams. That draw (replicate 0) turns out to have *understated* the nonlinear-versus-linear gap (F3):

| contrast at σ_q=1.052 | replicate 0 | replicate 1 | replicate 2 | pooled (3 reps × 5 seeds) | two-way SE |
|---|---:|---:|---:|---:|---:|
| gate − linear ΔW1 | −0.00089 | −0.00292 | −0.00362 | −0.00248 (**−25.6%**) | 0.00082 |
| stacked − linear ΔW1 | −0.00053 | −0.00333 | −0.00341 | −0.00242 (**−25.0%**) | 0.00095 |

For these contrasts the replicate SD (≈0.0014–0.0016) is 6–8× the seed SD (≈0.0002–0.0003). With only 3 replicates, the replicate component is itself estimated on 2 degrees of freedom. **Steps 1, 2 and 4 used only replicate 0**, so their SEs, like the earlier ones, do not include this component. Their paired *orderings* are still informative, but their margins should be read as one draw.

## Step 3 — Stage-2 replicates (F3)

Replicates 1 and 2 shift the NPE simulation bank (`--sbi-train-seed` +1, +2), the NPE seed and the posterior seed (+1000 each), and the test bank (seeds 110–119, 120–129). They were applied to every σ_q ∈ {0.20, 1.052} Stage-1 run. Pooled over 5 seeds × 3 replicates, with a two-way random-effects SE:

| contrast | ΔW1 | Δ / B | SE (two-way) | A better (seed×rep) |
|---|---:|---:|---:|---:|
| σ_q=0.20: gate − linear | −0.00150 | −15.0% | 0.00045 | 13/15 |
| σ_q=0.20: stacked − linear | −0.00091 | −9.1% | 0.00053 | 12/15 |
| σ_q=0.20: stacked − gate | +0.00059 | +7.0% | 0.00034 | 2/15 |
| σ_q=1.052: gate − linear | −0.00248 | −25.6% | 0.00082 | 15/15 |
| σ_q=1.052: stacked − linear | −0.00242 | −25.0% | 0.00095 | 15/15 |
| σ_q=1.052: stacked − gate | +0.00005 | +0.7% | 0.00026 | 7/15 |
| linear: 1.052 − 0.20 | −0.00030 | −3.0% | 0.00047 | 9/15 |
| gate: 1.052 − 0.20 | −0.00127 | −15.0% | 0.00037 | 15/15 |
| stacked: 1.052 − 0.20 | −0.00181 | −19.9% | 0.00046 | 15/15 |

**These revise `STAGE2_FINDINGS_20260913.md`. The following contradict it; they are stated, not reconciled:**

- *Fair nonlinear gain at the draft bandwidth* is ≈25%, not 10.3% (gate) and 6.2% (stacked). This is close to the 24.7% W1 reduction in the paper's abstract, which the earlier report had called overstated. That earlier criticism does not survive replication.
- *At σ_q=0.20* the gate's gain over linear is −15.0% (about 3.3 SE), where the earlier report found −6.5% and "not distinguishable from zero".
- *The bandwidth gain for linear* is not robust: −3.0% (SE 0.00047, 9/15) against the earlier −8.8% (4/5). For gate (−15.0%) and stacked (−19.9%) it holds on all 15 seed×replicate units.
- *Stacked vs gate:* stacked is behind at σ_q=0.20 (+7.0%, 2/15) and tied at σ_q=1.052 (+0.7%).

## Step 1 — 60k-step Stage 1 at σ_q=1.052 (F1; replicate 0)

Stage 1 was run for 60,000 steps (cosine decay from step 30,000, the same proportion as 10k/20k), otherwise under the same protocol.

- **Stage 1 does not improve.** Standardized exact-score MSE changes by +0.00006 (linear), +0.00208 (gate) and −0.00019 (stacked), with 60k better on only 1–2 of 5 seeds. Validation-best steps are late (52,600–57,100) in 3 seeds but near 31,000–33,000 in the other 2.
- **Stage 2 does not improve either.** ΔW1 (60k − 20k): linear −3.3% (SE 0.00017, 4/5), gate +1.1%, stacked +1.4%.
- **The bandwidth effect is unchanged.** σ_q=1.052 at 60k vs σ_q=0.20 at 20k gives −11.8% / −11.6% / −14.5% (linear / gate / stacked), against −8.8% / −12.5% / −15.7% at 20k.

**The earlier caveat — that an unconverged wide tube might understate the bandwidth effect — is not supported.** Note also that after 60k steps the replicate-0 method gaps shrink (gate − linear −6.2%, SE 0.00032; stacked − linear −1.6%). Given F3, this is one draw of the Stage-2 streams and should not be read as a training-length effect without replicates.

## Step 2 — bandwidth grid and simulation-only selection (F2; replicate 0)

**W1 vs bandwidth.** Every wider tube beats σ_q=0.20 for every method (F2c: better on 3–5 of 5 seeds in all 12 cells), even though Stage-1 score MSE rises monotonically (linear 0.041 → 0.155 → 0.459 → 0.500 → 0.780 across 0.20 → 0.5 → 0.977 → 1.052 → 2.0):

| method | 0.20 | 0.5 | 0.977 | 1.052 | 2.0 |
|---|---:|---:|---:|---:|---:|
| linear | 0.00941 | 0.00902 | **0.00845** | 0.00858 | 0.00887 |
| gate | 0.00880 | 0.00819 | 0.00770 | 0.00770 | **0.00744** |
| stacked | 0.00955 | 0.00824 | 0.00800 | 0.00805 | **0.00798** |

For linear, W1 is lowest near the measured draft value 0.977 (= √(2Σ_e)). For gate and stacked it is still falling or flat at 2.0, **the edge of the grid**, so their optimum is not bracketed.

**Simulation-only selection by held-out NPE log loss (NLL).** The score is −mean log q(π | context) on one fresh 20,000-simulation bank. Minimizing pooled NLL selects σ_q=1.052 for all three methods, whose W1 is within 0.9–3.5% of the W1-best bandwidth and 8.8–15.7% better than 0.20. **This agreement is not evidence that the criterion works.**

- Rank correlation between NLL and W1 across the 5 bandwidths is +0.60 (linear), −0.50 (gate) and −0.20 (stacked).
- Per-seed argmin agreement is 2/5, 0/5 and 1/5.
- In F2c the paired NLL difference against 0.20 has the "right" sign in only 4 of 12 cells, and is within ~1 SE in most. Gate at σ_q=2.0, for example, has the best W1 (−15.5%, 5/5) but a *worse* NLL (+0.00140).

**The criterion is too insensitive to rank bandwidths at this bank size.** F3 suggests why: a single NPE fit per (σ_q, seed) carries replicate-level noise of the same order as the bandwidth effect. A usable data-only criterion probably needs NLL averaged over several NPE fits and simulation banks, or a calibration-based check (next steps).

## Step 4a — p=3 Stage 2 (F4; replicate 0)

**Model and implementation.** β = (logit π, τ, log σ) with prior uniform on the Stage-1 anchor box (π ∈ [0.05, 0.70], τ ∈ [0.5, 2], σ ∈ [0.7, 1.4]); note that at p=1 the prior is uniform in π. The components are:

- *Pilot:* the box-constrained maximizer of the marginal composite likelihood.
- *Context:* (pilot β, frozen score at the pilot) ∈ ℝ⁶.
- *NPE:* a 3-d MDN.
- *Test bank:* 12 true β ({0.10, 0.30, 0.65} × {0.8, 1.5} × {0.85, 1.2}) × 10 datasets.
- *Exact posterior:* a 40³ grid over the box, then an 80³ grid over mean ± 8 SD; the largest marginal mass at a zoom-window edge is 7.0e-5.
- *Multivariate distance:* **sliced W1**, the mean over 100 fixed directions of the 1-d W1 in box-width-standardized coordinates. It stands in for the true multivariate W1, which is not computed. Marginal W1 is reported per coordinate.
- *Stage 1:* readout-matched (`--lr 1e-3 --branch-lr 1e-3`), σ_q = (0.20, 0.08, 0.04), 20k steps.

**Stage 1 (F4a, F4a′).** Gate and stacked both beat linear by 17% (−0.03993, SE 0.00227; −0.04006, SE 0.00149; 5/5 each) and **tie each other** (stacked − gate −0.00012, SE 0.00127, 3/5). This contradicts FINDINGS_20260912 §3, where stacked beat gate at p=3 (−0.01401, SE 0.00082, 5/5) with readouts at `lr 1e-4`. The earlier UNMATCHED gaps against linear (−0.01875 gate, −0.03276 stacked) understated the gate's matched gap.

**Stage 2 (F4b, F4c).**

| method | sliced W1 | vs linear | SE | better | W1 logit π | W1 τ | W1 log σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| pilot only | 0.09212 | +236% | — | — | 0.2648 | 0.2292 | 0.0383 |
| linear | 0.02742 | — | — | — | 0.1086 | 0.0594 | 0.0067 |
| gate | 0.02492 | −9.1% | 0.00074 | 4/5 | 0.1004 | 0.0515 | 0.0060 |
| stacked | 0.02430 | −11.4% | 0.00051 | 5/5 | 0.0961 | 0.0503 | 0.0057 |

- **Nonlinear aggregation helps at p=3 in posterior terms too.** Marginal-W1 gains over linear are 7.5–13.3% (gate) and 11.5–15.2% (stacked) per coordinate.
- **Stacked vs gate** (−2.5%, SE 0.00086, 3/5) is not distinguishable.
- **Score methods vs pilot only:** the score cuts sliced W1 by 70–74% relative to the pilot alone.
- **Posterior SD and coverage** are close to the exact posterior for all score methods.
- **Posterior-mean MSE against the truth** does not separate the methods; it is dominated by data noise, and the exact posterior's own MSE for logit π is 0.237.

**Caveats specific to p=3.**

- **Pilot boundary rate.** The pilot sits on the prior boundary for **42.5% of test datasets**: τ at its lower bound in 20.8%, π at its upper bound in 18.3% and π at its lower bound in 7.5%, reaching 50–70% at τ=0.8. The composite marginal likelihood is weakly identified along a small-τ/large-π ridge, so the pilot is poor there (the pilot-only NPE is 3.4× worse than linear). The score context compensates, but a better pilot could change the method gaps.
- **One draw of the Stage-2 streams** (see F3).
- **Tube not from the draft rule.** The p=3 tube is the packaged (0.20, 0.08, 0.04), not the draft rule's (0.821, 0.162, 0.050). Given Step 2, a wider p=3 tube may do better; this was not tried.

## Step 4b — Model 2 learning-rate audit, linear only (F5)

Option A ran Model 2 linear at both learning rates with the stacked-audit launcher (40×40 blocks, σ_q=0.15, constant rate, 20k steps).

- **Learning-rate effect.** lr 1e-3 lowers Stage-1 standardized MSE from 0.15750 to 0.09571: **−39.2%** (paired −0.06179, SE 0.00534, 5/5).
- **Reproduction.** The lr 1e-4 mean reproduces FINDINGS_20260912 §1 exactly (0.15750). **The lr 1e-3 mean does not** (0.09571 here vs 0.09163 there, so −39.2% vs −41.8%). This is not reconciled here. The chunked-feature change below alters standardization constants only at rounding level, and the 1e-4 arm, run through the same code, matches exactly.
- **Both rates under-train.** Validation-best steps are 19,400–20,000 at 1e-4 and 16,200–19,600 at 1e-3.
- **The Model 2 gate arm was not run.** It measured ~9 s/step on this CPU (~50 h per run), so the gate-versus-linear question for Model 2 remains open.

## Code and environment changes

- **`model2/stage1.py`: memory.** `featurize` and `make_feature_stats` now process 2,000 datasets at a time; the full 40k bank needed ~50 GB of float64 temporaries. Features are bit-identical. Feature statistics for banks larger than one chunk come from float64 running sums, which agree with the one-shot numpy values to ~1e-15 relative. Banks of 2,000 datasets or fewer use the original code path unchanged. Tests: `tests/test_model2_chunked_features.py`.
- **New module `model1_p3/stage2_npe.py`**, with launcher `scripts/run_model1_p3_stage2_npe.sh` and tests `tests/test_model1_p3_stage2.py`. The tests check the grid likelihood against `scores.log_likelihood`, the zoomed grid against a 120³ full-box grid, pilot determinism and recovery, sliced-W1 sanity, and a tiny end-to-end run.
- **New scripts:**
  - `scripts/heldout_npe_nll_model1.py`: held-out NPE log loss.
  - `scripts/run_model1_stage2_npe_stacked_rep.sh`: Stage-2 replicates.
  - `scripts/run_model1_followup_job.sh`, `scripts/run_model1_p3_followup_job.sh` and `scripts/run_job_queue.sh`: the job pipeline and queue.
  - `scripts/summarize_followup_findings.py`: regenerates this report's tables.
- **Wall time on CPU** (i9-9900, 4 jobs in parallel at 3 threads): the 45-job Model 1 queue ran 08:20 → 00:00 (15.7 h). p=1 Stage 1 took ~40 min at 20k steps and ~5 h at 60k; p=3 Stage 1 1.2–1.9 h; p=3 Stage 2 28–43 min. The Model 2 linear jobs took ~35 min each.

## Next steps (not run)

1. **More Stage-2 replicates** (≥5, ideally with NPE bank and test bank varied separately) before quoting any method or bandwidth margin. The replicate component dominates, and 3 replicates estimate it poorly.
2. **Extend the p=1 bandwidth grid past 2.0** for gate and stacked, whose W1 optimum is at the edge. Re-test simulation-only selection with NLL averaged over replicates, or with simulation-based calibration.
3. **p=3:** a better-identified pilot (multi-start, or one reparametrized along the π–τ ridge), the draft-rule wide tube, and Stage-2 replicates.
4. **Model 2 gate arm** on a GPU/MPS machine, together with the unreconciled lr 1e-3 linear value.

## Tables

Regenerated by `scripts/summarize_followup_findings.py`; do not edit by hand.

<!-- TABLES:BEGIN -->
### F1a. Stage-1 standardized MSE at σ_q=1.052: 20k vs 60k steps (negative Δ favours 60k)
| method | n seeds | 20k | 60k | Δ (60k − 20k) | SE | 60k better | 60k best steps |
|---|---:|---:|---:|---:|---:|---:|---|
| linear (ILSA) | 5 | 0.49999 | 0.50005 | +0.00006 | 0.00159 | 2/5 | 56400, 55600, 57100, 32700, 31000 |
| gate | 5 | 0.49546 | 0.49753 | +0.00208 | 0.00092 | 1/5 | 43600, 55600, 38000, 52600, 31000 |
| stacked m_dim=8 | 5 | 0.49542 | 0.49523 | -0.00019 | 0.00200 | 1/5 | 56400, 55600, 37600, 52600, 31000 |

### F1b. Stage-2 W1 to the exact posterior: paired differences (unit = Stage-1 seed)
| comparison (A − B) | n units | mean Δ | SE (units) | A better | Δ / B | n cells | SE (cells) |
|---|---:|---:|---:|---:|---:|---:|---:|
| pilot only: σ_q=1.052 60k − 1.052 20k | 5 | +0.00002 | 0.00001 | 1/5 | +0.1% | 300 | 0.00002 |
| pilot only: σ_q=1.052 60k − 0.20 20k | 5 | +0.00002 | 0.00001 | 1/5 | +0.1% | 300 | 0.00002 |
| linear (ILSA): σ_q=1.052 60k − 1.052 20k | 5 | -0.00029 | 0.00017 | 4/5 | -3.3% | 300 | 0.00015 |
| linear (ILSA): σ_q=1.052 60k − 0.20 20k | 5 | -0.00111 | 0.00045 | 4/5 | -11.8% | 300 | 0.00036 |
| gate: σ_q=1.052 60k − 1.052 20k | 5 | +0.00008 | 0.00034 | 2/5 | +1.1% | 300 | 0.00017 |
| gate: σ_q=1.052 60k − 0.20 20k | 5 | -0.00102 | 0.00030 | 5/5 | -11.6% | 300 | 0.00032 |
| stacked m_dim=8: σ_q=1.052 60k − 1.052 20k | 5 | +0.00011 | 0.00034 | 3/5 | +1.4% | 300 | 0.00018 |
| stacked m_dim=8: σ_q=1.052 60k − 0.20 20k | 5 | -0.00139 | 0.00030 | 5/5 | -14.5% | 300 | 0.00038 |

### F1c. Method contrasts after 60k-step Stage 1 (W1)
| comparison (A − B) | n units | mean Δ | SE (units) | A better | Δ / B | n cells | SE (cells) |
|---|---:|---:|---:|---:|---:|---:|---:|
| σ_q=1.052 60k: gate − linear (ILSA) | 5 | -0.00052 | 0.00032 | 3/5 | -6.2% | 300 | 0.00023 |
| σ_q=1.052 60k: stacked m_dim=8 − linear (ILSA) | 5 | -0.00013 | 0.00025 | 4/5 | -1.6% | 300 | 0.00027 |
| σ_q=1.052 60k: stacked m_dim=8 − gate | 5 | +0.00038 | 0.00020 | 2/5 | +4.9% | 300 | 0.00021 |

### F2a. Bandwidth grid: Stage-1 score MSE, posterior W1 and held-out NPE log loss (lower is better)
W1 and log loss are means over Stage-1 seeds; the log loss is −mean log q(π | context) on one fresh 20k bank.

| method | σ_q | n seeds | Stage-1 std MSE | W1 | W1 SE | cov90 | held-out NLL | NLL SE | n NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pilot only | 0.20 | 5 | — | 0.02279 | 0.00001 | 0.833 | -0.9030 | 0.0000 | 5 |
| pilot only | 0.5 | 5 | — | 0.02281 | 0.00000 | 0.833 | -0.9029 | 0.0000 | 5 |
| pilot only | 0.977 | 5 | — | 0.02281 | 0.00000 | 0.833 | -0.9029 | 0.0000 | 5 |
| pilot only | 1.052 | 5 | — | 0.02279 | 0.00001 | 0.833 | -0.9030 | 0.0000 | 5 |
| pilot only | 2.0 | 5 | — | 0.02281 | 0.00000 | 0.833 | -0.9029 | 0.0000 | 5 |
| linear (ILSA) | 0.20 | 5 | 0.04064 | 0.00941 | 0.00041 | 0.873 | -0.9453 | 0.0005 | 5 |
| linear (ILSA) | 0.5 | 5 | 0.15529 | 0.00902 | 0.00010 | 0.883 | -0.9446 | 0.0002 | 5 |
| linear (ILSA) | 0.977 | 5 | 0.45935 | 0.00845 | 0.00013 | 0.887 | -0.9464 | 0.0002 | 5 |
| linear (ILSA) | 1.052 | 5 | 0.49999 | 0.00858 | 0.00020 | 0.883 | -0.9473 | 0.0005 | 5 |
| linear (ILSA) | 2.0 | 5 | 0.78011 | 0.00887 | 0.00010 | 0.890 | -0.9446 | 0.0003 | 5 |
| gate | 0.20 | 5 | 0.02266 | 0.00880 | 0.00034 | 0.883 | -0.9501 | 0.0012 | 5 |
| gate | 0.5 | 5 | 0.14065 | 0.00819 | 0.00028 | 0.890 | -0.9493 | 0.0010 | 5 |
| gate | 0.977 | 5 | 0.45451 | 0.00770 | 0.00026 | 0.883 | -0.9500 | 0.0011 | 5 |
| gate | 1.052 | 5 | 0.49546 | 0.00770 | 0.00023 | 0.897 | -0.9513 | 0.0012 | 5 |
| gate | 2.0 | 5 | 0.77902 | 0.00744 | 0.00018 | 0.900 | -0.9487 | 0.0003 | 5 |
| stacked m_dim=8 | 0.20 | 5 | 0.02820 | 0.00955 | 0.00037 | 0.877 | -0.9491 | 0.0017 | 5 |
| stacked m_dim=8 | 0.5 | 5 | 0.14529 | 0.00824 | 0.00024 | 0.893 | -0.9470 | 0.0003 | 5 |
| stacked m_dim=8 | 0.977 | 5 | 0.45581 | 0.00800 | 0.00025 | 0.897 | -0.9485 | 0.0004 | 5 |
| stacked m_dim=8 | 1.052 | 5 | 0.49542 | 0.00805 | 0.00014 | 0.890 | -0.9493 | 0.0011 | 5 |
| stacked m_dim=8 | 2.0 | 5 | 0.77991 | 0.00798 | 0.00021 | 0.900 | -0.9477 | 0.0004 | 5 |

### F2b. Simulation-only bandwidth selection vs the exact-posterior ranking
‘selected’ minimizes pooled held-out NLL; ‘W1-best’ is the oracle choice. Rank correlation is over the 5 bandwidths.

| method | n seeds | selected σ_q | W1 at selected | W1-best σ_q | best W1 | W1 at 0.20 | selected vs 0.20 | per-seed agreement | rank corr (NLL, W1) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear (ILSA) | 5 | 1.052 | 0.00858 | 0.977 | 0.00845 | 0.00941 | -8.8% | 2/5 | +0.60 |
| gate | 5 | 1.052 | 0.00770 | 2.0 | 0.00744 | 0.00880 | -12.5% | 0/5 | -0.50 |
| stacked m_dim=8 | 5 | 1.052 | 0.00805 | 2.0 | 0.00798 | 0.00955 | -15.7% | 1/5 | -0.20 |

### F2c. Each bandwidth against σ_q=0.20, paired by Stage-1 seed (negative favours the wider tube)
The NLL difference is a log-density difference on the same 20k held-out bank; exp(−ΔNLL) is the density ratio.

| method | σ_q | n seeds | ΔW1 | SE | wider better | ΔW1 / W1(0.20) | ΔNLL | SE | wider better (NLL) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear (ILSA) | 0.5 | 5 | -0.00039 | 0.00031 | 4/5 | -4.2% | +0.00071 | 0.00063 | 2/5 |
| linear (ILSA) | 0.977 | 5 | -0.00096 | 0.00036 | 4/5 | -10.2% | -0.00108 | 0.00044 | 4/5 |
| linear (ILSA) | 1.052 | 5 | -0.00083 | 0.00051 | 4/5 | -8.8% | -0.00205 | 0.00029 | 5/5 |
| linear (ILSA) | 2.0 | 5 | -0.00054 | 0.00032 | 4/5 | -5.7% | +0.00066 | 0.00053 | 1/5 |
| gate | 0.5 | 5 | -0.00061 | 0.00046 | 3/5 | -6.9% | +0.00084 | 0.00209 | 2/5 |
| gate | 0.977 | 5 | -0.00110 | 0.00053 | 5/5 | -12.5% | +0.00011 | 0.00181 | 2/5 |
| gate | 1.052 | 5 | -0.00110 | 0.00040 | 5/5 | -12.5% | -0.00115 | 0.00156 | 4/5 |
| gate | 2.0 | 5 | -0.00136 | 0.00040 | 5/5 | -15.5% | +0.00140 | 0.00117 | 2/5 |
| stacked m_dim=8 | 0.5 | 5 | -0.00132 | 0.00022 | 5/5 | -13.8% | +0.00209 | 0.00167 | 2/5 |
| stacked m_dim=8 | 0.977 | 5 | -0.00156 | 0.00042 | 4/5 | -16.3% | +0.00063 | 0.00158 | 2/5 |
| stacked m_dim=8 | 1.052 | 5 | -0.00150 | 0.00039 | 5/5 | -15.7% | -0.00021 | 0.00144 | 3/5 |
| stacked m_dim=8 | 2.0 | 5 | -0.00158 | 0.00044 | 5/5 | -16.5% | +0.00139 | 0.00180 | 2/5 |

### F3. W1 contrasts over Stage-1 seeds × Stage-2 replicates (negative favours A)
Replicate 0 is the original run; replicates 1–2 change the NPE bank, NPE seed, posterior seed and test bank.
Each unit is one (seed, replicate) pair averaged over its 60 test datasets; Δ / B divides by B's mean W1.
SE (two-way) treats seeds and replicates as random effects; SD columns are the variance-component estimates.

| comparison (A − B) | seeds × reps | mean ΔW1 | Δ / B | SE over seeds | SE over reps | SE (two-way) | A better (seed×rep units) | mean Δ by replicate | SD seed | SD rep | SD resid |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| σ_q=0.20: gate − linear (ILSA) | 5×3 | -0.00150 | -15.0% | 0.00028 | 0.00045 | 0.00045 | 13/15 | -0.00061, -0.00187, -0.00202 | 0.00000 | 0.00055 | 0.00122 |
| σ_q=0.20: stacked m_dim=8 − linear (ILSA) | 5×3 | -0.00091 | -9.1% | 0.00018 | 0.00053 | 0.00053 | 12/15 | +0.00014, -0.00142, -0.00145 | 0.00000 | 0.00079 | 0.00101 |
| σ_q=0.20: stacked m_dim=8 − gate | 5×3 | +0.00059 | +7.0% | 0.00034 | 0.00009 | 0.00034 | 2/15 | +0.00076, +0.00045, +0.00057 | 0.00059 | 0.00000 | 0.00085 |
| σ_q=1.052: gate − linear (ILSA) | 5×3 | -0.00248 | -25.6% | 0.00018 | 0.00082 | 0.00082 | 15/15 | -0.00089, -0.00292, -0.00362 | 0.00018 | 0.00139 | 0.00060 |
| σ_q=1.052: stacked m_dim=8 − linear (ILSA) | 5×3 | -0.00242 | -25.0% | 0.00018 | 0.00095 | 0.00095 | 15/15 | -0.00053, -0.00333, -0.00341 | 0.00025 | 0.00162 | 0.00054 |
| σ_q=1.052: stacked m_dim=8 − gate | 5×3 | +0.00005 | +0.7% | 0.00021 | 0.00023 | 0.00026 | 7/15 | +0.00035, -0.00041, +0.00021 | 0.00027 | 0.00028 | 0.00066 |
| linear (ILSA): σ_q=1.052 − σ_q=0.20 | 5×3 | -0.00030 | -3.0% | 0.00026 | 0.00045 | 0.00047 | 9/15 | -0.00083, -0.00067, +0.00060 | 0.00027 | 0.00067 | 0.00090 |
| gate: σ_q=1.052 − σ_q=0.20 | 5×3 | -0.00127 | -15.0% | 0.00037 | 0.00023 | 0.00037 | 15/15 | -0.00110, -0.00172, -0.00099 | 0.00065 | 0.00000 | 0.00088 |
| stacked m_dim=8: σ_q=1.052 − σ_q=0.20 | 5×3 | -0.00181 | -19.9% | 0.00032 | 0.00038 | 0.00046 | 15/15 | -0.00150, -0.00257, -0.00136 | 0.00059 | 0.00058 | 0.00072 |

### F4a. p=3 Stage-1 standardized exact-score MSE (mean over coordinates and evaluation points)
| method | n seeds | std MSE | SE |
|---|---:|---:|---:|
| linear (ILSA) | 5 | 0.23429 | 0.00359 |
| gate | 5 | 0.19435 | 0.00158 |
| stacked m_dim=8 | 5 | 0.19423 | 0.00228 |

#### F4a′. p=3 Stage-1 paired differences (readouts matched at lr 1e-3; negative favours A)
| comparison (A − B) | n seeds | mean Δ | SE | A better | Δ / B |
|---|---:|---:|---:|---:|---:|
| stacked m_dim=8 − gate | 5 | -0.00012 | 0.00127 | 3/5 | -0.1% |
| gate − linear (ILSA) | 5 | -0.03993 | 0.00227 | 5/5 | -17.0% |
| stacked m_dim=8 − linear (ILSA) | 5 | -0.04006 | 0.00149 | 5/5 | -17.1% |

### F4b. p=3 Stage-2 posterior metrics, pooled over 12 true β × 10 test datasets
Sliced W1 is in box-width-standardized coordinates; marginal W1 and MSE are in (logit π, τ, log σ) units.

| method | n seeds | sliced W1 | SE | W1 u | W1 tau | W1 log_sigma | MSE u | MSE tau | MSE log_sigma | |SD err| u | |SD err| tau | |SD err| log_sigma | cov90 u | cov90 tau | cov90 log_sigma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exact posterior | 5 | 0.00000 | — | 0.00000 | 0.00000 | 0.00000 | 0.237196 | 0.022525 | 0.001367 | 0.00000 | 0.00000 | 0.00000 | 0.925 | 0.900 | 0.892 |
| pilot only | 5 | 0.09212 | 0.00000 | 0.26480 | 0.22923 | 0.03826 | 0.348441 | 0.078680 | 0.002883 | 0.09619 | 0.14294 | 0.01969 | 0.925 | 0.925 | 0.917 |
| linear (ILSA) | 5 | 0.02742 | 0.00014 | 0.10858 | 0.05936 | 0.00674 | 0.232741 | 0.028278 | 0.001397 | 0.03205 | 0.02996 | 0.00138 | 0.938 | 0.902 | 0.890 |
| gate | 5 | 0.02492 | 0.00085 | 0.10043 | 0.05146 | 0.00602 | 0.244403 | 0.029404 | 0.001390 | 0.03023 | 0.02945 | 0.00139 | 0.927 | 0.902 | 0.890 |
| stacked m_dim=8 | 5 | 0.02430 | 0.00057 | 0.09605 | 0.05032 | 0.00573 | 0.247234 | 0.028853 | 0.001397 | 0.03173 | 0.02874 | 0.00133 | 0.922 | 0.897 | 0.888 |

### F4c. p=3 paired contrasts (unit = Stage-1 seed; negative favours A)
| comparison (A − B) | n units | mean Δ | SE (units) | A better | Δ / B | n cells | SE (cells) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stacked m_dim=8 − gate (sliced W1) | 5 | -0.00062 | 0.00086 | 3/5 | -2.5% | 600 | 0.00057 |
| stacked m_dim=8 − gate (W1 u) | 5 | -0.00438 | 0.00355 | 5/5 | -4.4% | 600 | 0.00274 |
| stacked m_dim=8 − gate (W1 tau) | 5 | -0.00114 | 0.00235 | 2/5 | -2.2% | 600 | 0.00140 |
| stacked m_dim=8 − gate (W1 log_sigma) | 5 | -0.00029 | 0.00046 | 4/5 | -4.8% | 600 | 0.00020 |
| gate − linear (ILSA) (sliced W1) | 5 | -0.00250 | 0.00074 | 4/5 | -9.1% | 600 | 0.00090 |
| gate − linear (ILSA) (W1 u) | 5 | -0.00814 | 0.00386 | 4/5 | -7.5% | 600 | 0.00381 |
| gate − linear (ILSA) (W1 tau) | 5 | -0.00790 | 0.00238 | 5/5 | -13.3% | 600 | 0.00233 |
| gate − linear (ILSA) (W1 log_sigma) | 5 | -0.00072 | 0.00017 | 5/5 | -10.7% | 600 | 0.00024 |
| stacked m_dim=8 − linear (ILSA) (sliced W1) | 5 | -0.00312 | 0.00051 | 5/5 | -11.4% | 600 | 0.00093 |
| stacked m_dim=8 − linear (ILSA) (W1 u) | 5 | -0.01253 | 0.00541 | 4/5 | -11.5% | 600 | 0.00394 |
| stacked m_dim=8 − linear (ILSA) (W1 tau) | 5 | -0.00904 | 0.00198 | 5/5 | -15.2% | 600 | 0.00244 |
| stacked m_dim=8 − linear (ILSA) (W1 log_sigma) | 5 | -0.00101 | 0.00032 | 5/5 | -15.0% | 600 | 0.00024 |
| linear (ILSA) − pilot only (sliced W1) | 5 | -0.06470 | 0.00014 | 5/5 | -70.2% | 600 | 0.00248 |
| linear (ILSA) − pilot only (W1 u) | 5 | -0.15622 | 0.00274 | 5/5 | -59.0% | 600 | 0.00979 |
| linear (ILSA) − pilot only (W1 tau) | 5 | -0.16987 | 0.00115 | 5/5 | -74.1% | 600 | 0.00637 |
| linear (ILSA) − pilot only (W1 log_sigma) | 5 | -0.03152 | 0.00010 | 5/5 | -82.4% | 600 | 0.00121 |
| gate − pilot only (sliced W1) | 5 | -0.06720 | 0.00085 | 5/5 | -72.9% | 600 | 0.00242 |
| gate − pilot only (W1 u) | 5 | -0.16437 | 0.00358 | 5/5 | -62.1% | 600 | 0.00967 |
| gate − pilot only (W1 tau) | 5 | -0.17777 | 0.00230 | 5/5 | -77.6% | 600 | 0.00624 |
| gate − pilot only (W1 log_sigma) | 5 | -0.03224 | 0.00025 | 5/5 | -84.3% | 600 | 0.00121 |
| stacked m_dim=8 − pilot only (sliced W1) | 5 | -0.06782 | 0.00057 | 5/5 | -73.6% | 600 | 0.00239 |
| stacked m_dim=8 − pilot only (W1 u) | 5 | -0.16875 | 0.00349 | 5/5 | -63.7% | 600 | 0.00982 |
| stacked m_dim=8 − pilot only (W1 tau) | 5 | -0.17892 | 0.00151 | 5/5 | -78.1% | 600 | 0.00605 |
| stacked m_dim=8 − pilot only (W1 log_sigma) | 5 | -0.03253 | 0.00022 | 5/5 | -85.0% | 600 | 0.00120 |

Largest exact-grid window-edge marginal mass: 6.97e-05. Test-dataset pilot boundary rate: 0.425.

### F5a. Model 2 Stage-1 standardized MSE by learning rate
| method | lr (readout and gate) | n seeds | std MSE | SE | per seed |
|---|---:|---:|---:|---:|---|
| linear (ILSA) | 1e-3 | 5 | 0.09571 | 0.00397 | 0.10771, 0.08795, 0.10155, 0.09426, 0.08709 |
| linear (ILSA) | 1e-4 | 5 | 0.15750 | 0.00390 | 0.15383, 0.15573, 0.16989, 0.14664, 0.16142 |

### F5b. Model 2 paired differences (negative favours A)
| comparison (A − B) | n seeds | mean Δ | SE | A better | Δ / B |
|---|---:|---:|---:|---:|---:|
| linear: lr 1e-3 − lr 1e-4 | 5 | -0.06179 | 0.00534 | 5/5 | -39.2% |
<!-- TABLES:END -->
