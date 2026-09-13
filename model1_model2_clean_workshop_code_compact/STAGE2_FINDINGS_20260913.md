# Stage-2 findings — 2026-09-13

Model 1, p=1. First Stage-2 (posterior) results for this line of work: readout-matched Stage 1 at two tube bandwidths, then pilot+score NPE with linear (ILSA), the nonlinear gate (`radial`) and stacked `m_dim=8`, all evaluated against the exact posterior. **Every figure is over 5 Stage-1 training seeds (20260709–20260713) unless marked otherwise.** The tables at the end are regenerated from `runs/` by `python scripts/summarize_stage2_findings.py` (set `PYTHONIOENCODING=utf-8` on a Windows console when using `--stdout`).

## Headline answers

**1. Bandwidth: the draft rule `σ_q = 1.052` wins at Stage 2 for every method.** It is 12–22× worse in Stage-1 exact-score MSE (0.50 vs 0.023–0.041, T2), yet posterior W1 to the exact posterior is *lower* at σ_q=1.052:

| method | W1, σ_q=0.20 | W1, σ_q=1.052 | paired Δ (1.052 − 0.20) | SE (seeds) | 1.052 better |
|---|---:|---:|---:|---:|---:|
| linear (ILSA) | 0.00941 | 0.00858 | −0.00083 (−8.8%) | 0.00051 | 4/5 |
| gate | 0.00880 | **0.00770** | −0.00110 (−12.5%) | 0.00040 | 5/5 |
| stacked m_dim=8 | 0.00955 | 0.00805 | −0.00150 (−15.7%) | 0.00039 | 5/5 |

Coverage also moves toward the exact posterior's own 0.900: 0.873 / 0.883 / 0.877 at σ_q=0.20 versus 0.883 / 0.897 / 0.890 at σ_q=1.052, with the pilot-only floor at 0.833. The best configuration overall is **gate at σ_q=1.052** (W1 0.00770). This confirms the FINDINGS_20260912 §7 caveat: Stage-1 exact-score MSE at the true parameter cannot choose the bandwidth. σ_q=1.052 was fixed a priori by the draft's rule, not tuned. The measured rule value is close to it: the real pilot has Σ_e = 0.477 in u = logit π, so √(2Σ_e) = 0.977 (§Pilot error).

**2. Stage 1 → Stage 2: the ordering does not simply survive, and the gaps shrink sharply at the narrow tube.**

- *σ_q=0.20.* Stage 1 orders gate (0.02266) < stacked (0.02820) < linear (0.04064); **stacked ≤ gate does not hold** (stacked − gate +0.00555, SE 0.00091, gate better on 5/5). At Stage 2 the order is gate (0.00880) < linear (0.00941) < stacked (0.00955). The Stage-1 gate-over-linear gap of −44.3% becomes −6.5% in W1 (SE 0.00059, 3/5), and stacked's −30.6% becomes +1.5% (SE 0.00052, 2/5). Neither Stage-2 difference is distinguishable from zero.
- *σ_q=1.052.* Stage 1 differences are under 1% (all three on the smoothing floor), but Stage 2 separates them: gate (0.00770) < stacked (0.00805) < linear (0.00858). Here **the posterior gap is larger than the Stage-1 gap**.
- *Stacked vs gate at Stage 2:* stacked is worse at both bandwidths: +8.6% (SE 0.00030, stacked better on 1/5) at σ_q=0.20 and +4.6% (SE 0.00019, 1/5) at σ_q=1.052.

**3. Fair nonlinear-versus-linear gap (readouts matched at lr 1e-3).** At the draft bandwidth the gate reduces posterior W1 by **10.3%** relative to linear (paired −0.00089, SE 0.00017, 5/5) and stacked by **6.2%** (−0.00053, SE 0.00017, 5/5). At σ_q=0.20 the gate's −6.5% (SE 0.00059) and stacked's +1.5% (SE 0.00052) are within noise. Per true π (T4), the nonlinear gain is concentrated at small π. At σ_q=1.052, π=0.07: gate 0.00644 and stacked 0.00607 vs linear 0.00858. At π=0.65–0.68 linear is level with or better than both.

## Contradictions with earlier documents (stated, not reconciled)

- **Paper abstract (`main_style_revised.pdf`).** It reports a 24.7% W1 reduction for Model 1 from NLSA. With readouts matched and posterior W1 averaged over the same 6 π × 10 test datasets, the gate's reduction is 10.3% at σ_q=1.052 and 6.5% (not significant) at σ_q=0.20. The packaged protocol trains readouts at `lr 1e-4`, which FINDINGS_20260912 §1 showed under-trains linear.
- **FINDINGS_20260912 §2, stacked m_dim=8 vs gate.** That table found a Stage-1 tie (−0.00013, SE 0.00123, both readouts at 1e-4). With both readouts at 1e-3 the gate is clearly ahead at σ_q=0.20 (stacked − gate +0.00555, SE 0.00091, 0/5), and the gate stays ahead at Stage 2. Matching the rates at 1e-3 improved the gate (0.03010 → 0.02266) far more than stacked (0.02997 → 0.02820).
- **FINDINGS_20260912 §2, gate − linear.** The earlier UNMATCHED Stage-1 gap (−0.01054) *understated* the fair gap: matched, it is −0.01799 (SE 0.00207, 5/5). Linear itself reproduces exactly (five-seed mean 0.04064 in both sessions).
- **My own n=1 preview from earlier in this session** said the gate was bandwidth-insensitive (−1.5% on seed 20260709). Over 5 seeds the gate also gains from σ_q=1.052 (−12.5%, 5/5).

## Caveats

- **What the SEs measure.** The NPE training seed (54000), the NPE simulation bank (seed 20260723) and the test bank (seeds 100–109) are identical in all 10 Stage-2 runs, so seed-level SEs measure Stage-1 training variability only, not NPE or test-bank variability. The cell-level columns in T6/T7 (300 cells) add test-bank variability.
- **NPE run-to-run noise.** The pilot-only NPE does not read Stage 1, yet its per-cell W1 differs by up to 6.9e-4 across runs. Its seed-level SE is 0.00001, so it is negligible pooled. The difference is CPU floating-point nondeterminism: seed 20260709's Stage 2 ran with 6 threads and the others with 4. Method rows carry the same noise.
- **Wide-tube Stage 1 may not be converged.** At σ_q=1.052 the validation-best step is 18,000–19,700 of 20,000 for linear in all 5 seeds and for stacked in 4/5 (T1). If the wide tube is under-trained, its Stage-2 advantage may be understated. This is not tuned here; see next steps.
- **Coverage** is over 60 cells per seed, and the exact posterior's own 90% coverage varies by π (0.70–1.00, T5), so compare with the exact row, not the nominal 0.90.
- Only two bandwidths were run. The Stage-2 comparison says the wider one is better, not that 1.052 is optimal.

## Pilot error (Σ_e)

`scripts/measure_model1_pilot_error.py` uses 50,000 simulations with π ~ U(0.05, 0.70) and the Stage-2 marginal pilot on a 201-point grid; no checkpoint or posterior is read. Output is in `runs/pilot_error/pilot_error.json`.

| quantity | value |
|---|---:|
| Σ_e = E[(u_pilot − u)²] | 0.477 |
| same, interior roots only | 0.420 |
| pilot bias in u | −0.067 |
| pilot at the prior boundary | 17.5% |
| √(2Σ_e), measured pilot | **0.977** |
| E[I(u)⁻¹] (Monte Carlo Fisher information, uniform π) | 0.529 |
| √(2·E[I⁻¹]) | 1.029 |

The pilot is not efficient and is truncated to the prior box, and its error is uneven in π (MSE 0.63–0.67 for π < 0.31, 0.23 for π > 0.57). The draft's 1.052 is within 8% of the measured rule value.

## Protocol

- **Stage 1.** Each seed makes one invocation per bandwidth with `--methods linear,radial,stacked --m-dim 8 --lr 1e-3 --gate-lr 1e-3 --joint-rho-lr 1e-3`, cosine tail from step 10,000, 20,000 steps, validation-best raw checkpoint; then `model1.evaluate_stage1` (5,000 test sets per π). Before Stage 2, `scripts/check_model1_stage1_run.py` confirmed for all 10 directories that the three rates are 1e-3, all arms share one minibatch SHA-256, and the nesting error is ≤ 4.8e-7 (T1).
- **Stage 2.** `scripts/run_model1_stage2_npe_stacked.sh`, which is the packaged `run_model1_stage2_npe.sh` with `--methods pilot,linear,radial,stacked`: 50,000 NPE simulations, MDN (64 hidden, 8 components), test π ∈ {0.07, 0.10, 0.30, 0.50, 0.65, 0.68} × test seeds 100–109, posterior n=5,000, exact grid of 5,000 points. One pilot cache (`runs/s2_sq0.20_20260709/pilot_cache.npz`) is shared by all runs. The pilot reads only Y, τ and the grid settings, never a Stage-1 checkpoint. The exact posterior is used only for evaluation.
- **Code.** `model1/stage2_npe.py` gains `stacked` (seed index 3; existing indices unchanged). `paired_w1_comparisons` now reports radial-vs-others followed by stacked-vs-others, and is unchanged when stacked is absent. Tests: `tests/test_model1_stage2_stacked.py` and the updated `tests/test_clean_protocol.py`; 89 tests pass.
- **Driver.** `scripts/run_model1_stage2_handoff_plan.sh` ran seeds 20260710–13; seed 20260709 used the same commands by hand.
- **Environment.** Windows 11, CPU only (i9-9900; no CUDA/MPS device). Python 3.12.13 (uv), torch 2.5.1+cpu, numpy 2.3.5, scipy 1.18.1, sbi 0.24.0. Two differences from the source environment: arviz is pinned to 0.23.4 because arviz 1.x breaks `sbi` 0.24.0 imports, and pytest needs `--basetemp` because of a Windows temp-directory permission error. Wall time: Stage 1 ~40 min per invocation (two in parallel, 3 threads each); Stage 2 5–12 min per run.

## Next steps (not run)

1. **Longer wide-tube Stage 1.** Rerun σ_q=1.052 with a 60k-step budget and validation-only checkpoint selection, to test whether the unconverged Stage 1 understates the bandwidth effect.
2. **Bandwidth selection without the exact posterior.** Add σ_q ∈ {0.5, 0.977, 2.0} and choose by NPE validation loss or a simulation-based calibration check, then confirm against the exact posterior only afterwards.
3. **NPE-seed and test-bank replication**, so the SEs cover NPE variability as well.
4. **Out of scope unless approved:** p=3 Stage 2 (needs a 3-d pilot, an ~80³ exact-posterior grid and a multivariate W1) and the Model 2 nonlinear learning-rate audit (`scripts/run_model2_lr_audit.sh`).

## Tables

Regenerated by `scripts/summarize_stage2_findings.py`; do not edit by hand.

<!-- TABLES:BEGIN -->
### T1. Stage-1 provenance
| σ_q | seed | lr / gate_lr / joint_rho_lr | one minibatch SHA | max nesting err | best step |
|---|---:|---|---|---:|---|
| 0.20 | 20260709 | 0.001 / 0.001 / 0.001 | yes | 2.4e-07 | linear:11700, radial:12100, stacked:12100 |
| 0.20 | 20260710 | 0.001 / 0.001 / 0.001 | yes | 2.4e-07 | linear:17900, radial:6600, stacked:11800 |
| 0.20 | 20260711 | 0.001 / 0.001 / 0.001 | yes | 2.4e-07 | linear:14400, radial:10400, stacked:14400 |
| 0.20 | 20260712 | 0.001 / 0.001 / 0.001 | yes | 3.6e-07 | linear:12700, radial:12700, stacked:12700 |
| 0.20 | 20260713 | 0.001 / 0.001 / 0.001 | yes | 4.8e-07 | linear:16100, radial:10500, stacked:11900 |
| 1.052 | 20260709 | 0.001 / 0.001 / 0.001 | yes | 3.6e-07 | linear:18100, radial:16400, stacked:16400 |
| 1.052 | 20260710 | 0.001 / 0.001 / 0.001 | yes | 4.8e-07 | linear:19700, radial:19500, stacked:19700 |
| 1.052 | 20260711 | 0.001 / 0.001 / 0.001 | yes | 4.8e-07 | linear:19600, radial:18200, stacked:18000 |
| 1.052 | 20260712 | 0.001 / 0.001 / 0.001 | yes | 2.4e-07 | linear:19000, radial:19000, stacked:19000 |
| 1.052 | 20260713 | 0.001 / 0.001 / 0.001 | yes | 3.6e-07 | linear:19300, radial:11500, stacked:19300 |

### T2. Stage-1 frozen exact-score standardized MSE (pooled over π ∈ {0.10, 0.30, 0.50, 0.65})
| σ_q | method | n seeds | std MSE (mean) | SE |
|---|---|---:|---:|---:|
| 0.20 | linear (ILSA) | 5 | 0.04064 | 0.00214 |
| 0.20 | gate | 5 | 0.02266 | 0.00220 |
| 0.20 | stacked m_dim=8 | 5 | 0.02820 | 0.00216 |
| 1.052 | linear (ILSA) | 5 | 0.49999 | 0.00212 |
| 1.052 | gate | 5 | 0.49546 | 0.00130 |
| 1.052 | stacked m_dim=8 | 5 | 0.49542 | 0.00117 |

#### T2b. Stage-1 paired differences in standardized MSE (negative favours A; SE over seeds)
| comparison (A − B) | n seeds | mean Δ std MSE | SE | A better | Δ / B |
|---|---:|---:|---:|---:|---:|
| σ_q=0.20: stacked m_dim=8 − gate | 5 | +0.00555 | 0.00091 | 0/5 | +24.5% |
| σ_q=0.20: gate − linear (ILSA) | 5 | -0.01799 | 0.00207 | 5/5 | -44.3% |
| σ_q=0.20: stacked m_dim=8 − linear (ILSA) | 5 | -0.01244 | 0.00125 | 5/5 | -30.6% |
| σ_q=1.052: stacked m_dim=8 − gate | 5 | -0.00004 | 0.00118 | 2/5 | -0.0% |
| σ_q=1.052: gate − linear (ILSA) | 5 | -0.00454 | 0.00198 | 4/5 | -0.9% |
| σ_q=1.052: stacked m_dim=8 − linear (ILSA) | 5 | -0.00457 | 0.00265 | 4/5 | -0.9% |

### T3. Stage-2 posterior metrics, pooled over 6 true π × 10 test datasets
Means over Stage-1 seeds of each seed's 60-cell average; RMSE is per seed, then averaged.

| σ_q | method | n seeds | post-mean MSE vs truth | RMSE vs exact mean | W1 to exact | W1 SE (seeds) | post SD | |SD − exact SD| | cov90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | exact posterior | 5 | 0.010124 | 0 | 0 | — | 0.0970 | 0 | 0.900 |
| 0.20 | pilot only | 5 | 0.012720 | 0.02858 | 0.02279 | 0.00001 | 0.1050 | 0.00956 | 0.833 |
| 0.20 | linear (ILSA) | 5 | 0.010619 | 0.01095 | 0.00941 | 0.00041 | 0.0961 | 0.00471 | 0.873 |
| 0.20 | gate | 5 | 0.010435 | 0.00968 | 0.00880 | 0.00034 | 0.0972 | 0.00567 | 0.883 |
| 0.20 | stacked m_dim=8 | 5 | 0.010684 | 0.01097 | 0.00955 | 0.00037 | 0.0978 | 0.00574 | 0.877 |
| 1.052 | exact posterior | 5 | 0.010124 | 0 | 0 | — | 0.0970 | 0 | 0.900 |
| 1.052 | pilot only | 5 | 0.012720 | 0.02858 | 0.02279 | 0.00001 | 0.1050 | 0.00956 | 0.833 |
| 1.052 | linear (ILSA) | 5 | 0.010786 | 0.00917 | 0.00858 | 0.00020 | 0.0969 | 0.00433 | 0.883 |
| 1.052 | gate | 5 | 0.010391 | 0.00751 | 0.00770 | 0.00023 | 0.0967 | 0.00495 | 0.897 |
| 1.052 | stacked m_dim=8 | 5 | 0.010425 | 0.00805 | 0.00805 | 0.00014 | 0.0961 | 0.00460 | 0.890 |

### T4. Stage-2 W1 to the exact posterior, per true π
| σ_q | method | n seeds | π=0.07 | π=0.1 | π=0.3 | π=0.5 | π=0.65 | π=0.68 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | pilot only | 5 | 0.02430 | 0.02633 | 0.02074 | 0.01575 | 0.02498 | 0.02465 |
| 0.20 | linear (ILSA) | 5 | 0.01136 | 0.01070 | 0.01008 | 0.00900 | 0.00807 | 0.00725 |
| 0.20 | gate | 5 | 0.00931 | 0.00951 | 0.00958 | 0.00950 | 0.00748 | 0.00741 |
| 0.20 | stacked m_dim=8 | 5 | 0.01047 | 0.00976 | 0.00960 | 0.00948 | 0.00912 | 0.00890 |
| 1.052 | pilot only | 5 | 0.02430 | 0.02633 | 0.02074 | 0.01575 | 0.02498 | 0.02465 |
| 1.052 | linear (ILSA) | 5 | 0.00858 | 0.00832 | 0.01090 | 0.00975 | 0.00747 | 0.00649 |
| 1.052 | gate | 5 | 0.00644 | 0.00771 | 0.00881 | 0.00883 | 0.00745 | 0.00696 |
| 1.052 | stacked m_dim=8 | 5 | 0.00607 | 0.00710 | 0.01085 | 0.00934 | 0.00813 | 0.00683 |

#### T5 (π = 0.07). All posterior metrics
| σ_q | method | n seeds | post-mean MSE vs truth | RMSE vs exact mean | W1 to exact | W1 SE (seeds) | post SD | |SD − exact SD| | cov90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | exact posterior | 5 | 0.009424 | 0 | 0 | — | 0.0833 | 0 | 0.900 |
| 0.20 | pilot only | 5 | 0.013318 | 0.03058 | 0.02430 | 0.00001 | 0.0926 | 0.01063 | 0.700 |
| 0.20 | linear (ILSA) | 5 | 0.010335 | 0.01476 | 0.01136 | 0.00041 | 0.0841 | 0.00672 | 0.800 |
| 0.20 | gate | 5 | 0.009464 | 0.01134 | 0.00931 | 0.00071 | 0.0814 | 0.00689 | 0.820 |
| 0.20 | stacked m_dim=8 | 5 | 0.010025 | 0.01364 | 0.01047 | 0.00055 | 0.0831 | 0.00708 | 0.800 |
| 1.052 | exact posterior | 5 | 0.009424 | 0 | 0 | — | 0.0833 | 0 | 0.900 |
| 1.052 | pilot only | 5 | 0.013318 | 0.03058 | 0.02430 | 0.00001 | 0.0926 | 0.01063 | 0.700 |
| 1.052 | linear (ILSA) | 5 | 0.010471 | 0.01007 | 0.00858 | 0.00016 | 0.0828 | 0.00484 | 0.820 |
| 1.052 | gate | 5 | 0.009110 | 0.00575 | 0.00644 | 0.00035 | 0.0809 | 0.00528 | 0.900 |
| 1.052 | stacked m_dim=8 | 5 | 0.009413 | 0.00510 | 0.00607 | 0.00012 | 0.0815 | 0.00506 | 0.900 |

#### T5 (π = 0.1). All posterior metrics
| σ_q | method | n seeds | post-mean MSE vs truth | RMSE vs exact mean | W1 to exact | W1 SE (seeds) | post SD | |SD − exact SD| | cov90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | exact posterior | 5 | 0.007174 | 0 | 0 | — | 0.0892 | 0 | 1.000 |
| 0.20 | pilot only | 5 | 0.009565 | 0.03291 | 0.02633 | 0.00001 | 0.0972 | 0.01018 | 1.000 |
| 0.20 | linear (ILSA) | 5 | 0.007971 | 0.01369 | 0.01070 | 0.00052 | 0.0890 | 0.00583 | 1.000 |
| 0.20 | gate | 5 | 0.007449 | 0.01140 | 0.00951 | 0.00055 | 0.0868 | 0.00764 | 1.000 |
| 0.20 | stacked m_dim=8 | 5 | 0.007755 | 0.01247 | 0.00976 | 0.00051 | 0.0879 | 0.00702 | 1.000 |
| 1.052 | exact posterior | 5 | 0.007174 | 0 | 0 | — | 0.0892 | 0 | 1.000 |
| 1.052 | pilot only | 5 | 0.009565 | 0.03291 | 0.02633 | 0.00001 | 0.0972 | 0.01018 | 1.000 |
| 1.052 | linear (ILSA) | 5 | 0.008030 | 0.00860 | 0.00832 | 0.00038 | 0.0885 | 0.00509 | 1.000 |
| 1.052 | gate | 5 | 0.007222 | 0.00684 | 0.00771 | 0.00048 | 0.0868 | 0.00657 | 1.000 |
| 1.052 | stacked m_dim=8 | 5 | 0.007262 | 0.00579 | 0.00710 | 0.00015 | 0.0871 | 0.00605 | 1.000 |

#### T5 (π = 0.3). All posterior metrics
| σ_q | method | n seeds | post-mean MSE vs truth | RMSE vs exact mean | W1 to exact | W1 SE (seeds) | post SD | |SD − exact SD| | cov90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | exact posterior | 5 | 0.002502 | 0 | 0 | — | 0.1208 | 0 | 1.000 |
| 0.20 | pilot only | 5 | 0.003624 | 0.02622 | 0.02074 | 0.00000 | 0.1236 | 0.00682 | 1.000 |
| 0.20 | linear (ILSA) | 5 | 0.002409 | 0.00970 | 0.01008 | 0.00062 | 0.1187 | 0.00586 | 1.000 |
| 0.20 | gate | 5 | 0.002313 | 0.00865 | 0.00958 | 0.00032 | 0.1198 | 0.00643 | 1.000 |
| 0.20 | stacked m_dim=8 | 5 | 0.002333 | 0.00917 | 0.00960 | 0.00032 | 0.1199 | 0.00636 | 1.000 |
| 1.052 | exact posterior | 5 | 0.002502 | 0 | 0 | — | 0.1208 | 0 | 1.000 |
| 1.052 | pilot only | 5 | 0.003624 | 0.02622 | 0.02074 | 0.00000 | 0.1236 | 0.00682 | 1.000 |
| 1.052 | linear (ILSA) | 5 | 0.002161 | 0.01124 | 0.01090 | 0.00029 | 0.1189 | 0.00548 | 1.000 |
| 1.052 | gate | 5 | 0.002232 | 0.00810 | 0.00881 | 0.00053 | 0.1188 | 0.00572 | 1.000 |
| 1.052 | stacked m_dim=8 | 5 | 0.002192 | 0.01064 | 0.01085 | 0.00069 | 0.1179 | 0.00599 | 1.000 |

#### T5 (π = 0.5). All posterior metrics
| σ_q | method | n seeds | post-mean MSE vs truth | RMSE vs exact mean | W1 to exact | W1 SE (seeds) | post SD | |SD − exact SD| | cov90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | exact posterior | 5 | 0.004644 | 0 | 0 | — | 0.1111 | 0 | 1.000 |
| 0.20 | pilot only | 5 | 0.004276 | 0.01993 | 0.01575 | 0.00001 | 0.1146 | 0.00516 | 1.000 |
| 0.20 | linear (ILSA) | 5 | 0.004942 | 0.00879 | 0.00900 | 0.00031 | 0.1082 | 0.00426 | 1.000 |
| 0.20 | gate | 5 | 0.004895 | 0.00962 | 0.00950 | 0.00068 | 0.1113 | 0.00470 | 1.000 |
| 0.20 | stacked m_dim=8 | 5 | 0.004732 | 0.00971 | 0.00948 | 0.00020 | 0.1116 | 0.00496 | 1.000 |
| 1.052 | exact posterior | 5 | 0.004644 | 0 | 0 | — | 0.1111 | 0 | 1.000 |
| 1.052 | pilot only | 5 | 0.004276 | 0.01993 | 0.01575 | 0.00001 | 0.1146 | 0.00516 | 1.000 |
| 1.052 | linear (ILSA) | 5 | 0.005010 | 0.00975 | 0.00975 | 0.00035 | 0.1099 | 0.00435 | 1.000 |
| 1.052 | gate | 5 | 0.004709 | 0.00890 | 0.00883 | 0.00039 | 0.1110 | 0.00427 | 1.000 |
| 1.052 | stacked m_dim=8 | 5 | 0.004748 | 0.00913 | 0.00934 | 0.00032 | 0.1097 | 0.00429 | 1.000 |

#### T5 (π = 0.65). All posterior metrics
| σ_q | method | n seeds | post-mean MSE vs truth | RMSE vs exact mean | W1 to exact | W1 SE (seeds) | post SD | |SD − exact SD| | cov90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | exact posterior | 5 | 0.017014 | 0 | 0 | — | 0.0895 | 0 | 0.800 |
| 0.20 | pilot only | 5 | 0.020356 | 0.02992 | 0.02498 | 0.00002 | 0.1014 | 0.01194 | 0.700 |
| 0.20 | linear (ILSA) | 5 | 0.017469 | 0.00860 | 0.00807 | 0.00078 | 0.0893 | 0.00327 | 0.800 |
| 0.20 | gate | 5 | 0.017590 | 0.00800 | 0.00748 | 0.00053 | 0.0928 | 0.00443 | 0.800 |
| 0.20 | stacked m_dim=8 | 5 | 0.017887 | 0.01008 | 0.00912 | 0.00064 | 0.0930 | 0.00473 | 0.800 |
| 1.052 | exact posterior | 5 | 0.017014 | 0 | 0 | — | 0.0895 | 0 | 0.800 |
| 1.052 | pilot only | 5 | 0.020356 | 0.02992 | 0.02498 | 0.00002 | 0.1014 | 0.01194 | 0.700 |
| 1.052 | linear (ILSA) | 5 | 0.018003 | 0.00785 | 0.00747 | 0.00033 | 0.0914 | 0.00356 | 0.800 |
| 1.052 | gate | 5 | 0.017861 | 0.00751 | 0.00745 | 0.00051 | 0.0923 | 0.00422 | 0.800 |
| 1.052 | stacked m_dim=8 | 5 | 0.017907 | 0.00857 | 0.00813 | 0.00054 | 0.0910 | 0.00363 | 0.800 |

#### T5 (π = 0.68). All posterior metrics
| σ_q | method | n seeds | post-mean MSE vs truth | RMSE vs exact mean | W1 to exact | W1 SE (seeds) | post SD | |SD − exact SD| | cov90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | exact posterior | 5 | 0.019987 | 0 | 0 | — | 0.0879 | 0 | 0.700 |
| 0.20 | pilot only | 5 | 0.025182 | 0.03004 | 0.02465 | 0.00002 | 0.1005 | 0.01264 | 0.600 |
| 0.20 | linear (ILSA) | 5 | 0.020588 | 0.00768 | 0.00725 | 0.00097 | 0.0875 | 0.00235 | 0.640 |
| 0.20 | gate | 5 | 0.020897 | 0.00769 | 0.00741 | 0.00075 | 0.0912 | 0.00391 | 0.680 |
| 0.20 | stacked m_dim=8 | 5 | 0.021375 | 0.00964 | 0.00890 | 0.00071 | 0.0914 | 0.00431 | 0.660 |
| 1.052 | exact posterior | 5 | 0.019987 | 0 | 0 | — | 0.0879 | 0 | 0.700 |
| 1.052 | pilot only | 5 | 0.025182 | 0.03004 | 0.02465 | 0.00002 | 0.1005 | 0.01264 | 0.600 |
| 1.052 | linear (ILSA) | 5 | 0.021040 | 0.00659 | 0.00649 | 0.00049 | 0.0896 | 0.00264 | 0.680 |
| 1.052 | gate | 5 | 0.021211 | 0.00689 | 0.00696 | 0.00065 | 0.0908 | 0.00365 | 0.680 |
| 1.052 | stacked m_dim=8 | 5 | 0.021030 | 0.00705 | 0.00683 | 0.00061 | 0.0892 | 0.00260 | 0.640 |

### T6. Paired W1 differences between methods (negative favours A)
| comparison (A − B) | n seeds | mean ΔW1 | SE (seeds) | A better (seeds) | Δ / W1(B) | n cells | SE (cells) | A better (cells) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| σ_q=0.20: stacked m_dim=8 − gate | 5 | +0.00076 | 0.00030 | 1/5 | +8.6% | 300 | 0.00022 | 133/300 |
| σ_q=0.20: gate − linear (ILSA) | 5 | -0.00061 | 0.00059 | 3/5 | -6.5% | 300 | 0.00031 | 161/300 |
| σ_q=0.20: stacked m_dim=8 − linear (ILSA) | 5 | +0.00014 | 0.00052 | 2/5 | +1.5% | 300 | 0.00030 | 150/300 |
| σ_q=0.20: linear (ILSA) − pilot only | 5 | -0.01338 | 0.00040 | 5/5 | -58.7% | 300 | 0.00112 | 226/300 |
| σ_q=0.20: gate − pilot only | 5 | -0.01399 | 0.00034 | 5/5 | -61.4% | 300 | 0.00110 | 222/300 |
| σ_q=0.20: stacked m_dim=8 − pilot only | 5 | -0.01324 | 0.00037 | 5/5 | -58.1% | 300 | 0.00111 | 210/300 |
| σ_q=1.052: stacked m_dim=8 − gate | 5 | +0.00035 | 0.00019 | 1/5 | +4.6% | 300 | 0.00016 | 143/300 |
| σ_q=1.052: gate − linear (ILSA) | 5 | -0.00089 | 0.00017 | 5/5 | -10.3% | 300 | 0.00025 | 173/300 |
| σ_q=1.052: stacked m_dim=8 − linear (ILSA) | 5 | -0.00053 | 0.00017 | 5/5 | -6.2% | 300 | 0.00025 | 152/300 |
| σ_q=1.052: linear (ILSA) − pilot only | 5 | -0.01421 | 0.00019 | 5/5 | -62.3% | 300 | 0.00107 | 224/300 |
| σ_q=1.052: gate − pilot only | 5 | -0.01509 | 0.00023 | 5/5 | -66.2% | 300 | 0.00102 | 236/300 |
| σ_q=1.052: stacked m_dim=8 − pilot only | 5 | -0.01474 | 0.00013 | 5/5 | -64.7% | 300 | 0.00105 | 220/300 |

### T7. Paired W1 differences between bandwidths, within method (negative favours σ_q=1.052)
| comparison (A − B) | n seeds | mean ΔW1 | SE (seeds) | A better (seeds) | Δ / W1(B) | n cells | SE (cells) | A better (cells) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear (ILSA): σ_q=1.052 − σ_q=0.20 | 5 | -0.00083 | 0.00051 | 4/5 | -8.8% | 300 | 0.00034 | 160/300 |
| gate: σ_q=1.052 − σ_q=0.20 | 5 | -0.00110 | 0.00040 | 5/5 | -12.5% | 300 | 0.00030 | 164/300 |
| stacked m_dim=8: σ_q=1.052 − σ_q=0.20 | 5 | -0.00150 | 0.00039 | 5/5 | -15.7% | 300 | 0.00036 | 165/300 |

Pilot-only W1 max abs difference across 10 Stage-2 runs: 6.94e-04.
<!-- TABLES:END -->
