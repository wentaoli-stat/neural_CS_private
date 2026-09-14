# Stage-2 input comparison — Model 1, p=1 (2026-09-13)

> **Read with `FOLLOWUP_FINDINGS_20260913.md` (F3).** Every figure here comes from one draw of the Stage-2
> random streams: NPE simulation bank 20260723, NPE seed 54000, test seeds 100–109. Replicating those
> streams showed they contribute more variance to method contrasts than Stage-1 training, and this draw
> understates the nonlinear-versus-linear gap: at σ_q = 1.052 the pooled gate − linear reduction is
> 25.6% (15/15 seed × replicate units), not the 10.9% below. The ± values here cover Stage-1 variability
> only. The tier gaps (unstructured scores 150–180% worse than the gate) are an order of magnitude larger
> than that replicate noise; margins between linear, gate and stacked are not reliable from this draw.

Which input should the Stage-2 NPE condition on? Eight arms, all at the draft tube bandwidth σ_q = 1.052
(the one that won in `STAGE2_FINDINGS_20260913.md`), all paired on one NPE simulation bank, one NPE
training seed, one data-only pilot and the same 60 test datasets. W1 is the 1-d Wasserstein distance, in
π units, between NPE posterior samples and the exact grid posterior. Tables are regenerated from `runs/ic/`
by `python scripts/summarize_input_comparison.py`.

## Headline

**1. Three tiers, not a gradient.**

| tier | input to NPE | W1 to exact |
|---|---|---:|
| worse than the pilot alone | NPE on all subscores (401 dims) | 0.03662 ± 0.00257 |
| | NPE on raw Y (400 dims) | 0.02966 ± 0.00086 |
| reference | pilot only | 0.02284 |
| barely better than the pilot | raw-input Fisher score | 0.02167 ± 0.00052 |
| | all-subscores Fisher score | 0.02142 ± 0.00034 |
| aggregated local scores | linear ILSA score | 0.00858 ± 0.00027 |
| | NLSA stacked m_dim=8 score | 0.00792 ± 0.00014 |
| | **NLSA gate score (paper)** | **0.00774 ± 0.00022** |

**2. Aggregation is what matters, not how much information the input carries.**

- *Direct high-dimensional contexts are worse than the pilot.* Raw Y and every subscore both contain
  strictly more information than the pilot, yet an MDN with 50,000 simulations extracts less from them.
- *Learning an unstructured Fisher score from them helps only a little*: 5% (raw input) and 6%
  (all subscores) below the pilot-only W1. The two inputs are indistinguishable (0.02167 vs 0.02142), so
  the input representation is not the bottleneck.
- *Aggregating the local scores cuts W1 by 60%*: linear ILSA 0.00858 against 0.0214 for the unstructured
  scores. The paper's architecture gets essentially all of its advantage over the unstructured
  baselines from the transform–sum–readout structure, before any nonlinearity.

**3. The nonlinear local map is a second-order refinement.** Gate − linear: +0.00084 (SE 0.00031) in
linear's disfavour, gate better on 4/5 seeds, 10.9%. Stacked − gate: +0.00018 (SE 0.00034), 3/5: no
detectable difference.

**4. Stage-1 accuracy barely separates tiers that Stage 2 separates 2.5-fold.** At this bandwidth the
unstructured scores have Stage-1 standardized MSE 0.56–0.57 against 0.49–0.51 for the aggregated scores,
only ~13% worse, yet their posterior W1 is 2.5× larger. This is the same lesson as the bandwidth result:
Stage-1 score MSE is a poor proxy for posterior quality.

**5. The unstructured scores overfit almost immediately.** Validation picked lr 1e-4 on every seed, and
the selected checkpoint sits at steps 1,000–2,100 of 20,000 (29,953 parameters, 40,000 simulations). The
aggregated scores select near the end of the run.

## Reproducibility across machines

The 2-dimensional-context arms reproduce the other machine's independent σ_q = 1.052 runs
(`STAGE2_FINDINGS_20260913.md`, Windows/i9) to within 0.00013 in W1: pilot 0.02284 vs 0.02279, linear
0.00858 vs 0.00858, gate 0.00774 vs 0.00770, stacked 0.00792 vs 0.00805.

The 400-dimensional raw-Y NPE does not. On byte-identical data and identical exact posteriors it gives
W1 0.03003 here against 0.02831 in the archived CUDA run, with single test datasets moving by up to 0.035.
High-dimensional NPE training is sensitive to platform and thread nondeterminism, so the two direct-NPE
arms are replicated over 5 NPE seeds (54000–54004) and their numbers should not be compared across
machines.

## Caveats

- **The ± values measure different things.** Score arms vary the Stage-1 training seed with the NPE seed
  fixed; direct-NPE arms vary the NPE seed with nothing to train in Stage 1. Neither covers test-bank
  variability (the 60 datasets are shared).
- **The direct NPEs use the shared recipe**: an MDN with independent z-scoring and no embedding network.
  A permutation-invariant embedding net might narrow the gap. That would be a structured summary learned
  inside the NPE, not a different conclusion about unstructured inputs.
- **Subscores are evaluated at the data-only pilot**, the only data-only choice of parameter value.
- **Two bandwidths.** Every Fisher-score arm was also run at σ_q = 0.20; see *Narrow versus wide bandwidth*.
- **Coverage** is over 60 datasets per arm (1/60 ≈ 0.017), so only the direct-NPE undercoverage (0.73–0.76
  against ~0.89 for the aggregated arms) is large enough to read.

## Protocol

- **Stage 1** (per seed 20260709–13, one bank shared by all score arms).
  NLSA: `model1.stage1 --methods linear,radial,stacked --m-dim 8 --lr 1e-3 --gate-lr 1e-3
  --joint-rho-lr 1e-3 --checkpoint-selection raw`; checked by `scripts/check_model1_stage1_run.py` (rates
  1e-3, one minibatch SHA-256, nesting ≤ 2.4e-7). Unstructured: `common.raw_fsm stage1
  --architecture {direct_flat_mlp, subscore_flat_mlp} --checkpoint-selection raw`, lr ∈ {1e-4, 1e-3} chosen
  per seed by validation FSM loss only.
- **Stage 2.** 50,000 NPE simulations (seed 20260723), MDN (64 hidden, 8 components), NPE seed 54000,
  posterior n = 5,000, exact grid 5,000, test π ∈ {0.07, 0.10, 0.30, 0.50, 0.65, 0.68} × test seeds
  100–109. One pilot cache (`scripts/make_model1_pilot_cache.py`) is shared by every arm; it matches the
  NPE bank draw, and its boundary rate (17.5%) matches the other machine's independent measurement.
- **Launchers:** `scripts/run_model1_input_comparison_seed.sh SEED`,
  `scripts/run_model1_direct_npe_baselines.sh` (with `SBI_SEEDS`).
- **Code:** `common/raw_fsm.py` gains `subscore_flat_mlp`, `--checkpoint-selection` and `--pilot-cache`;
  `common/raw_data_npe.py` gains `--context subscores_at_pilot`. A pre-existing bug — `raw_fsm.pilot_args`
  omitted the `device` field that `model1.stage2_npe.data_only_pilot` reads, breaking Model 1 raw-FSM
  Stage 2 whenever the pilot was computed rather than loaded — is fixed, with a regression test. 111 tests
  pass.

## Narrow versus wide bandwidth

Every Fisher-score arm was rerun at the packaged σ_q = 0.20 on the same five Stage-1 seeds, NPE bank,
NPE seed, pilot and test datasets, so each method's two bandwidths are paired by seed. Tables are
regenerated by `python scripts/compare_input_bandwidths.py`.

**1. The wide bandwidth helps the aggregated scores and does nothing for the unstructured ones.**
Wide − narrow in W1: NLSA gate −10.8% (SE 0.00036, 5/5), stacked −16.9% (SE 0.00042, 5/5), linear ILSA
−11.5% (SE 0.00072, 4/5); raw-input Fisher score −1.7% (SE 0.00064, 3/5) and all-subscores Fisher score
+0.2% (SE 0.00037, 2/5), both indistinguishable from zero. Pilot-only is bit-identical at both bandwidths,
as it must be, since it never reads Stage 1.

**2. For the unstructured scores, Stage-1 accuracy moves a lot while the posterior does not move at all.**
Narrowing the tube halves the all-subscores score's Stage-1 error (standardized MSE 0.56 → 0.26) and makes
it twice as accurate as the raw-input score (0.26 vs 0.52), yet their posterior W1 is unchanged and equal
(0.0214 and 0.0220). An unstructured score's value accuracy does not reach the posterior.

**3. The tiers hold at both bandwidths.** Unstructured scores are +146% to +154% worse than the gate at
σ_q = 0.20 and +177% to +180% at 1.052; linear ILSA is +11.7% and +10.9%. The aggregation gap and the
nonlinear gain do not depend on the bandwidth.

**4. Stacked versus gate does depend on it.** At σ_q = 0.20 stacked is worse than the gate (+9.8%, SE
0.00039, gate better on 4/5); at 1.052 they are indistinguishable (+2.3%, SE 0.00034, 3/5). Stacked
gains the most from the wide tube (−16.9%).

**5. Cross-machine reproduction.** The narrow NLSA arms match the other machine's σ_q = 0.20 runs to within
0.0003: gate 0.00868 vs 0.00880, stacked 0.00953 vs 0.00955, linear 0.00970 vs 0.00941. So do the
within-method bandwidth effects: gate −10.8% vs −12.5%, stacked −16.9% vs −15.7%, linear −11.5% vs −8.8%.

The unstructured scores chose lr 1e-4 by validation on every seed at both bandwidths.


## Tables

Stage-2 input comparison, Model 1 p=1, σ_q=1.052. 40 posterior files checked: all share one exact posterior on 60 test datasets.

### Posterior quality, pooled over 6 true π × 10 test datasets

| input to NPE | replicates | W1 to exact | SE | post-mean MSE vs truth | RMSE vs exact mean | |SD − exact SD| | cov90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pilot only | 5 Stage-1 seeds | 0.02284 | 0.00000 | 0.012731 | 0.02863 | 0.00959 | 0.833 |
| NPE on raw Y | 5 NPE seeds | 0.02966 | 0.00086 | 0.013117 | 0.03549 | 0.00943 | 0.730 |
| NPE on all subscores | 5 NPE seeds | 0.03662 | 0.00257 | 0.013790 | 0.04427 | 0.01030 | 0.760 |
| raw-input Fisher score | 5 Stage-1 seeds | 0.02167 | 0.00052 | 0.012544 | 0.02689 | 0.00953 | 0.837 |
| all-subscores Fisher score | 5 Stage-1 seeds | 0.02142 | 0.00034 | 0.012653 | 0.02647 | 0.00931 | 0.847 |
| linear ILSA score | 5 Stage-1 seeds | 0.00858 | 0.00027 | 0.010783 | 0.00918 | 0.00436 | 0.887 |
| NLSA gate score (paper) | 5 Stage-1 seeds | 0.00774 | 0.00022 | 0.010379 | 0.00756 | 0.00504 | 0.890 |
| NLSA stacked m_dim=8 score | 5 Stage-1 seeds | 0.00792 | 0.00014 | 0.010407 | 0.00787 | 0.00461 | 0.890 |

### W1 to exact, per true π

| input to NPE | π=0.07 | π=0.1 | π=0.3 | π=0.5 | π=0.65 | π=0.68 |
|---|---:|---:|---:|---:|---:|---:|
| pilot only | 0.02431 | 0.02633 | 0.02070 | 0.01593 | 0.02500 | 0.02473 |
| NPE on raw Y | 0.03699 | 0.03913 | 0.02416 | 0.02455 | 0.02711 | 0.02601 |
| NPE on all subscores | 0.02692 | 0.02748 | 0.03585 | 0.04305 | 0.04412 | 0.04233 |
| raw-input Fisher score | 0.02546 | 0.02786 | 0.01951 | 0.01558 | 0.02092 | 0.02071 |
| all-subscores Fisher score | 0.02664 | 0.02841 | 0.01971 | 0.01528 | 0.01943 | 0.01906 |
| linear ILSA score | 0.00869 | 0.00854 | 0.01072 | 0.00976 | 0.00740 | 0.00640 |
| NLSA gate score (paper) | 0.00667 | 0.00801 | 0.00882 | 0.00876 | 0.00739 | 0.00680 |
| NLSA stacked m_dim=8 score | 0.00591 | 0.00708 | 0.01071 | 0.00922 | 0.00791 | 0.00669 |

### Paired W1 against NLSA gate score (paper) (positive = worse than the paper's method)

| comparison | replicates | mean ΔW1 | SE | paper method better | Δ / W1(paper) |
|---|---:|---:|---:|---:|---:|
| pilot only − NLSA gate score (paper) | 5 Stage-1 seeds | +0.01509 | 0.00022 | 5/5 | +195.0% |
| NPE on raw Y − NLSA gate score (paper) | 5 NPE seeds | +0.02192 | 0.00086 | 5/5 | +283.1% |
| NPE on all subscores − NLSA gate score (paper) | 5 NPE seeds | +0.02888 | 0.00257 | 5/5 | +373.1% |
| raw-input Fisher score − NLSA gate score (paper) | 5 Stage-1 seeds | +0.01393 | 0.00049 | 5/5 | +179.9% |
| all-subscores Fisher score − NLSA gate score (paper) | 5 Stage-1 seeds | +0.01368 | 0.00046 | 5/5 | +176.7% |
| linear ILSA score − NLSA gate score (paper) | 5 Stage-1 seeds | +0.00084 | 0.00031 | 4/5 | +10.9% |
| NLSA stacked m_dim=8 score − NLSA gate score (paper) | 5 Stage-1 seeds | +0.00018 | 0.00034 | 3/5 | +2.3% |

### Stage-1 Fisher-score accuracy of each score arm (frozen exact-score standardized MSE)

| score | seed | lr (by validation) | best step | std MSE |
|---|---:|---:|---:|---:|
| raw-input Fisher score | 20260709 | 1e-4 | 1200 | 0.57071 |
| raw-input Fisher score | 20260710 | 1e-4 | 1300 | 0.57038 |
| raw-input Fisher score | 20260711 | 1e-4 | 1400 | 0.57397 |
| raw-input Fisher score | 20260712 | 1e-4 | 1600 | 0.56455 |
| raw-input Fisher score | 20260713 | 1e-4 | 1400 | 0.56745 |
| all-subscores Fisher score | 20260709 | 1e-4 | 1000 | 0.57362 |
| all-subscores Fisher score | 20260710 | 1e-4 | 2000 | 0.56075 |
| all-subscores Fisher score | 20260711 | 1e-4 | 2100 | 0.56068 |
| all-subscores Fisher score | 20260712 | 1e-4 | 1600 | 0.56037 |
| all-subscores Fisher score | 20260713 | 1e-4 | 1600 | 0.55858 |
| linear ILSA score | 20260709 | 1e-3 | — | 0.50788 |
| NLSA gate score (paper) | 20260709 | 1e-3 | — | 0.49797 |
| NLSA stacked m_dim=8 score | 20260709 | 1e-3 | — | 0.49435 |
| linear ILSA score | 20260710 | 1e-3 | — | 0.49729 |
| NLSA gate score (paper) | 20260710 | 1e-3 | — | 0.49460 |
| NLSA stacked m_dim=8 score | 20260710 | 1e-3 | — | 0.49280 |
| linear ILSA score | 20260711 | 1e-3 | — | 0.49676 |
| NLSA gate score (paper) | 20260711 | 1e-3 | — | 0.49865 |
| NLSA stacked m_dim=8 score | 20260711 | 1e-3 | — | 0.49969 |
| linear ILSA score | 20260712 | 1e-3 | — | 0.50101 |
| NLSA gate score (paper) | 20260712 | 1e-3 | — | 0.49460 |
| NLSA stacked m_dim=8 score | 20260712 | 1e-3 | — | 0.49582 |
| linear ILSA score | 20260713 | 1e-3 | — | 0.49703 |
| NLSA gate score (paper) | 20260713 | 1e-3 | — | 0.49466 |
| NLSA stacked m_dim=8 score | 20260713 | 1e-3 | — | 0.49444 |


## Tables: narrow versus wide bandwidth

Tube bandwidth σ_q = 0.20 (narrow) vs 1.052 (wide, draft rule), Model 1 p=1, Stage 2. Paired within method by Stage-1 seed.

### Posterior W1 to exact and Stage-1 score accuracy, by bandwidth

| score | seeds (narrow) | W1 σ_q=0.20 | SE | seeds (wide) | W1 σ_q=1.052 | SE | Stage-1 std MSE σ_q=0.20 | Stage-1 std MSE σ_q=1.052 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pilot only | 5 | 0.02284 | 0.00000 | 5 | 0.02284 | 0.00000 | — | — |
| raw-input Fisher score | 5 | 0.02204 | 0.00039 | 5 | 0.02167 | 0.00052 | 0.5172 | 0.5694 |
| all-subscores Fisher score | 5 | 0.02138 | 0.00048 | 5 | 0.02142 | 0.00034 | 0.2625 | 0.5628 |
| linear ILSA score | 5 | 0.00970 | 0.00049 | 5 | 0.00858 | 0.00027 | 0.0406 | 0.5000 |
| NLSA gate score (paper) | 5 | 0.00868 | 0.00025 | 5 | 0.00774 | 0.00022 | 0.0227 | 0.4961 |
| NLSA stacked m_dim=8 score | 5 | 0.00953 | 0.00037 | 5 | 0.00792 | 0.00014 | 0.0282 | 0.4954 |

### Paired W1, wide − narrow (negative favours the draft's wide bandwidth)

| score | seeds | mean ΔW1 | SE | wide better | Δ / W1(narrow) |
|---|---:|---:|---:|---:|---:|
| pilot only | 5 | +0.00000 | 0.00000 | 0/5 | +0.0% |
| raw-input Fisher score | 5 | -0.00037 | 0.00064 | 3/5 | -1.7% |
| all-subscores Fisher score | 5 | +0.00004 | 0.00037 | 2/5 | +0.2% |
| linear ILSA score | 5 | -0.00111 | 0.00072 | 4/5 | -11.5% |
| NLSA gate score (paper) | 5 | -0.00094 | 0.00036 | 5/5 | -10.8% |
| NLSA stacked m_dim=8 score | 5 | -0.00161 | 0.00042 | 5/5 | -16.9% |

### Paired W1 against the NLSA gate at σ_q = 0.20 (positive = gate better)

| comparison | seeds | mean ΔW1 | SE | gate better | Δ / W1(gate) |
|---|---:|---:|---:|---:|---:|
| pilot only − gate | 5 | +0.01416 | 0.00025 | 5/5 | +163.1% |
| raw-input Fisher score − gate | 5 | +0.01336 | 0.00055 | 5/5 | +154.0% |
| all-subscores Fisher score − gate | 5 | +0.01270 | 0.00064 | 5/5 | +146.3% |
| linear ILSA score − gate | 5 | +0.00102 | 0.00057 | 4/5 | +11.7% |
| NLSA stacked m_dim=8 score − gate | 5 | +0.00085 | 0.00039 | 4/5 | +9.8% |

### Paired W1 against the NLSA gate at σ_q = 1.052 (positive = gate better)

| comparison | seeds | mean ΔW1 | SE | gate better | Δ / W1(gate) |
|---|---:|---:|---:|---:|---:|
| pilot only − gate | 5 | +0.01509 | 0.00022 | 5/5 | +195.0% |
| raw-input Fisher score − gate | 5 | +0.01393 | 0.00049 | 5/5 | +179.9% |
| all-subscores Fisher score − gate | 5 | +0.01368 | 0.00046 | 5/5 | +176.7% |
| linear ILSA score − gate | 5 | +0.00084 | 0.00031 | 4/5 | +10.9% |
| NLSA stacked m_dim=8 score − gate | 5 | +0.00018 | 0.00034 | 3/5 | +2.3% |
