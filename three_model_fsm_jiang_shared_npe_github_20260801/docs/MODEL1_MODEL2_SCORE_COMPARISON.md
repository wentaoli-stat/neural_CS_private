# Model 1/2 paired FSM versus official Jiang

Date: 2026-07-20

## Conclusion

On matched full-dataset score diagnostics and the same 100 repeated observed
datasets, the global comparison is

```text
nonlinear-gate FSM > linear-inner-map FSM > official Jiang Round 1
```

for both Model 1 and Model 2.  Here `>` means lower full-score standardized
MSE and lower score-root RMSE to the exact MLE.

This does **not** yet establish that our method beats Jiang's complete
data-dependent two-round procedure.  On the single observed dataset used to
construct each Jiang Round-2 proposal, Jiang is highly competitive and gives
the best root in Model 2.  Valid repeated-data evaluation of the final Jiang
procedure requires retraining Round 2 separately for every observed dataset.

## What is paired

- The data-generating models, parameter support, block size, signal strength,
  and number of iid blocks are identical.
- One length-20 raw block is one outer iid observation for Jiang.
- A full observed dataset contains 20 blocks in Model 1 and 40 blocks in
  Model 2.
- Score diagnostics use the same 1,000 newly simulated full datasets at each
  of `p = 0.10, 0.30, 0.50, 0.65`.
- Every score is converted to the same `u = logit(p)` full-dataset score.
- Root diagnostics use the same 100 datasets at `p_true = 0.30` and the same
  evaluation-only exact MLE for every paired row.
- The five frozen linear/radial FSM checkpoints are unchanged.  The Jiang
  checkpoint uses the authors' default batch size 10 and otherwise the
  frozen official implementation.

The two FSM arms here are the original Model-1/2 positive nonlinear local-gate
and linear-inner-map estimators with a DeepSets block readout.  This is not the
later HMM composite-score/gate/GRU experiment.

## Full-dataset score accuracy

The metric is

```text
mean[(estimated full u-score - exact full u-score)^2]
------------------------------------------------------
             variance(exact full u-score)
```

Lower is better.  FSM values average four parameter points and five training
seeds.  Jiang values average the same four parameter points for one training
seed.

| Model | Linear FSM | Nonlinear FSM | Jiang R1 | Jiang R2 local diagnostic |
|---|---:|---:|---:|---:|
| Model 1 | 0.0978 | **0.0847** | 0.4035 | 0.2327 |
| Model 2 | 0.1521 | **0.1140** | 0.4541 | 0.3700 |

Relative to linear FSM, the nonlinear gate reduces mean standardized MSE by
13.4% in Model 1 and 25.0% in Model 2.  Relative to Jiang Round 1, nonlinear
FSM reduces it by 79.0% and 74.9%, respectively.

The Round-2 column is only descriptive outside the proposal's local region.
That network was trained from one observed-data-dependent proposal and is not
a second global amortized model.

## Score roots on the same 100 datasets

The main learned-estimator error is RMSE to the exact MLE of each same
dataset.  RMSE to the generating parameter also contains irreducible sampling
variation and can obscure differences between learned score fields.

| Model | Method | RMSE to true `p` | RMSE to exact MLE | Boundary rate | Paired win rate versus Jiang R1 |
|---|---|---:|---:|---:|---:|
| Model 1 | Linear FSM, 5-seed mean | 0.1423 | 0.0340 | 1.2% | 67.0% |
| Model 1 | **Nonlinear FSM, 5-seed mean** | **0.1400** | **0.0304** | **0.6%** | **69.4%** |
| Model 1 | Jiang R1, 1 seed | 0.1468 | 0.0569 | 5.0% | — |
| Model 1 | Exact MLE | 0.1330 | 0 | — | — |
| Model 2 | Linear FSM, 5-seed mean | **0.1143** | 0.0278 | 0.8% | 66.8% |
| Model 2 | **Nonlinear FSM, 5-seed mean** | 0.1150 | **0.0235** | **0.4%** | **74.0%** |
| Model 2 | Jiang R1, 1 seed | 0.1198 | 0.0489 | 1.0% | — |
| Model 2 | Exact MLE | 0.1145 | 0 | — | — |

The nonlinear gate reduces root RMSE to exact MLE relative to linear FSM by
10.7% in Model 1 and 15.7% in Model 2.  In Model 2, linear has a marginally
smaller RMSE to the generating parameter even though nonlinear is closer to
the exact MLE.  This is finite-sample cancellation, not evidence that the
linear learned score is more accurate.

Every one of the five nonlinear checkpoints has lower root RMSE to exact MLE
than the one official Jiang Round-1 checkpoint in both models.  Nevertheless,
Jiang training has only one seed here, so this is not yet a replicated
between-method significance result.

## The fixed observed dataset and Jiang Round 2

This table explains why the result must not be summarized as “our method
already beats full two-round Jiang.”  FSM entries average the five frozen
checkpoint roots; Jiang has one fitted checkpoint per round.

| Model | Exact MLE | Linear FSM abs. error | Nonlinear FSM abs. error | Jiang R1 abs. error | Jiang R2 abs. error |
|---|---:|---:|---:|---:|---:|
| Model 1 | 0.3631 | 0.0378 | 0.0323 | **0.0086** | 0.0113 |
| Model 2 | 0.5501 | 0.0110 | 0.0201 | 0.0468 | **0.0015** |

Jiang's local second round substantially improves Model 2 on this particular
dataset.  Model 1 Round 1 is already excellent and its Round 2 is slightly
worse, though still better than the mean FSM root.  One selected dataset
cannot replace a repeated-data comparison.

## Jiang confidence intervals

The following are frozen-Round-1 frequentist 95% normal intervals over the
same 100 datasets.  Our original Model-1/2 experiment did not construct the
same three confidence sets, so there is no honest width/coverage winner table
yet.

| Model | Jiang covariance | Coverage | Mean width |
|---|---|---:|---:|
| Model 1 | score outer product | 85% | 0.3866 |
| Model 1 | curvature | 83% | 0.4407 |
| Model 1 | sandwich | 85% | 0.5201 |
| Model 2 | score outer product | 83% | 0.3347 |
| Model 2 | curvature | 87% | 0.3772 |
| Model 2 | sandwich | 90% | 0.4342 |

These intervals under-cover in this one-seed frozen-model evaluation.  This
does not measure unconditional two-round Jiang coverage.

## Simulation budget and replication caveats

- Each FSM checkpoint uses 40,000 training full datasets: about 800,000 raw
  blocks in Model 1 and 1,600,000 in Model 2, before validation data.
- One Jiang round uses 10,000 ordinary training blocks plus about 600,000
  repeated-reference blocks for Fisher/debias estimation.  Two rounds use
  about 1.22 million raw blocks.
- Therefore Round-1 simulator budget favors FSM, especially in Model 2; the
  complete two-round budgets are of the same order but still not identical.
- FSM has five training seeds; official Jiang currently has one.
- FSM targets a Gaussian-smoothed likelihood score.  Jiang targets the direct
  likelihood score with curvature and mean correction.
- The FSM representation uses analytic marginal/pairwise likelihood-ratio
  subscores.  It does not use the exact full likelihood, full score, latent
  states, or generating test parameter, but it is not a raw simulator-only
  representation.
- NPE posterior errors and Jiang confidence-set width are different objects
  and are not merged in this table.

## Defensible claim

The current evidence supports:

> On these two block-mixture models, the proposed nonlinear composite-score
> FSM gives a more accurate globally amortized full score and a more accurate
> global score root than both the linear-inner-map ablation and one official
> Jiang Round-1 fit.

It does not yet support:

> The proposed method uniformly outperforms Jiang's complete two-round
> estimator and confidence-set procedure.

That stronger claim requires several Jiang training seeds and a repeated
`train R1 -> infer R1 -> train R2 -> infer R2` experiment, with Round 2
retrained separately for every observed dataset.

## Reproduction and artifacts

The paired evaluator is
`src/khoo_vs_jiang/block_mixture_compare.py`.  It loads the original FSM
checkpoints read-only and the official Jiang checkpoints, regenerates the
paired datasets from recorded seeds, and writes all per-parameter and
per-training-seed rows.

The publication bundle retains the paired JSON summaries under
`results/model1/`, `results/model2/`, and `results/jiang/`. Large checkpoints,
logs, and regenerated arrays are intentionally excluded; their expected local
layout is recorded in `artifacts/checkpoints/README.md`.
