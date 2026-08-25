# Model 1/2 shared-NPE posterior comparison

Date: 2026-07-23

## Bottom line

When every score estimator is passed to the same amortized NPE backend, both
models give the same posterior-quality ordering:

```text
ours nonlinear gate > ours linear ablation > Jiang official Round 1 > pilot only
```

Here `>` means closer to the exact posterior of the same held-out complete
dataset. The headline metric is posterior-mean RMSE to the exact posterior
mean, not error to the simulated parameter.

| Model | Pilot only | Jiang R1 | Ours linear, 5-seed mean | Ours nonlinear, 5-seed mean |
|---|---:|---:|---:|---:|
| Model 1 posterior-mean RMSE | 0.03740 | 0.03014 | 0.01642 ± 0.00038 | **0.01482 ± 0.00132** |
| Model 2 posterior-mean RMSE | 0.06808 | 0.03272 | 0.02534 ± 0.00090 | **0.02189 ± 0.00092** |

Relative to Jiang R1, the nonlinear context reduces posterior-mean RMSE by
50.8% in Model 1 and 33.1% in Model 2.

## Matched experimental protocol

- Model 1 complete datasets contain 20 iid length-20 blocks.
- Model 2 complete datasets contain 40 iid length-20 blocks.
- Jiang is the frozen official Round-1 ELU MLP with its trained debias
  regression. Its single-block scores are summed to form the complete-dataset
  score.
- Our linear and nonlinear arms are the original frozen full-dataset score
  checkpoints, with five Stage-1 seeds per arm.
- Every method receives the same data-only pilot. The score contexts are
  `(pilot_u(D), estimated full-dataset u-score at pilot_u(D))`.
- The NPE training table contains the same 50,000 stratified-prior complete
  datasets for every method. The test table contains the same 100 held-out
  complete datasets.
- All methods use the same MDN architecture, optimizer, split, and NPE seed:
  64 hidden features, 8 mixture components, batch size 256, learning rate
  `5e-4`, at most 300 epochs, and early-stop patience 20.
- Each learned posterior is evaluated using 5,000 samples. The reference
  posterior uses a 5,000-point exact grid.
- The true simulated parameter is an NPE target during training, never a
  context input. Exact likelihood calculations are used only for held-out
  reference-posterior evaluation.

Jiang Round 2 is deliberately absent. It is fitted from a proposal centered
on a particular observed dataset and is therefore not one frozen amortized
mapping that can be applied to all 100 held-out datasets. This experiment
tests the information carried by the globally amortized Stage-1 score under a
common posterior backend; it is not presented as Jiang's published
score-root/confidence-set procedure.

## Full posterior metrics

All metrics below are lower-is-better. The uncertainty following an ours
entry is the standard deviation across five frozen Stage-1 fits; the same 100
test datasets are used for every fit.

### Model 1

| Method | Mean RMSE | Mean MAE | Posterior-SD MAE | 90% endpoint MAE | Width MAE | W1 |
|---|---:|---:|---:|---:|---:|---:|
| Pilot only | 0.03740 | 0.02941 | 0.00788 | 0.02606 | 0.02711 | 0.02981 |
| Jiang official R1 | 0.03014 | 0.02468 | 0.00738 | 0.02225 | 0.02542 | 0.02523 |
| Ours linear | 0.01642 ± 0.00038 | 0.01277 ± 0.00004 | 0.00491 ± 0.00026 | 0.01260 ± 0.00032 | 0.01667 ± 0.00070 | 0.01359 ± 0.00010 |
| **Ours nonlinear** | **0.01482 ± 0.00132** | **0.01135 ± 0.00098** | **0.00479 ± 0.00024** | **0.01173 ± 0.00079** | **0.01593 ± 0.00084** | **0.01237 ± 0.00093** |

### Model 2

| Method | Mean RMSE | Mean MAE | Posterior-SD MAE | 90% endpoint MAE | Width MAE | W1 |
|---|---:|---:|---:|---:|---:|---:|
| Pilot only | 0.06808 | 0.05547 | 0.02233 | 0.05544 | 0.07360 | 0.05667 |
| Jiang official R1 | 0.03272 | 0.02603 | 0.00958 | 0.02587 | 0.03238 | 0.02674 |
| Ours linear | 0.02534 ± 0.00090 | 0.01992 ± 0.00053 | 0.00495 ± 0.00009 | 0.01908 ± 0.00041 | 0.01664 ± 0.00047 | 0.02039 ± 0.00053 |
| **Ours nonlinear** | **0.02189 ± 0.00092** | **0.01751 ± 0.00065** | **0.00453 ± 0.00015** | **0.01687 ± 0.00056** | **0.01521 ± 0.00051** | **0.01787 ± 0.00067** |

## Seed-level robustness

Every one of the five nonlinear Stage-1 checkpoints beats the single Jiang R1
checkpoint in posterior-mean RMSE and W1 in both models.

| Model | Method | Seed-level posterior-mean RMSE range | Seed-level W1 range |
|---|---|---:|---:|
| Model 1 | Jiang R1 | 0.03014 | 0.02523 |
| Model 1 | Ours linear | 0.01597–0.01701 | 0.01347–0.01372 |
| Model 1 | Ours nonlinear | **0.01300–0.01630** | **0.01103–0.01330** |
| Model 2 | Jiang R1 | 0.03272 | 0.02674 |
| Model 2 | Ours linear | 0.02458–0.02676 | 0.01976–0.02110 |
| Model 2 | Ours nonlinear | **0.02062–0.02309** | **0.01687–0.01861** |

Jiang has only one Stage-1 fit in this comparison. The result is therefore a
strong five-seed robustness check for our method, but not yet a multi-seed
between-method uncertainty analysis.

## Interpretation

The pilot-only rows show that the pilot itself is not sufficient to explain
the result. Jiang's score improves the common NPE substantially over the
pilot, so the Jiang implementation is carrying real inferential information.
Our learned full-dataset score improves it further, and the nonlinear gate
consistently improves on the linear ablation. This is the intended ablation
pattern rather than a failure of the shared NPE backend.

The defensible claim is:

> On Model 1 and Model 2, under identical amortized posterior training and
> evaluation, the proposed nonlinear composite-score context recovers the
> exact posterior more accurately than its linear ablation and one official
> Jiang Round-1 score fit.

It is not a claim that NPE is part of Jiang's original method, nor that this
experiment compares against a separately retrained Jiang Round 2 for every
test dataset.

## Artifacts

The implementation is
`src/khoo_vs_jiang/block_mixture_stage2_npe.py`, with focused tests in
`tests/test_block_mixture_stage2_npe.py`.

The compact publication output is under `results/model1/shared_npe/` and
`results/model2/shared_npe/`. Each retained model directory contains:

- `posterior_summary_by_stage1_fit.csv`: one row per fitted context;
- `posterior_summary_by_family.csv`: the headline family-level table;
- `protocol.json`: portable provenance and the common Stage-2 protocol.

The per-dataset arrays, shared contexts, and fitted NPE checkpoints are large
regenerated artifacts and are intentionally not part of the Git repository.

The run was executed on a remote GPU instance. The downloaded archive SHA-256 is
`13ea72f112c0df398d28c78c97d7727bef27cd0c4483ce6006cdda8511dc5ef5`.
