# Jiang architecture and NPE ablation on nonlinear emissions

## Scientific question

This experiment separates three possible explanations for performance on the
nonlinear-emission HMM:

1. Jiang's published flattened MLP and official score-root inference;
2. replacing only Jiang's single-trajectory score architecture with a GRU;
3. placing the same neural posterior estimator (NPE) after different frozen
   score summaries;
4. the additional contribution of local composite scores and the positive
   nonlinear gate.

The data-generating process is the sparse-scale-mixture HMM specified in
`NONLINEAR_EMISSION_GATE_EXPERIMENT_20260724.md`.

## Arms and estimands

The frequentist score-root comparison is:

| Arm | Single-observation score model | Training/inference |
|---|---|---|
| A | Jiang's published flattened ELU MLP | Official direct SM, Fisher penalty, conditional debias, two rounds, root and confidence sets |
| B | Raw-sequence GRU | The same Jiang objectives, budgets, debiasing, two rounds, root and confidence sets |

The posterior comparison uses three auxiliary arms:

| Arm | Frozen Round-1 summary | Stage-2 inference |
|---|---|---|
| A-NPE | Jiang published MLP score | Shared NPE |
| C | Jiang raw-sequence GRU score | Shared NPE |
| D | Local composite scores, nonlinear gate and GRU | Shared NPE |

The A-to-B comparison measures the effect of changing Jiang's architecture
under its own inference procedure.  The C-to-D comparison measures the
additional value of local composite structure and the nonlinear gate after
holding the NPE fixed.  A-NPE is retained to determine whether NPE alone can
rescue a weak published-MLP score.

## Controls

- One complete length-50 trajectory is one outer observation.  Outer
  observations are iid; coordinates within a trajectory follow an HMM.
- A and B use the same 10,000-parameter paper profile, batch size, optimizer,
  Fisher/debias budgets, training seed, observed dataset and official
  checkpoint-selection rules.
- All NPE arms use observation-independent Round-1 score checkpoints.  A
  Round-2 checkpoint is tied to the observed dataset that formed its proposal
  and is not treated as an amortized feature extractor.
- The NPE arms share every simulated parameter and trajectory, the same
  data-only adjacent-pair pilot, two-dimensional context, MDN architecture,
  optimizer settings, training seed and posterior sampling seed.
- Jiang's physical-parameter score is converted to the common
  `u=logit(p)` coordinate using

  ```text
  score_u = p(1-p) score_p.
  ```

- Exact scores and exact HMM posteriors are evaluation-only.
- Posterior quality is measured against the exact posterior, not against the
  generating parameter.

## Validation status

The local and remote test suites pass 11 tests.  End-to-end smoke runs confirm
that the architecture-adapted GRU can complete both Jiang rounds and that all
three frozen-score features can train and evaluate the same NPE.  Smoke
numbers are integrity checks and are not scientific results.

## Formal results

Two training tracks are kept separate:

- `published-batch`: batch size 10, exactly matching the Jiang paper profile;
- `vectorized-paper`: both MLP and GRU use batch size 64 while retaining the
  same 10,000 simulated parameters, epoch limits, Fisher reference sizes,
  debiasing and two-round procedure.  This paired track is a controlled
  architecture comparison but is not labelled as the published Jiang
  hyperparameter configuration.
- `reduced-curvature`: the compute-feasible paired track keeps the 10,000
  parameter direct-SM table and batch size 64, while both architectures use
  200 Fisher anchors with 100 reference trajectories, 5 Fisher epochs, 100
  debias epochs and 5 debias-curvature epochs.  It retains the same objectives
  and two-round procedure but is explicitly not a paper-budget result.

The published-batch MLP baseline was completed at full paper budget.  Its
Round-1 root hit the upper bound at `0.99000` without score convergence.  Its
Round-2 learned-score root was `0.98894`, despite an exact MLE of `0.93594` on
the same observed dataset.  Round-2 standardized score MSE was `0.99336` and
score correlation was `0.09935`.  Thus the published MLP result is a failure
on this DGP, not a baseline that was weakened for the ablation.

The batch-10 GRU paper track demonstrated substantially better direct-SM
validation loss but made the full Fisher stage computationally prohibitive:
one joint-score epoch took about 148 seconds and one 500-reference Fisher
epoch took several minutes.  It was not used as a partially trained result.
The headline architecture comparison below is therefore the explicitly
paired `reduced-curvature` track.

### Jiang root and confidence-set ablation

Both reduced-curvature arms used the same 10,000-parameter score table,
batch size 64, direct-SM objective, five Fisher epochs with 200 anchors and
100 reference observations, conditional debiasing, proposal rule, two rounds,
seed and observed dataset.  Only the single-trajectory score architecture
changed.

| Metric | Jiang MLP | Jiang raw-sequence GRU |
|---|---:|---:|
| Round-1 best joint validation SM | -7.39 | **-254.44** |
| Round-2 best joint validation SM | -1.24 | **-41.20** |
| Round-1 score stdMSE | 0.9984 | **0.9701** |
| Round-1 score correlation | 0.0536 | **0.2634** |
| Round-2 score stdMSE | 0.9954 | **0.4636** |
| Round-2 score correlation | 0.0752 | **0.7694** |
| Round-2 estimate | 0.9900 | 0.9900 |
| Round-2 score-root converged | no | no |
| Round-2 sandwich width | 0.4087 | **0.0724** |

The GRU dramatically improves the shape of the learned score, especially in
Round 2.  It nevertheless does not repair the score zero: both methods end at
the upper boundary and fail the score-residual criterion.  The GRU interval is
narrower but is centered at the wrong boundary estimate, so narrowness is not
evidence of valid inference.  This isolates the remaining failure as a
mean-offset/root-calibration problem rather than merely an inability to encode
temporal dependence.

### Shared-NPE posterior ablation

The posterior experiment used 50,000 shared Stage-2 simulations, a shared
data-only pilot, the same two-dimensional context, MDN, optimizer, seed and
5,000 posterior draws.  Evaluation comprised 25 observations at each of four
parameter values.  Every number below is relative to the exact posterior for
the corresponding observation.

| Exact-posterior metric | Jiang MLP + NPE | Jiang GRU + NPE | Structured gate + GRU + NPE |
|---|---:|---:|---:|
| Posterior-mean RMSE | 0.01426 | 0.01415 | **0.00991** |
| Posterior-mean MAE | 0.01237 | 0.01226 | **0.00755** |
| Posterior-SD MAE | 0.00352 | 0.00390 | **0.00271** |
| 90% endpoint MAE | 0.00892 | 0.00912 | **0.00652** |
| Absolute 90% width error | 0.01139 | 0.01272 | **0.00851** |
| Wasserstein-1 | 0.01243 | 0.01230 | **0.00773** |

Adding a GRU before the shared NPE changes posterior-mean RMSE by only `0.7%`
and W1 by only `1.0%` relative to the MLP-score NPE.  It slightly worsens the
posterior-SD and interval metrics.  Thus NPE largely calibrates the two weak
Jiang summaries back toward the information already present in their common
pilot; the large score-shape improvement does not translate into a comparably
large posterior improvement.

Relative to Jiang-GRU plus the same NPE, the structured local-composite gate
arm reduces posterior-mean RMSE by `30.0%`, posterior-mean MAE by `38.4%`,
posterior-SD MAE by `30.5%`, endpoint MAE by `28.6%`, width error by `33.1%`,
and W1 by `37.2%`.

## Conclusion and limits

The experiment answers the motivating question in two parts:

1. A GRU makes Jiang's learned score much more correlated with the exact HMM
   score, but GRU alone does not repair Jiang's official score root or
   confidence set.
2. NPE can hide much of the root failure by learning a posterior calibration,
   but Jiang-MLP and Jiang-GRU become almost indistinguishable after the same
   NPE.  The structured local-composite/gated score remains materially more
   informative and wins every exact-posterior metric.

The root/CI table is a one-observed-dataset architecture ablation and does not
estimate coverage.  The posterior table uses one Jiang training seed and one
structured-score training seed over 100 test observations.  Also, the
structured Stage-1 checkpoint used its own 40,000-simulation full training
profile whereas the reduced-curvature Jiang score table used 10,000 primary
simulations plus reference simulations.  Therefore the final comparison is a
method-level ablation under shared Stage-2 budget, not a claim that the entire
gain is caused by the nonlinear gate alone.  The separate linear-versus-gate
experiment supplies the matched gate-specific attribution.
