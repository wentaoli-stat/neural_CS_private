# Nonlinear-emission experiment for isolating the local gate

For the full method specification, observation-unit conventions, Direct-FSM
derivation, exact training/inference configurations, Jiang comparison,
reproduction commands, and reporting limitations, see
`SINGLE_PARAMETER_NONLINEAR_HMM_METHOD_AND_EXPERIMENT_20260720.md`.

## Question

The Gaussian common-factor HMM did not show that the positive nonlinear local
gate was essential: with the same GRU, its standardized score MSE was 0.211946
versus 0.215631 for the linear-local arm.  This experiment changes only the
emission model to test the gate under stronger local nonlinearity.

## Data-generating process

The binary HMM transition model and inferred parameter remain unchanged.  We
fix `P01=0.06`, infer `P11` over `[0.80,0.99]`, and use trajectories of length
50 with 10 coordinates at every time point.  Conditional on the hidden state,

```text
B_t = 0:  Y_tj ~ N(0,1)
B_t = 1:  Y_tj ~ (1-rho) N(0,1) + rho N(0,scale^2), independently over j.
```

The selected full-budget setting is `rho=0.20` and `scale=3`.  Its exact
coordinate emission log likelihood ratio is

```text
ell(y) = log[(1-rho) + (rho/scale)
             exp{0.5 y^2 (1-scale^-2)}].
```

Consequently, the exact full emission evidence is the sum of the coordinate
log ratios, and the exact HMM likelihood and transition score are available by
the forward-backward algorithm.  Exact scores are used only for evaluation.

The local mixture subscore passed to both learned models is

```text
s(y) = sigmoid(logit(pi_ref) + ell(y)) - pi_ref.
```

Recovering additive evidence from this bounded local score requires

```text
ell(y) = logit(s(y) + pi_ref) - logit(pi_ref),
```

a nonlinear, sign-preserving map.  This is representable by the method's
positive gate `s -> s * m_eta(s)` but not by the identity local map.  The GRU,
training target, data, budget, hidden size, initialization, and optimizer are
otherwise matched.  The gated model strictly nests the fitted linear model at
the start of gate training.

## Parameter screen

The pilot screen used one training seed.  The first row used 10,000 training
trajectories and 1,000 iterations; the remaining rows used 8,000 and 800.

| rho | scale | Linear stdMSE | Gate stdMSE | Gate improvement |
|---:|---:|---:|---:|---:|
| 0.05 | 8.0 | 0.2364 | 0.2365 | -0.1% |
| 0.10 | 5.0 | 0.2895 | 0.2716 | 6.2% |
| 0.10 | 3.0 | 0.2851 | 0.2650 | 7.1% |
| **0.20** | **3.0** | **0.3491** | **0.3184** | **8.8%** |
| 0.30 | 2.5 | 0.3684 | 0.3433 | 6.8% |

The selected setting was fixed before the full-budget training-seed check.

## Full-budget exact-score results

Each arm used 40,000 training trajectories, 8,000 validation trajectories,
3,000 updates, the same 64-unit GRU, and the same Direct-FSM proposal target.

| Training seed | Linear stdMSE | Gate stdMSE | Improvement | Linear corr. | Gate corr. |
|---:|---:|---:|---:|---:|---:|
| 20260721 | 0.2262 | 0.1738 | 23.2% | 0.8872 | 0.9113 |
| 20260722 | 0.1929 | 0.1346 | 30.2% | 0.9041 | 0.9347 |
| 20260723 | 0.2150 | 0.1562 | 27.3% | 0.8983 | 0.9225 |
| **Mean** | **0.2113** | **0.1549** | **26.9%** | **0.8965** | **0.9228** |

Thus the gate's score-level advantage is large and replicated across all
three training seeds.  The learned multipliers also moved materially away
from one: their maximum values ranged from 1.14 to 1.75 for marginal pieces
and from 1.33 to 2.13 for pairwise pieces.

## Repeated-dataset score-root inference

For each fitted training seed, both arms were evaluated on the same 100 newly
simulated observed datasets, each containing 100 independent complete HMM
trajectories.  Results below use the learned-score root and sandwich interval.

| Training seed | Linear RMSE | Gate RMSE | Linear MAE | Gate MAE | Linear coverage | Gate coverage | Linear width | Gate width |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260721 | 0.00998 | **0.00813** | 0.00781 | **0.00661** | 0.94 | 0.94 | 0.03219 | 0.03029 |
| 20260722 | **0.00691** | 0.00796 | **0.00580** | 0.00661 | 0.97 | 0.86 | 0.02610 | 0.02487 |
| 20260723 | 0.01008 | **0.00801** | 0.00798 | **0.00622** | 0.84 | 0.94 | 0.02706 | 0.02874 |
| **Mean** | **0.00899** | **0.00803** | **0.00720** | **0.00648** | **0.917** | **0.913** | **0.02845** | **0.02797** |

Pooling the equally sized evaluations gives root RMSE 0.00911 for the linear
arm and 0.00803 for the gate arm, an 11.8% reduction.  Gate root RMSE is also
more stable across the three fitted networks.  However, the second training
seed is a real counterexample: its gate has better score MSE but a positive
score offset, worse root RMSE, and undercoverage.  Therefore the current
result supports an average downstream advantage, not uniform dominance over
training seeds.

## Interpretation

This experiment succeeds at its main mechanistic objective: under a
non-Gaussian sparse-scale-mixture emission, the nonlinear local gate provides
a large and reproducible improvement in likelihood-score approximation.  The
improvement also transfers on average to point inference while retaining
similar mean sandwich coverage and width.

It is not yet a final Jiang comparison.  Jiang has not been run on this new
emission model, and the training-seed-dependent score offset must remain
visible in any report.  The defensible claim at this stage is that the new DGP
isolates and validates the gate mechanism; it does not establish that the
gate improves every fitted estimator.

## Frozen Stage-2 posterior results

Stage 2 follows the existing Model-3 construction exactly in form.  For every
complete trajectory, an adjacent-time composite pilot is computed without the
generating parameter.  The frozen Stage-1 model is evaluated at that pilot,
and the NPE context is

```text
(pilot u_hat(Y), frozen learned score S_hat(Y; u_hat(Y))).
```

For each Stage-1 training seed, the linear and gated arms use the same 50,000
Stage-2 simulations, pilot, MDN architecture, hyperparameters, and NPE random
seed.  Evaluation uses 25 observations at each of four parameter values, for
100 complete-trajectory observations per fitted Stage-1 model.  The exact HMM
posterior under the same uniform prior is the reference distribution.

The Stage-2 RMSE below is **not** posterior-mean error relative to the
generating parameter.  It is

```text
sqrt(mean[(learned posterior mean - exact posterior mean)^2]).
```

| Stage-1 seed | Linear mean RMSE to exact | Gate mean RMSE to exact | Linear W1 | Gate W1 | Linear endpoint MAE | Gate endpoint MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 20260721 | 0.01253 | **0.01005** | 0.01066 | **0.00777** | 0.00808 | **0.00635** |
| 20260722 | 0.01174 | **0.00936** | 0.00989 | **0.00775** | 0.00732 | **0.00538** |
| 20260723 | 0.01212 | **0.01075** | 0.01020 | **0.00867** | 0.00755 | **0.00660** |
| **Mean** | **0.01213** | **0.01005** | **0.01025** | **0.00806** | **0.00765** | **0.00611** |

The complete exact-posterior comparison is:

| Metric relative to exact posterior | Linear | Gate | Gate reduction |
|---|---:|---:|---:|
| Posterior-mean RMSE | 0.01213 | **0.01005** | **17.1%** |
| Posterior-mean MAE | 0.01013 | **0.00790** | **22.0%** |
| Posterior-SD MAE | 0.00304 | **0.00254** | **16.6%** |
| 90% interval endpoint MAE | 0.00765 | **0.00611** | **20.1%** |
| Absolute 90% interval-width error | 0.01020 | **0.00789** | **22.7%** |
| Wasserstein-1 distance | 0.01025 | **0.00806** | **21.3%** |

Unlike score-root inference, where one gated fit exhibited a mean score offset,
the frozen Stage-2 NPE improves every exact-posterior metric for all three
Stage-1 seeds.  The NPE can calibrate an overall score offset from its shared
training simulations, while the gated score supplies a more informative local
summary.  Frequentist error of the posterior mean relative to the generating
parameter is deliberately not used as a Stage-2 posterior-approximation metric
in this table.
