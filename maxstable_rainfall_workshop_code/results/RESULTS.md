# Selected 47x79, Stage-1 10k results

These are the retained results for **Identity versus Score Gate**. The code
and result artifacts retain the internal key `positive_anchor` for Score Gate.
The old unconditioned/geometry gate was removed from this snapshot.

## Prior-wide Stage-2 test

The 100 held-out datasets and rough-500 pilots are shared. The metric is error
of the posterior mean to the generating normalized parameter; an exact
79-site posterior is unavailable.

| Method | RMSE u1 | RMSE u2 | RMSE u3 | Joint L2 RMSE |
|---|---:|---:|---:|---:|
| Identity | 0.04657 | 0.04864 | 0.03473 | 0.07577 |
| Score Gate | **0.04403** | **0.04405** | 0.03514 | **0.07151** |

Score Gate lowers joint L2 RMSE by 5.61% in this single-seed run.

## Fixed-truth original covariance-scale MSE

Each regime contains 100 paired test datasets. Entries are MSE for
`(Sigma_11, Sigma_12, Sigma_22)`.

| Truth | Method | Sigma11 MSE | Sigma12 MSE | Sigma22 MSE | Sum MSE |
|---|---|---:|---:|---:|---:|
| low | Full all-pair MPLE | 776.34 | 205.98 | 286.97 | 1269.29 |
| low | Identity | 214.76 | 41.92 | 55.70 | 312.37 |
| low | Score Gate | **138.24** | **40.04** | **36.81** | **215.08** |
| center | Full all-pair MPLE | 1860.96 | 474.63 | 814.87 | 3150.45 |
| center | Identity | 518.34 | 137.53 | 193.82 | 849.69 |
| center | Score Gate | **465.96** | **124.42** | **174.63** | **765.01** |
| high | Full all-pair MPLE | 2648.96 | 1037.86 | 917.64 | 4604.46 |
| high | Identity | 577.31 | 259.76 | 255.08 | 1092.15 |
| high | Score Gate | **555.11** | **165.62** | **168.13** | **888.85** |

Relative reductions in summed MSE are 31.15% (low), 9.97% (center), and
18.61% (high).

## Mean uncertainty scale

NPE entries are mean posterior SD; MPLE entries are mean sandwich asymptotic
SE, so their interpretation is related but not identical.

| Truth | Method | Sigma11 | Sigma12 | Sigma22 |
|---|---|---:|---:|---:|
| low | MPLE asymptotic SE | 24.95 | 13.52 | 14.95 |
| low | Identity posterior SD | 16.22 | 8.19 | 9.16 |
| low | Score Gate posterior SD | 14.19 | 8.28 | 8.69 |
| center | MPLE asymptotic SE | 41.00 | 22.99 | 24.38 |
| center | Identity posterior SD | 19.82 | 11.29 | 11.84 |
| center | Score Gate posterior SD | 18.65 | 10.62 | 10.80 |
| high | MPLE asymptotic SE | 49.59 | 33.40 | 26.65 |
| high | Identity posterior SD | 31.31 | 21.55 | 19.69 |
| high | Score Gate posterior SD | 30.45 | 20.41 | 18.07 |

## Extremal-coefficient recovery

Score Gate reduces integrated squared error of the Smith extremal coefficient
by 16.31% (low), 13.98% (center), and 18.20% (high). Full dataset-level rows,
paired bootstrap intervals, orientation errors, and axis-ratio errors are in
`results/extremal/summary.json`.

## Raw-data contextual baselines

The two additional baselines use the same 100 fixed-truth datasets, Stage-1
seed 20260819, Stage-2 simulation seed 20260731, NPE seed 54000, and posterior
sampling seed 97000 as the retained paper comparison. No checkpoint was
retrained for this evaluation.

| Truth | Method | Sigma11 MSE | Sigma12 MSE | Sigma22 MSE | Sum MSE | Extremal ISE |
|---|---|---:|---:|---:|---:|---:|
| low | Raw-data NPE | 11669.0 | 6436.4 | 3738.7 | 21844.1 | 0.00359308 |
| low | Raw-data FSM | 547.5 | 72.3 | 222.6 | 842.4 | 0.00023146 |
| center | Raw-data NPE | 139.9 | 22.5 | 53.2 | 215.6 | 0.00006632 |
| center | Raw-data FSM | 1519.0 | 272.3 | 615.2 | 2406.5 | 0.00047116 |
| high | Raw-data NPE | 16812.5 | 16514.8 | 5022.2 | 38349.5 | 0.00286575 |
| high | Raw-data FSM | 1370.3 | 519.2 | 361.6 | 2251.1 | 0.00021454 |

Raw-data NPE returns nearly the same posterior mean at all three truths, close
to the prior center. Its small center-truth error is therefore coincidental; the
low and high truths expose the collapse. Raw-data FSM uses more information,
but its covariance MSE and extremal ISE exceed both Identity and Score Gate at
every truth. Dataset-level rows and full protocol provenance are in
`results/raw_baselines_paper_fixed_truth/`.

## Scope

This is one Stage-1 training seed and 100 paired tests per reported cell. It
supports a successful selected experiment, not a multi-seed general claim.
The exact posterior is unavailable; MPLE is a comparator, not ground truth.
