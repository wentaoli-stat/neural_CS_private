# Model 2 amortized FSM: fixed-sigma formal results

## Setting

-   Training seeds: `20260709, 20260710, 20260711, 20260712, 20260713`
-   `n_train=40000`, `n_val=8000`, `n_test=5000` per diagnostic anchor
-   `K=40`, `m=20`, `tau=1.0`
-   Continuous stratified anchors: `pi in [0.05, 0.7]`
-   Fixed FSM proposal width: `sigma_q=0.15` in logit coordinate
-   Architecture: `silu` rho head, positive signed anchor-conditioned radial gates
-   Exact full score is used only for the following post-training diagnostics.

## Exact-score standardized MSE

|   pi | linear mean +/- sd | radial mean +/- sd | radial improvement | wins |
|-----:|-------------------:|-------------------:|-------------------:|-----:|
| 0.07 |  0.2959 +/- 0.0984 |  0.1923 +/- 0.0378 |              28.6% |  4/5 |
| 0.10 |  0.1987 +/- 0.0395 |  0.1432 +/- 0.0244 |              25.5% |  4/5 |
| 0.30 |  0.0978 +/- 0.0086 |  0.0713 +/- 0.0044 |              26.9% |  5/5 |
| 0.50 |  0.1137 +/- 0.0072 |  0.0833 +/- 0.0055 |              26.7% |  5/5 |
| 0.65 |  0.1760 +/- 0.0148 |  0.1410 +/- 0.0118 |              19.7% |  5/5 |
| 0.68 |  0.2004 +/- 0.0282 |  0.1602 +/- 0.0200 |              19.8% |  5/5 |

## Overall paired result

-   Radial wins: **28/30** seed-anchor comparisons.
-   Mean paired relative reduction in exact-score std-MSE: **24.5%** (SD across pairs 14.1 percentage points).
-   After averaging the six anchors within each independent training seed, radial wins **5/5** seeds.
-   Exploratory paired seed-level test: `t(4)=3.22`, `p=0.0324`. The seed, not the seed-anchor row, is treated as the independent unit.
-   Five seeds satisfy the planned formal replication count, although wider architecture claims should still be checked in another model setting.

## Fixed-grid FSM objective

|   pi | linear relative MSE | radial relative MSE | radial improvement | wins |
|-----:|--------------------:|--------------------:|-------------------:|-----:|
| 0.07 |   0.9739 +/- 0.0547 |   0.9711 +/- 0.0521 |              0.27% |  4/5 |
| 0.10 |   0.9521 +/- 0.0304 |   0.9485 +/- 0.0310 |              0.38% |  5/5 |
| 0.20 |   0.9372 +/- 0.0131 |   0.9355 +/- 0.0127 |              0.18% |  5/5 |
| 0.30 |   0.9077 +/- 0.0294 |   0.9057 +/- 0.0288 |              0.21% |  4/5 |
| 0.40 |   0.9198 +/- 0.0403 |   0.9174 +/- 0.0393 |              0.26% |  5/5 |
| 0.50 |   0.8834 +/- 0.0313 |   0.8799 +/- 0.0317 |              0.40% |  5/5 |
| 0.60 |   0.9450 +/- 0.0183 |   0.9425 +/- 0.0155 |              0.26% |  4/5 |
| 0.65 |   0.9667 +/- 0.0407 |   0.9649 +/- 0.0403 |              0.18% |  5/5 |
| 0.68 |   0.9367 +/- 0.0307 |   0.9345 +/- 0.0304 |              0.24% |  3/5 |

The FSM regression target has variance `1/sigma_q^2`, so differences in its relative MSE are expected to be much smaller than differences in the noiseless exact-score diagnostic.
