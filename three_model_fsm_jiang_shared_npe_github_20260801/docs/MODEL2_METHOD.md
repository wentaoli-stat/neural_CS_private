# Model 2: Blockwise Common-Factor Variance Mixture

## Model, amortized FSM method, and current results

Date: 2026-07-11

This document records the current Model 2 method and the formal results that
should be used in subsequent comparisons. It distinguishes:

1. the exact toy model and its structural marginal/pairwise decomposition;
2. likelihood-free amortized FSM training;
3. Mode A consumers of the frozen score field;
4. Mode B pilot-plus-score SBI with NPE and conditional diffusion;
5. a separate fixed-center score-calibration diagnostic.

The exact likelihood and exact score are available only because this is a toy
model. They are used for evaluation, numerical sanity checks, and oracle
references. They are not used in the FSM training loss.

---

## 1. Why Model 2 is needed

Model 1 tests nonlinear aggregation of marginal evidence. Model 2 adds a
second structural issue: the latent state changes within-block dependence, so
one-observation marginals do not contain all the information in the full
block.

Under the active state, observations in a block share a Gaussian factor. The
full score depends on cross-products $Y_{ki}Y_{kj}$. Marginal subscores only
contain $Y_{ki}^2$, while same-block pairwise subscores contain

$$
(Y_{ki}+Y_{kj})^2
=
Y_{ki}^2+Y_{kj}^2+2Y_{ki}Y_{kj}.
$$

Thus Model 2 separates two questions:

1. **channel selection:** are marginal subscores alone sufficient, or is the
   pairwise channel required?
2. **nonlinear calibration:** after both channels are present, does a learned
   nonlinear inner map improve over the identity inner map?

This is a more demanding and more realistic composite-score example than
Model 1.

---

## 2. Generative model

There are $K$ independent blocks and $m$ observations in each block. For block
$k$,

$$
Z_k\sim\operatorname{Bernoulli}(\pi).
$$

If $Z_k=0$,

$$
Y_k\sim N(0,I_m).
$$

If $Z_k=1$, the observations share a common factor:

$$
Y_{ki}=U_k+\varepsilon_{ki},
\qquad
U_k\sim N(0,\tau^2),
\qquad
\varepsilon_{ki}\stackrel{iid}{\sim}N(0,1).
$$

Therefore

$$
Y_k\mid Z_k=1
\sim
N\left(0,I_m+\tau^2\mathbf 1_m\mathbf 1_m^\top\right).
$$

The unknown parameter is

$$
\pi\in[0.05,0.70],
$$

and training uses

$$
u=\operatorname{logit}(\pi).
$$

The common-factor scale $\tau$ is fixed in each experiment.

---

## 3. Exact likelihood and Fisher score

### 3.1 Full block likelihood ratio

The alternative-to-null likelihood ratio for block $k$ is

$$
R_{m,k}
=
(1+m\tau^2)^{-1/2}
\exp\left\{
\frac{\tau^2}{2(1+m\tau^2)}
\left(\sum_{i=1}^{m}Y_{ki}\right)^2
\right\}.
$$

The block density is

$$
p_\pi(Y_k)
=
f_0(Y_k)\{1-\pi+\pi R_{m,k}\}.
$$

### 3.2 Exact score in $\pi$

The exact constrained-coordinate score is

$$
S_\pi^{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^{K}
\frac{R_{m,k}-1}{1-\pi+\pi R_{m,k}}.
$$

If

$$
w_k
=
P(Z_k=1\mid Y_k,\pi)
=
\frac{\pi R_{m,k}}{1-\pi+\pi R_{m,k}},
$$

then

$$
S_\pi^{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^{K}
\frac{w_k-\pi}{\pi(1-\pi)}.
$$

### 3.3 Exact score in the implemented logit coordinate

The implementation predicts

$$
S_u^{\mathrm{full}}(Y;u)
=
\pi(1-\pi)S_\pi^{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^{K}(w_k-\pi).
$$

All current Stage-1 score errors are reported in this $u$ coordinate.

---

## 4. Marginal and pairwise composite subscores

### 4.1 Marginal channel

For one observation,

$$
R_{1,ki}
=
(1+\tau^2)^{-1/2}
\exp\left\{
\frac{\tau^2Y_{ki}^2}{2(1+\tau^2)}
\right\}.
$$

At anchor $u_a$ with $\pi_a=\operatorname{sigmoid}(u_a)$, the code uses the
local $u$-subscore

$$
s_{ki}^{(1)}(u_a)
=
\operatorname{sigmoid}
\left(u_a+\log R_{1,ki}\right)
-\pi_a.
$$

This channel contains information about $Y_{ki}^2$ but not the cross-products.

### 4.2 Same-block pairwise channel

For a same-block pair $i<j$,

$$
R_{2,kij}
=
(1+2\tau^2)^{-1/2}
\exp\left\{
\frac{\tau^2(Y_{ki}+Y_{kj})^2}{2(1+2\tau^2)}
\right\}.
$$

The local pairwise $u$-subscore is

$$
s_{kij}^{(2)}(u_a)
=
\operatorname{sigmoid}
\left(u_a+\log R_{2,kij}\right)
-\pi_a.
$$

Only same-block pairs are needed. Cross-block pairs are redundant because the
blocks are independent.

---

## 5. Why the two channels can recover the full block score

For a $\pi$-subscore $s$, define

$$
A_\pi(s)
=
\frac{1+(1-\pi)s}{1-\pi s}.
$$

Then

$$
\ell_{ki}^{(1)}
=
\log A_\pi\left(s_{ki}^{(1,\pi)}\right)
$$

recovers an affine function of $Y_{ki}^2$, and

$$
\ell_{kij}^{(2)}
=
\log A_\pi\left(s_{kij}^{(2,\pi)}\right)
$$

recovers an affine function of $(Y_{ki}+Y_{kj})^2$.

The identity

$$
\sum_{i<j}(Y_{ki}+Y_{kj})^2
=
(m-2)\sum_iY_{ki}^2
+
\left(\sum_iY_{ki}\right)^2
$$

implies

$$
\left(\sum_iY_{ki}\right)^2
=
\sum_{i<j}(Y_{ki}+Y_{kj})^2
-(m-2)\sum_iY_{ki}^2.
$$

Consequently, suitable nonlinear transforms of the marginal and pairwise
subscores can recover the statistic required by $R_{m,k}$ and therefore the
exact full score.

An nCS-style representation is

$$
U_{1,k}
=
\sum_i\varphi_1\left(s_{ki}^{(1)}\right),
\qquad
U_{2,k}
=
\sum_{i<j}\varphi_2\left(s_{kij}^{(2)}\right),
$$

followed by

$$
S^{\mathrm{full}}(Y;\pi)
=
\sum_{k=1}^{K}
H_{\pi,\tau,m}(U_{1,k},U_{2,k}).
$$

This exact representation is the theoretical motivation for separate learned
inner maps for the two channels and a nonlinear block readout.

---

## 6. Information-map design checks

Before the full experiment, the marginal-channel ceiling was evaluated by

$$
R_{\mathrm{marg}}
=
\frac{
\operatorname{Var}
\left{
E[g_{\mathrm{full}}(Y_k)\mid A_k]
\right}
}{
\operatorname{Var}\{g_{\mathrm{full}}(Y_k)\}
},
$$

where $A_k$ denotes the marginal-only block information. This is the fraction
of full-score variance recoverable from that channel.

The precheck found the following approximate pattern:

| $\tau$ | Marginal/full information fraction |
|---:|---:|
| 0.2 | 0.08 |
| 0.3 | 0.15 |
| 0.5 | 0.34 |
| 0.75 | 0.51 |
| 1.0 | 0.62 |
| 1.5 | 0.73 |

Thus small $\tau$ makes the pairwise channel especially necessary but also
makes the entire dataset weakly informative. Larger $\tau$ increases total
information and makes the local likelihood-ratio geometry more nonlinear, but
the marginal channel itself becomes more informative. The current
$\tau=1.0,K=40$ design is primarily the nonlinear-calibration design point; the
earlier $\tau=0.5,K=40$ design was the cleaner channel-ceiling experiment.

The FSM smoothing width was also selected by an exact one-dimensional
smoothing-floor calculation. At $\tau=1.0$, the inherited $\sigma_q=0.5$ gave
an exact-score floor near 0.26, which was too large. Reducing the logit-space
width to

$$
\sigma_q=0.15
$$

reduced the floor to about 0.008. This is why the current formal experiment
uses a stronger signal together with a narrower FSM proposal.

---

## 7. Current amortized FSM method

### 7.1 Anchors and likelihood-free target

The anchor is sampled over the deployment interval:

$$
\pi_a\in[0.05,0.70],
\qquad
u_a=\operatorname{logit}(\pi_a).
$$

The local proposal and simulator are

$$
u\mid u_a\sim N(u_a,\sigma_q^2),
\qquad
Y\sim p(\cdot\mid u).
$$

The FSM regression target is

$$
T_{\mathrm{FSM}}
=
\frac{u-u_a}{\sigma_q^2}.
$$

Each method minimizes

$$
\mathcal L_{\mathrm{FSM}}
=
E\left[
\left{
\widehat S(Y,u_a)
-
\frac{u-u_a}{\sigma_q^2}
\right}^2
\right].
$$

This is likelihood-free and does not use the exact Fisher score.

### 7.2 Feature standardization

Marginal and pairwise subscores are standardized separately using frozen
training-set statistics:

$$
\widetilde s_{ki}^{(1)}
=
\frac{s_{ki}^{(1)}-\mu_1}{\sigma_1},
\qquad
\widetilde s_{kij}^{(2)}
=
\frac{s_{kij}^{(2)}-\mu_2}{\sigma_2}.
$$

The two block channels are then standardized with shared frozen block-level
statistics. Both methods use exactly the same statistics.

### 7.3 Linear-inner-map CS with anchor-conditioned DeepSets

The linear block feature is

$$
C_k
=
\left(
\frac1m\sum_i\widetilde s_{ki}^{(1)},
\frac1{\binom{m}{2}}\sum_{i<j}\widetilde s_{kij}^{(2)}
\right).
$$

The score model is

$$
\widehat S_{\mathrm{lin}}(Y,u_a)
=
\sum_{k=1}^{K}
\rho_\psi(C_k,\widetilde u_a).
$$

The term "linear" refers only to the identity inner maps

$$
\phi_1(s)=s,
\qquad
\phi_2(s)=s.
$$

The outer $\rho_\psi$ is a nonlinear MLP shared across blocks.

### 7.4 Signed positive-gate CS with the same DeepSets readout

Model 2 uses separate learned gates for the marginal and pairwise channels:

$$
\phi_{\eta,1}(s,\widetilde u_a)
=
s\,m_{\eta,1}(s,\widetilde u_a),
$$

$$
\phi_{\eta,2}(s,\widetilde u_a)
=
s\,m_{\eta,2}(s,\widetilde u_a),
$$

with

$$
m_{\eta,r}(s,\widetilde u_a)
=
\operatorname{softplus}
\left\{
\operatorname{MLP}_{\eta,r}(s,\widetilde u_a)
\right\}>0.
$$

The signed input removes the old odd-function restriction caused by
$m(|s|)$. Positivity preserves the direction of local evidence without
imposing monotonicity on the multiplier.

The gated block feature is

$$
\widetilde C_k
=
\left(
\frac1m\sum_i\phi_{\eta,1}(\widetilde s_{ki}^{(1)},\widetilde u_a),
\frac1{\binom{m}{2}}\sum_{i<j}
\phi_{\eta,2}(\widetilde s_{kij}^{(2)},\widetilde u_a)
\right),
$$

and

$$
\widehat S_{\mathrm{rad}}(Y,u_a)
=
\sum_{k=1}^{K}
\rho_\psi(\widetilde C_k,\widetilde u_a).
$$

### 7.5 Strict nesting

The two gate outputs are initialized at one and the trained linear
$\rho_\psi$ is copied into the radial model. Therefore

$$
\widehat S_{\mathrm{rad}}(Y,u_a)
=
\widehat S_{\mathrm{lin}}(Y,u_a)
$$

at initialization up to numerical precision. The radial model then receives
400 gate-only steps followed by lower-learning-rate joint fine-tuning. EMA,
gradient clipping, weight decay, early stopping, and restored best checkpoints
are enabled.

---

## 8. Formal Stage-1 setting

| Item | Value |
|---|---:|
| Blocks $K$ | 40 |
| Block size $m$ | 20 |
| Signal $\tau$ | 1.0 |
| Parameter range | $\pi\in[0.05,0.70]$ |
| Feature mode | evaluated at sampled anchor |
| FSM width $\sigma_q$ | 0.15 in logit coordinates |
| Train / validation / test | 40,000 / 8,000 / 5,000 |
| Independent Stage-1 seeds | 20260709--20260713 |
| Optimization steps | 3,000 |
| Batch size | 512 |
| $\rho$ network | width 64, depth 2, SiLU |
| Two gate networks | width 16, signed and anchor-conditioned |
| Main learning rate | $3\times10^{-4}$ |
| Gate/joint fine-tuning rate | $10^{-4}$ |
| Weight decay | $10^{-3}$ |
| EMA | 0.995 |
| Early-stopping patience | 20 checks |

---

## 9. Stage-1 formal results

Exact full scores are evaluation-only. The table averages five independently
trained checkpoints.

| Test $\pi$ | Linear std-MSE | Radial std-MSE | Radial reduction | Radial wins |
|---:|---:|---:|---:|---:|
| 0.07 | $0.2959\pm0.0984$ | $0.1923\pm0.0378$ | 28.6% | 4/5 |
| 0.10 | $0.1987\pm0.0395$ | $0.1432\pm0.0244$ | 25.5% | 4/5 |
| 0.30 | $0.0978\pm0.0086$ | $0.0713\pm0.0044$ | 26.9% | 5/5 |
| 0.50 | $0.1137\pm0.0072$ | $0.0833\pm0.0055$ | 26.7% | 5/5 |
| 0.65 | $0.1760\pm0.0148$ | $0.1410\pm0.0118$ | 19.7% | 5/5 |
| 0.68 | $0.2004\pm0.0282$ | $0.1602\pm0.0200$ | 19.8% | 5/5 |

Across all seed-anchor pairs:

- radial wins 28/30 comparisons;
- the mean paired reduction is 24.5%;
- after averaging anchors within seed, radial wins 5/5 seeds;
- the seed-level paired test gives $t(4)=3.22$, $p=0.0324$.

The raw FSM validation objective shows a much smaller difference, typically
about 0.2--0.4%, because the regression target variance is

$$
\operatorname{Var}(T_{\mathrm{FSM}})
=
\frac{1}{\sigma_q^2}
\approx44.44.
$$

The exact-score diagnostic removes that target noise and is therefore the
appropriate metric for the representational comparison.

---

## 10. Mode A: direct score-field consumers

### 10.1 MLE and score roots

Five checkpoints were evaluated on 20 shared datasets at each of
$\pi\in\{0.10,0.30,0.50,0.65\}$.

| Method | RMSE to true $\pi$ | RMSE to exact MLE | RMSE to smoothed-oracle MLE | Boundary match |
|---|---:|---:|---:|---:|
| Exact MLE | 0.09616 | 0 | 0.00093 | 1.0000 |
| Composite pilot | 0.11932 | 0.07815 | 0.07817 | 0.8750 |
| Linear learned field | 0.10030 | 0.02897 | 0.02892 | 0.9175 |
| Radial learned field | 0.09920 | 0.02514 | 0.02497 | 0.9325 |

Radial reduces RMSE to the exact MLE by 13.2%, wins all five Stage-1 seeds,
and gives paired $p=0.0109$.

### 10.2 Score-only quasi-Newton

The score-only secant update is

$$
\widehat H_t
=
\frac{
\widehat S(Y,u_t)-\widehat S(Y,u_{t-1})
}{u_t-u_{t-1}},
$$

$$
u_{t+1}
=
u_t-
\frac{\widehat S(Y,u_t)}{\widehat H_t}.
$$

Support clipping, step clipping, backtracking, KKT boundary checks, and a
score-ascent fallback make the procedure safe. It does not differentiate the
score network.

| Score | Global-grid RMSE to exact MLE | Quasi-Newton RMSE | Mean quasi-Newton iterations |
|---|---:|---:|---:|
| Linear | 0.028969 | 0.028971 | 2.098 |
| Radial | 0.025141 | 0.025141 | 2.095 |

The quasi-Newton root agrees with the global learned-score root to about
$10^{-5}$ on average, with zero boundary/status mismatches. Thus the learned
field is usable in the score-only deployment mode proposed by FSM-MLE.

### 10.3 Deterministic posterior integration

The learned score field is integrated in $u$ and combined with the uniform
prior on $\pi$, including the logit Jacobian.

| Method | Posterior-mean RMSE | Posterior SD | 90% coverage | $W_1$ to exact |
|---|---:|---:|---:|---:|
| Exact likelihood | 0.08606 | 0.08230 | 0.9000 | 0 |
| Smoothed oracle | 0.08631 | 0.08552 | 0.9125 | 0.00318 |
| Linear field | 0.08772 | 0.09146 | 0.9500 | 0.02156 |
| Radial field | 0.08592 | 0.08962 | 0.9475 | 0.01781 |

Radial reduces $W_1$ to exact by 17.4%, wins all five checkpoints, and gives
paired $p=0.00169$. It also reduces the absolute posterior-SD error by 19.4%.

### 10.4 ULA

ULA samples from the score-integrated surrogate posterior using the learned
likelihood score plus the prior score.

| Method | Sample-mean RMSE | 90% coverage | $W_1$ sampler to surrogate | $W_1$ sampler to exact | Mean $\widehat R$ | Max $\widehat R$ |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.08762 | 0.950 | 0.00453 | 0.02246 | 1.0081 | 1.0332 |
| Radial | 0.08601 | 0.950 | 0.00481 | 0.01912 | 1.0076 | 1.0280 |

Radial reduces ULA-to-exact $W_1$ by 14.9% with paired $p=0.00140$. The
sampler-only error remains much smaller than the learned-field error.

### 10.5 Prior-predictive SBC

Each of the five Stage-1 checkpoints is evaluated on the same 500
prior-predictive datasets.

| Method | PIT mean | 90% coverage | Mean $W_1$ to exact |
|---|---:|---:|---:|
| Exact likelihood | 0.5283 | 0.9120 | 0 |
| Linear field | 0.5222 | 0.9232 | 0.02060 |
| Radial field | 0.5229 | 0.9252 | 0.01701 |

Radial reduces SBC mean $W_1$ by 17.5%. It is better for all five training
seeds, with paired $p=1.46\times10^{-4}$. Both learned posteriors have coverage
close to the exact reference.

---

## 11. Mode B: marginal-pilot plus frozen score

For deployable amortized SBI, a data-only marginal composite pilot is computed
on a 201-point grid:

$$
\widehat u(Y)
=
\operatorname*{argroot}_{u}
C_{\mathrm{marg}}(Y,u).
$$

The contexts are

$$
T_{\mathrm{pilot}}(Y)=\widehat u(Y),
$$

$$
T_{\mathrm{lin}}(Y)
=
\left(
\widehat u(Y),
\widehat S_{\mathrm{lin}}(Y,\widehat u(Y))
\right),
$$

and

$$
T_{\mathrm{rad}}(Y)
=
\left(
\widehat u(Y),
\widehat S_{\mathrm{rad}}(Y,\widehat u(Y))
\right).
$$

The true simulation parameter is used only as the posterior-model target and
never appears in the context.

### 11.1 Matched Stage-2 setting

- one frozen Stage-1 checkpoint, seed 20260709;
- 50,000 shared simulations;
- prior $\pi\sim\operatorname{Uniform}(0.05,0.70)$;
- test $\pi\in\{0.07,0.10,0.30,0.50,0.65,0.68\}$;
- ten paired datasets per test value;
- 5,000 posterior samples per method and dataset;
- exact likelihood-grid posterior for evaluation;
- identical cached contexts for NPE and diffusion.

### 11.2 NPE and diffusion results

| Backend | Context | Mean $W_1$ to exact | RMSE posterior mean to exact | MSE to true $\pi$ | Mean posterior SD | 90% coverage |
|---|---|---:|---:|---:|---:|---:|
| NPE | Pilot only | 0.05816 | 0.07081 | 0.01268 | 0.10261 | 0.867 |
| NPE | Linear pilot + score | 0.01557 | 0.01940 | 0.00564 | 0.07805 | 0.933 |
| NPE | Radial pilot + score | **0.01377** | **0.01736** | **0.00527** | 0.07614 | 0.967 |
| Diffusion | Pilot only | 0.06351 | 0.07403 | 0.01349 | 0.11211 | 0.900 |
| Diffusion | Linear pilot + score | **0.01693** | 0.02171 | **0.00579** | 0.08031 | 0.950 |
| Diffusion | Radial pilot + score | 0.01756 | **0.02142** | 0.00581 | 0.08027 | 0.967 |

For NPE, radial reduces $W_1$ relative to linear by 11.55%, wins 40/60 paired
datasets, and gives $p=0.0060$. For diffusion, radial and linear are tied:
radial is 3.71% worse in pooled W1, with $p=0.436$.

The backend-robust result is that adding either learned score to the pilot
reduces W1 by about 72--73% relative to pilot only. The finer radial-versus-
linear ranking is backend-sensitive.

---

## 12. Separate fixed-center score-calibration diagnostic

This section is a supplementary experiment, not the amortized five-seed run
above. It uses a fixed center $\pi_0=0.3$, $\tau=1.0$, $K=40$, $m=20$,
$\sigma_q=0.15$, and one Stage-1 training seed. It asks whether a downstream
task that consumes the numerical score scale, rather than only a one-
dimensional summary ordering, benefits from the radial map.

### 12.1 Information diagnostic

| Method | Relative efficiency |
|---|---:|
| Marginal-only linear DeepSets | 0.603 |
| Marginal-only radial DeepSets | 0.609 |
| Marginal+pairwise linear DeepSets | 0.833 |
| Marginal+pairwise radial DeepSets | **0.885** |
| Smoothed-score oracle | 0.983 |

The marginal-only methods plateau close to the precomputed information ceiling
near 0.62. Adding the pairwise channel breaks that ceiling. The radial map then
improves numerical score quality within the two-channel family.

### 12.2 One-step estimation without post-hoc calibration

| Method | RMSE | 90% coverage |
|---|---:|---:|
| Smoothed-score oracle | 0.1090 | 0.906 |
| Radial marginal+pairwise DeepSets | **0.1127** | 0.916 |
| Linear marginal+pairwise ridge | 0.1325 | 0.886 |
| Linear marginal+pairwise DeepSets | 0.1493 | 0.878 |

This result shows why direct score consumers are useful: NPE can be insensitive
to invertible rescalings of a scalar context, while a one-step estimator or
Fisher-information estimate needs the score's numerical calibration to be
correct.

Because this diagnostic uses a single fixed-center checkpoint, it should be
reported as supporting evidence rather than merged with the formal amortized
five-seed tables.

---

## 13. What the current evidence supports

The current evidence supports four distinct statements:

1. **Pairwise information matters.** Marginal-only methods exhibit the
   predicted information ceiling, while marginal+pairwise methods exceed it.
2. **The signed positive inner map improves score approximation.** In the
   formal amortized run it reduces exact-score std-MSE by 24.5% on average and
   wins 28/30 seed-anchor comparisons.
3. **The improved field helps direct score consumers.** Radial improves exact-
   MLE distance, integrated-posterior W1, ULA-to-exact W1, and prior-predictive
   SBC W1 across five checkpoints.
4. **Pilot-plus-score SBI is successful, but the fine ranking depends on the
   posterior backend.** NPE gives a modest significant radial advantage;
   conditional diffusion gives a tie.

The comparison is not "nonlinear network versus linear network." Both primary
methods use the same nonlinear anchor-conditioned DeepSets head. The formal
comparison isolates the learned nonlinear inner subscore map.

---

## 14. Limitations

### 14.1 Mode-B network replication

The formal Stage-1 and Mode-A results use five independent checkpoints. The
50k NPE/diffusion comparison uses one Stage-1 checkpoint and one fit per
backend. Multiple posterior-network seeds are still needed for a backend-level
architecture claim.

### 14.2 Fixed smoothing width

The current amortized field uses fixed $\sigma_q=0.15$ across anchors. This is
simple and preserves a coherent smoothed likelihood field, but it is not
guaranteed to be optimal at every anchor.

### 14.3 Smoothing versus exact-score error

FSM targets a smoothed score. The selected width makes the oracle smoothing
floor small, but it is not zero. Results should distinguish approximation
error to the smoothed target from smoothing bias relative to the exact score.

### 14.4 Backend sensitivity

The radial-vs-linear difference is positive under NPE and absent under the
current diffusion configuration. This should be reported directly rather than
selecting only the favorable backend.

### 14.5 Separate roles of channel and nonlinearity

The pairwise-channel result and the radial-inner-map result answer different
questions. Pairwise versus marginal tests information availability; radial
versus linear tests representation and calibration after the channel is
available.

---

## 15. Code and artifact map

### 15.1 Core Stage-1 implementation

- `run_blockwise_common_factor_amortized_fsm_experiment.py`
- `model2_amortized_score_runtime.py`
- `summarize_amortized_fixed_sigma_results.py`

### 15.2 Formal Stage-1 results

- `runs/formal40k_fixed_sigma_seed20260709/`
- `runs/formal40k_fixed_sigma_seed20260710/`
- `runs/formal40k_fixed_sigma_seed20260711/`
- `runs/formal40k_fixed_sigma_seed20260712/`
- `runs/formal40k_fixed_sigma_seed20260713/`
- `runs/formal40k_fixed_sigma_summary.md`

### 15.3 Mode-A consumers

- `run_model2_mode_a_mle.py`
- `run_model2_mode_a_posterior_grid.py`
- `run_model2_mode_a_ula.py`
- `run_model2_mode_a_sbc.py`
- `run_model2_mode_a_quasi_newton_formal.sh`
- `model2_mode_a_quasi_newton_results_20260710.md`

### 15.4 Mode-B consumers

- `run_model2_pilot_score_sbi_npe.py`
- `run_model2_pilot_score_torch_diffusion.py`
- `model2_marginal_pilot_50k_npe_vs_diffusion_20260711.md`

### 15.5 Supplementary score-consuming diagnostic

- `run_model2_score_consuming_tasks.py`
- `autodl_results/model2_score_consuming_20260706/`

---

## 16. Current bottom line

Model 2 supplies the clearest structural argument in the current project.
Marginal subscores cannot recover the common-factor cross-products; same-block
pairwise subscores add the missing channel; a signed positive nonlinear inner
map then improves score calibration within the complete two-channel
representation.

In the formal five-seed amortized run, radial lowers exact-score standardized
MSE by 24.5%, integrated-posterior $W_1$ by 17.4%, ULA-to-exact $W_1$ by 14.9%,
and prior-predictive SBC $W_1$ by about 17.5%. It also lowers quasi-Newton RMSE
to the exact MLE by 13.2%.

The deployable 50k Mode-B result confirms that a data-only pilot plus either
learned score is much better than the pilot alone. NPE retains an 11.6% radial
advantage; the present conditional diffusion fit does not. The careful summary
is therefore:

> Model 2 validates the pairwise-channel theory and shows a replicated
> amortized score-approximation benefit from the signed positive inner map.
> That benefit transfers robustly to direct score consumers and to NPE, while
> its size in generic posterior networks remains backend-dependent.
