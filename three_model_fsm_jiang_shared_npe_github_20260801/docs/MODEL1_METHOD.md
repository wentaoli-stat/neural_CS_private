# Model 1: Latest Method and Experiment Specification

Date: 2026-07-24

## 1. Scope

This document freezes the latest Model 1 experiment used in the
shared-posterior comparison. It describes:

1. the blockwise mean-shift data-generating process;
2. the proposed linear-inner-map and nonlinear local-gate score estimators;
3. likelihood-free Direct Fisher score matching (Direct-FSM) training;
4. the data-only pilot and shared neural posterior estimator (NPE);
5. the Jiang Round-1 comparison arm;
6. the exact-posterior evaluation protocol and current results.

The 20 outer blocks are iid and unordered, so the full-dataset score is aggregated with a permutation-invariant DeepSets sum. 没有保留不同blocks之间的顺序信息.

## 2. Statistical model

The unknown scalar parameter is the mixture probability
\[
p\in[0.05,0.70],
\qquad
u=\operatorname{logit}(p).
\]

因为要用Gaussian local proposal perturb,为了保证产生合法参数, map到logit空间, u属于R

One complete dataset contains \(K=20\) iid blocks. Each block contains
\(m=20\) scalar coordinates:
\[
B_k\sim\operatorname{Bernoulli}(p),
\]

\[
Y_{kj}=\tau B_k+\varepsilon_{kj},
\qquad
\tau=0.5,
\qquad
\varepsilon_{kj}\overset{\text{iid}}{\sim}N(0,1).
\]

Therefore:

- different blocks \(Y_k\) are iid;
- the 20 coordinates within a block share \(B_k\) and are dependent after
  marginalizing \(B_k\);
- one complete observation has shape \(20\times20\).

If \(B_k=0\), then

\[
Y_{kj}\mid B_k=0
\overset{\mathrm{iid}}{\sim}
\mathcal N(0,1),
\qquad j=1,\ldots,m.
\]

If \(B_k=1\), then

\[
Y_{kj}\mid B_k=1
\overset{\mathrm{iid}}{\sim}
\mathcal N(\tau,1),
\qquad j=1,\ldots,m.
\]

Thus, conditional on the block-level latent variable \(B_k\), the observations within block \(k\) are independent and identically distributed. After marginalizing out \(B_k\), however, the observations within the same block are generally dependent because they share the same latent variable.





## 3. Analytic local composite-score input

For one coordinate, the active-versus-inactive marginal log likelihood ratio
is

\[
\ell_{kj}
=
\log\frac{f_1(Y_{kj})}{f_0(Y_{kj})}
=
\tau Y_{kj}-\frac{\tau^2}{2}.
\]

At a candidate anchor \(u_a\), with \(p_a=\sigma(u_a)\), the coordinate-level
marginal score in the logit coordinate is

\[
s_{kj}(u_a)
=
\sigma(u_a+\ell_{kj})-p_a.
\]

The proposed models receive the standardized array

\[
\left\{s_{kj}(u_a):
k=1,\ldots,20,\ j=1,\ldots,20\right\},
\]

not the raw \(Y\) array.

These features are analytic **marginal composite subscores**. They are not the
exact block score. The exact block likelihood ratio would use

\[
\ell_k^{\mathrm{block}}
=
\tau\sum_{j=1}^{20}Y_{kj}
-\frac{20\tau^2}{2},
\]

which is used only for held-out diagnostics and the exact posterior reference.
The true full likelihood, true full score, latent \(B_k\), and generating test
parameter are not inputs to Stage 1 or inference.

## 4. Linear-inner-map ablation

After global feature standardization, the linear-inner-map arm forms one
summary per block:

\[
C_k(u_a)
=
\frac{1}{20}\sum_{j=1}^{20}s_{kj}^{\mathrm{std}}(u_a).
\]

The block summary and standardized anchor are passed through a shared MLP
\(\rho_\psi\), and the 20 contributions are summed:

\[
\widehat S_{\mathrm{linear}}(Y,u_a)
=
\sum_{k=1}^{20}
\rho_\psi\!\left(C_k(u_a),\widetilde u_a\right).
\]

The readout has:

- input dimension 2;
- two hidden layers;
- hidden width 64;
- SiLU activations;
- one scalar output per block;
- a DeepSets sum over the 20 iid blocks.

The name `linear` refers only to the absence of a nonlinear transformation
between the local subscore and its within-block average. The complete network
is not a globally linear function because \(\rho_\psi\) is an MLP.

The production checkpoint name is `model_linear.pt`.

## 5. Proposed positive nonlinear local gate

The proposed arm inserts an anchor-conditioned positive multiplier before
within-block aggregation:

\[
\widetilde s_{kj}(u_a)
=
s_{kj}^{\mathrm{std}}(u_a)
\,
g_\eta\!\left(
s_{kj}^{\mathrm{std}}(u_a),
\widetilde u_a
\right),
\]

where

\[
g_\eta(\cdot)>0.
\]

The multiplier is a two-hidden-layer SiLU MLP with hidden width 16 and a
softplus output. Positivity preserves the sign of the original local score
while allowing nonlinear, anchor-dependent magnitude reweighting.

The gated block statistic is

\[
\widetilde C_k(u_a)
=
\frac{1}{20}\sum_{j=1}^{20}\widetilde s_{kj}(u_a),
\]

followed by the same form of DeepSets readout:

\[
\widehat S_{\mathrm{nonlinear}}(Y,u_a)
=
\sum_{k=1}^{20}
\rho_\psi\!\left(\widetilde C_k(u_a),\widetilde u_a\right).
\]

The implementation calls this arm `radial` for historical reasons. In current
reports it is labelled `ours nonlinear-gate`.

### Strict nesting and training schedule

The nonlinear arm is initialized exactly at the trained linear arm:

1. the gate output is initialized to one;
2. the trained linear DeepSets readout is copied into the nonlinear arm;
3. the initial nonlinear prediction is numerically checked against the
   linear prediction;
4. the gate alone is trained for 400 steps;
5. the gate and readout are then jointly fine-tuned.

Thus the nonlinear experiment begins from the linear solution rather than
from an unrelated random model.


\[
\begin{aligned}
\text{Linear:}\qquad
Y_{kj}
&\longrightarrow
\ell_{kj}
\longrightarrow
s_{kj}(u_a)
\longrightarrow
\text{within-block averaging}
\longrightarrow
\text{DeepSets over blocks}
\longrightarrow
\widehat S_{\mathrm{linear}}(Y,u_a),
\\[1em]
\text{Nonlinear:}\qquad
Y_{kj}
&\longrightarrow
\ell_{kj}
\longrightarrow
s_{kj}(u_a)
\longrightarrow
\text{positive nonlinear gate}
\longrightarrow
\text{within-block aggregation}
\longrightarrow
\text{DeepSets over blocks}
\longrightarrow
\widehat S_{\mathrm{nonlinear}}(Y,u_a).
\end{aligned}
\]




## 6. Stage-1 Direct-FSM training

The learned score field is globally amortized over anchors. For every Stage-1
row:

\[
p_a\sim\text{stratified }U[0.05,0.70],
\qquad
u_a=\operatorname{logit}(p_a),
\]

\[
u=u_a+\sigma_q\epsilon,
\qquad
\epsilon\sim N(0,1),
\qquad
\sigma_q=0.2,
\]

\[
Y\sim P_u.
\]

Because the perturbation is an untruncated Gaussian in \(u\), the valid
Direct-FSM regression target is

\[
T(u,u_a)
=
\frac{u-u_a}{\sigma_q^2}.
\]

Both arms minimize

\[
\mathcal L_{\mathrm{FSM}}
=
\mathbb E\left[
\left\{
\widehat S(Y,u_a)
-
\frac{u-u_a}{\sigma_q^2}
\right\}^2
\right].
\]

This is a likelihood-free proposal-target regression. The exact score is not
a training label.

### Formal Stage-1 configuration

| Quantity | Value |
|---|---:|
| Training complete datasets | 40,000 |
| Validation complete datasets | 8,000 |
| Diagnostic test datasets | 5,000 |
| Blocks per dataset | 20 |
| Coordinates per block | 20 |
| Mean shift \(\tau\) | 0.5(既不太容易、也不接近不可识别的中等难度setting) |
| Anchor support | \([0.05,0.70]\) |
| Proposal SD in \(u\) | 0.2 |
| Main hidden width/depth | 64 / 2 |
| Gate hidden width | 16 |
| Batch size | 512 |
| Maximum optimization steps | 3,000 |
| Linear learning rate | \(3\times10^{-4}\) |
| Gate and joint-readout learning rates | \(10^{-4}\) |
| Weight decay | \(10^{-3}\) |
| Gradient clipping | 5.0 |
| EMA decay | 0.995 |
| Early-stop patience | 20 |
| Stage-1 seeds | 20260709–20260713 |

Each complete Stage-1 dataset contains 20 raw blocks, so one 40,000-row fit
uses approximately 800,000 simulated raw blocks before validation.

40000个complete datasets,每个dataset 有 20个blocks, 每个block有20个measures

### Losses that are not used

The proposed Model 1 Stage-1 objective does **not** contain:

- Jiang's curvature/Fisher penalty;
- Jiang's post-training conditional mean-zero debias regression;
- a supervised true-score loss;
- an exact-likelihood loss.

Those components belong to the Jiang comparison arm, not the proposed
Direct-FSM estimator.

## 7. Data-only pilot

For each complete dataset \(D\), the pilot is computed without the generating
parameter. On a 201-point grid over \(p\in[0.05,0.70]\), the implementation
evaluates the marginal-composite estimating equation

\[
G(D,u)
=
\sum_{k=1}^{20}
\frac{1}{20}
\sum_{j=1}^{20}
s_{kj}(u).
\]

The constrained grid root/mode is denoted

\[
\widehat u_{\mathrm{pilot}}(D).
\]

The same pilot is used by pilot-only, Jiang, linear, and nonlinear NPE
arms.

\(\boxed{ \text{不使用未知真值或oracle信息，只使用observed data和已知local model。} }\)

## 8. Frozen score context and Stage-2 NPE

The Stage-1 network is frozen and evaluated at the dataset-specific pilot:

\[
\widehat S\!\left(
D,\widehat u_{\mathrm{pilot}}(D)
\right).
\]

Every score-based method receives the two-dimensional context

\[
c(D)
=
\left(
\widehat u_{\mathrm{pilot}}(D),
\widehat S(D,\widehat u_{\mathrm{pilot}}(D))
\right).
\]

The pilot-only control receives only

\[
c_{\mathrm{pilot}}(D)
=
\widehat u_{\mathrm{pilot}}(D).
\]

The NPE learns

\[
q_\psi(p\mid c(D))
\]

from simulated complete datasets.

### Formal Stage-2 configuration

| Quantity | Value |
|---|---:|
| Stage-2 training complete datasets | 50,000 |
| Stage-2 parameter distribution | stratified \(U[0.05,0.70]\) |
| Held-out complete datasets | 100 |
| NPE family | MDN |
| Hidden features | 64 |
| Mixture components | 8 |
| Batch size | 256 |
| Learning rate | \(5\times10^{-4}\) |
| Maximum epochs | 300 |
| Validation fraction | 0.10 |
| Early-stop patience | 20 |
| Shared NPE seed | 54000 |
| Posterior samples per test dataset | 5,000 |
| Exact posterior grid points | 5,000 |

All methods use the same Stage-2 simulations, test datasets, pilot, NPE
architecture, optimizer settings, and NPE seed.

## 9. Comparison arms

The latest Model 1 posterior table contains five families.

### 9.1 Pilot-only

\[
\widehat u_{\mathrm{pilot}}(D)\rightarrow\text{shared NPE}.
\]

This measures how much posterior information is already present in the common
pilot.

### 9.2 Ours: linear-inner-map

\[
\left(
\widehat u_{\mathrm{pilot}},
\widehat S_{\mathrm{linear}}(D,\widehat u_{\mathrm{pilot}})
\right)
\rightarrow\text{shared NPE}.
\]

### 9.3 Ours: positive nonlinear local gate

\[
\left(
\widehat u_{\mathrm{pilot}},
\widehat S_{\mathrm{nonlinear}}(D,\widehat u_{\mathrm{pilot}})
\right)
\rightarrow\text{shared NPE}.
\]

### 9.4 Jiang official Round 1

Jiang's official ELU MLP learns one debiased score per raw iid block. The
complete-dataset score sums all 20 block scores, converts the result to the
common logit coordinate, and passes it to the same NPE.

Jiang Round 2 is excluded from this amortized table because Round 2 retrains
around a particular observed dataset. The label is therefore explicitly
`Jiang R1 + shared NPE`, not Jiang's original end-to-end
Round-1/round-2/root/confidence-set procedure.

## 10. Exact posterior evaluation

For held-out evaluation only, the exact block likelihood is available:

\[
p(Y_k\mid p)
=
(1-p)f_0(Y_k)+pf_1(Y_k).
\]

Under the uniform prior on \([0.05,0.70]\), the reference posterior is

\[
p(p\mid D)
\propto
\prod_{k=1}^{20}
\left\{(1-p)f_0(Y_k)+pf_1(Y_k)\right\}.
\]

The exact grid posterior is never used for Stage-1 training, pilot
construction, NPE context construction, or NPE optimization. It is used only
to calculate held-out posterior errors.

The headline quantity is

\[
\sqrt{
\frac1{100}
\sum_{r=1}^{100}
\left(
\widehat{\mathbb E}[p\mid D_r]
-
\mathbb E_{\mathrm{exact}}[p\mid D_r]
\right)^2
}.
\]

It is not error relative to the simulated parameter.

## 11. Latest results

All metrics are lower-is-better. Uncertainty is the standard deviation across
five frozen Stage-1 fits for the proposed arms. Jiang currently has one
Stage-1 fit.

| Method | Posterior-mean RMSE | Mean MAE | Posterior-SD MAE | 90% endpoint MAE | Width MAE | W1 |
|---|---:|---:|---:|---:|---:|---:|
| Pilot only | 0.03740 | 0.02941 | 0.00788 | 0.02606 | 0.02711 | 0.02981 |
| Jiang official R1 | 0.03014 | 0.02468 | 0.00738 | 0.02225 | 0.02542 | 0.02523 |
| Ours linear | 0.01642 ± 0.00038 | 0.01277 ± 0.00004 | 0.00491 ± 0.00026 | 0.01260 ± 0.00032 | 0.01667 ± 0.00070 | 0.01359 ± 0.00010 |
| **Ours nonlinear** | **0.01482 ± 0.00132** | **0.01135 ± 0.00098** | **0.00479 ± 0.00024** | **0.01173 ± 0.00079** | **0.01593 ± 0.00084** | **0.01237 ± 0.00093** |

The nonlinear arm reduces posterior-mean RMSE by:

- 9.7% relative to the matched linear-inner-map ablation;
- 50.8% relative to Jiang Round 1 plus the same NPE.

The current ordering is

```text
ours nonlinear > ours linear > Jiang R1 > pilot only,
```

where `>` means closer to the exact posterior on these held-out Model 1
datasets.

## 12. Fairness and limitations

The shared-posterior comparison controls the following:

- identical Stage-2 simulated datasets;
- identical held-out datasets;
- identical data-only pilot;
- identical context dimension for every score arm;
- identical NPE architecture, optimizer, and seed;
- identical exact-posterior reference.

It does not control every possible dimension:

- the raw Stage-1 simulator budgets are not exactly equal across the proposed
  and Jiang training procedures;
- the proposed arms have five Stage-1 seeds, while Jiang currently has one;
- the proposed method uses known analytic marginal likelihood-ratio
  components, whereas Jiang receives raw blocks;
- Jiang Round 2 is not part of the amortized NPE table.

The supported claim is therefore:

> On Model 1, under a common frozen-score, pilot-plus-score NPE protocol and
> exact-posterior evaluation, the positive nonlinear composite-score context
> is more informative than the matched no-gate context and one Jiang Round-1
> score adaptation.

It is not a universal same-budget dominance claim over the complete original
Jiang workflow.

## 13. Code map

### Proposed Stage 1

- `code/stage1/run_blockwise_mean_shift_amortized_fsm_experiment.py`  
  Simulator, analytic marginal subscores, linear and nonlinear architectures,
  Direct-FSM training, strict nested initialization, and frozen checkpoints.

- `code/stage1/model1_amortized_score_runtime.py`  
  Differentiable evaluation of a frozen score field at arbitrary candidate
  anchors.

- `code/stage1/run_model1_pilot_score_sbi_npe.py`  
  Model 1 data-only pilot and pilot-plus-score context construction.

- `code/stage1/run_model1_mode_a_mle.py` and
  `code/stage1/model1_mode_a_reference.py`  
  Constrained grid-root utilities and evaluation-only Mode-A references used
  by the pilot code.

- `code/stage1/run_blockwise_mean_shift_two_stage_sbi_npe_experiment.py` and
  `code/stage1/run_blockwise_mean_shift_fsm_experiment.py`  
  Shared scalar NPE and legacy fixed-anchor utilities imported by the
  production pilot entrypoint.

### Shared comparison

- `code/khoo_original/`  
  Thin read-only import adapter for Khoo et al.'s original sequential
  FSM-MLE Algorithm 1. It refits a fresh local affine score at every parameter
  iteration and does not use a cached grid, interpolation, or NPE. No formal
  Model 1 result from this runner is included in the current table.

- `code/khoo_vs_jiang/block_mixture_stage2_npe.py`  
  Matched Stage-2 datasets, shared contexts, NPE training, exact posterior, and
  posterior metrics.

- `code/khoo_vs_jiang/block_mixture_jiang.py`  
  Model 1 raw-block simulator and paper-native Jiang experiment adapter.

- `code/khoo_vs_jiang/jiang_official.py`  
  Frozen official Jiang training/inference adapter.

- `code/khoo_vs_jiang/block_mixture_compare.py`  
  Frozen Stage-1 checkpoint selection and paired score/root diagnostics.

- `code/shared_npe_backend.py`  
  Focused copy of the generic scalar NPE training, sampling, and Wasserstein
  utilities used by the formal run.

### Frozen configurations and summaries

- `configs/stage1/`: the five exact Stage-1 JSON configurations;
- `configs/stage2/`: the formal shared-NPE protocol;
- `results/shared_npe/`: pilot, Jiang, linear, and nonlinear summaries;
- `tests/`: focused model, protocol, and fidelity tests.





|       | best epoch | Energy-W2 ↓         | Torus-W2 ↓          | TICA-W2 ↓           | ESS ↑                 |
| ----- | ---------- | ------------------- | ------------------- | ------------------- | --------------------- |
| Ala-2 | ours e299  | **0.4849 ± 0.0925** | **0.4393 ± 0.0188** | **1.0104 ± 0.0718** | **0.02052 ± 0.00243** |
|       | plain e299 | 0.9157 ± 0.1105     | 0.5059 ± 0.0685     | 1.0588 ± 0.0542     | 0.01762 ± 0.00103     |
| Ala-3 | ours e399  | **0.2872 ± 0.0816** | **0.5135 ± 0.0103** | 0.1614 ± 0.0019     | **0.03368 ± 0.00159** |
|       | plain e399 | 0.4700 ± 0.0856     | 0.5843 ± 0.0189     | **0.1606 ± 0.0008** | 0.02725 ± 0.00182     |
|       |            |                     |                     |                     |                       |
|       |            |                     | —                   | —                   | —                     |
