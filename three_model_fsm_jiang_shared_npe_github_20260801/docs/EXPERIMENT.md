# 三个模型上的 Ours Linear / Ours Nonlinear / Jiang R1 + Shared NPE

日期：2026-07-25

## 1. 文档目的

本文统一记录三个一维参数实验：

1. Model 1：blockwise mean-shift mixture；
2. Model 2：blockwise common-factor variance mixture；
3. Model 3：nonlinear-emission HMM。

每个模型都比较三条主要方法臂：

1. `Ours linear + shared NPE`；
2. `Ours nonlinear + shared NPE`；
3. `Jiang official Round 1 score + shared NPE`。

比较分成两个阶段：

- **NPE 之前**：比较冻结 score estimator 与 exact full score 的
  standardized MSE；
- **NPE 之后**：将各方法的冻结 score 放入相同的
  pilot-plus-score NPE，比较 learned posterior 与 exact posterior。

本文不把 Jiang Round 2 放入 shared-NPE 表。Jiang Round 2 的 proposal
依赖某一个固定 observed dataset，不能作为一个 observation-independent
的 frozen amortized score extractor 复用于所有测试数据集。

---

## 2. 统一的比较对象

### 2.1 参数坐标

三个模型都在 logit 坐标中比较 score：

$$
u=\operatorname{logit}(p)
=\log\frac{p}{1-p},
\qquad
p=\sigma(u).
$$

如果 Jiang 网络输出 physical-$p$ score，则转换为

$$
\widehat S_{J,u}(D,p)
=
p(1-p)\widehat S_{J,p}(D,p).
$$

因此所有 score-MSE 表比较的都是同一个 $u$ -score estimand。

### 2.2 Ours 的 Direct-FSM Stage 1

三个模型中的 proposed score estimator 都使用 anchored Direct Fisher
score matching。对每个 Stage-1 training row：
$$
u_a=\operatorname{logit}(p_a),
\qquad
u\mid u_a\sim N(u_a,\sigma_q^2),
\qquad
D\sim P_{\sigma(u)}.
$$

likelihood-free regression target 为

$$
T_{\mathrm{FSM}}(u,u_a)
=
\frac{u-u_a}{\sigma_q^2}.
$$

两条 proposed arms 都最小化

$$
\mathcal L_{\mathrm{FSM}}
=
\mathbb E
\left[
\left\{
\widehat S(D,u_a)
-
\frac{u-u_a}{\sigma_q^2}
\right\}^2
\right].
$$

训练不使用：

- exact full likelihood；
- exact full score；
- test dataset 的生成参数；
- latent states；
- Jiang 的 curvature/Fisher penalty；
- Jiang 的 conditional mean-zero debias regression。

有限 $\sigma_q$ 下，Direct-FSM 的 population target 是 Gaussian-smoothed
likelihood score。报告的 exact-score MSE 同时包含 smoothing bias、
representation error 和 neural approximation error。Linear 与 nonlinear
使用相同 $\sigma_q$，所以它们之间仍是 matched ablation。

\(\sigma_q\) 太大：训练比较稳定，但学到的是较强平滑后的 score, \(\sigma_q\) 太小：smoothed score 更接近 exact score, 单噪声大

暂时还没关于sigma q调参

### 2.3 Shared pilot-plus-score NPE

对一个完整数据对象 $D$，先只用数据计算公共 pilot：

$$
\widehat u_{\mathrm{pilot}}(D).
$$

然后冻结 Stage-1 score network，在 pilot 处计算：

$$
\widehat S_m
\left(
D,\widehat u_{\mathrm{pilot}}(D)
\right),
$$

其中 $m$ 表示 linear、nonlinear 或 Jiang R1。NPE context 统一为

$$
c_m(D)
=
\left(
\widehat u_{\mathrm{pilot}}(D),
\widehat S_m
\left(D,\widehat u_{\mathrm{pilot}}(D)\right)
\right).
$$

NPE 学习

$$
q_{\psi,m}(p\mid c_m(D)).
$$

在同一个模型内部，所有方法使用相同的：

- Stage-2 simulated parameters；
- Stage-2 simulated datasets/trajectories；
- data-only pilot；
- context dimension；
- NPE family 和 architecture；
- optimizer、split、training seed；
- held-out test objects；
- posterior sampling seed；
- exact-posterior evaluation reference。

模拟参数 $p$ 是 NPE 的 supervised target，但从不进入 inference context。
这与标准 amortized simulation-based inference 一致。

### 2.4 两类指标

#### Pre-NPE exact-score standardized MSE

在固定参数点 $p$ 上：

$$
\operatorname{stdMSE}(p)
=
\frac{
\mathbb E
\left[
\left\{
\widehat S_u(D,p)-S_u^*(D,p)
\right\}^2
\right]
}{
\operatorname{Var}\{S_u^*(D,p)\}
}.
$$

数值越小，表示冻结 score estimator 越接近 exact score。Exact score
只在训练完成后的 held-out diagnostic 中使用。

#### Post-NPE posterior metrics

对每个 held-out $D_r$，使用 exact posterior
$\pi_{\mathrm{exact}}(p\mid D_r)$ 作为 reference。主要指标包括：

- posterior-mean RMSE；
- posterior-mean MAE；
- posterior-SD MAE；
- 90% interval endpoint MAE；
- 90% interval-width MAE；
- Wasserstein-1 distance。

例如 posterior-mean RMSE 为

$$
\sqrt{
\frac1R
\sum_{r=1}^R
\left[
\widehat{\mathbb E}_m(p\mid D_r)
-
\mathbb E_{\mathrm{exact}}(p\mid D_r)
\right]^2
}.
$$

它不是 posterior mean 到生成参数的 RMSE。

---

## 3. 三个模型与 observation unit 总览

| 项目 | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Unknown parameter | mixture probability $p$ | mixture probability $p$ | HMM persistence $p=p_{11}$ |
| Parameter support | $[0.05,0.70]$ | $[0.05,0.70]$ | $[0.80,0.99]$ |
| Reference truth | $0.30$ | $0.30$ | $0.94$ |
| Complete NPE object | 20 iid blocks | 40 iid blocks | one dependent $50\times10$ trajectory |
| One Jiang score unit | one length-20 block | one length-20 block | one complete trajectory, flattened to 500 |
| Ours aggregation | within block + DeepSets over blocks | marginal/pairwise within block + DeepSets | marginal/pairwise per time + GRU |
| Ours Stage-1 rows | 40,000 complete datasets | 40,000 complete datasets | 40,000 complete trajectories |
| Ours Stage-1 seeds | 5 | 5 | 3 |
| Jiang R1 fits in tables | 1 | 1 | 1 |

Model 1/2 的 outer blocks 是 iid 且无序，因此使用 permutation-invariant
DeepSets sum。Model 3 的时间点有顺序和 HMM dependence，因此使用 GRU，
不会把时间点错误地视为 iid observations。

---

# Part I. Model 1

## 4. Model 1：blockwise mean-shift mixture

### 4.1 Data-generating process

未知参数为

$$
p\in[0.05,0.70].
$$

一个完整 dataset 有 $K=20$ 个 iid blocks，每个 block 有 $m=20$
个 coordinates：

$$
B_k\sim\operatorname{Bernoulli}(p),
$$

$$
Y_{kj}
=
\tau B_k+\varepsilon_{kj},
\qquad
\tau=0.5,
\qquad
\varepsilon_{kj}\overset{\mathrm{iid}}{\sim}N(0,1).
$$

条件于 $B_k$，同一个 block 内的 coordinates iid；积分掉 $B_k$ 后，
同一个 block 内的 coordinates 因共享 $B_k$ 而相关。不同 blocks iid。

### 4.2 Analytic marginal composite subscores

单坐标 active-vs-inactive log likelihood ratio 为

$$
\ell_{kj}
=
\log\frac{f_1(Y_{kj})}{f_0(Y_{kj})}
=
\tau Y_{kj}-\frac{\tau^2}{2}.
$$

在 candidate anchor $u_a$、$p_a=\sigma(u_a)$ 处，coordinate-level
logit subscore 为

$$
s_{kj}(u_a)
=
\sigma(u_a+\ell_{kj})-p_a.
$$

这些是 marginal composite subscores，不是 exact block score。Exact
block log likelihood ratio

$$
\ell_k^{\mathrm{block}}
=
\tau\sum_{j=1}^{20}Y_{kj}
-\frac{20\tau^2}{2}
$$

只用于 held-out exact-score 和 exact-posterior reference。

## 5. Model 1 方法实现

### 5.1 Ours linear

标准化后，先在每个 block 内做 identity aggregation：

$$
C_k(u_a)
=
\frac1{20}
\sum_{j=1}^{20}
s_{kj}^{\mathrm{std}}(u_a).
$$

block summary 与标准化 anchor 输入共享的 nonlinear readout
$\rho_\psi$：

$$
\widehat S_{\mathrm{lin}}(D,u_a)
=
\sum_{k=1}^{20}
\rho_\psi(C_k(u_a),\widetilde u_a).
$$

$\rho_\psi$ 是 width 64、两层 SiLU MLP。不同 blocks 共享同一个
readout，并通过 DeepSets sum 聚合。

这里的 `linear` 只表示 local subscore 在 block average 前没有 nonlinear
inner map；完整模型仍包含 nonlinear MLP。

### 5.2 Ours nonlinear

在 block average 前加入 anchor-conditioned positive gate：

$$
\widetilde s_{kj}(u_a)
=
s_{kj}^{\mathrm{std}}(u_a)
g_\eta
\left(
s_{kj}^{\mathrm{std}}(u_a),
\widetilde u_a
\right),
\qquad
g_\eta(\cdot)>0.
$$

gate 是 width 16、两层 SiLU MLP，最后使用 softplus，随后

$$
\widetilde C_k(u_a)
=
\frac1{20}
\sum_{j=1}^{20}\widetilde s_{kj}(u_a),
$$

$$
\widehat S_{\mathrm{nonlin}}(D,u_a)
=
\sum_{k=1}^{20}
\rho_\psi(\widetilde C_k(u_a),\widetilde u_a).
$$

nonlinear arm 严格嵌套 linear arm：

1. 先训练 linear；
2. 复制 linear readout；
3. gate 初始化为恒等 multiplier 1；
4. 前 400 steps 只训练 gate；
5. 之后联合微调 gate 与 readout。

### 5.3 Model 1 proposed Stage-1 setting

| 项目 | 设置 |
|---|---:|
| Parameter support | $p\in[0.05,0.70]$ |
| Complete training datasets | 40,000 |
| Complete validation datasets | 8,000 |
| Blocks per dataset | 20 |
| Coordinates per block | 20 |
| Signal | $\tau=0.5$ |
| Proposal SD | $\sigma_q=0.20$ in $u$ |
| DeepSets readout | width 64, depth 2, SiLU |
| Gate | width 16, depth 2, SiLU + softplus |
| Batch size | 512 |
| Maximum steps | 3,000 per arm |
| Stage-1 seeds | 20260709--20260713 |

一条 Stage-1 row 是一个完整 20-block dataset，所以每个 40,000-row
fit 使用约 800,000 个 raw blocks，未计 validation。

### 5.4 Jiang R1 + NPE

Jiang R1 的 outer iid observation 是一个 raw length-20 block。官方 ELU
MLP 输入：

$$
(p,Y_k)\in\mathbb R^{21}
$$

并输出单-block physical-$p$ score
$\widehat s_{J,p}(p,Y_k)$。完整 dataset score 为

$$
\widehat S_{J,u}(D,p)
=
p(1-p)
\sum_{k=1}^{20}
\widehat s_{J,p}(p,Y_k).
$$

该 R1 score 在 data-only pilot 处计算，然后与 pilot 一起输入 shared NPE。

Jiang R1 使用官方 repository 的：

1. coordinate-wise direct score matching；
2. joint direct score matching；
3. Fisher/Bartlett curvature penalty；
4. conditional mean-score debias regression；
5. debias-curvature continuation。

R1 proposal 为 $U[0.05,0.70]$。网络为两层 width-64 ELU MLP。核心配置为：

| 项目 | Jiang R1 |
|---|---:|
| Primary training parameters/observations | 10,000 |
| Batch size | 10 |
| Coordinate epochs | 100 |
| Joint epochs | 100 |
| Fisher epochs | 50 |
| Fisher anchors | 1,000 |
| Reference observations per anchor | 500 |
| Debias epochs | 500 |
| Debias-curvature epochs | 50 |
| Jiang Stage-1 fits in results | 1 |

Jiang 不使用我们的 composite subscores、gate 或 DeepSets。Exact score
不是 Jiang 的监督训练 target。

## 6. Model 1 pre-NPE score experiment

### 6.1 Setting

| 项目 | 设置 |
|---|---|
| Exact estimand | full-dataset $u$-score |
| Parameter points | $p=0.10,0.30,0.50,0.65$ |
| Held-out datasets per point | 1,000 |
| Dataset size | 20 iid blocks, each length 20 |
| Ours checkpoints | 5 linear + 5 nonlinear |
| Jiang checkpoints | 1 official R1 |
| Metric | exact-score standardized MSE and correlation |
| Exact-score role | evaluation only |

Ours 的结果先对相同参数点和 held-out datasets 计算，再对五个 Stage-1
fits 汇总；Jiang 使用一个 official R1 fit。

### 6.2 Result

| Method | Mean score stdMSE | Mean score correlation |
|---|---:|---:|
| Jiang official R1 | 0.4035 | 0.8858 |
| Ours linear | 0.0978 | 0.9584 |
| **Ours nonlinear** | **0.0847** | **0.9652** |

相对 Jiang R1，ours nonlinear 将 stdMSE 降低约 79.0%。相对 matched
linear，nonlinear 降低约 13.4%。

## 7. Model 1 shared-NPE posterior experiment

### 7.1 Setting

| 项目 | 设置 |
|---|---|
| Prior / Stage-2 parameter distribution | stratified $U[0.05,0.70]$ |
| Stage-2 training complete datasets | 50,000 |
| Held-out complete datasets | 100 |
| Pilot | 201-point marginal-composite grid root/mode |
| Context | $(\widehat u_{\rm pilot},\widehat S(D,\widehat u_{\rm pilot}))$ |
| NPE | MDN, hidden width 64, 8 components |
| Batch size / learning rate | 256 / $5\times10^{-4}$ |
| Maximum epochs / patience | 300 / 20 |
| Shared NPE seed | 54000 |
| Posterior draws per dataset | 5,000 |
| Exact posterior | 5,000-point likelihood grid, evaluation only |
| Ours replication | 5 Stage-1 fits |
| Jiang replication | 1 Stage-1 fit |

`±` 表示五个 frozen Stage-1 fits 之间的标准差，不是 100 个 test datasets
上的 standard error。

### 7.2 Result

| Method | Mean RMSE | Mean MAE | Posterior-SD MAE | 90% endpoint MAE | Width MAE | W1 |
|---|---:|---:|---:|---:|---:|---:|
| Jiang official R1 + NPE | 0.03014 | 0.02468 | 0.00738 | 0.02225 | 0.02542 | 0.02523 |
| Ours linear + NPE | 0.01642 ± 0.00038 | 0.01277 ± 0.00004 | 0.00491 ± 0.00026 | 0.01260 ± 0.00032 | 0.01667 ± 0.00070 | 0.01359 ± 0.00010 |
| **Ours nonlinear + NPE** | **0.01482 ± 0.00132** | **0.01135 ± 0.00098** | **0.00479 ± 0.00024** | **0.01173 ± 0.00079** | **0.01593 ± 0.00084** | **0.01237 ± 0.00093** |

Ours nonlinear 的 posterior-mean RMSE：

- 相对 ours linear 降低约 9.7%；
- 相对 Jiang R1 + NPE 降低约 50.8%。

---

# Part II. Model 2

## 8. Model 2：blockwise common-factor variance mixture

### 8.1 Data-generating process

未知参数为

$$
p\in[0.05,0.70].
$$

一个完整 dataset 有 $K=40$ 个 iid blocks，每个 block 有 $m=20$
coordinates：

$$
B_k\sim\operatorname{Bernoulli}(p).
$$

如果 $B_k=0$：

$$
Y_k\sim N(0,I_m).
$$

如果 $B_k=1$：

$$
Y_{ki}
=
\tau Z_k+\varepsilon_{ki},
\qquad
Z_k\sim N(0,1),
\qquad
\varepsilon_{ki}\overset{\mathrm{iid}}{\sim}N(0,1),
\qquad
\tau=1.
$$

因此

$$
Y_k\mid B_k=1
\sim
N\left(0,I_m+\tau^2\mathbf1_m\mathbf1_m^\top\right).
$$

Model 2 中 parameter 不改变均值，而改变同一 block 内的 covariance。
cross-products $Y_{ki}Y_{kj}$ 携带关键信息，所以只使用 marginal
coordinates 不足以恢复 full block score。

### 8.2 Marginal and pairwise composite subscores

单坐标 likelihood ratio：

$$
R_{1,ki}
=
(1+\tau^2)^{-1/2}
\exp
\left\{
\frac{\tau^2Y_{ki}^2}{2(1+\tau^2)}
\right\}.
$$

单坐标 local $u$-subscore：

$$
s_{ki}^{(1)}(u_a)
=
\sigma\left(u_a+\log R_{1,ki}\right)-p_a.
$$

同一 block 内一对 coordinates 的 likelihood ratio：

$$
R_{2,kij}
=
(1+2\tau^2)^{-1/2}
\exp
\left\{
\frac{\tau^2(Y_{ki}+Y_{kj})^2}{2(1+2\tau^2)}
\right\},
\qquad i<j.
$$

pairwise local subscore：

$$
s_{kij}^{(2)}(u_a)
=
\sigma\left(u_a+\log R_{2,kij}\right)-p_a.
$$

pairwise channel 包含

$$
(Y_{ki}+Y_{kj})^2
=
Y_{ki}^2+Y_{kj}^2+2Y_{ki}Y_{kj},
$$

因此补回 marginal channel 缺少的 cross-product information。

## 9. Model 2 方法实现

### 9.1 Ours linear

两个通道分别标准化。每个 block 的 identity-inner-map feature 为

$$
C_k
=
\left(
\frac1m\sum_i \widetilde s_{ki}^{(1)},
\frac1{\binom m2}\sum_{i<j}\widetilde s_{kij}^{(2)}
\right).
$$

完整 score 为

$$
\widehat S_{\mathrm{lin}}(D,u_a)
=
\sum_{k=1}^{40}
\rho_\psi(C_k,\widetilde u_a),
$$

其中 $\rho_\psi$ 是共享的 width-64、两层 SiLU DeepSets readout。

### 9.2 Ours nonlinear

对 marginal 和 pairwise channel 分别使用 positive gates：

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

$$
m_{\eta,r}(\cdot)>0.
$$

gated block feature 为

$$
\widetilde C_k
=
\left(
\frac1m\sum_i
\phi_{\eta,1}
\left(\widetilde s_{ki}^{(1)},\widetilde u_a\right),
\frac1{\binom m2}\sum_{i<j}
\phi_{\eta,2}
\left(\widetilde s_{kij}^{(2)},\widetilde u_a\right)
\right),
$$

$$
\widehat S_{\mathrm{nonlin}}(D,u_a)
=
\sum_{k=1}^{40}
\rho_\psi(\widetilde C_k,\widetilde u_a).
$$

两个 gate 都是 width-16 positive-multiplier MLP。Nonlinear arm 从
identity gates 和复制的 linear readout 开始，先 gate-only 训练，再联合微调。

### 9.3 Model 2 proposed Stage-1 setting

| 项目 | 设置 |
|---|---:|
| Parameter support | $p\in[0.05,0.70]$ |
| Complete training datasets | 40,000 |
| Complete validation datasets | 8,000 |
| Blocks per dataset | 40 |
| Coordinates per block | 20 |
| Signal | $\tau=1.0$ |
| Proposal SD | $\sigma_q=0.15$ in $u$ |
| DeepSets readout | width 64, depth 2, SiLU |
| Gates | two width-16 positive gates |
| Batch size | 512 |
| Maximum steps | 3,000 per arm |
| Stage-1 seeds | 20260709--20260713 |

一条 Stage-1 row 是一个完整 40-block dataset，所以每个 40,000-row
fit 使用约 1,600,000 个 raw blocks，未计 validation。

### 9.4 Jiang R1 + NPE

Jiang 仍把一个 raw length-20 block 作为一个 iid observation。官方 MLP
学习单-block physical-$p$ score，并通过

$$
\widehat S_{J,u}(D,p)
=
p(1-p)
\sum_{k=1}^{40}
\widehat s_{J,p}(p,Y_k)
$$

构造 full-dataset $u$-score。

Jiang 不把 block 内 coordinates 当成 iid observations；相关结构完整保留
在 raw 20-dimensional block input 中。R1 使用与 Model 1 相同的 official
paper profile，只把 simulator、sample size 和 observation distribution
替换为 Model 2。

## 10. Model 2 pre-NPE score experiment

### 10.1 Setting

| 项目 | 设置 |
|---|---|
| Exact estimand | full-dataset $u$-score |
| Parameter points | $p=0.10,0.30,0.50,0.65$ |
| Held-out datasets per point | 1,000 |
| Dataset size | 40 iid blocks, each length 20 |
| Ours checkpoints | 5 linear + 5 nonlinear |
| Jiang checkpoints | 1 official R1 |
| Metric | exact-score standardized MSE and correlation |
| Exact-score role | evaluation only |

### 10.2 Result

| Method | Mean score stdMSE | Mean score correlation |
|---|---:|---:|
| Jiang official R1 | 0.4541 | 0.8640 |
| Ours linear | 0.1521 | 0.9343 |
| **Ours nonlinear** | **0.1140** | **0.9478** |

相对 Jiang R1，ours nonlinear 将 stdMSE 降低约 74.9%。相对 matched
linear，nonlinear 降低约 25.0%。

## 11. Model 2 shared-NPE posterior experiment

### 11.1 Setting

| 项目 | 设置 |
|---|---|
| Prior / Stage-2 parameter distribution | stratified $U[0.05,0.70]$ |
| Stage-2 training complete datasets | 50,000 |
| Held-out complete datasets | 100 |
| Pilot | 201-point marginal-composite grid root/mode |
| Context | $(\widehat u_{\rm pilot},\widehat S(D,\widehat u_{\rm pilot}))$ |
| NPE | MDN, hidden width 64, 8 components |
| Batch size / learning rate | 256 / $5\times10^{-4}$ |
| Maximum epochs / patience | 300 / 20 |
| Shared NPE seed | 54000 |
| Posterior draws per dataset | 5,000 |
| Exact posterior | 5,000-point likelihood grid, evaluation only |
| Ours replication | 5 Stage-1 fits |
| Jiang replication | 1 Stage-1 fit |

### 11.2 Result

| Method | Mean RMSE | Mean MAE | Posterior-SD MAE | 90% endpoint MAE | Width MAE | W1 |
|---|---:|---:|---:|---:|---:|---:|
| Jiang official R1 + NPE | 0.03272 | 0.02603 | 0.00958 | 0.02587 | 0.03238 | 0.02674 |
| Ours linear + NPE | 0.02534 ± 0.00090 | 0.01992 ± 0.00053 | 0.00495 ± 0.00009 | 0.01908 ± 0.00041 | 0.01664 ± 0.00047 | 0.02039 ± 0.00053 |
| **Ours nonlinear + NPE** | **0.02189 ± 0.00092** | **0.01751 ± 0.00065** | **0.00453 ± 0.00015** | **0.01687 ± 0.00056** | **0.01521 ± 0.00051** | **0.01787 ± 0.00067** |

Ours nonlinear 的 posterior-mean RMSE：

- 相对 ours linear 降低约 13.6%；
- 相对 Jiang R1 + NPE 降低约 33.1%。

---

# Part III. Model 3

## 12. Model 3：nonlinear-emission HMM

### 12.1 Observation unit

一个 NPE observation 是一条完整 trajectory：

$$
Y=(Y_1,\ldots,Y_T),
\qquad
T=50,
\qquad
Y_t\in\mathbb R^{10}.
$$

raw shape 为 $50\times10$。时间点之间遵循 HMM dependence，不能把
500 个 raw coordinates 当成 500 个 iid observations。

### 12.2 Hidden-state transition model

$$
B_1\sim\operatorname{Bernoulli}(0.5),
$$

$$
P(p)
=
\begin{pmatrix}
1-p_{01} & p_{01}\\
1-p & p
\end{pmatrix},
\qquad
p_{01}=0.06,
\qquad
p=p_{11}\in[0.80,0.99].
$$

reference truth 为 $p^*=0.94$。

### 12.3 Nonlinear sparse-scale-mixture emission

若 $B_t=0$：

$$
Y_{tj}\mid B_t=0
\sim N(0,1).
$$

若 $B_t=1$：

$$
Y_{tj}\mid B_t=1
\sim
(1-\rho)N(0,1)+\rho N(0,a^2),
$$

其中

$$
\rho=0.20,
\qquad
a=3.
$$

给定 state 后，coordinates 条件独立。state-1 与 state-0 的单坐标
likelihood ratio 为

$$
r(y)
=
(1-\rho)
+
\frac{\rho}{a}
\exp
\left\{
\frac{y^2}{2}(1-a^{-2})
\right\},
$$

$$
\ell(y)=\log r(y).
$$

少量 tail coordinates 会产生很大的 $\ell(y)$，因此该模型专门检验
nonlinear local calibration 是否能在 aggregation 前保留强 evidence。

## 13. Model 3 方法实现

### 13.1 Local marginal and pairwise features

使用固定 feature reference

$$
\pi_{\mathrm{ref}}=0.30.
$$

它不是未知 transition parameter、真实参数或 prior。单坐标 local subscore：

$$
s_{tj}^{(1)}
=
\sigma
\left(
\operatorname{logit}(\pi_{\mathrm{ref}})
+\ell(Y_{tj})
\right)
-\pi_{\mathrm{ref}}.
$$

同一时间点的 coordinate-pair subscore：

$$
s_{tjk}^{(2)}
=
\sigma
\left(
\operatorname{logit}(\pi_{\mathrm{ref}})
+\ell(Y_{tj})+\ell(Y_{tk})
\right)
-\pi_{\mathrm{ref}},
\qquad j<k.
$$

每个时间点有 10 个 marginal 和 45 个 pairwise subscores。这里的
pairwise 是同一时间点的 cross-sectional pair，不是相邻时间 pair。

### 13.2 Ours linear

标准化后，在每个时间点先做 identity aggregation：

$$
c_{t,1}^{\mathrm{lin}}
=
\frac1{10}\sum_{j=1}^{10}z_{tj}^{(1)},
$$

$$
c_{t,2}^{\mathrm{lin}}
=
\frac1{45}\sum_{j<k}z_{tjk}^{(2)}.
$$

标准化 anchor $\widetilde u_a$ 被重复输入全部 50 个时间点：

$$
x_t
=
\left(
\widetilde c_{t,1},
\widetilde c_{t,2},
\widetilde u_a
\right).
$$

one-layer, one-direction GRU 使用 hidden width 64，取最后 hidden state，
再通过 `64 -> SiLU -> 1` head 输出一条完整 trajectory 的 score：

$$
\widehat S_{\mathrm{lin}}(Y,u_a).
$$

`linear` 仅表示 local gate 是 identity；GRU 和 final head 都是 nonlinear。

### 13.3 Ours nonlinear

两个通道分别使用 positive gates：

$$
g_{tj}^{(1)}
=
z_{tj}^{(1)}m_{\eta_1}(z_{tj}^{(1)}),
$$

$$
g_{tjk}^{(2)}
=
z_{tjk}^{(2)}m_{\eta_2}(z_{tjk}^{(2)}),
\qquad
m_{\eta_r}(z)>0.
$$

每个 gate 是

```text
1 -> 16 -> SiLU -> 16 -> SiLU -> 1 -> softplus
```

再计算

$$
c_{t,1}^{\mathrm{gate}}
=
\frac1{10}\sum_jg_{tj}^{(1)},
$$

$$
c_{t,2}^{\mathrm{gate}}
=
\frac1{45}\sum_{j<k}g_{tjk}^{(2)},
$$

并输入与 linear 相同的 GRU 和 head。

Model 3 的 gate 本身不输入 anchor；anchor dependence 由后面的 GRU
承担。该 nonlinear model 同样从复制的 linear GRU 和 identity gates
开始训练。

### 13.4 Model 3 proposed Stage-1 setting

| 项目 | 设置 |
|---|---:|
| Unknown parameter | $p_{11}$ |
| Fixed transition | $p_{01}=0.06$ |
| Parameter support | $[0.80,0.99]$ |
| Trajectory shape | $50\times10$ |
| State-1 emission | $0.8N(0,1)+0.2N(0,3^2)$ |
| Training trajectories | 40,000 |
| Validation trajectories | 8,000 |
| Proposal SD | $\sigma_q=0.25$ in $u$ |
| GRU | one layer, hidden width 64 |
| Gates | two width-16 positive gates |
| Batch size | 512 |
| Maximum steps | 3,000 per arm |
| Stage-1 seeds | 20260721, 20260722, 20260723 |

每条 Stage-1 row 是一条完整 trajectory，不是一个含 100 条 trajectory
的 frequentist dataset。

### 13.5 Jiang R1 + NPE

Jiang official MLP 将一条完整 $50\times10$ trajectory flatten 为
500-dimensional raw observation：

$$
\operatorname{vec}(Y)\in\mathbb R^{500}.
$$

网络输入为 scaled physical parameter 和 flattened trajectory，输出单轨迹
physical-$p$ score。它使用与 Model 1/2 相同的 official R1 objectives 和
paper profile：

- direct score matching；
- curvature/Fisher penalty；
- conditional mean-zero debias regression；
- width-64、两层 ELU MLP；
- 10,000 primary R1 training trajectories；
- batch size 10。

在 shared-NPE experiment 中只冻结并使用 observation-independent R1
checkpoint：

$$
\widehat S_{J,u}(Y,p)
=
p(1-p)\widehat S_{J,p}(Y,p).
$$

Jiang R2 不进入该表。

## 14. Model 3 pre-NPE score experiment

### 14.1 Setting

| 项目 | 设置 |
|---|---|
| Exact estimand | one-complete-trajectory $u$-score |
| Parameter points | $p=0.84,0.90,0.94,0.97$ |
| Held-out trajectories per point | 2,000 |
| Trajectory shape | $50\times10$ |
| Ours checkpoints | 3 linear + 3 nonlinear |
| Jiang checkpoints | 1 official MLP R1 |
| Metric | exact-score standardized MSE and correlation |
| Exact-score computation | forward-backward, evaluation only |

Model 3 的 exact HMM score 通过 forward-backward 计算，但不进入任何
Stage-1 training target 或 NPE context。

### 14.2 Result

| Method | Mean score stdMSE | Mean score correlation |
|---|---:|---:|
| Jiang official MLP R1 | 0.7976 | 0.4818 |
| Ours linear + GRU | 0.2113 | 0.8965 |
| **Ours nonlinear gate + GRU** | **0.1549** | **0.9228** |

相对 Jiang official MLP R1，ours nonlinear 将 stdMSE 降低约 80.6%。
相对 matched linear，nonlinear 降低约 26.9%，并在三个 Stage-1 seeds
上都降低 stdMSE、提高 correlation。

## 15. Model 3 shared-NPE posterior experiment

### 15.1 Data-only pilot

对每条完整 trajectory，在 201 个 $u$-grid points 上计算 adjacent-time
pairwise-composite score field

$$
C_{\mathrm{temp}}(Y,u).
$$

使用 constrained grid root/mode 得到

$$
\widehat u_{\mathrm{pilot}}(Y).
$$

pilot 只使用 observed trajectory、candidate grid 和已知模型结构，不使用
生成参数。所有 score arms 使用同一个 pilot。

### 15.2 Setting

| 项目 | 设置 |
|---|---|
| Prior / Stage-2 parameter distribution | stratified $U[0.80,0.99]$ |
| Stage-2 training trajectories | 50,000 |
| Test parameter points | $0.84,0.90,0.94,0.97$ |
| Test trajectories | 25 per point, total 100 |
| Complete NPE object | one $50\times10$ trajectory |
| Context | $(\widehat u_{\rm pilot},\widehat S(Y,\widehat u_{\rm pilot}))$ |
| NPE | MDN, hidden width 64, 5 components |
| Batch size / learning rate | 256 / $5\times10^{-4}$ |
| Maximum epochs / patience | 300 / 20 |
| Stage-2 train seed | 20260725 |
| NPE seed | 54000 |
| Posterior draws per trajectory | 5,000 |
| Exact posterior | 5,000-point forward-likelihood grid, evaluation only |
| Ours replication | 3 Stage-1 fits |
| Jiang replication | 1 Stage-1 fit |

### 15.3 Result

| Method | Mean RMSE | Mean MAE | Posterior-SD MAE | 90% endpoint MAE | Width MAE | W1 |
|---|---:|---:|---:|---:|---:|---:|
| Jiang official MLP R1 + NPE | 0.01426 | 0.01237 | 0.00352 | 0.00892 | 0.01139 | 0.01243 |
| Ours linear + NPE | 0.01213 | 0.01013 | 0.00304 | 0.00765 | 0.01020 | 0.01025 |
| **Ours nonlinear + NPE** | **0.01005** | **0.00790** | **0.00254** | **0.00611** | **0.00789** | **0.00806** |

Ours nonlinear 的 posterior-mean RMSE：

- 相对 ours linear 降低约 17.1%；
- 相对 Jiang official MLP R1 + NPE 降低约 29.5%。

Model 3 另有 `Jiang raw-sequence GRU R1 + NPE` secondary adaptation，
posterior-mean RMSE 为 0.01415；它不是 Jiang published architecture，
因此不放入主三臂表。

---

# Part IV. 跨模型结果与解释

## 16. Pre-NPE score 总表

下面每行只能在同一个模型内部比较。三个模型的参数范围、数据维度和 exact
score variance 不同，即使使用 standardized MSE，也不应把不同模型的绝对
数值解释成统一难度刻度。

| Model | Jiang R1 stdMSE | Ours linear stdMSE | Ours nonlinear stdMSE | Nonlinear reduction vs Jiang |
|---|---:|---:|---:|---:|
| Model 1 | 0.4035 | 0.0978 | **0.0847** | 79.0% |
| Model 2 | 0.4541 | 0.1521 | **0.1140** | 74.9% |
| Model 3 | 0.7976 | 0.2113 | **0.1549** | 80.6% |

三个模型的 pre-NPE ordering 一致：

```text
ours nonlinear score > ours linear score > Jiang official R1 score
```

这里 `>` 表示 exact-score standardized MSE 更低。

## 17. Post-NPE posterior 总表

| Model | Jiang R1 + NPE mean RMSE | Ours linear + NPE | Ours nonlinear + NPE | Nonlinear reduction vs Jiang |
|---|---:|---:|---:|---:|
| Model 1 | 0.03014 | 0.01642 | **0.01482** | 50.8% |
| Model 2 | 0.03272 | 0.02534 | **0.02189** | 33.1% |
| Model 3 | 0.01426 | 0.01213 | **0.01005** | 29.5% |

三个模型的 post-NPE ordering 也一致：

```text
ours nonlinear + NPE > ours linear + NPE > Jiang R1 + NPE
```

这里 `>` 表示 posterior mean 更接近同一 held-out object 的 exact
posterior mean。

## 18. 为什么 score-MSE 差距大于 posterior 差距

NPE 并不只看到 score，还看到共同 pilot：

$$
c(D)
=
\left(
\widehat u_{\mathrm{pilot}}(D),
\widehat S(D,\widehat u_{\mathrm{pilot}}(D))
\right).
$$

此外，NPE 可以从 Stage-2 simulations 中学习 score 的：

- scale calibration；
- mean offset correction；
- nonlinear transformation；
- 与 pilot 的联合关系。

因此，一个全局 exact-score MSE 较高但仍保留部分参数排序信息的 score，
经过 NPE 后可能被显著校准。特别是 Model 3 中，Jiang R1 score 的
stdMSE 较高，但 pilot-plus-score NPE 仍得到可用 posterior。

这不意味着 pre-NPE score MSE 不重要。相反，两张表回答不同问题：

- score-MSE 表衡量 score estimator 本身的数值正确性；
- posterior 表衡量 score 作为 frozen summary 经过共同 NPE 后的最终信息量。

## 19. Fairness controls

### 19.1 已控制

在每个模型内部，post-NPE comparison 控制：

- 相同 complete observation definition；
- 相同 Stage-2 simulations；
- 相同 held-out objects；
- 相同 data-only pilot；
- 相同 2-dimensional context interface；
- 相同 NPE architecture、optimizer 和 random seed；
- 相同 posterior sample size；
- 相同 exact-posterior reference；
- Jiang 只使用 observation-independent R1。

pre-NPE score comparison控制：

- 相同 score coordinate $u$；
- 相同 exact full-score estimand；
- 相同 parameter grid；
- 相同 observation unit；
- exact score 只作为 held-out diagnostic。

### 19.2 尚未完全控制

以下差异需要在论文中明确披露：

1. Ours 与 Jiang 的 Stage-1 simulator budgets 不完全相同；
2. Ours 使用 5/5/3 个 Stage-1 fits，Jiang 当前每个模型只有 1 个 fit；
3. Ours 使用已知 analytic local likelihood-ratio subscores；
4. Jiang 接收 raw block 或 raw flattened trajectory；
5. Ours Direct-FSM 学习 Gaussian-smoothed score；
6. Jiang 使用 direct score matching、curvature penalty 和 debias regression；
7. Model 3 的 nonlinear-emission setting 经过 pilot screen，目的是突出
   nonlinear local calibration；
8. 三个模型的 NPE 分别训练；`shared NPE` 表示同一个模型内各方法共享
   backend 和数据，不表示一个 NPE 同时处理三个模型。

## 20. Jiang R1 + NPE 的正确命名

`Jiang R1 + shared NPE` 是为了比较 frozen score summary informativeness
而构造的 amortized adaptation。NPE 不是 Jiang 原论文的一部分。

Jiang 原始流程是：

```text
train R1
-> infer on fixed observed dataset
-> construct data-dependent R2 proposal
-> train R2
-> score root
-> frequentist confidence set
```

本文的 posterior comparison 是：

```text
train Jiang R1 once
-> freeze R1
-> evaluate R1 full score at shared data-only pilot
-> shared NPE
-> amortized posterior
```

因此本文支持：

> 在三个当前一维 benchmark 上，在相同的 pilot-plus-frozen-score NPE
> posterior protocol 下，proposed nonlinear composite-score summary
> 比 matched linear ablation 和一个 Jiang official R1 score fit 更接近
> exact posterior。

本文不支持：

> Proposed method 已经在 repeated-dataset frequentist coverage 上系统性
> 击败 Jiang 完整 R1+R2 confidence-set procedure。

---

## 21. Source and artifact map

### Model 1 method and posterior

- `docs/MODEL1_METHOD.md`
- `experiments/block_models/current_model1_fsm_method_package/`
- `configs/model1/stage1/`
- `results/model1/`

### Model 2 method and posterior

- `docs/MODEL2_METHOD.md`
- `experiments/block_models/current_model2_fsm_method_package/`
- `configs/model2/stage1/`
- `results/model2/`

### Model 1/2 matched score comparison

- `docs/MODEL1_MODEL2_SCORE_COMPARISON.md`
- `docs/MODEL1_MODEL2_SHARED_NPE.md`
- `src/khoo_vs_jiang/block_mixture_compare.py`
- `src/khoo_vs_jiang/block_mixture_stage2_npe.py`

### Model 3

- `docs/MODEL3_METHOD.md`
- `docs/MODEL3_GATE_EXPERIMENT.md`
- `docs/MODEL3_JIANG_NPE_ABLATION.md`
- `src/khoo_vs_jiang/nonlinear_emission_screen.py`
- `src/khoo_vs_jiang/nonlinear_emission_stage2.py`
- `results/model3/`

### Combined posterior summary

- `results/headline_posterior_mean_rmse.csv`

### Jiang implementation

- Official repository commit:
  `fb273f0e1bbfca2d1d752c97d1f9d431dd1039c9`
- `src/khoo_vs_jiang/jiang_official.py`
- `src/khoo_vs_jiang/block_mixture_jiang.py`
- `SOURCE_SNAPSHOT.md`
