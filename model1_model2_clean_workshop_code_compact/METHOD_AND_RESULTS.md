# Method and current results

## 当前使用的 nonlinear gate

本文统一把三项实验的非线性方法称为 **Nonlinear gate**。历史代码和 checkpoint
中的内部键分别是 Model 1 的 `radial`、Model 2 的 `shared_radial` 和
Max-stable 的 `positive_anchor`；这些只是实现标识。`positive multiplier` 只描述
下面的数学约束 $m_\eta>0$，不再作为方法名称。

对 pooling 之前的每个 local score $s$，Nonlinear gate 做

\[
\boxed{\phi_\eta(s,a)=s\,m_\eta(s,a)},
\qquad
m_\eta(s,a)=\operatorname{softplus}\!\bigl(g_\eta(s,\widetilde a)\bigr)>0,
\]

其中 $a=\operatorname{logit}(\pi_a)$ 是当前 anchor，$\widetilde a$ 是用 training anchors 的均值和标准差得到的标准化 anchor。$g_\eta$ 是两层隐藏层的 SiLU MLP：

```text
(local score, standardized anchor)
    -> Linear -> SiLU -> Linear -> SiLU -> Linear -> softplus
    -> positive multiplier m_eta
```

因为 $m_\eta>0$，这个变换可以改变 local score 的幅度，但不改变其符号：正 score 仍为正，负 score 仍为负，零仍为零。最后一层在初始化时使用零权重和令 `softplus(bias)=1` 的 bias，因此初始时逐点满足

\[
m_\eta(s,a)=1,
\qquad
\phi_\eta(s,a)=s.
\]

所以 Nonlinear gate 在初始化时精确退化为 Linear；两者的区别只是在 pooling **之前**，Nonlinear gate 多了这个可学习的非线性校准。之后两者使用相同的 block pooling 和同类型的 nonlinear readout：

\[
\widehat S(Y,a)
=\sum_{k=1}^{K}
\rho\!\left(\operatorname{pool}_{j}\phi_\eta(s_{kj},a),\widetilde a\right).
\]

- **Model 1 `radial`**：gate 作用于标准化后的 marginal local score；每个 block 内取均值后交给 $\rho$。
- **Model 2 `shared_radial`**：先把 marginal 和 pairwise 两个 channel 各自还原到 raw local-score 坐标，再用**同一个** $m_\eta(s,a)$ 分别校准两个 channel；两个 channel 仍分别 pooling，随后一起交给 $\rho$。`shared` 表示共享 gate 参数，不表示把两个 channel 合并。
- **Linear**：等价于固定 $m_\eta\equiv1$。它没有 pre-pooling gate，但后面的 $\rho$ 仍是 MLP，因此“Linear”并不表示整个预测器是线性的。

这个 gate 的形式由网络学习；训练代码没有代入解析 inverse link，也没有把 exact full score 或真实参数提供给 gate。

## 共同协议

未知参数是混合概率 `pi`，训练和 score evaluation 使用 `u = logit(pi)`。Stage 1 在连续 stratified anchor 上抽样，并用

```text
u | a ~ Normal(a, sigma_q²)
target = (u - a) / sigma_q²
```

训练 anchored amortized FSM field。exact full score 从不用于优化或 checkpoint selection，只用于冻结后的测试诊断。

两个模型都使用 shared block readout `sum_k rho(block_k, anchor)`。因此 “Linear” 并不是全网络线性，而是没有额外的 pre-pooling gate。

Stage 2 的 context 统一为：

```text
(u_pilot(Y), S_frozen(Y, u_pilot(Y)))
```

`u_pilot` 完全由观测数据得到；真 `pi` 只作为 NPE target。exact likelihood grid 只评估 posterior fidelity。

## Model 1

- DGP：`Y_kj = eps_kj + B_k tau`。
- 当前设置：`K=20, m=20, tau=.5, sigma_q=.20`。
- Linear local summary：marginal mixture u-scores 的 block mean。
- Nonlinear gate（内部键 `radial`）：在 standardized local scores 上施加 anchor-conditioned positive multiplier，再 pooling。
- 比较：同 cache、同 minibatch stream、identity gate、matched untrained `rho`，各 20k updates；历史报告 checkpoint 是 validation-best raw。

五个独立 Stage-1 training seeds（`20260709`--`20260713`）的确认性汇总：

| Stage | Linear | Nonlinear gate | 相对降低 |
|---|---:|---:|---:|
| Stage 1 mean standardized MSE | 0.076983 | 0.045731 | 40.60% |
| Stage 2 mean W1 to exact | 0.010772 | 0.008107 | 24.74% |

### Stage 1：按真实参数拆分的 score MSE

以下是五个 validation-best 冻结 checkpoint 在各自 exact-score test set 上的均值。每个 training seed、每个真实 `pi` 有 5,000 条独立测试数据。`standardized MSE` 是 raw score MSE 除以该参数点 exact score 的方差，因此更适合跨参数比较。

| 真实 `pi` | Linear score MSE | Nonlinear-gate score MSE | Nonlinear gate 相对降低 |
|---:|---:|---:|---:|
| 0.10 | 0.189196 | 0.110531 | 41.58% |
| 0.30 | 0.088288 | 0.046034 | 47.86% |
| 0.50 | 0.077965 | 0.049597 | 36.39% |
| 0.65 | 0.101952 | 0.070853 | 30.50% |
| **四个参数平均** | **0.114350** | **0.069254** | **39.44%**。 |

### Stage 2：按真实参数拆分的 posterior 结果

每个 Stage-1 seed、每个真实 `pi` 使用相同的 10 个 paired observation datasets（seeds 100--109）。下表先在每个 Stage-1 seed 内按固定真值平均，再对五个 Stage-1 seeds 等权平均，因此不会把不同真值混在一起。五次 Stage 2 固定使用相同 NPE training seed；这些结果确认了 Stage-1 training randomness 下的稳定性，尚未覆盖独立 NPE-seed 变异。

#### Posterior-mean MSE vs true parameter

对每个固定真值，计算

```text
mean_j [(posterior_mean_j - pi_true)^2].
```

| 真实 `pi` | Exact posterior | Pilot-only NPE | Linear NPE | Nonlinear-gate NPE | Nonlinear gate 相对 Linear |
|---:|---:|---:|---:|---:|---:|
| 0.07 | 0.009424 | 0.014467 | 0.011095 | **0.010051** | **降低 9.41%** |
| 0.10 | 0.007174 | 0.010358 | 0.008185 | **0.007351** | **降低 10.18%** |
| 0.30 | 0.002502 | 0.003625 | **0.002311** | 0.002315 | 增加 0.20% |
| 0.50 | 0.004644 | 0.004467 | 0.005016 | **0.004806** | **降低 4.19%** |
| 0.65 | 0.017014 | 0.019251 | 0.018945 | **0.017716** | **降低 6.49%** |
| 0.68 | 0.019987 | 0.023267 | 0.022184 | **0.020872** | **降低 5.91%** |
| **六个参数平均** | **0.010124** | **0.012573** | **0.011289** | **0.010519** | **降低 6.83%** |

这里的 exact 列也是 exact posterior mean 对生成真值的 MSE



**Posterior-mean MSE vs exact posterior mean**

对每个测试数据集，先计算 NPE posterior mean 与 exact-likelihood posterior mean 的平方差，再在同一固定真值内平均。

| 真实 `pi` | Pilot-only NPE | Linear NPE | Nonlinear-gate NPE | Nonlinear gate 相对 Linear |
|---:|---:|---:|---:|---:|
| 0.07 | 0.001190 | 0.000200 | **0.000130** | **降低 34.93%** |
| 0.10 | 0.001258 | 0.000229 | **0.000106** | **降低 53.88%** |
| 0.30 | 0.000632 | 0.000176 | **0.000096** | **降低 45.63%** |
| 0.50 | 0.000347 | 0.000092 | **0.000063** | **降低 31.66%** |
| 0.65 | 0.000611 | 0.000107 | **0.000047** | **降低 56.18%** |
| 0.68 | 0.000596 | 0.000103 | **0.000041** | **降低 59.97%** |
| **六个参数平均** | **0.000772** | **0.000151** | **0.000080** | **降低 46.80%** |

Nonlinear gate 在 6/6 个固定真值上更准确地复现 exact posterior mean。

#### W1 distance vs exact posterior

`W1` 比较整条一维 posterior，因而同时反映 posterior 的位置、宽度和形状；其单位是原始 `pi` 尺度。

| 真实 `pi` | Pilot-only NPE | Linear NPE | Nonlinear-gate NPE | Nonlinear gate 相对 Linear |
|---:|---:|---:|---:|---:|
| 0.07 | 0.028068 | 0.011519 | **0.009400** | **降低 18.39%** |
| 0.10 | 0.027939 | 0.013667 | **0.009664** | **降低 29.29%** |
| 0.30 | 0.019922 | 0.012113 | **0.009604** | **降低 20.72%** |
| 0.50 | 0.014142 | 0.008833 | **0.007609** | **降低 13.86%** |
| 0.65 | 0.021058 | 0.009523 | **0.006525** | **降低 31.49%** |
| 0.68 | 0.020345 | 0.008976 | **0.005841** | **降低 34.93%** |
| **六个参数平均** | **0.021912** | **0.010772** | **0.008107** | **降低 24.74%** |

Nonlinear gate 在 6/6 个固定真值上的整条 posterior 都更接近 exact posterior。

#### Posterior SD vs exact posterior

第一张表报告每个固定真值下的平均 posterior SD：

| 真实 `pi` | Exact posterior | Pilot-only NPE | Linear NPE | Nonlinear-gate NPE |
|---:|---:|---:|---:|---:|
| 0.07 | 0.08330 | 0.09604 | 0.08583 | **0.08350** |
| 0.10 | 0.08917 | 0.09984 | 0.09081 | **0.08878** |
| 0.30 | 0.12079 | 0.12345 | 0.12026 | **0.12010** |
| 0.50 | 0.11107 | 0.11240 | **0.11061** | 0.11102 |
| 0.65 | 0.08946 | 0.09894 | 0.09255 | **0.09252** |
| 0.68 | 0.08791 | 0.09849 | 0.09072 | **0.09065** |
| **六个参数平均** | **0.09695** | **0.10486** | **0.09846** | **0.09776** |

平均 SD 的正负误差会抵消，因此第二张表使用逐数据集 posterior SD 对 exact posterior SD 的平均绝对误差：

| 真实 `pi` | Linear SD error | Nonlinear-gate SD error | Nonlinear gate 相对变化 |
|---:|---:|---:|---:|
| 0.07 | 0.005621 | **0.005481** | 降低 2.50% |
| 0.10 | 0.005792 | **0.005368** | 降低 7.32% |
| 0.30 | **0.004765** | 0.005195 | 增加 9.01% |
| 0.50 | **0.004128** | 0.004174 | 增加 1.12% |
| 0.65 | **0.003843** | 0.004027 | 增加 4.78% |
| 0.68 | **0.003008** | 0.003139 | 增加 4.34% |
| **六个参数平均** | **0.004526** | **0.004564** | **增加 0.83%** |

五-seed 均值中，Nonlinear gate 的 posterior SD error 只在 2/6 个参数点更低，六点平均与 Linear 基本相同（增加 0.83%）。因此稳定的改善主要体现在 posterior mean 和整条 posterior，而不是 posterior width。

### Stage 2：naive raw-data NPE baseline

为检查收益是否只是来自 NPE 本身，另用完全相同的 50,000 份 Stage-2
training simulations、NPE 配置和测试 seeds，直接把完整
`Y.shape=(20,20)` 按 canonical order 展平成 400 维 context。该 baseline 不读取
pilot、FSM score、解析 summary 或真实参数。

| 真实 `pi` | Raw NPE MSE vs truth | Raw NPE MSE vs exact mean | Raw NPE W1 vs exact | Raw NPE posterior SD |
|---:|---:|---:|---:|---:|
| 0.07 | 0.015315 | 0.00171598 | 0.036826 | 0.09269 |
| 0.10 | 0.011252 | 0.00169948 | 0.035496 | 0.09612 |
| 0.30 | 0.003313 | 0.00078140 | 0.023155 | 0.11398 |
| 0.50 | 0.004570 | 0.00052769 | 0.020134 | 0.10765 |
| 0.65 | 0.018050 | 0.00088916 | 0.027775 | 0.09396 |
| 0.68 | 0.022260 | 0.00082799 | 0.026452 | 0.09215 |

六个真值等权汇总：

| Stage-2 context | MSE vs truth | RMSE vs exact mean | Mean W1 vs exact | Average posterior SD | Coverage 90% |
|---|---:|---:|---:|---:|---:|
| Raw `flatten(Y)` | 0.012460 | 0.032766 | 0.028306 | 0.099424 | 0.750 |
| Pilot + Linear score | 0.011593 | 0.013324 | 0.011909 | 0.098643 | 0.883 |
| Pilot + Nonlinear-gate score | **0.010552** | **0.007769** | **0.006933** | 0.098007 | 0.883 |

因此 Model 1 的 raw NPE 确实能从原始数据学习部分信息，但在 posterior mean、
整条 posterior 和 coverage 上都不如结构化 score context。这个比较是实用的
end-to-end baseline；由于输入维度和 inductive bias 不同，它不是 Stage-1
Linear/Nonlinear-gate 架构效应的替代因果实验。





### Direct amortized raw-FSM baseline

真正的 direct baseline 将完整数据一次性送入同一个 amortized score network：

```text
concatenate(flatten(Y), standardized anchor) -> 64 -> 64 -> scalar score
```

它不使用 `phi`、gate、pooling、DeepSets 或 blockwise sum。其余 40k/8k FSM
simulation bank、20k updates、validation checkpoint rule、data-only pilot、50k NPE
和 60-case test 均保持不变。Model 1 网络有 29,953 个参数；validation-best 是
step 1,700 raw。

stage 1

| 真实 `pi` | Direct raw score MSE | Direct raw standardized MSE |
|---:|---:|---:|
| 0.10 | 0.764660 | 0.862536 |
| 0.30 | 0.990653 | 0.379283 |
| 0.50 | 0.964768 | 0.303915 |
| 0.65 | 1.130964 | 0.395932 |
| **四个参数平均** | **0.962761** | **0.485416** |

| Stage-2 context | MSE vs truth | RMSE vs exact mean | Mean W1 vs exact | Posterior SD | Coverage 90% |
|---|---:|---:|---:|---:|---:|
| Pilot-only | 0.012843 | 0.028999 | 0.023149 | 0.105720 | 0.850 |
| Direct raw-data NPE `flatten(Y)` | 0.012460 | 0.032766 | 0.028306 | 0.099424 | 0.750 |
| Pilot + direct raw-FSM score | 0.012643 | 0.027405 | 0.022227 | 0.102606 | 0.833 |
| Pilot + Linear score | 0.011593 | 0.013324 | 0.011909 | 0.098643 | 0.883 |
| Pilot + Nonlinear-gate score | **0.010552** | **0.007769** | **0.006933** | 0.098007 | 0.883 |

Direct raw-FSM 的 W1 接近 pilot-only 的 `0.023149`。在固定 Stage-1 simulation
budget 下，未结构化的 raw MLP 几乎没有提供可替代 composite score 的额外信息；
参数更多并未抵消其缺少 permutation/block inductive bias 的样本效率问题。

## Model 2

- DGP：`Y_kj = eps_kj + B_k tau Z_k`。
- 当前设置：`K=40, m=40, tau=1, sigma_q=.15`。
- Linear local summary：marginal 和全部 unordered pairwise mixture u-scores，各自取 block mean。
- Nonlinear gate（内部键 `shared_radial`）：恢复两个 channel 的 raw local score，使用同一个 anchor-conditioned positive multiplier，映回各自标准化坐标后再 pooling。
- gate 在 identity initialization 时逐浮点嵌套 Linear；两臂使用同 cache、相同 minibatch stream、matched untrained `rho`、共同 `1e-4` constant LR，最多训练 20k updates。本节只报告 validation-best raw checkpoint。



Stage 1：按真实参数拆分的 score MSE

每个 training seed、每个真实 `pi` 有 5,000 条独立 exact-score test data。validation-best raw 只按 validation FSM loss 选择，不使用 test exact score。下表报告五个 Stage-1 seeds 的逐参数均值；各 seed 的原始逐参数 CSV 均保存在 artifacts 中。

#### validation-best raw

| 真实 `pi` | Linear raw MSE | Nonlinear-gate raw MSE | Nonlinear gate 相对降低 | Linear stdMSE | Nonlinear-gate stdMSE | Nonlinear gate 相对降低 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.07 | 0.347772 | 0.281849 | 18.96% | 0.215685 | 0.174820 | 18.95% |
| 0.10 | 0.362347 | 0.286198 | 21.02% | 0.162591 | 0.128245 | 21.12% |
| 0.30 | 0.497447 | 0.321675 | 35.33% | 0.095515 | 0.061719 | 35.38% |
| 0.50 | 0.616006 | 0.368024 | 40.26% | 0.108688 | 0.064966 | 40.23% |
| 0.65 | 0.780004 | 0.460298 | 40.99% | 0.172037 | 0.101530 | 40.98% |
| 0.68 | 0.800835 | 0.465150 | 41.92% | 0.190505 | 0.110542 | 41.97% |
| **六个参数平均** | **0.567402** | **0.363866** | **35.87%** | **0.157503** | **0.106970** | **32.08%** |

五个 training seeds 中，Nonlinear gate 均在 6/6 个固定真值上降低 score MSE。原始 seed `20260709` 和 `20260710` 的 validation-best steps 记录仍保留在各自 `training_info.json` 中；新增 seeds 同样只使用 validation FSM loss 选择 checkpoint。

### Stage 2：按真实参数拆分的 posterior 结果

Stage 2 使用五个 validation-best raw Stage-1 checkpoint pairs。每个 Stage-1 seed 共有 6 个真实 `pi`，每个真值使用相同的 10 个 paired observation datasets（seeds 100--109）。表中先在每个 Stage-1 seed 内按固定真值平均，再对五个 Stage-1 seeds 等权平均。所有 Stage 2 固定使用相同 NPE training seed；Exact 与 Pilot-only 不消费 Linear/Nonlinear-gate Stage-1 checkpoint。

#### Posterior-mean MSE vs true parameter

| Stage-1 policy | 真实 `pi` | Exact | Pilot-only | Linear | Nonlinear gate | Nonlinear gate 相对 Linear |
|---|---:|---:|---:|---:|---:|---:|
| validation-best raw | 0.07 | 0.004202 | 0.009315 | 0.005259 | **0.004985** | **降低 5.21%** |
| validation-best raw | 0.10 | 0.005229 | 0.007786 | 0.005253 | **0.005212** | **降低 0.78%** |
| validation-best raw | 0.30 | 0.003994 | 0.009907 | 0.005829 | **0.005505** | **降低 5.55%** |
| validation-best raw | 0.50 | 0.010048 | 0.016778 | **0.010084** | 0.010446 | 增加 3.59% |
| validation-best raw | 0.65 | 0.003526 | 0.014915 | 0.004249 | **0.003735** | **降低 12.09%** |
| validation-best raw | 0.68 | 0.005612 | 0.020233 | 0.006776 | **0.005957** | **降低 12.10%** |
| validation-best raw | **平均** | **0.005435** | **0.013156** | **0.006242** | **0.005973** | **降低 4.30%** |

五-seed 均值中 Nonlinear gate 在 5/6 个真值上降低 MSE-to-truth，平均改善 4.30%；因此 point-estimation MSE 的收益仍远小于 Stage-1 score MSE 与 posterior-fidelity 指标的收益。

#### Posterior-mean MSE vs exact posterior mean

| Stage-1 policy | 真实 `pi` | Pilot-only | Linear | Nonlinear gate | Nonlinear gate 相对 Linear |
|---|---:|---:|---:|---:|---:|
| validation-best raw | 0.07 | 0.001734 | 0.000425 | **0.000395** | **降低 7.03%** |
| validation-best raw | 0.10 | 0.001705 | **0.000385** | 0.000388 | 增加 0.86% |
| validation-best raw | 0.30 | 0.004569 | 0.000844 | **0.000564** | **降低 33.23%** |
| validation-best raw | 0.50 | 0.005595 | 0.000376 | **0.000280** | **降低 25.51%** |
| validation-best raw | 0.65 | 0.006935 | 0.000416 | **0.000113** | **降低 72.77%** |
| validation-best raw | 0.68 | 0.007475 | 0.000390 | **0.000107** | **降低 72.72%** |
| validation-best raw | **平均** | **0.004669** | **0.000473** | **0.000308** | **降低 34.89%** |

五-seed 均值中 Nonlinear gate 在 5/6 个固定真值上更准确地复现 exact posterior mean，平均 MSE 降低 34.89%。

#### W1 distance vs exact posterior

| Stage-1 policy | 真实 `pi` | Pilot-only | Linear | Nonlinear gate | Nonlinear gate 相对 Linear |
|---|---:|---:|---:|---:|---:|
| validation-best raw | 0.07 | 0.034115 | 0.017639 | **0.017568** | **降低 0.40%** |
| validation-best raw | 0.10 | 0.036335 | 0.016932 | **0.016587** | **降低 2.04%** |
| validation-best raw | 0.30 | 0.055032 | 0.025670 | **0.020007** | **降低 22.06%** |
| validation-best raw | 0.50 | 0.064704 | 0.015273 | **0.013607** | **降低 10.91%** |
| validation-best raw | 0.65 | 0.059587 | 0.015897 | **0.009455** | **降低 40.53%** |
| validation-best raw | 0.68 | 0.062955 | 0.015347 | **0.009053** | **降低 41.01%** |
| validation-best raw | **平均** | **0.052121** | **0.017793** | **0.014379** | **降低 19.19%** |

五-seed 均值中 Nonlinear gate 的 W1 在 6/6 个固定真值上更低，平均降低 19.19%。

#### Posterior SD vs exact posterior

平均 posterior SD：

| Stage-1 policy | 真实 `pi` | Exact | Pilot-only | Linear | Nonlinear gate |
|---|---:|---:|---:|---:|---:|
| validation-best raw | 0.07 | 0.051256 | 0.079444 | 0.059985 | **0.059254** |
| validation-best raw | 0.10 | 0.057393 | 0.083736 | 0.064203 | **0.063884** |
| validation-best raw | 0.30 | 0.084978 | 0.112510 | 0.090215 | **0.089029** |
| validation-best raw | 0.50 | 0.087461 | 0.112840 | 0.090704 | **0.090201** |
| validation-best raw | 0.65 | 0.065243 | 0.092940 | 0.066006 | **0.064349** |
| validation-best raw | 0.68 | 0.062198 | 0.090572 | 0.063352 | **0.061524** |
| validation-best raw | **平均** | **0.068088** | **0.095340** | **0.072411** | **0.071373** |

为避免平均 SD 的正负偏差互相抵消，下面使用逐数据集 posterior SD 对 exact posterior SD 的平均绝对误差：

| Stage-1 policy | 真实 `pi` | Pilot-only | Linear | Nonlinear gate | Nonlinear gate 相对 Linear |
|---|---:|---:|---:|---:|---:|
| validation-best raw | 0.07 | 0.028188 | 0.009908 | **0.009211** | **降低 7.03%** |
| validation-best raw | 0.10 | 0.026344 | 0.008123 | **0.007746** | **降低 4.65%** |
| validation-best raw | 0.30 | 0.027532 | 0.006857 | **0.005355** | **降低 21.90%** |
| validation-best raw | 0.50 | 0.027373 | 0.005430 | **0.003499** | **降低 35.56%** |
| validation-best raw | 0.65 | 0.027697 | 0.005545 | **0.003552** | **降低 35.94%** |
| validation-best raw | 0.68 | 0.029397 | 0.005983 | **0.003748** | **降低 37.36%** |
| validation-best raw | **平均** | **0.027755** | **0.006974** | **0.005518** | **降低 20.87%** |

五-seed 均值中 Nonlinear gate 的 posterior SD absolute error 在 6/6 个固定真值上均更低，平均降低 20.87%。

### Stage 2：naive raw-data NPE baseline

同样使用 50,000 份训练模拟和相同测试 seeds，把完整
`Y.shape=(40,40)` 直接展平为 1,600 维 NPE context；不使用 pilot、marginal /
pairwise score 或真实参数。

| 真实 `pi` | Raw NPE MSE vs truth | Raw NPE MSE vs exact mean | Raw NPE W1 vs exact | Raw NPE posterior SD |
|---:|---:|---:|---:|---:|
| 0.07 | 0.027259 | 0.01101639 | 0.100098 | 0.11459 |
| 0.10 | 0.019148 | 0.00700552 | 0.080147 | 0.11690 |
| 0.30 | 0.005017 | 0.00430468 | 0.064131 | 0.13153 |
| 0.50 | 0.010548 | 0.00590045 | 0.064581 | 0.12387 |
| 0.65 | 0.025825 | 0.01443312 | 0.095297 | 0.10763 |
| 0.68 | 0.035221 | 0.01550923 | 0.102176 | 0.10767 |

六个真值等权汇总：

| Stage-2 context | MSE vs truth | RMSE vs exact mean | Mean W1 vs exact | Average posterior SD | Coverage 90% |
|---|---:|---:|---:|---:|---:|
| Raw `flatten(Y)` | 0.020503 | 0.098463 | 0.084405 | 0.117031 | 0.683 |
| Pilot + Linear score | 0.006183 | 0.021266 | 0.017715 | 0.073069 | 0.867 |
| Pilot + Nonlinear-gate score | **0.005930** | 0.017743 | 0.014943 | 0.071077 | **0.900** |

在 1,600 维 Model-2 原始输入上，naive flat NPE 明显不如两个 score-summary
NPE，且在 prior 边缘退化最严重。这说明当前收益不能简单解释为“任何 NPE
直接看 raw data 都能得到”；结构化 composite-score bottleneck 在固定 50k
simulation budget 下提供了很强的 inductive bias。它仍不是与专门设计的
raw-data DeepSets/Transformer encoder 的比较，后者应作为另一项更强但计算量
不同的 baseline。



### Direct amortized raw-FSM baseline

Direct Model-2 baseline 使用：

```text
concatenate(flatten(Y[40,40]), standardized anchor) -> 64 -> 64 -> scalar score
```

它有 106,753 个参数，但不使用任何 block/coordinate weight sharing。validation-best
由 FSM validation loss 在 step 500 EMA 选出；exact score 仍只用于冻结后的诊断。

| 真实 `pi` | Direct raw score MSE | Direct raw standardized MSE |
|---:|---:|---:|
| 0.07 | 1.658290 | 1.012917 |
| 0.10 | 2.285166 | 1.006266 |
| 0.30 | 5.053721 | 0.985342 |
| 0.50 | 5.669404 | 1.004057 |
| 0.65 | 4.667697 | 1.021549 |
| 0.68 | 4.412715 | 1.015084 |
| **六个参数平均** | **3.957832** | **1.007536** |

| Stage-2 context | MSE vs truth | RMSE vs exact mean | Mean W1 vs exact | Posterior SD | Coverage 90% |
|---|---:|---:|---:|---:|---:|
| Direct raw-data NPE `flatten(Y)` | 0.020503 | 0.098463 | 0.084405 | 0.117031 | 0.683 |
| Pilot-only | 0.013142 | 0.068372 | 0.052220 | 0.095387 | 0.850 |
| Pilot + direct raw-FSM score | 0.013230 | 0.068637 | 0.052751 | 0.094709 | 0.817 |
| Pilot + Linear score | 0.006183 | 0.021266 | 0.017715 | 0.073069 | 0.867 |
| Pilot + Nonlinear-gate score | **0.005930** | **0.017743** | **0.014943** | 0.071077 | **0.900** |

Direct raw score 的 Stage-1 standardized MSE 约为 1，平均相关系数仅约 `0.084`；
Stage 2 与 pilot-only 几乎重合。这个结果不是参数不足：它比 Nonlinear gate 多约22倍
参数。它显示在固定 simulation budget 下，无结构的 1,600 维 raw network 非常
sample-inefficient，而解析 local composite scores 提供了关键归纳偏置。

## Max-stable rainfall-79

### Model and estimated parameters

One observation is a complete dataset

\[
Y=\{Y_{t,i}:t=1,\ldots,47;\ i=1,\ldots,79\},
\]

containing 47 iid annual-maxima fields at 79 Swiss rainfall sites. Dependence
is described by the two-dimensional Smith max-stable model. The scientific
parameter is the positive-definite covariance matrix

\[
\boldsymbol\Sigma
=
\left[
\begin{array}{cc}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{12} & \Sigma_{22}
\end{array}
\right],
\]

which is parameterized as

\[
\Sigma_{11}=\sigma_x^2,
\qquad
\Sigma_{22}=\sigma_y^2,
\qquad
\Sigma_{12}=\rho\sigma_x\sigma_y,
\]

with σₓ, σᵧ > 0 and −1 < ρ < 1. The experiment infers all three
dependence coordinates. The seven remaining marginal/GEV coordinates in the
simulator are held fixed at their rainfall reference values.

The neural inference is carried out through the bounded normalized coordinate
**u = (u₁, u₂, u₃) ∈ [0.15, 0.85]³**. Let
**Σ₀ = (Σ₁₁,₀, Σ₁₂,₀, Σ₂₂,₀)** be the rainfall reference covariance,

\[
(\Sigma_{11,0},\Sigma_{12,0},\Sigma_{22,0})
=(332.1527,70.3982,184.6266),
\]

and define

\[
\sigma_{x,0}=\sqrt{\Sigma_{11,0}},
\qquad
\sigma_{y,0}=\sqrt{\Sigma_{22,0}},
\qquad
\rho_0=\frac{\Sigma_{12,0}}{\sigma_{x,0}\sigma_{y,0}}.
\]

The normalized coordinates map to the covariance through

\[
\sigma_x(\boldsymbol u)
=\sigma_{x,0}\exp\!\left[0.70(u_1-0.5)\right],
\]

\[
\sigma_y(\boldsymbol u)
=\sigma_{y,0}\exp\!\left[0.70(u_2-0.5)\right],
\]

\[
\rho(\boldsymbol u)
=\tanh\!\left[
\operatorname{arctanh}(\rho_0)+1.40(u_3-0.5)
\right].
\]

Thus u₁ controls the horizontal dependence scale, u₂ controls the vertical
dependence scale, and u₃ controls correlation/orientation through the Fisher-z
coordinate. The center **u = (0.5, 0.5, 0.5)** maps exactly to Σ₀. The
exponential and tanh maps guarantee σₓ, σᵧ > 0 and |ρ| < 1, hence a
positive-definite Σ.

For FSM training and score evaluation, each bounded coordinate is transformed
to an unconstrained coordinate

\[
w_r=\operatorname{logit}\!\left(\frac{u_r-0.15}{0.70}\right),
\qquad
u_r=0.15+0.70\,\operatorname{sigmoid}(w_r).
\]

Here **u** is a three-dimensional normalized covariance coordinate and **w**
is its unconstrained transform. This notation is specific to the Max-stable
experiment: unlike Model 1/2, uᵣ itself is **not** a logit-transformed mixture
probability.

### Pairwise score representation and the nonlinear treatment

The full 79-site likelihood is not used. For year t and site pair (i,j), the
tractable input is the exact three-dimensional bivariate likelihood score
evaluated at anchor wₐ,

\[
\boldsymbol s_{t,ij}(\boldsymbol w_a)
=
\left.
\nabla_{\boldsymbol w}
\log f_{ij}\!\left(Y_{t,i},Y_{t,j}\mid\boldsymbol w\right)
\right|_{\boldsymbol w=\boldsymbol w_a}
\in\mathbb R^3.
\]

Every annual field supplies all 3,081 pairwise score vectors. Linear and the Nonlinear gate receive
exactly the same pair scores and anchors. Their only architectural difference
is the local map applied before pair pooling:

\[
\text{Linear:}\qquad
\phi_L(\boldsymbol s,\boldsymbol w_a)=\boldsymbol s.
\]

\[
\text{Nonlinear gate:}\qquad
\phi_N(\boldsymbol s,\boldsymbol w_a)
=\boldsymbol s\odot
\boldsymbol m_\eta(\boldsymbol s,\boldsymbol w_a),
\qquad
\boldsymbol m_\eta>\boldsymbol0.
\]

The Nonlinear gate jointly receives the three raw-score coordinates after
train-only standardization and the three standardized anchor coordinates. A
two-hidden-layer width-16 SiLU MLP outputs
three `softplus` multipliers. Its last layer is initialized so that
mη ≡ 1; therefore the Nonlinear-gate model is exactly
equal to Linear at initialization. The gate does not receive pair distance,
direction, or other geometry.

After the local map, the 3,081 pairs are assigned to 20 fixed equal-count
distance bins. Within each year and bin, transformed pair scores are averaged:
\[
\overline{\boldsymbol\phi}_{t,g}
=\frac{1}{|G_g|}
\sum_{(i,j)\in G_g}
\phi(\boldsymbol s_{t,ij},\boldsymbol w_a),
\qquad g=1,\ldots,20.
\]

The ordered 20 × 3 = 60 bin summary and the three-dimensional standardized
anchor are passed to the same width-64, two-hidden-layer annual MLP ρ.
The 47 iid annual contributions are then summed:

\[
\widehat{\boldsymbol S}(Y,\boldsymbol w_a)
=
\sum_{t=1}^{47}
\rho\!\left(
\overline{\boldsymbol\phi}_{t,1},\ldots,
\overline{\boldsymbol\phi}_{t,20},
\widetilde{\boldsymbol w}_a
\right)
\in\mathbb R^3.
\]

Consequently, `Linear` means only that the pre-pooling local map is the
identity. Its annual readout ρ is still nonlinear.

### Stage 1 and Stage 2

Stage 1 draws stratified anchors over [0.15, 0.85]³, transforms them to wₐ,
and uses the isotropic proposal

\[
\boldsymbol w\mid\boldsymbol w_a
\sim
\mathcal N(\boldsymbol w_a,0.20^2\boldsymbol I_3),
\qquad
\boldsymbol T
=\frac{\boldsymbol w-\boldsymbol w_a}{0.20^2}.
\]

Linear and the Nonlinear gate are trained on the same 10,000 complete simulated
datasets, 8,000 validation datasets, anchors, FSM targets, minibatches, and
10,000 optimizer updates. The retained checkpoint is fixed-final EMA. Exact
full likelihood scores are not used for training or checkpoint selection.

At Stage 2, both methods first compute the same data-only rough pairwise MPLE
pilot using a fixed stratified subset of 500 pairs: 25 pairs from each of the
20 distance bins. Safeguarded multistart Newton optimization returns
ŵ_pilot. The frozen Stage-1 field is then
evaluated at that pilot, producing the six-dimensional NPE context

\[
\left(
\widehat{\boldsymbol w}_{\mathrm{pilot}},
\widehat{\boldsymbol S}
(Y,\widehat{\boldsymbol w}_{\mathrm{pilot}})
\right)\in\mathbb R^6.
\]

Each arm uses the same 10,000 Stage-2 simulations and the same 8-component
MDN posterior estimator. The generating u is used only as the
NPE target and synthetic evaluation truth; it is not included in the context.
The full 3,081-pair MPLE is an evaluation comparator only. An exact 79-site
posterior is unavailable.

### Fixed-truth, coordinate-specific covariance MSE

The three fixed truths are synthetic generating parameters. Consequently, the
generating covariance is known even though the exact 79-site posterior is not
available. Each truth uses 100 paired simulated datasets shared by Linear and
the Nonlinear gate. The normalized truths and their corresponding covariance coordinates
are:

| Truth label | Normalized truth u | (Σ₁₁, Σ₁₂, Σ₂₂) |
|---|---:|---:|
| Low | (0.25, 0.25, 0.25) | (234.064, −10.052, 130.104) |
| Center | (0.50, 0.50, 0.50) | (332.153, 70.398, 184.627) |
| High | (0.75, 0.75, 0.75) | (471.347, 199.071, 261.998) |

For covariance coordinate r, the reported metric is

\[
\operatorname{MSE}_r
=
\frac{1}{100}
\sum_{d=1}^{100}
\left(
\widehat\Sigma_r^{(d)}-\Sigma_{r,\mathrm{true}}
\right)^2.
\]

Here the estimate is the posterior mean on the original covariance scale for
test dataset d.

| Normalized truth | Covariance coordinate | Linear MSE | Nonlinear-gate MSE | Nonlinear gate relative reduction |
|---|---|---:|---:|---:|
| (0.25, 0.25, 0.25) | Σ₁₁ | 214.76 | **138.24** | **35.63%** |
|  | Σ₁₂ | 41.92 | **40.04** | **4.48%** |
|  | Σ₂₂ | 55.70 | **36.81** | **33.92%** |
| (0.50, 0.50, 0.50) | Σ₁₁ | 518.34 | **465.96** | **10.10%** |
|  | Σ₁₂ | 137.53 | **124.42** | **9.53%** |
|  | Σ₂₂ | 193.82 | **174.63** | **9.90%** |
| (0.75, 0.75, 0.75) | Σ₁₁ | 577.31 | **555.11** | **3.85%** |
|  | Σ₁₂ | 259.76 | **165.62** | **36.24%** |
|  | Σ₂₂ | 255.08 | **168.13** | **34.09%** |

The Nonlinear gate has lower MSE in all 3 × 3 = 9 truth-by-coordinate cells. Because
the three covariance coordinates have different physical scales, their MSEs
should be interpreted separately. For completeness, the sums of the three
coordinate MSEs are:

| Normalized truth | Linear summed MSE | Nonlinear-gate summed MSE | Nonlinear gate relative reduction |
|---|---:|---:|---:|
| (0.25, 0.25, 0.25) | 312.37 | **215.08** | **31.15%** |
| (0.50, 0.50, 0.50) | 849.69 | **765.01** | **9.97%** |
| (0.75, 0.75, 0.75) | 1092.15 | **888.85** | **18.61%** |

### Max-stable-specific metric: extremal coefficient

For a spatial displacement vector h, the Smith model's pairwise
extremal coefficient is

\[
\delta_\Sigma(\boldsymbol h)
=
2\Phi\!\left(
\frac{1}{2}
\sqrt{
\boldsymbol h^\top\Sigma^{-1}\boldsymbol h
}
\right),
\]

where Φ is the standard-normal CDF. The extremal coefficient summarizes
the strength of pairwise extremal dependence at a given spatial separation:

- δΣ(h) = 1 corresponds to complete extremal
  dependence;
- δΣ(h) = 2 corresponds to extremal independence;
- intermediate values describe partial extremal dependence.

This functional metric is scientifically more direct than covariance-element
error alone: it measures whether the inferred covariance reproduces the
distance- and direction-dependent extremal association implied by the Smith
model.

The evaluation grid contains 12 radii and 8 directions, giving G = 96 locked
displacement vectors. For each test dataset, the integrated squared error of
the posterior-derived mean extremal-coefficient surface is

\[
\operatorname{ISE}_{\delta}^{(d)}
=
\frac{1}{G}
\sum_{g=1}^{G}
\left[
\widehat\delta^{(d)}(\boldsymbol h_g)
-
\delta_{\Sigma_{\mathrm{true}}}(\boldsymbol h_g)
\right]^2.
\]

The table reports the mean of this dataset-level ISE over the 100 paired test
datasets at each fixed truth.

| Normalized truth | Linear extremal-coefficient MSE | Nonlinear-gate extremal-coefficient MSE | Nonlinear gate relative reduction |
|---|---:|---:|---:|
| (0.25, 0.25, 0.25) | 6.9748 × 10⁻⁵ | **5.8374 × 10⁻⁵** | **16.31%** |
| (0.50, 0.50, 0.50) | 1.4116 × 10⁻⁴ | **1.2142 × 10⁻⁴** | **13.98%** |
| (0.75, 0.75, 0.75) | 1.2348 × 10⁻⁴ | **1.0100 × 10⁻⁴** | **18.20%** |
| **All 300 paired datasets** | **1.1146 × 10⁻⁴** | **9.3599 × 10⁻⁵** | **16.03%** |

For the absolute paired difference `Linear ISE - Nonlinear-gate ISE`, the paired
bootstrap 95% intervals are [3.36 × 10⁻⁶, 2.00 × 10⁻⁵] at the low
truth, [6.93 × 10⁻⁶, 3.31 × 10⁻⁵] at the center truth, and
[9.78 × 10⁻⁶, 3.59 × 10⁻⁵] at the high truth. All three intervals
are above zero. The retained Max-stable results also report covariance-axis
orientation and major/minor axis-ratio errors, but the extremal-coefficient ISE
is the primary model-specific functional metric.

These Max-stable metrics compare posterior summaries with known synthetic
generating truths; they do not compare the learned posterior with an exact
79-site posterior. The current evidence uses one Stage-1 training seed and 100
paired test datasets per fixed truth. Source artifacts are retained in
`maxstable_rainfall79_10k_clean/results/fixed_truth` and
`maxstable_rainfall79_10k_clean/results/extremal`.

## 理论解释与限制

Model 2 的 local mixture score link 是 `s = sigmoid(a + ell) - sigmoid(a)`，严格单调但在 pooling 前会饱和。Nonlinear gate（内部键 `shared_radial`）可以近似 channel-independent inverse link，把 `s` 校准回 local log-ratio `ell`；marginal 与 pairwise pooled log-ratios共同恢复 block 的平方和统计量。Linear 先平均 bounded scores，通常会产生不可逆 collision，因此 outer `rho` 再强也无法恢复被 pooling 丢掉的证据强度。

这解释的是 local-summary bottleneck。有限 `sigma_q` 下的 FSM target还包含跨 block interaction，而当前网络仍强制 block-additive；Nonlinear gate 不能消除这部分 smoothing / function-class mismatch。
