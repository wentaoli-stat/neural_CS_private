# Model 1 / Model 2 selected package

这是论文当前使用的 Model 1 / Model 2 精简 workshop 包。它保留正式选中的
checkpoint、主结果、源码和测试，但不包含约 61 MB 的 confirmatory 多种子历史
实验。正式问题只有一个：在相同的
anchored Direct-FSM 与 Stage-2 NPE 协议下，pooling 前增加 nonlinear gate
是否优于 identity local map。

完整三模型运行教程见根目录的
[`RUN_THREE_MODELS.ipynb`](../RUN_THREE_MODELS.ipynb)，推导和逐参数结果见
[`METHODOLOGY_PAPER.md`](METHODOLOGY_PAPER.md) 与
[`METHOD_AND_RESULTS.md`](METHOD_AND_RESULTS.md)。

## Selected protocols

| Model | Data shape | DGP | Linear | Nonlinear | Main checkpoint |
|---|---:|---|---|---|---|
| Model 1 | `20 × 20` | mean shift, `tau=.5` | identity marginal-score map | Nonlinear gate (`radial`) | validation-best raw |
| Model 2 | `40 × 40` | common-factor variance, `tau=1` | identity marginal/pairwise maps | Nonlinear gate (`shared_radial`) | validation-best raw |

两个模型都在 `u=logit(pi)` 坐标训练。Stage 1 使用

```text
u | anchor ~ Normal(anchor, sigma_q²)
FSM target = (u - anchor) / sigma_q²
```

Nonlinear gate 是 pooling 前的

$$
\phi_\eta(s,a)=s\,m_\eta(s,a),
\qquad m_\eta(s,a)>0.
$$

`Linear` 等价于固定 $m_\eta\equiv1$；两边后面的 block readout `rho` 都是
nonlinear MLP，因此 “Linear” 只描述 local pre-pooling map。Model 2 的
内部键 `shared_radial` 表示 marginal 和 pairwise channel 在 raw-score 坐标共用
同一个 nonlinear gate，两个 channel 并没有合并。为兼容已有 checkpoint，代码中
继续保留内部键 `radial` 和 `shared_radial`；文档展示名称统一为
**Nonlinear gate**。

Stage 2 对每个模拟或观测数据都使用完全 data-only 的 context：

```text
(u_pilot(Y), frozen_score(Y, u_pilot(Y)))
```

真实 `pi` 只作为 NPE target；exact likelihood/score 只用于冻结后的模拟评测。

## Canonical directory layout

```text
common/
  npe.py                 shared MDN-NPE helpers
  pilot.py               constrained one-dimensional pilot root
model1/
  stage1.py              simulator, features, Linear/nonlinear-gate training
  evaluate_stage1.py     separate frozen exact-score evaluation
  runtime.py             frozen Stage-1 replay
  stage2_npe.py          pilot-score NPE and exact evaluation
model2/
  stage1.py              simulator, marginal/pairwise features, training
  evaluate_stage1.py     separate frozen exact-score evaluation
  runtime.py             frozen Linear/nonlinear-gate replay
  stage2_npe.py          equal-channel pilot-score NPE and exact evaluation
scripts/
  run_model1_stage1.sh
  run_model1_stage2_npe.sh
  run_model2_stage1_40x40.sh
  run_model2_stage2_validation_best_npe.sh
  summarize_results.py
  verify_package.py
tests/                    protocol, nesting, exact-reference and runtime tests
artifacts/                selected checkpoints and compact core results
manifest/                 selected-checkpoint provenance
```

The files above are the canonical run surface. `common/raw_data_npe.py`,
`common/raw_fsm.py`, raw baseline launchers, capacity/sweep launchers, and
the full package's `artifacts/confirmatory_20260820/` are optional diagnostics or
historical provenance. The 61 MB confirmatory tree is intentionally omitted from
this compact workshop archive; it is not imported by the selected Stage-1/Stage-2
launchers and is not required by the tutorial. In particular, the exploratory
structured raw-DeepSets FSM is not part of the reported main comparison.

## Installation

Python 3.12 was used for the packaged run.

```bash
cd model1_model2_clean_workshop_code_compact
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Verify the frozen package before running anything expensive:

```bash
python scripts/verify_package.py
pytest -q
python scripts/summarize_results.py
```

## Model 1

Stage 1 trains Linear and the Nonlinear gate on the same simulation/validation banks and
minibatch stream. Both use the common selected `--iters 20000` budget. The launcher
then invokes a separate frozen exact-score evaluator; the training module itself
never generates held-out exact-score data:

```bash
DEVICE=cuda SEED=20260709 \
  scripts/run_model1_stage1.sh runs/model1_seed20260709
```

A combined Stage-1 directory contains both checkpoints, so Stage 2 accepts that
single directory (the nonlinear checkpoint uses internal key `radial`):

```bash
DEVICE=cuda scripts/run_model1_stage2_npe.sh \
  runs/model1_seed20260709 \
  runs/model1_seed20260709_stage2
```

Important outputs:

```text
runs/model1_seed20260709/model_linear.pt
runs/model1_seed20260709/model_radial.pt
runs/model1_seed20260709/score_summary_by_pi.csv
runs/model1_seed20260709_stage2/posterior_by_seed.csv
runs/model1_seed20260709_stage2/posterior_full_metrics_by_pi.csv
```

## Model 2

The selected size is `40 × 40`; the old `40 × 20` result is not part of this
package's headline protocol. The selected launcher uses a constant learning rate,
so cosine-tail-only arguments are omitted, and it explicitly requests the packaged
final-EMA validation artifact. As in Model 1, exact-score evaluation is a separate
module invoked only after Stage-1 training has finished.

```bash
DEVICE=cuda SEED=20260709 OMP_NUM_THREADS=8 \
  scripts/run_model2_stage1_40x40.sh runs/model2_seed20260709
```

Run Stage 2 from the two validation-best checkpoints saved in that directory:

```bash
DEVICE=cuda scripts/run_model2_stage2_validation_best_npe.sh \
  runs/model2_seed20260709 \
  runs/model2_seed20260709_stage2
```

Important outputs:

```text
runs/model2_seed20260709/model_linear.pt
runs/model2_seed20260709/model_shared_radial.pt
runs/model2_seed20260709/score_summary_by_pi.csv
runs/model2_seed20260709_stage2/posterior_by_seed.csv
runs/model2_seed20260709_stage2/posterior_summary.csv
```

The canonical Stage-2 runner loads validation-best checkpoints only. Earlier
fixed-20k EMA posterior results remain archived for provenance, but their
checkpoint-override launcher is no longer part of the current interface.

## Existing results and inference boundary

The full research bundle contains five independent Stage-1 training seeds for
both Model 1 and Model 2. Those confirmatory files are omitted here to keep the
workshop archive small; the reported aggregate numbers remain documented in
[`METHOD_AND_RESULTS.md`](METHOD_AND_RESULTS.md). The five Stage-2 repetitions
deliberately hold the NPE training seed
and paired test bank fixed, so the reported seed-level intervals measure
Stage-1 training variability, not independent NPE-seed variability.

The exact numerical tables, including per-parameter Stage-1 MSE, posterior-mean
MSE, W1 and posterior-SD error, are in
[`METHOD_AND_RESULTS.md`](METHOD_AND_RESULTS.md).

## Max-stable

The selected Max-stable implementation is a separate package:
[`../maxstable_rainfall79_10k_clean`](../maxstable_rainfall79_10k_clean).
Do not use the older `maxstable_exactsmith_experiment_20260818` directory for
the current Linear-versus-Positive result.
