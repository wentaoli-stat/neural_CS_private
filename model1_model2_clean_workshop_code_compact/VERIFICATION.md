# Verification

## 一键验证

```bash
python scripts/verify_package.py
pytest -q
python scripts/summarize_results.py
```

第一条检查精简包 manifest 和正式 checkpoint SHA；第二条检查 import closure、exact reference、identity nesting、artifact runtime replay 和 fixed checkpoint provenance；第三条从包内 CSV 重算核心结果。完整五种子汇总需要完整版中的 `artifacts/confirmatory_20260820/`；精简包会明确跳过该部分。

## 已执行的检查

- Python compile/import：Model 1 和 Model 2 的 Stage 1、runtime、Stage 2、NPE helper 全部可导入。
- launcher：六个 shell script 全部通过 `bash -n`。
- method surface：Model 1 Stage 1 支持 `linear/radial/stacked`，Stage 2 仍为 `linear/radial`；Model 2 仍为 `linear/shared_radial`；另有 data-only `pilot`、naive `flatten(Y)` 和 raw-FSM representation baselines。
- legacy closure：新版源码没有 import 旧 fixed-center、two-stage monolith、Mode-A、Jiang、anchor-score 或 score-only 文件。
- frozen runtime：打包的 Model 1/2 Linear/Nonlinear-gate validation-best/fixed-20k checkpoints 均通过 forward replay、finite-difference derivative、block permutation 和 within-block permutation checks；对应内部键分别为 `radial` 与 `shared_radial`。
- Model 2 fixed policy：显式 checkpoint 必须同时满足 `selection=fixed_milestone`、`checkpoint_source=ema` 和 `checkpoint_step=config.iters=20000`。
- exact reference：打包 exact posterior helper直接使用当前 DGP 的 full block log-ratio，仅供 evaluation。
- raw-data baseline：检查 context 是 canonical literal flatten、没有参数输入，模拟器逐元素复现正式 Stage-2 DGP，并审计两套 50k artifacts 各含完整 60-case evaluation。
- raw-FSM baseline：检查 raw block encoder 的 block/within-block permutation symmetry、anchor conditioning、runtime checkpoint replay、data-only pilot 以及 Stage-2 context 契约。
- compact scope：不包含 `artifacts/confirmatory_20260820/` 多种子历史树；正式选中的 checkpoint、核心结果与其 SHA provenance 均保留。

## Manifest 规则

`MANIFEST.sha256`：

- 使用 relative POSIX paths；
- 按路径排序；
- 覆盖所有交付文件但不包含自身；
- 可由 `scripts/verify_package.py` 在 macOS/Linux 一致校验。
- 不包含生成的 `runs/`、Python/test caches、SBI logs 或 Finder `.DS_Store`。

## Stacked initialization and compatibility checks

The new Model-1 tests check zero initial learned features, nonzero learned-feature
readout weights, matched ILSA predictions for scalar/vector features and with/without
the constant channel, successive-step gradient flow without weight decay,
permutation invariance, checkpoint/runtime replay, raw-only checkpoint selection,
and a small three-method training/evaluation cycle. Existing packaged-runtime and
artifact-contract tests remain unchanged. A separate five-step regression against
the pre-change source confirms bitwise-identical legacy Linear/radial weights and
training traces with the unchanged default checkpoint policy.

Before this extension, all 32 baseline tests passed; the original manifest failed
on the pre-existing methodology hash mismatch and `.DS_Store`. The methodology
content is preserved, its manifest digest is refreshed, and Finder metadata is
excluded. No artifact is rehashed to conceal a content change: all archived
artifact digests must still equal their pre-change values.

原始 artifact `config.json` 中可能保留远端绝对路径，这是运行 provenance，不是 portable launcher 参数。可移植的 parent/checkpoint 关系由 `manifest/selections.json` 给出。
