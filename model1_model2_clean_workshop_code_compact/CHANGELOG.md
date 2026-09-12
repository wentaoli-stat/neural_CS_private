# Changelog

## 2026-09-12 Model 1 stacked Stage-1 comparison

- Add the optional stacked local map with zero initial features and nonzero
  learned-feature readout columns, matched initialization, frozen runtime replay,
  feature-scale diagnostics, and a three-method Stage-1 launcher. Preserve
  existing architecture defaults and packaged checkpoints. See
  [STACKED_NLSA_RESULTS.md](STACKED_NLSA_RESULTS.md) for the single-seed comparison.
- Add an explicit raw-only checkpoint-selection option; the legacy default
  remains raw-or-EMA. The new launcher uses raw-only as specified in the prompt.
- Exclude generated `runs/` outputs and Finder `.DS_Store` metadata from package
  verification. Refresh the delivery manifest for source changes, including the
  already-existing methodology math-formatting change; artifact hashes remain
  unchanged.

## 2026-08-22 interface cleanup

- 统一将论文和 README 中的非线性方法展示为 `Nonlinear gate`，同时保留旧
  checkpoint 所需的内部键。
- 删除 Stage-1 CLI、函数签名和新配置中从未参与 clean protocol 计算的
  `pi_ref`，并删除只有 `anchor` 一个合法选择的 `feature_pi_mode` 开关。
- 删除只能分别取 `1`、`matched_random` 和 `0` 的
  `gate_condition_on_anchor`、`radial_init`、`gate_only_steps` CLI 假开关；
  三项正式设置改为代码常量并继续写入 checkpoint config。
- local score 现在在代码中直接、明确地使用每条数据的 `anchor_pi`；runtime
  仍兼容带有旧 `feature_pi_mode=anchor` 元数据的冻结 checkpoint。
- 修复 Model 2 `--save-final-ema-validation` 原先 `store_true` 却默认已为真的
  假开关；现在通用 CLI 默认关闭，正式 launcher 明确开启，正式行为不变。
- 精简正式 launchers 中被同值 override、同目录 override 或 constant-LR 分支
  完全忽略的参数；通用 parser 仍保留真正可用于消融的不同训练预算、flow 与
  cosine-tail 选项，并在 help 中标清 model-specific 条件。
- 将 Model 1 Stage-2 parser 的 simulation/NPE seed 默认值对齐正式 launcher
  (`20260723/54000`)，并将 Model 2 默认 exact-diagnostic grid 对齐正式六个
  `pi` 点；因此直接调用 canonical parser 与通过 launcher 的默认协议一致。
- 删除 Model 2 Stage-2 最后一个完全无效的 `--score-batch-size`：旧实现虽然
  将它传入 `pilot_score_contexts`，函数从未读取；真正控制该路径分批的是
  `--pilot-batch-size`。Model 1 的同名参数确实控制 frozen-score forward，保留。
- 删除 Model 1/2 Stage-1 中未用于 checkpoint selection、Stage-2 或主结果的
  fixed-anchor FSM curve，以及对应的 `--validation-pi-values` 和
  `--n-validation-per-anchor`；历史 artifacts 中原有 CSV 保留为旧运行记录。
- 删除 Model 1 历史遗留的 `--linear-iters`、`--radial-iters` 不等预算开关；
  两个方法现在与 Model 2 一样统一由 `--iters` 控制，正式 20k 行为不变。
- 将 Model 1/2 的 exact-score test 从 Stage-1 trainer 拆到独立的
  `evaluate_stage1.py`；trainer 不再暴露 `--n-test`、diagnostic grid 或
  `--skip-exact-diagnostics`，并只保存 held-out RNG 起始状态。正式 launcher
  在训练完全结束后才调用 evaluator，因此测试隔离在代码结构上强制成立。
- 删除 Model 1/2 Stage 2 的 `--stage1-run-dirs` 方法级目录覆盖层；canonical
  Stage 1 已在同一目录保存两个方法，因此 Stage 2 现在只接受一个
  `--stage1-run-dir`。Model 1 launcher 也由三个位置参数精简为两个，并删除只为
  混合旧目录服务的 Model 2 helper；历史 artifacts 不作改写。
- 统一 Model 1/2 Stage 2 的 `--context-score-batch-size` 与 `--pilot-cache`；
  Model 1 pilot 直接复用 `--device`。Model 2 的新运行固定为 validation-best、
  equal-channel 和 Torch，删除 `--stage1-checkpoints`、`--pilot-mode`、
  `--pilot-backend` 以及 fixed-20k launcher。底层历史 checkpoint/NumPy reference
  仍可用于 artifact 测试，但不再暴露为 canonical CLI。
- 删除把 pilot-root 附近小 score 误判为网络死亡的 `--min-mean-abs-score`；
  sanity guard 仍检查确定性、pilot 相关性、有限数值和爆炸上限。

## 2026-08-20 raw-data baselines

- 增加 Model 1 / Model 2 各五个独立 Stage-1 training seeds 的 validation-best Stage-1/Stage-2 确认性汇总，并将所有逐 `pi` 主结果表更新为五-seed 等权均值。
- 增加以 Stage-1 training seed 为配对推断单位的置信区间和检验；明确五次 Stage 2 共用同一 NPE training seed，尚未覆盖 NPE-seed 变异。
- 增加 Model 1 / Model 2 共用的 literal `flatten(Y)` Stage-2 NPE runner；不使用 pilot、score 或真实参数 context。
- 按正式 50k simulation、NPE 和 60-case test 协议运行并打包两套 artifacts。
- 在结果文档中加入逐 `pi` 与 pooled raw/Linear/Nonlinear-gate 对照，并扩展测试、结果重算和 manifest。
- 将原 `raw Y -> 32D phi -> block pooling -> score` 明确重命名为 structured raw DeepSets exploratory；它不是 direct raw baseline。
- 增加真正的 amortized direct raw-FSM：`concatenate(flatten(Y), anchor) -> 2x64 MLP -> score`，不含 phi、gate、pooling、DeepSets 或 blockwise sum；按同一 Stage-1/Stage-2 协议运行并打包 Model 1/2 artifacts。

## 2026-08-18 clean edition

- 新建独立目录；没有覆盖或删除旧 package。
- 只保留 Model 1 `20×20` 和 Model 2 `40×40` 的当前 Stage-1/runtime/data-only NPE 主链。
- 从旧 two-stage 文件抽取最小 SBI helper，从 Mode-A 抽取纯一维 pilot root helper；正式 Stage 2 不再有 legacy import closure。
- Model 1 CLI 默认值改为正式 `tau=.5, sigma_q=.20, 20k, matched_random`。
- Model 2 CLI 默认值改为正式 `40×40, sigma_q=.15, 20k, shared_radial, matched_random`。
- Model 2 删除 separate `radial`、`bounded_shared` 以及旧 feature-mode 的公开入口；Model 1 删除 Khoo/Jiang 入口。
- 新 launcher 检查 `feature_stats.npz`，拒绝覆盖输出，并显式区分 Model 2 fixed-20k EMA 与 validation-best Stage 2。
- 只携带 Model 1 当前 artifacts，以及 Model 2 两个 40×40 Stage-1 seeds和两套 40×40 NPE；没有混入旧 40×20 / diffusion 结果。
- 删除新包内冗余的 Model 1 posterior sample arrays；它们仍可从未改动的旧 package 恢复。
- 增加 checkpoint selection manifest、全文件 SHA-256 manifest、结果重算脚本和测试。
