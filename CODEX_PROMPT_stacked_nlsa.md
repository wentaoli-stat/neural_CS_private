# Codex task: add the stacked NLSA local map `φ(s,β) = (1, s, m_η(s,β))` to Model 1 / Model 2 and run the comparison

## Repository and working directory

Repo root: `/Users/wentao/Code/Research/neural_CS_private`
Work inside: `model1_model2_clean_workshop_code_compact/`

Read these first, in this order:

- `model1_model2_clean_workshop_code_compact/README.md` — selected protocols and launchers
- `model1_model2_clean_workshop_code_compact/METHOD_AND_RESULTS.md` — the current headline numbers you must beat / compare against
- `model1_model2_clean_workshop_code_compact/model1/stage1.py`, `model1/runtime.py`, `model1/stage2_npe.py`
- `model1_model2_clean_workshop_code_compact/model2/stage1.py`, `model2/runtime.py`, `model2/stage2_npe.py`
- `model1_model2_clean_workshop_code_compact/tests/` (all of it)

Do not modify anything under `maxstable_rainfall_workshop_code/`, `three_model_fsm_jiang_shared_npe_github_20260801/`, or `model1_model2_stage1_code_20260710/`.

### Local execution environment

An isolated Python 3.12.13 environment is installed at the repository root in
`.venv-stacked-nlsa/`. From the repository root, activate it before using the
launchers:

```bash
source .venv-stacked-nlsa/bin/activate
export OMP_NUM_THREADS=8
cd model1_model2_clean_workshop_code_compact
```

The original requirements cannot be resolved as written: `sbi==0.24.0` requires
`torch<2.6`, whereas `requirements.txt` requests `torch>=2.8,<2.9`. This local
environment retains SBI 0.24.0 and uses PyTorch 2.5.1. Record that version
difference when comparing new runs with archived results; do not claim an
identical software environment. The existing requirements file remains intact.

The installed direct dependencies are NumPy 2.3.5, SciPy 1.18.1, Matplotlib
3.10.9, pytest 8.4.2, and tqdm 4.70.1. SBI's imports also require the older
ArviZ/PyMC API: use ArviZ 0.23.4 and PyMC 5.28.5, rather than ArviZ 1.x and
PyMC 6.x. To recreate the environment from the repository root:

```bash
uv venv --python 3.12.13 .venv-stacked-nlsa
uv pip install --python .venv-stacked-nlsa/bin/python \
  numpy==2.3.5 torch==2.5.1 scipy==1.18.1 matplotlib==3.10.9 \
  sbi==0.24.0 pytest==8.4.2 tqdm==4.70.1 arviz==0.23.4 pymc==5.28.5
```

Pre-implementation baseline: all 32 existing tests pass in this environment.
The manifest verifier already reports a hash mismatch for
`METHODOLOGY_PAPER.md` and an unmanifested `.DS_Store`; these are pre-existing
issues, not changes introduced by the stacked architecture. CUDA is unavailable
on this Apple Silicon host, so the specified experiment fallback is CPU.

## Background: what exists today vs. what the paper says

The paper (`main_style_revised.pdf`, Section 3.2 and Appendix I.1) defines the transform–sum–readout surrogate score

```
S_ω(X, β) = R_ψ(R_1, ..., R_G, β),    R_g(X, β) = Σ_{j ∈ U_g} φ_{η,g}(s_j(X; β), β)
```

and specifies the local map in **stacked** form:

```
φ_{η,g}(s, a) = [ s ; m_{η,g}(s, a) ]        (Appendix I.1: "stacking, rather than elementwise multiplication, is intended")
ILSA init:  φ_{0,g}(s, a) = [ s ; 0 ]        i.e. m_{η,g} ≡ 0
```

The current code implements something different and strictly less flexible: a **multiplicative positive gate**

```
φ_η(s, a) = s · m_η(s, a),    m_η = softplus(MLP) > 0
```

(`RadialCSBetaDeepSets` in `model1/stage1.py`, `SharedRawRadialCSBetaDeepSets` in `model2/stage1.py`; internal keys `radial` and `shared_radial`). It nests the identity map at initialization (`m_η ≡ 1`), which is why it can be warm-started from ILSA, but because `m_η > 0` it can only rescale the magnitude of each local score and can never change its sign or represent a map that is not of the form `s · (positive function)`.

**Goal of this task:** implement the paper's stacked map, which also nests ILSA exactly at initialization (`m_η ≡ 0`, matched identity/anchor readout columns, nonzero learned-feature readout columns, and a zero constant-channel column) but is as flexible as a generic MLP, and measure whether it gives better numerical performance than the multiplicative gate on the Model 1 and Model 2 toy examples.

Use the stacked map in the form the methodology writes it, including the constant channel:

```
φ_{η,g}(s, β) = ( 1, s, m_{η,g}(s, β) )
```

Note and document that, after within-block pooling with a fixed block size, the constant channel pools to a constant and therefore only contributes a fixed bias to the readout; keep it for fidelity to the methodology but make it a flag so its effect can be checked.

## What to implement

Add a **new** architecture with internal key `stacked` (display label: `stacked NLSA local map`). **Extend, do not replace.** `linear` and `radial` / `shared_radial` must keep working bit-for-bit, and every packaged checkpoint under `artifacts/` must still load and pass its runtime replay test.

### Model 1 (`model1/stage1.py`)

Add `StackedCSBetaDeepSets(nn.Module)` alongside `LinearCSBetaDeepSets` and `RadialCSBetaDeepSets`.

- New module `LocalFeatureMLP` (do not reuse `PositiveMLPMultiplier`): same shape as the existing gate MLP — `Linear(in, gate_hidden) -> SiLU -> Linear -> SiLU -> Linear(gate_hidden, m_dim)` — but **no softplus**, and the final layer initialized with **zero weight and zero bias**, so `m_η ≡ 0` exactly at initialization. Input is `(s_z, anchor_z)` i.e. the same anchor-conditioned input convention the current `radial` gate uses (standardized local score, standardized anchor). Output dimension `m_dim` (see `--m-dim` below).
- Forward pass:
  - identity channel: reuse the **existing** standardized `block` feature verbatim (the one `linear` consumes) — do not recompute it from `s`. This makes the ILSA nesting numerically exact with no delta-form trick.
  - learned channel: `m = LocalFeatureMLP(s_z, anchor_z)` of shape `(n, K, m_channels_per_local, m_dim)`, pooled within block with the **same** `mean(dim=2)` pooling the identity channel uses.
  - constant channel: a column of ones (pools to 1), included iff `include_constant_channel`.
  - readout input: `concat([ones?, block, m_pooled, anchor])` into `rho`, then `.squeeze(-1).sum(dim=1)` over blocks, exactly as now.
- Therefore `tensors_for_method("stacked")` needs input keys `("block", "s", "anchor_z")`, and `forward_method` dispatches accordingly. Register `stacked` in `ARCHITECTURES` with `needs_subscores=True`, `needs_raw_y=False`, `within_block_perm_invariant=True`, `parameter_input="anchor_z"`, a fresh `seed_offset` (e.g. `33`), and `warm_start_from="linear"`.

### Model 2 (`model2/stage1.py`)

Add `SharedStackedCSBetaDeepSets(nn.Module)` and register `"stacked"` in `GATED_METHODS`-style dispatch (extend `tensors_for_method` / `forward_method` / `build_model`; input keys `("block", "s1", "s2", "anchor_z")`).

- **One shared** `LocalFeatureMLP` applied to both the marginal and the pairwise channel, in **raw local-score coordinates** (de-standardize with the `s1_mean/s1_sd`, `s2_mean/s2_sd` buffers exactly as `SharedRawRadialCSBetaDeepSets` does), so the only thing that changes relative to the existing `shared_radial` is stacking vs. multiplication.
- Identity channels: reuse the existing standardized two-column `block` feature verbatim.
- Pool `m(s1)` and `m(s2)` separately within block (mean over `dim=2`), keep them as separate readout columns.
- Readout input: `concat([ones?, block(2 cols), m1_pooled, m2_pooled, anchor])`.

### Exact ILSA nesting at initialization (hard requirement)

The existing `matched_random` protocol must extend to `stacked`:

1. Build the untrained `linear` model under seed `args.seed + ARCHITECTURES["linear"].seed_offset` and save its `rho` state (this already happens in `run()`).
2. Build `stacked`, then copy the linear `rho` weights into the corresponding columns/rows of the stacked `rho`: the first `Linear` layer's weight columns for the identity `block` channel(s) and the `anchor` channel come from the linear `rho`; **retain the ordinary seeded `nn.Linear` random initialization for every learned-feature column `W_m`, and zero only the optional constant-channel column**; biases and all deeper layers are copied verbatim. Write this as an explicit, tested helper (e.g. `load_matched_linear_rho(stacked_rho, linear_rho, column_map)`), not as ad-hoc slicing buried in `run()`.
3. Because `m_η ≡ 0` at init, `W_m m_η = 0` even though `W_m` is nonzero. With the constant-channel column zero and the other weights matched, `stacked(...)` must reproduce the untrained linear prediction to machine precision. Reuse the existing nesting assertion in `run()` (`nesting_error > 1e-5 -> RuntimeError`) and record `nested_initialization_max_abs_diff` in `training_info`.
4. Verify staged gradient flow: the zero-initialized final `m` layer receives a nonzero loss gradient on the first step through nonzero `W_m`; `W_m` receives zero loss gradient on that first step because its input is zero, but receives a nonzero gradient after `m` starts producing nonzero features. Add a test of actual loss gradients and updates over successive steps, distinguishing learning from AdamW weight decay. Earlier hidden layers of `m` can begin learning once its final-layer weights become nonzero. When included, the zero-initialized constant-channel column can receive a gradient immediately because its input is one.

Initialization correction approved by the user: do **not** initialize both `m_η` and `W_m` to zero. Doing so gives `∇_{W_m} L = δ m_η^T = 0` and `∇_η L = (∂m_η/∂η)^T W_m^T δ = 0`, permanently disabling the learned-feature branch under the specified optimizer. Nonzero `W_m` preserves the exact ILSA initialization while removing this gradient deadlock.

### Optimizer groups

Mirror the current two-group setup: put the `LocalFeatureMLP` parameters in the group currently governed by `--gate-lr`, and `rho` in the `--joint-rho-lr` group, so that `stacked` and `radial` are trained under identical learning-rate schedules. Keep `--grad-clip 5` as is. `m_η` is now unbounded (no softplus), so log `max |m|` and the pooled-`m` scale to `training_trace.csv` every `--print-every` steps; if a run diverges, report it rather than silently changing hyperparameters.

### New CLI flags (both models)

- `--m-dim` (int, default `1`) — output dimension of `m_{η,g}`.
- `--include-constant-channel` (0/1, default `1`) — the `1` in `(1, s, m)`.

Keep every existing default unchanged, including `--methods` (`linear,radial` for Model 1, `linear,shared_radial` for Model 2) and every argument asserted in `tests/test_clean_protocol.py::test_formal_defaults`. Persist the new flags into `config.json` so `runtime.py` can rebuild the model.

### Runtime, evaluation, Stage 2

- `model1/runtime.py` / `model2/runtime.py`: the score path currently branches `if self.method == "linear": ... else: <gated>`. Make it a three-way dispatch so `stacked` gets both the standardized `block` feature and the local `s` (Model 1) / `s1, s2` (Model 2) tensors. `parse_method_list` and the `method not in {...}` guards must accept `stacked`. The anchor-conditioning guard (`gate_condition_on_anchor`) applies to `stacked` too.
- `model1/evaluate_stage1.py`, `model2/evaluate_stage1.py`: they already iterate over `config["methods"]`, so they should work once the runtime accepts the key — verify, don't assume.
- `model1/stage2_npe.py`, `model2/stage2_npe.py`: add `"stacked"` to `METHOD_LABELS` and `METHOD_SEED_INDEX` (new index, e.g. `3`; do not renumber existing entries — that would change NPE seeds and break comparability with the archived results). In `model1/stage2_npe.py` the paired-comparison block hard-codes `radial` as the reference method; generalize it so the reference is configurable and so `stacked` appears both as a competitor to `radial` and, ideally, as its own reference row. Keep the default behaviour identical when `stacked` is absent.

### Tests

Update/extend `tests/`:

- `tests/test_clean_protocol.py`: the method-surface assertions become `{"linear", "radial", "stacked"}` and `{"linear", "shared_radial", "stacked"}`; stage-2 label sets gain `"stacked"`. Keep every other assertion as is.
- `tests/test_identity_nesting.py`: add `test_model1_stacked_identity_matches_linear` and `test_model2_stacked_identity_matches_linear` — build the stacked model, apply the matched-rho loader, and assert `assert_close(stacked(...), linear(block, anchor), atol=1e-6, rtol=0)`. Add the trainability test described above.
- New `tests/test_stacked_local_map.py`: assert `m_η ≡ 0` at init; assert the constant channel is genuinely constant after pooling; assert `--m-dim > 1` changes readout input width and still nests exactly; assert a `stacked` checkpoint round-trips through `runtime.AmortizedScoreRuntime` and reproduces `stage1.predict` on the same inputs.
- `tests/test_packaged_runtimes.py` and `tests/test_artifact_contract.py` must pass **unchanged**.

Run `pytest -q` and `python scripts/verify_package.py` before and after; both must pass.

## Experiment protocol

The comparison must be paired and matched. Train all three local maps in a **single** Stage-1 invocation per model per seed, so they share the simulation cache, the validation bank, the minibatch stream, the iteration budget, and the matched untrained `rho` initialization.

### Launchers

Add (do not edit the existing ones):

- `scripts/run_model1_stage1_stacked.sh` — copy of `run_model1_stage1.sh` with `--methods linear,radial,stacked` and the new flags.
- `scripts/run_model1_stage2_npe_stacked.sh` — copy of `run_model1_stage2_npe.sh` with `--methods pilot,linear,radial,stacked` and the added `model_stacked.pt` precondition check.
- `scripts/run_model2_stage1_40x40_stacked.sh` — copy of `run_model2_stage1_40x40.sh` with `--methods linear,shared_radial,stacked`.
- `scripts/run_model2_stage2_validation_best_npe_stacked.sh` — copy with `--methods pilot,linear,shared_radial,stacked`.

Every other hyperparameter stays exactly as in the selected launchers (Model 1: `K=20, m=20, tau=0.5, sigma_q=0.20`, `--iters 20000`, `--hidden 64 --depth 2 --gate-hidden 16`, `--lr-schedule cosine_tail --lr-decay-start-step 10000 --lr-min-ratio 0.1`; Model 2: `K=40, m=40, tau=1.0, sigma_q=0.15`, `--iters 20000`, `--lr-schedule constant`, `--milestone-steps 10000,15000,20000`, `--save-final-ema-validation`). `checkpoint selection remains validation-best raw`, and the exact score must stay out of training and out of checkpoint selection.

### Runs

Primary (do these first, in this order):

1. Model 1, `SEED=20260709`: Stage 1 (3 methods) + `evaluate_stage1`, then Stage 2 (4 methods).
2. Model 2, `SEED=20260709`: Stage 1 (3 methods) + `evaluate_stage1`, then Stage 2 (4 methods).

Then, if the primary runs are stable and the wall-clock budget allows:

3. Repeat both models for `SEED=20260710` and `SEED=20260711` so there are three Stage-1 seeds, matching the confirmatory style of the archived five-seed tables.

Secondary ablation (Model 1 only, seed `20260709`, Stage 1 + `evaluate_stage1` only — Stage 2 not required):

4. `--m-dim 4` — does a vector-valued `m_{η,g}` help?
5. `--include-constant-channel 0` — confirm the constant channel is numerically irrelevant at fixed block size.

Write all outputs under `runs/` (which is gitignored); **never** write into `artifacts/` and never overwrite an existing run directory (the launchers already refuse to).

Use `DEVICE=cuda` if a GPU is visible, otherwise `DEVICE=cpu` with `OMP_NUM_THREADS=8`. Report actual wall-clock per run.

## Deliverables

1. The code changes described above, with `pytest -q` and `scripts/verify_package.py` green.
2. A new file `STACKED_NLSA_RESULTS.md` in `model1_model2_clean_workshop_code_compact/` containing:
   - a short statement of the two maps being compared (`s · m_η(s,a)` with `m_η > 0` vs. stacked `(1, s, m_η(s,a))` with `m_η ≡ 0` at init) and why the stacked one is strictly more expressive;
   - the verified ILSA nesting error at initialization for each run;
   - **Stage 1**, per true `pi` (Model 1: `0.10, 0.30, 0.50, 0.65`; Model 2: `0.07, 0.10, 0.30, 0.50, 0.65, 0.68`), the raw and standardized score MSE for `linear` / gate / `stacked`, plus the relative reduction of `stacked` over both — same table format as `METHOD_AND_RESULTS.md`;
   - **Stage 2**, per true `pi`, posterior-mean MSE vs. the true parameter, posterior-mean MSE vs. the exact posterior mean, W1 to the exact posterior, and posterior-SD error, for all four methods;
   - the paired `stacked` − gate differences with their across-seed spread, and an explicit statement of which comparisons are paired (same Stage-1 cache/stream, same NPE training seed, same 10 paired test datasets, seeds `100`–`109`) and which are not;
   - the `--m-dim 4` and `--include-constant-channel 0` ablation numbers;
   - the training-stability observations for the unbounded `m_η` (max `|m|`, pooled-`m` scale, any divergence).
3. A short section appended to `METHOD_AND_RESULTS.md` and a line in `CHANGELOG.md` pointing to the new results file. Do **not** rewrite the existing headline numbers — the packaged gate results stay as the current record until we decide to switch.
4. A plain-language verdict at the top of `STACKED_NLSA_RESULTS.md`: does the stacked map beat the multiplicative gate, on which model, on which metric, and by how much — and if it does not, say so plainly with the numbers.

## Constraints

- Do not change the DGPs, the anchor/`sigma_q` protocol, the FSM target, the Stage-2 context definition, or any existing default.
- The exact score and the true `pi` remain unavailable to training and to checkpoint selection; they are used only for frozen post-hoc evaluation.
- Do not tune hyperparameters for `stacked` beyond the flags above. If it underperforms under the matched protocol, that is the result — report it. If you believe a specific hyperparameter is the binding constraint, say so in the results file and propose a follow-up run; do not silently run it.
- Commit nothing unless asked. Leave the working tree with the changes and the new results file, and summarize what you ran and what you found.
