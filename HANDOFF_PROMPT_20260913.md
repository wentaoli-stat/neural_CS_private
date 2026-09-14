# Handoff: Stage-2 posterior inference for Model 1 (p=1)

You are picking up a research session from another machine. The code is in this git repo; the
experiment outputs (`runs/`) were deliberately **not** synced, so you will regenerate everything
you need. Work inside `model1_model2_clean_workshop_code_compact/`.

## 1. Read first

1. `model1_model2_clean_workshop_code_compact/new_results/FINDINGS_20260912.md`. This holds the previous session's
   results. **All of them are Stage 1 only** (frozen exact-score MSE); no posterior was ever computed.
   Pay attention to the "Read this first" section.
2. `main_style_revised.pdf` (repo root), §3.2–3.3: the NLSA architecture and the tube FSM
   bandwidth rule `Σ_q = 2Σ_e`.
3. `model1_model2_clean_workshop_code_compact/README.md` and `scripts/run_model1_stage2_npe.sh`:
   the packaged Stage-2 protocol.

`scripts/summarize_session_findings.py` reads `runs/`, which does not exist here. Do not try to run it.

Three facts from the previous session that decide how to run this:

- **The packaged `lr 1e-4` under-trains.** In Stage 1, linear improved 27–55% at `lr 1e-3`.
- **Readout learning rates were never matched.** In `model1/stage1.py`, linear trains with `--lr`,
  while gate (`radial`) and `stacked` train their readout with `--joint-rho-lr` and their local branch
  with `--gate-lr`. Earlier runs raised only linear's rate, so every nonlinear-versus-linear Stage-1
  gap in the findings is confounded. Stacked vs gate was matched and is trustworthy.
- **The draft bandwidth looks terrible at Stage 1 but may not be.** With `σ_q = 1.052` (the draft's
  `2Σ_e` rule) instead of the packaged `0.20`, Stage-1 exact-score MSE rose ~20× and all three methods
  collapsed to the same value (a smoothing-bias floor). Stage-1 MSE measures error at the true
  parameter, where a narrow tube is favoured by construction; Stage 2 evaluates the score at a
  *pilot*, which is what the wide tube is designed for. **Settling this is the main purpose of this
  handoff.**

## 2. Setup

- Python 3.12 venv. **Don't** `pip install -r requirements-dev.txt` as-is: `requirements.txt` pins
  `torch>=2.8,<2.9`, but every previous result came from torch 2.5.1. Install the same list with torch
  replaced:

  ```bash
  pip install "torch==2.5.1" "numpy>=2.3,<2.4" "scipy>=1.18,<1.19" "matplotlib>=3.10,<3.11" \
              "sbi==0.24.0" tqdm "pytest>=8,<9"
  ```

  The source environment was Python 3.12.13, torch 2.5.1, numpy 2.3.5, sbi 0.24.0. If pip refuses
  that combination on your platform, report it instead of silently upgrading torch.
- Run `pytest -q` from the package directory. Everything should pass (86 tests at handoff). Stop and
  report if not.
- `python scripts/verify_package.py` is expected to **fail** on hash mismatches: the manifest predates
  this work and is only regenerated at release. Don't "fix" it.
- Detect the device (`cuda` > `mps` > `cpu`) and use it for training; keep `--data-device cpu`.
- **Time before committing compute.** No Stage-2 wall time was ever recorded. Run one Stage-1 seed
  and one Stage-2 invocation with `--n-sbi-train 5000` first, extrapolate, and tell the user the
  projected total before launching the full plan. Previous CPU timings (Apple M3 Max, contended):
  Stage-1 p=1 linear ~1 min, gate and stacked ~4–80 min each depending on load.

## 3. Code change: add `stacked` to Model 1 Stage 2

`model1/stage2_npe.py` supports only `pilot`, `linear` and `radial`. The Stage-1 trainer and
`model1/runtime.py` already support `stacked`.

- Add `"stacked"` to `METHOD_LABELS` and to `METHOD_SEED_INDEX` with index **3**. Do not renumber
  existing entries: that would change NPE seeds and break comparability with archived results.
- `paired_w1_comparisons` hard-codes `radial` as the reference (around line 293). Generalize it so it
  reports stacked−radial, radial−linear and stacked−linear, and behaves exactly as before when stacked
  is absent.
- Add a launcher that copies `scripts/run_model1_stage2_npe.sh`, requires `model_stacked.pt`, and
  passes `--methods pilot,linear,radial,stacked`.
- Update `tests/test_clean_protocol.py` for the new method surface and add a smoke test that pushes a
  tiny stacked checkpoint through Stage 2. `pytest -q` must pass.

## 4. Stage 1: retrain with matched readouts, at two bandwidths

Every arm at the same rates, in **one** invocation per seed so all methods share the simulation bank
and minibatch stream:

```bash
python -u -m model1.stage1 \
  --n-train 40000 --n-val 8000 --n-blocks 20 --block-size 20 --tau 0.5 \
  --anchor-pi-min 0.05 --anchor-pi-max 0.70 --sigma-q "$SIGMA_Q" --seed "$SEED" \
  --methods linear,radial,stacked --hidden 64 --depth 2 --gate-hidden 16 --m-dim 8 \
  --iters 20000 --batch-size 512 --lr 1e-3 --gate-lr 1e-3 --joint-rho-lr 1e-3 \
  --lr-schedule cosine_tail --lr-decay-start-step 10000 --lr-min-ratio 0.1 \
  --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 --checkpoint-selection raw \
  --patience 0 --print-every 100 --device "$DEVICE" --output-dir "runs/s1_sq${SIGMA_Q}_${SEED}"
python -u -m model1.evaluate_stage1 --run-dir "runs/s1_sq${SIGMA_Q}_${SEED}" \
  --n-test 5000 --pi-values 0.10,0.30,0.50,0.65 --batch-size 512 --device "$DEVICE"
```

- `SIGMA_Q ∈ {0.20, 1.052}`; `SEED ∈ {20260709 … 20260713}`. Start with one seed for both bandwidths.
- Before Stage 2, confirm from each `config.json` that `lr`, `gate_lr` and `joint_rho_lr` all equal
  `1e-3`, and from `training_info.json` that every arm shares one `minibatch_sha256` and that the
  nesting error is ≤ 1e-5.
- `1.052` is `√(2·E[I⁻¹])` over the anchor prior, assuming an efficient pilot. If there's time,
  measure the real pilot's error variance `Σ_e` by simulation and report `√(2Σ_e)` next to it.

## 5. Stage 2

Follow `scripts/run_model1_stage2_npe.sh` exactly (50k NPE simulations, MDN, test π ∈
{0.07, 0.10, 0.30, 0.50, 0.65, 0.68}, test seeds 100–109, fixed NPE and posterior seeds), with
`--methods pilot,linear,radial,stacked`, once per Stage-1 directory.

- Keep the NPE training seed and the test bank identical across every run, so all differences are
  paired.
- The launcher accepts `PILOT_CACHE`. Before reusing one cache across Stage-1 runs, check whether the
  pilot depends on the Stage-1 checkpoint (the selected pilot mode is `marginal`). Reuse only if it
  doesn't.
- The exact posterior grid is evaluation-only. It must not influence training, NPE, or any selection.

## 6. What to report

Write `model1_model2_clean_workshop_code_compact/STAGE2_FINDINGS_<date>.md` together with a script that
regenerates its tables from `runs/`. For each bandwidth × method, per true π and pooled:

- posterior-mean MSE vs truth, RMSE vs the exact posterior mean
- W1 to the exact posterior
- posterior SD vs the exact posterior SD
- 90% coverage

Plus paired W1 differences with SE and win counts: stacked−gate, gate−linear, stacked−linear, and
**σ_q=1.052 − σ_q=0.20 within each method**. Include the pilot-only row as the floor.

Answer these directly, with numbers:

1. **Bandwidth.** Does the draft `σ_q = 1.052`, ~20× worse in Stage-1 score MSE, recover at Stage 2,
   or even win? This is the headline question.
2. **Stage 1 → Stage 2.** Does the Stage-1 ordering (stacked ≤ gate < linear) survive in posterior W1,
   and is the gap larger or smaller?
3. **Fair nonlinear-versus-linear gap.** With readouts matched, what's the Stage-2 gain of gate and
   stacked over linear?

State the number of seeds behind every figure. Label anything with n=1 as indicative. If a result
contradicts `FINDINGS_20260912.md`, say so plainly, and don't reconcile it by retuning.

## 7. Constraints

- Don't modify anything under `artifacts/`, and don't overwrite existing run directories (the
  launchers refuse to).
- Don't tune hyperparameters by looking at posterior or exact-score metrics. If you think a setting is
  binding, say so and propose a follow-up run.
- Don't commit or push unless the user asks.
- **Out of scope unless the user approves:** p=3 Stage 2 (needs a 3-d pilot, a ~80³ exact-posterior
  grid and a multivariate W1, none of which exist yet), and the Model 2 nonlinear learning-rate audit
  (`scripts/run_model2_lr_audit.sh`, about 1.8 h per gate run on Apple MPS). Mention them as next steps
  only.
