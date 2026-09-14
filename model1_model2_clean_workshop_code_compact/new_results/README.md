# New results (from 2026-09-11 onward)

Everything in this folder was produced after commit `978121b`, the state of the repository when this
line of work began. Results that existed before it — uploaded by another contributor — are untouched
in their original places: `../METHOD_AND_RESULTS.md`, `../METHODOLOGY_PAPER.md`, `../artifacts/`,
`../manifest/`, and the sibling packages `three_model_fsm_jiang_shared_npe_github_20260801/` and
`maxstable_rainfall_workshop_code/`.

`METHOD_AND_RESULTS.md` had one section appended during this work, pointing to
`STACKED_NLSA_RESULTS.md`. That section is removed and the file restored byte-for-byte to its `978121b`
version, so it again holds only the contributor's results.

## Documents

Newest and most reliable first.

| file | covers | produced by | commit | status |
|---|---|---|---|---|
| `STAGE2_INPUT_COMPARISON_20260913.md` | Model 1 p=1 Stage 2: NPE on raw Y and on all subscores; raw-input, all-subscores, linear, gate and stacked Fisher scores; σ_q = 0.20 and 1.052; 5 seeds | Claude Code, laptop | `3ea303c` | current |
| `STAGE2_FINDINGS_20260913.md` | Model 1 p=1 Stage 2: linear, gate, stacked at both bandwidths, readouts matched; measured pilot error Σ_e | Claude Code, desktop | `1960884` | current; reproduced by the file above |
| `FINDINGS_20260912.md` | Stage 1 only: learning-rate audit (Model 1 p=1 and p=3, Model 2 linear), p=3 model, `m_dim` sweeps, input representations, simulation and optimization budgets, bandwidth at Stage 1 | Claude Code, laptop | `43a4b2e` | partly outdated, see below |
| `STACKED_NLSA_RESULTS.md` | first stacked-map comparison: Model 1 p=1, Stage 1, one seed | Codex | `dda60b8` | superseded |

### What is outdated in `FINDINGS_20260912.md`

- Every row marked **UNMATCHED**: the readout learning rates of the two arms differ.
- "Stacked ties the gate at p=1": an artifact of the `lr 1e-4` readout. With readouts matched at 1e-3
  the gate is better (`STAGE2_FINDINGS_20260913.md`).
- "Stacked beats the gate at p=3": matched, but at readout 1e-4, the setting that produced the false
  p=1 tie. Not yet rerun at 1e-3.
- §7 says the wide tube's score is "accurate where the pilot lands". A later check showed it is less
  accurate than the narrow tube's at every displacement tested; the wide tube wins at Stage 2 for
  another reason.

## Not yet written to any file

- **Learned-score MLE check** (seed 20260709 only): the root of the narrow-tube score is as good an
  estimator as the exact MLE (RMSE 0.633 vs 0.637 in logit π), while the wide tube's is worse (0.68–0.69);
  and the narrow-tube score stays accurate out to a displacement of 1.0 in logit π.
- **Convergence curves** (p=3, Stage 1): https://claude.ai/code/artifact/6a273a4c-464b-43fb-9df7-363e1da5899f

## Raw run outputs

Not in git (`runs/` is gitignored). Everything under `../runs/` on the laptop is new; the desktop has
its own `runs/` (`s1_sq*`, `s2_sq*`, `pilot_error/`). `runs/` stays at the package root because the
launchers and summarizers use that path. Tables regenerate from it:

| document | regenerate with |
|---|---|
| `STAGE2_INPUT_COMPARISON_20260913.md` | `scripts/summarize_input_comparison.py`, `scripts/compare_input_bandwidths.py` |
| `STAGE2_FINDINGS_20260913.md` | `scripts/summarize_stage2_findings.py` (needs the desktop's `runs/`) |
| `FINDINGS_20260912.md` | `scripts/summarize_session_findings.py` |
| `STACKED_NLSA_RESULTS.md` | `scripts/summarize_model1_stacked.py` |

All four write into this folder.

## Code with no results yet

Model 2 stacked arms (`../model2/stage1.py`), the p=3 Model 2 module (`../model2_p3/`), the Model 2
nonlinear learning-rate audit (`../scripts/run_model2_lr_audit.sh`), and the desktop's follow-ups (p=3
Stage 2, held-out NPE likelihood, follow-up job queues).
