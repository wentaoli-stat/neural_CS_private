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
| `FOLLOWUP_FINDINGS_20260913.md` | Stage-2 replicates of the random streams; 60k-step Stage 1; bandwidth grid σ_q ∈ {0.20, 0.5, 0.977, 1.052, 2.0} with held-out-NLL selection; p=3 Stage 2; Model 2 linear learning-rate audit | Claude Code, desktop | `06487d3` | **current; revises the two Stage-2 reports below** |
| `STAGE2_INPUT_COMPARISON_20260913.md` | Model 1 p=1 Stage 2: NPE on raw Y and on all subscores; raw-input, all-subscores, linear, gate and stacked Fisher scores; σ_q = 0.20 and 1.052; 5 seeds | Claude Code, laptop | `3ea303c` | one Stage-2 draw; tiers reliable, margins between linear / gate / stacked are not |
| `STAGE2_FINDINGS_20260913.md` | Model 1 p=1 Stage 2: linear, gate, stacked at both bandwidths, readouts matched; measured pilot error Σ_e | Claude Code, desktop | `1960884` | one Stage-2 draw; margins revised by the follow-up |
| `FINDINGS_20260912.md` | Stage 1 only: learning-rate audit (Model 1 p=1 and p=3, Model 2 linear), p=3 model, `m_dim` sweeps, input representations, simulation and optimization budgets, bandwidth at Stage 1 | Claude Code, laptop | `43a4b2e` | partly outdated, see below |
| `STACKED_NLSA_RESULTS.md` | first stacked-map comparison: Model 1 p=1, Stage 1, one seed | Codex | `dda60b8` | superseded |

### What the follow-up revises

- **SEs in both Stage-2 reports were too small.** They held the NPE bank, NPE seed and test bank fixed,
  so they covered Stage-1 variability only, and that component is the smaller one.
- **Fair nonlinear gain at σ_q = 1.052 is ≈25%** (gate −25.6%, stacked −25.0%, 15/15), not the 10.3%
  and 6.2% reported from one draw. It agrees with the paper abstract's 24.7%.
- **At σ_q = 0.20 the gate beats linear by 15%** (≈3.3 SE), where one draw said "not significant".
- **The bandwidth gain holds for gate (−15.0%) and stacked (−19.9%)** on all 15 units, but **not robustly
  for linear** (−3.0%, 9/15).
- **The wide tube's advantage is not a convergence artifact**: 60k-step Stage 1 changes nothing.

### What is outdated in `FINDINGS_20260912.md`

- Every row marked **UNMATCHED**: the readout learning rates of the two arms differ.
- "Stacked ties the gate at p=1": an artifact of the `lr 1e-4` readout. With readouts matched at 1e-3
  the gate is better at Stage 1.
- "Stacked beats the gate at p=3": **contradicted** by the follow-up (F4). With readouts matched at 1e-3
  they tie at Stage 1, and at Stage 2.
- §7 says the wide tube's score is "accurate where the pilot lands". A later check showed it is less
  accurate than the narrow tube's at every displacement tested; the wide tube wins at Stage 2 for
  another reason.
- §1's Model 2 linear value at `lr 1e-3` (0.09163) is **not reproduced** by the follow-up (0.09571); the
  `lr 1e-4` value reproduces exactly. Unreconciled.

## Not yet written to any file

- **Learned-score MLE check** (seed 20260709 only): the root of the narrow-tube score is as good an
  estimator as the exact MLE (RMSE 0.633 vs 0.637 in logit π), while the wide tube's is worse (0.68–0.69);
  and the narrow-tube score stays accurate out to a displacement of 1.0 in logit π.
- **Convergence curves** (p=3, Stage 1): https://claude.ai/code/artifact/6a273a4c-464b-43fb-9df7-363e1da5899f

## Raw run outputs

Not in git (`runs/` is gitignored). The laptop's `../runs/` holds the input comparison (`runs/ic/`) and
the Stage-1 studies; the desktop's `runs/` holds the Stage-2 reports, the replicates, the bandwidth grid,
p=3 Stage 2 and the Model 2 audit. `runs/` stays at the package root because the launchers and
summarizers use that path. Tables regenerate from it:

| document | regenerate with | runs needed |
|---|---|---|
| `FOLLOWUP_FINDINGS_20260913.md` | `scripts/summarize_followup_findings.py` | desktop |
| `STAGE2_INPUT_COMPARISON_20260913.md` | `scripts/summarize_input_comparison.py`, `scripts/compare_input_bandwidths.py` | laptop |
| `STAGE2_FINDINGS_20260913.md` | `scripts/summarize_stage2_findings.py` | desktop |
| `FINDINGS_20260912.md` | `scripts/summarize_session_findings.py` | laptop |
| `STACKED_NLSA_RESULTS.md` | `scripts/summarize_model1_stacked.py` | laptop |

All five write into this folder.

## Code with no results yet

Model 2 gate and stacked arms (`../model2/stage1.py`; ~50 h per gate run on the desktop CPU), the p=3
Model 2 module (`../model2_p3/`), and a p=3 Stage 2 with the draft-rule wide tube.
