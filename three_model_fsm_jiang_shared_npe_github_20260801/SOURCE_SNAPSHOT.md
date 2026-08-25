# Source snapshot

This directory was assembled as a publication-oriented snapshot of the three-model comparison recorded on 2026-07-25.

## Included code

- `src/khoo_vs_jiang/`: comparison orchestration, DGPs, Jiang adapters, score diagnostics, and shared-NPE experiments.
- `src/shared_npe_backend.py`: the common `sbi` loading, NPE training, posterior sampling, and Wasserstein helpers used by all Stage-2 runners.
- `experiments/block_models/current_model1_fsm_method_package/`: Model 1 Direct-FSM implementation used by the frozen runs.
- `experiments/block_models/current_model2_fsm_method_package/`: Model 2 Direct-FSM implementation used by the frozen runs.
- `experiments/model3_upstream/`: the frozen Model 3 support files imported by the comparison package.

The three Stage-1 Model 3 files are checked by SHA-256 in `src/khoo_vs_jiang/upstreams.py`; the additional two-stage helper supports the documented two-parameter diagnostic.

## External Jiang implementation

Repository:

```text
https://github.com/Haoyu-Jiang/Structured_Score_Matching.git
```

Frozen commit:

```text
fb273f0e1bbfca2d1d752c97d1f9d431dd1039c9
```

The comparison imports `MLE/utils_sm.py` from that checkout. The external repository is not copied or modified by this bundle.

## Result provenance

The report-level numerical claims are backed by the retained files under `results/`. In particular:

- Model 1/2 score results: `results/model{1,2}/score_by_seed/` and `jiang_paired_score.json`;
- Model 1/2 shared-NPE results: `results/model{1,2}/shared_npe/`;
- Model 3 Stage-1 score results: `results/model3/stage1/` plus `jiang_r1_score_diagnostic.json`;
- Model 3 shared-NPE results: `results/model3/shared_npe/` and the Jiang summary referenced in `docs/EXPERIMENT.md`.

Checkpoint paths in configs and protocols were rewritten to repository-relative placeholders. Numerical outputs were not changed.
