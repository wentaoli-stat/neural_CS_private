# Three-model Direct-FSM vs Jiang R1 + shared NPE

This repository is a clean, uploadable snapshot of the experiments comparing:

- **Ours linear**: anchored Direct Fisher score matching with linear local-score aggregation;
- **Ours nonlinear**: the matched estimator with a learned nonlinear gate;
- **Jiang R1 + shared NPE**: the official Jiang Round-1 score estimator, followed by the same pilot-plus-score NPE used for our methods.

The three data-generating models are a blockwise mean-shift mixture (Model 1), a blockwise common-factor mixture (Model 2), and a nonlinear-emission HMM (Model 3). The full method, estimands, fairness controls, and numerical tables are documented in [docs/EXPERIMENT.md](docs/EXPERIMENT.md).

## Headline results

Pre-NPE score error is standardized MSE against the exact full-data score on held-out simulations. Post-NPE RMSE compares the learned posterior mean with the **exact posterior mean**, not with the generating parameter.

| Model | Jiang R1 score stdMSE | Ours linear | Ours nonlinear |
|---|---:|---:|---:|
| Model 1 | 0.4035 | 0.0978 | **0.0847** |
| Model 2 | 0.4541 | 0.1521 | **0.1140** |
| Model 3 | 0.7976 | 0.2113 | **0.1549** |

| Model | Jiang R1 + NPE posterior-mean RMSE | Ours linear + NPE | Ours nonlinear + NPE |
|---|---:|---:|---:|
| Model 1 | 0.03014 | 0.01642 | **0.01482** |
| Model 2 | 0.03272 | 0.02534 | **0.02189** |
| Model 3 | 0.01426 | 0.01213 | **0.01005** |

Machine-readable copies are in [results/headline_score_stdmse.csv](results/headline_score_stdmse.csv) and [results/headline_posterior_mean_rmse.csv](results/headline_posterior_mean_rmse.csv). Seed-level and method-level outputs are retained under `results/`.

## Repository layout

```text
configs/       Frozen experiment configurations and portable checkpoint paths
docs/          Combined report plus model-specific method notes
experiments/   Frozen Model 1/2 training code and the Model 3 upstream snapshot
results/       Reported CSV/JSON results; no model weights
src/           Jiang adapters, DGPs, score diagnostics, and shared-NPE runners
tests/         Unit and method-fidelity tests
artifacts/     Expected local checkpoint layout; large files are Git-ignored
scripts/       Package integrity check
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[npe,test]'
```

The Jiang adapter imports the authors' repository without modifying it. Clone the exact revision used for these experiments:

```bash
git clone https://github.com/Haoyu-Jiang/Structured_Score_Matching.git external/Structured_Score_Matching
git -C external/Structured_Score_Matching checkout fb273f0e1bbfca2d1d752c97d1f9d431dd1039c9
```

The default path is `external/Structured_Score_Matching`. A different checkout can be selected with:

```bash
export JIANG_SSM_REPO=/absolute/path/to/Structured_Score_Matching
```

The three frozen Model 3 support files are already included in `experiments/model3_upstream/` and are hash-checked when loaded. `KHOO_FSM_REPO` can override that location.

## Checkpoints and full reproduction

Large PyTorch weights and generated arrays are intentionally excluded from this GitHub bundle. Place or regenerate checkpoints using the layout in [artifacts/checkpoints/README.md](artifacts/checkpoints/README.md). The retained JSON configurations identify every formal seed and hyperparameter.

Useful entry points are:

```bash
# Jiang paper-native Round 1/2 smoke run for a block model
python -m khoo_vs_jiang.block_mixture_jiang \
  --model model1 --profile smoke --output-dir runs/jiang_model1_smoke

# Formal Model 3 Direct-FSM Stage 1 (repeat for the three declared seeds)
python -m khoo_vs_jiang.nonlinear_emission_screen \
  --output-dir runs/model3_seed20260721 --seed 20260721 \
  --rho 0.2 --scale 3 --n-train 40000 --n-val 8000 \
  --sigma-q 0.25 --iters 3000 --gate-only-steps 400

# Shared NPE for Model 1 after checkpoints have been installed
FSM_CHECKPOINT_ROOT="$PWD/artifacts/checkpoints/ours" \
python -m khoo_vs_jiang.block_mixture_stage2_npe \
  --model model1 --output-dir runs/model1_shared_npe

# Shared NPE for one Model 3 Stage-1 checkpoint
python -m khoo_vs_jiang.nonlinear_emission_stage2 \
  --checkpoint artifacts/checkpoints/ours/model3/seed20260721/models.pt \
  --output-dir runs/model3_shared_npe_seed20260721
```

For Models 1 and 2, the frozen training programs are under `experiments/block_models/`; exact formal settings are in `configs/model1/stage1/` and `configs/model2/stage1/`. See the model-specific documents before launching the expensive runs.

## Validation

The repository can be audited without checkpoints:

```bash
python scripts/check_package.py
python -m compileall -q src experiments tests
pytest -q tests/test_dgp.py
```

Tests that import Jiang require the external checkout above. Full posterior reruns additionally require the Stage-1 checkpoints and `sbi`.

## Interpretation and fairness notes

- Jiang contributes its official **Round-1** score estimator. Its per-block scores are summed for Models 1/2; for Model 3 one complete trajectory is the score unit.
- Every post-NPE arm uses the same simulated datasets, data-only pilot, two-dimensional context, NPE family, optimizer, seed, test objects, and exact-posterior reference.
- Jiang Round 2 is intentionally absent from the amortized NPE table because its proposal and retrained score are tied to one observed dataset.
- Exact likelihoods and exact scores are used only for held-out evaluation and exact-posterior references, never as Stage-1 training inputs.
- The proposed estimator targets a Gaussian-smoothed score at finite proposal scale `sigma_q`; exact-score MSE therefore also contains smoothing bias.
- Ours uses five Stage-1 seeds in Models 1/2 and three in Model 3; the Jiang headline is one frozen R1 fit. The seed counts are disclosed in the detailed tables.

## Source and licensing note

[SOURCE_SNAPSHOT.md](SOURCE_SNAPSHOT.md) records where every component came from and the frozen Jiang commit. The external Jiang repository is not vendored here and remains subject to its own license. This bundle deliberately does not invent a license or author metadata; add the intended `LICENSE` and citation information before making the GitHub repository public.
