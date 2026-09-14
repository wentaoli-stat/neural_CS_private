# Stacked NLSA: Model 1, Stage 1, one seed

Verdict: stacked NLSA has 88.03% higher mean raw Fisher-score MSE than the multiplicative gate (0.108617 versus 0.057767), and beats it at 0/4 tested parameter values. The relative reduction in mean standardized MSE is -86.40% (negative means worse). This is one training seed; it does not establish stability across training seeds.

## Scope and protocol

Seed `20260709`; `K=20`, `m=20`, `tau=0.5`, `sigma_q=0.20`. All three methods use one shared bank of 40,000 training and 8,000 validation datasets, the same anchors and FSM targets, matched untrained linear readout weights, the same minibatch stream, and 20,000 updates each. The readout has two width-64 SiLU layers; the local network has two width-16 SiLU layers. Batch size 512, learning rate 1e-4 for both parameter groups, cosine decay after step 10,000 to ratio 0.1, weight decay 1e-3, gradient clipping 5, EMA decay 0.995.

Checkpoint selection is explicitly validation-best **raw**, using only validation FSM loss. The pre-existing trainer's default actually considers both raw and EMA; that default remains unchanged, while the new launcher explicitly uses `--checkpoint-selection raw` for all three methods as required by the experiment prompt. EMA loss is logged but cannot select a checkpoint.

The exact full score in the `u=logit(pi)` coordinate is used only after all training has finished. Each parameter point has 5,000 shared test datasets. Standardized MSE is raw MSE divided by the exact score variance at that parameter. Mean rows average the four parameter points equally.

No Model 2, Stage 2, extra seeds, or architecture ablations were run.

## Local maps and initialization

The gate uses $s_z\,\operatorname{softplus}(g_\eta(s_z,a_z))$. The stacked map uses $(1,s_z,m_\eta(s_z,a_z))$ with an unbounded learned channel, `m_dim=1`, and the constant included. Its identity input to the readout is the existing standardized block feature verbatim. The learned local features are mean-pooled separately.

The added channel can learn signed functions and nonzero values at zero local score; the positive multiplicative map cannot. Retaining the identity alongside a learned feature enlarges the summary representation. This does not guarantee better finite-sample optimization or an exact containment theorem for these fixed-width networks.

The feature MLP's final layer starts at zero. Its readout columns retain seeded nonzero `nn.Linear` initialization; the constant readout column starts at zero. Thus initial predictions match ILSA while gradients reach the feature MLP. The constant pools to 1 and is representationally redundant with the first-layer bias, but making its coefficient trainable can change optimization and regularization. Numerical irrelevance of training with versus without it is therefore not assumed.

| Method | Initial max abs difference from ILSA | Parameters | Selected raw step | Training wall time (s) |
| --- | --- | --- | --- | --- |
| Linear | 0 | 4417 | 12100 | 33.04 |
| Multiplicative gate | 2.38e-07 | 4754 | 9800 | 239.99 |
| Stacked NLSA | 0 | 4882 | 12100 | 253.72 |

## Fisher-score approximation error

### Raw MSE

| True pi | Linear | Multiplicative gate | Stacked NLSA | Stacked reduction vs linear | Stacked reduction vs gate |
| --- | --- | --- | --- | --- | --- |
| 0.10 | 0.188079 | 0.100674 | 0.185249 | 1.50% | -84.01% |
| 0.30 | 0.081581 | 0.030823 | 0.079454 | 2.61% | -157.77% |
| 0.50 | 0.077679 | 0.038980 | 0.077495 | 0.24% | -98.81% |
| 0.65 | 0.095951 | 0.060591 | 0.092269 | 3.84% | -52.28% |
| Mean | 0.110823 | 0.057767 | 0.108617 | 1.99% | -88.03% |

### Standardized MSE

| True pi | Linear | Multiplicative gate | Stacked NLSA | Stacked reduction vs linear | Stacked reduction vs gate |
| --- | --- | --- | --- | --- | --- |
| 0.10 | 0.212153 | 0.113560 | 0.208961 | 1.50% | -84.01% |
| 0.30 | 0.031234 | 0.011801 | 0.030420 | 2.61% | -157.77% |
| 0.50 | 0.024470 | 0.012279 | 0.024412 | 0.24% | -98.81% |
| 0.65 | 0.033591 | 0.021212 | 0.032302 | 3.84% | -52.28% |
| Mean | 0.075362 | 0.039713 | 0.074024 | 1.78% | -86.40% |

## Paired test-set differences

Negative differences favor stacked NLSA. SE is the standard error of the paired squared-error difference across the 5,000 test datasets, conditional on these fitted models. It is **not** an across-training-seed uncertainty estimate; that spread is unavailable with one seed. No NPE pairing or posterior comparisons apply to this Stage-1-only run.

| True pi | Baseline | Stacked minus baseline raw MSE | Paired test SE |
| --- | --- | --- | --- |
| 0.10 | Linear | -0.002830 | 0.000115 |
| 0.10 | Multiplicative gate | 0.084576 | 0.002254 |
| 0.30 | Linear | -0.002126 | 0.000080 |
| 0.30 | Multiplicative gate | 0.048631 | 0.001180 |
| 0.50 | Linear | -0.000184 | 0.000065 |
| 0.50 | Multiplicative gate | 0.038515 | 0.000923 |
| 0.65 | Linear | -0.003682 | 0.000079 |
| 0.65 | Multiplicative gate | 0.031677 | 0.000996 |

## Stability, timing, and reproducibility

Maximum logged |m|: 2.271216; maximum pooled-m RMS: 0.617111; maximum pooled-m SD: 0.092300. These diagnostics use the same first 256 validation datasets at every logging step (step 0, step 1, then every 100 updates). They are not maxima over all training data. All training and validation losses were finite; no divergence or hyperparameter changes occurred.

Stage 1 plus exact evaluation wall time: 529.12 seconds. Training times above include each method's validation passes. Report replay is additional. Hardware: Apple M3 Max, 64 GiB RAM; CPU, 8 threads. Python 3.12.13, PyTorch 2.5.1, NumPy 2.3.5. PyTorch 2.5.1 satisfies SBI 0.24.0's declared dependency constraint, whereas the archived requirements ask for the incompatible PyTorch 2.8. These are newly matched fits on one environment, not bitwise reproductions of the historical five-seed tables.

Initial gradient-flow and runtime-roundtrip tests pass. All three frozen runtime sanity checks pass. The replay reproduces the evaluator's dataset hashes and MSEs. Training-data fingerprints are in `config.json`; equal minibatch hashes are in `training_info.json`; frozen checkpoint and test-data hashes are in `exact_evaluation.json`.

Run directory: `runs/model1_stacked_seed20260709_20260911`. Raw results: `score_summary_by_pi.csv`, `score_predictions.npz`, `training_trace.csv`, and `stacked_summary.json`. The log is the adjacent `.log` file.

Reproduce from the package directory:

```bash
OMP_NUM_THREADS=8 DEVICE=cpu SEED=20260709 \
PYTHON_BIN=../.venv-stacked-nlsa/bin/python \
bash scripts/run_model1_stage1_stacked.sh runs/NEW_UNUSED_DIRECTORY
../.venv-stacked-nlsa/bin/python scripts/summarize_model1_stacked.py \
  --run-dir runs/NEW_UNUSED_DIRECTORY
```

Existing packaged headline results and all files under `artifacts/` remain unchanged.
