# Model 1 and Model 2 Amortized FSM Stage 1

This folder contains only the code needed to reproduce the current formal
Stage-1 experiments. It intentionally excludes NPE, diffusion, posterior
integration, ULA, SBC, and MLE/root-finding code.

## Contents

```text
model1_model2_stage1_code_20260710/
├── README.md
├── requirements.txt
├── summarize_amortized_stage1_results.py
├── model1/
│   ├── run_blockwise_mean_shift_amortized_fsm_experiment.py
│   └── run_model1_stage1_formal.sh
└── model2/
    ├── run_blockwise_common_factor_amortized_fsm_experiment.py
    └── run_model2_stage1_formal.sh
```

Both Python training scripts are self-contained. They implement their own
simulator, anchored FSM data generation, composite-score features, linear and
radial networks, nested warm start, paired training, EMA, early stopping, and
exact-score post-training diagnostics.

## Installation

```bash
python -m pip install -r requirements.txt
```

CUDA is recommended for the formal 40k runs. Select a CUDA-enabled PyTorch
build appropriate for the machine when installing PyTorch.

## Model 1 formal Stage 1

Setting:

```text
K=20, m=20, tau=0.5, sigma_q=0.2
n_train=40000, n_val=8000, n_test=5000
five seeds: 20260709--20260713
```

Run:

```bash
bash model1/run_model1_stage1_formal.sh
```

## Model 2 formal Stage 1

Setting:

```text
K=40, m=20, tau=1.0, sigma_q=0.15
n_train=40000, n_val=8000, n_test=5000
five seeds: 20260709--20260713
```

Run:

```bash
bash model2/run_model2_stage1_formal.sh
```

## Per-seed outputs

Each run directory contains:

```text
config.json
feature_stats.npz
model_linear.pt
model_radial.pt
training_trace.csv
training_info.json
fixed_grid_fsm_loss.csv
score_summary_by_pi.csv
```

True scores are used only to create `score_summary_by_pi.csv` after training.
The FSM training objective itself uses only simulator draws, anchors, and the
known Gaussian proposal score.
