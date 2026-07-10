#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEEDS=(20260709 20260710 20260711 20260712 20260713)
for seed in "${SEEDS[@]}"; do
  "${PYTHON_BIN}" model1/run_blockwise_mean_shift_amortized_fsm_experiment.py \
    --n-train 40000 \
    --n-val 8000 \
    --n-test 5000 \
    --n-blocks 20 \
    --block-size 20 \
    --tau 0.5 \
    --pi-ref 0.3 \
    --feature-pi-mode anchor \
    --anchor-pi-min 0.05 \
    --anchor-pi-max 0.7 \
    --sigma-q 0.2 \
    --seed "${seed}" \
    --methods linear,radial \
    --hidden 64 \
    --depth 2 \
    --gate-hidden 16 \
    --gate-condition-on-anchor 1 \
    --iters 3000 \
    --batch-size 512 \
    --lr 3e-4 \
    --gate-lr 1e-4 \
    --joint-rho-lr 1e-4 \
    --gate-only-steps 400 \
    --weight-decay 1e-3 \
    --grad-clip 5 \
    --ema-decay 0.995 \
    --patience 20 \
    --print-every 100 \
    --validation-pi-values 0.07,0.10,0.20,0.30,0.40,0.50,0.60,0.65,0.68 \
    --n-validation-per-anchor 1000 \
    --diagnostic-pi-values 0.1,0.3,0.5,0.65 \
    --device cuda \
    --output-dir "runs/model1_tau0p5_sigma0p2_seed${seed}"
done

"${PYTHON_BIN}" summarize_amortized_stage1_results.py \
  --runs-dir runs \
  --prefix model1_tau0p5_sigma0p2_seed \
  --output-prefix runs/model1_tau0p5_sigma0p2_summary
