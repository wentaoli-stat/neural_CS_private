#!/usr/bin/env bash
# Learning-rate audit for Model 2. The packaged launcher trains every arm at
# 1e-4; the Model 1 audit and the Model 2 linear arm both show that setting
# under-trains badly (linear improves 41.8% at 1e-3 across 5 seeds), so the
# archived gate-versus-linear margin is measured against an under-trained
# baseline. This re-runs both arms at both rates.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
for seed in ${SEEDS:-20260709 20260710 20260711 20260712 20260713}; do
  for lr in ${RATES:-1e-4 1e-3}; do
    out="${OUT_PREFIX:-runs/m2audit}_${lr}_${seed}"
    [[ -e "${out}" ]] && { echo "skip existing ${out}"; continue; }
    SEED="${seed}" LR="${lr}" GATE_LR="${lr}" \
      METHODS="${METHODS:-linear,shared_radial}" \
      DEVICE="${DEVICE:-mps}" PYTHON_BIN="${python_bin}" \
      bash scripts/run_model2_stage1_stacked.sh "${out}"
  done
done
