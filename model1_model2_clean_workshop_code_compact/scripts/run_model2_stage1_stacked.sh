#!/usr/bin/env bash
# Model 2 Stage 1 with the stacked NLSA local map alongside the packaged
# linear and multiplicative-gate arms. All four share one simulation bank,
# one minibatch stream, and the matched untrained linear readout.
#
# The gate and stacked arms apply their local MLP to 780 pairwise scores per
# block, so a run costs ~6 h on CPU and ~1.8 h on Metal. DEVICE defaults to mps.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
output_dir="$1"
python_bin="${PYTHON_BIN:-python}"
seed="${SEED:-20260709}"
cd "${repo_root}"
if [[ -e "${output_dir}" ]]; then
  echo "refusing to overwrite output directory: ${output_dir}" >&2
  exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
"${python_bin}" -u -m model2.stage1 \
  --n-train 40000 --train-cache-replicates 1 --n-val 8000 \
  --n-blocks 40 --block-size 40 --tau 1.0 \
  --anchor-pi-min 0.05 --anchor-pi-max 0.70 \
  --sigma-q 0.15 --seed "${seed}" \
  --methods "${METHODS:-linear,shared_radial,stacked_shared,stacked_split}" \
  --hidden 64 --depth 2 --gate-hidden 16 \
  --m-dim "${M_DIM:-8}" --include-constant-channel "${INCLUDE_CONSTANT_CHANNEL:-1}" \
  --iters 20000 --batch-size 512 \
  --lr "${LR:-1e-3}" --gate-lr "${GATE_LR:-1e-3}" --joint-rho-lr "${LR:-1e-3}" \
  --lr-schedule constant \
  --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 \
  --milestone-steps 10000,15000,20000 \
  --patience 0 --print-every 100 \
  --save-final-ema-validation \
  --device "${DEVICE:-mps}" \
  --output-dir "${output_dir}"

"${python_bin}" -u -m model2.evaluate_stage1 \
  --run-dir "${output_dir}" \
  --n-test 5000 --pi-values 0.07,0.10,0.30,0.50,0.65,0.68 \
  --batch-size 512 --device "${DEVICE:-mps}"
