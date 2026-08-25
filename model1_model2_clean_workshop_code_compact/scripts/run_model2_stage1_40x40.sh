#!/usr/bin/env bash
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

if [[ -e "${output_dir}" ]]; then
  echo "refusing to overwrite output directory: ${output_dir}" >&2
  exit 2
fi

cd "${repo_root}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" "${python_bin}" -u -m model2.stage1 \
  --n-train 40000 --train-cache-replicates 1 --n-val 8000 \
  --n-blocks 40 --block-size 40 --tau 1.0 \
  --anchor-pi-min 0.05 --anchor-pi-max 0.70 \
  --sigma-q 0.15 --seed "${seed}" \
  --methods linear,shared_radial \
  --hidden 64 --depth 2 --gate-hidden 16 \
  --iters 20000 --batch-size 512 \
  --lr 1e-4 --gate-lr 1e-4 --joint-rho-lr 1e-4 \
  --lr-schedule constant \
  --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 \
  --milestone-steps 10000,15000,20000 \
  --patience 0 --print-every 100 \
  --save-final-ema-validation \
  --device "${DEVICE:-auto}" \
  --output-dir "${output_dir}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" "${python_bin}" -u -m model2.evaluate_stage1 \
  --run-dir "${output_dir}" \
  --n-test 5000 --pi-values 0.07,0.10,0.30,0.50,0.65,0.68 \
  --batch-size 512 --device "${DEVICE:-auto}"
