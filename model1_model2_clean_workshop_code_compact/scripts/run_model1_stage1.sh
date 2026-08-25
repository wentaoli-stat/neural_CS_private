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
"${python_bin}" -u -m model1.stage1 \
  --n-train 40000 --n-val 8000 \
  --n-blocks 20 --block-size 20 --tau 0.5 \
  --anchor-pi-min 0.05 --anchor-pi-max 0.70 \
  --sigma-q 0.20 --seed "${seed}" \
  --methods linear,radial \
  --hidden 64 --depth 2 --gate-hidden 16 \
  --iters 20000 \
  --batch-size 512 \
  --lr 1e-4 --gate-lr 1e-4 --joint-rho-lr 1e-4 \
  --lr-schedule cosine_tail --lr-decay-start-step 10000 --lr-min-ratio 0.1 \
  --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 \
  --patience 0 --print-every 100 \
  --device "${DEVICE:-auto}" \
  --output-dir "${output_dir}"

"${python_bin}" -u -m model1.evaluate_stage1 \
  --run-dir "${output_dir}" \
  --n-test 5000 --pi-values 0.10,0.30,0.50,0.65 \
  --batch-size 512 --device "${DEVICE:-auto}"
