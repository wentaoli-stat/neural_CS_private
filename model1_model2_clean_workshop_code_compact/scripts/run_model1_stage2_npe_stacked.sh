#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 STAGE1_DIR OUTPUT_DIR" >&2
  exit 2
fi

stage1_dir="$1"
output_dir="$2"
for required in \
  "${stage1_dir}/config.json" "${stage1_dir}/feature_stats.npz" \
  "${stage1_dir}/model_linear.pt" "${stage1_dir}/model_radial.pt" \
  "${stage1_dir}/model_stacked.pt"; do
  [[ -f "${required}" ]] || { echo "required file does not exist: ${required}" >&2; exit 2; }
done
if [[ -e "${output_dir}" ]]; then
  echo "refusing to overwrite output directory: ${output_dir}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
pilot_cache_args=()
if [[ -n "${PILOT_CACHE:-}" ]]; then
  pilot_cache_args=(--pilot-cache "${PILOT_CACHE}")
fi
cd "${repo_root}"
"${python_bin}" -u -m model1.stage2_npe \
  --stage1-run-dir "${stage1_dir}" \
  --methods pilot,linear,radial,stacked \
  --pi-prior-min 0.05 --pi-prior-max 0.70 \
  --n-sbi-train 50000 --sbi-train-seed 20260723 \
  --sbi-model mdn --sbi-hidden-features 64 --sbi-num-components 8 \
  --sbi-batch-size 256 --sbi-lr 5e-4 \
  --max-epochs 300 --validation-fraction 0.10 --stop-after-epochs 20 \
  --sbi-seed 54000 --shared-npe-seed 1 \
  --test-pi-values 0.07,0.10,0.30,0.50,0.65,0.68 \
  --test-seeds 100,101,102,103,104,105,106,107,108,109 \
  --posterior-n 5000 --posterior-seed 87000 --shared-posterior-seed 1 \
  --grid-size 5000 --context-score-batch-size 256 \
  --pilot-grid-size 201 --pilot-batch-size 512 \
  --pilot-grid-chunk-size 16 --pilot-progress-every 10 \
  --pilot-sanity-n 64 --sanity-seed 20260710 \
  --save-training-context 1 \
  "${pilot_cache_args[@]}" \
  --device "${DEVICE:-auto}" --data-device cpu \
  --output-dir "${output_dir}"
