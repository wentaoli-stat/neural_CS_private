#!/usr/bin/env bash
# p=3 Model 1 Stage-2 pilot+score NPE, mirroring scripts/run_model1_stage2_npe_stacked.sh
# (50k NPE simulations, MDN 64 hidden / 8 components, fixed NPE and posterior seeds,
# test seeds 100-109) with 12 test parameter values and an 80^3 exact-posterior grid.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 STAGE1_DIR OUTPUT_DIR" >&2
  exit 2
fi
stage1_dir="$1"
output_dir="$2"
for required in \
  "${stage1_dir}/config.json" "${stage1_dir}/feature_stats.npz" \
  "${stage1_dir}/model_linear.pt" "${stage1_dir}/model_gate.pt" "${stage1_dir}/model_stacked.pt"; do
  [[ -f "${required}" ]] || { echo "required file does not exist: ${required}" >&2; exit 2; }
done
if [[ -e "${output_dir}" ]]; then
  echo "refusing to overwrite output directory: ${output_dir}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
cd "${repo_root}"
"${python_bin}" -u -m model1_p3.stage2_npe \
  --stage1-run-dir "${stage1_dir}" \
  --methods pilot,linear,gate,stacked \
  --n-sbi-train 50000 --sbi-train-seed 20260723 \
  --sbi-model mdn --sbi-hidden-features 64 --sbi-num-components 8 \
  --sbi-batch-size 256 --sbi-lr 5e-4 \
  --max-epochs 300 --validation-fraction 0.10 --stop-after-epochs 20 \
  --sbi-seed 54000 \
  --test-betas "0.1:0.8:0.85;0.1:0.8:1.2;0.1:1.5:0.85;0.1:1.5:1.2;0.3:0.8:0.85;0.3:0.8:1.2;0.3:1.5:0.85;0.3:1.5:1.2;0.65:0.8:0.85;0.65:0.8:1.2;0.65:1.5:0.85;0.65:1.5:1.2" \
  --test-seeds 100,101,102,103,104,105,106,107,108,109 \
  --posterior-n 5000 --posterior-seed 87000 \
  --coarse-grid-size 40 --grid-size 80 --sliced-directions 100 --sliced-seed 20260913 \
  --pilot-grid-points 9 --pilot-steps 300 --pilot-lr 0.01 --pilot-batch-size 1024 \
  --pilot-sanity-n 128 --sanity-seed 20260710 \
  --device "${DEVICE:-cpu}" --data-device cpu \
  --output-dir "${output_dir}"
