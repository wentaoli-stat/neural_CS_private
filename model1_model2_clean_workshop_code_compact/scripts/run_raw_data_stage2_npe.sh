#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {model1|model2} OUTPUT_DIR" >&2
  exit 2
fi

model="$1"
output_dir="$2"
case "${model}" in
  model1|model2) ;;
  *) echo "model must be model1 or model2" >&2; exit 2 ;;
esac
if [[ -e "${output_dir}" ]]; then
  echo "refusing to overwrite output directory: ${output_dir}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
cd "${repo_root}"
"${python_bin}" -u -m common.raw_data_npe \
  --model "${model}" \
  --pi-prior-min 0.05 --pi-prior-max 0.70 \
  --n-sbi-train 50000 --sbi-train-seed 20260723 \
  --sbi-model mdn --sbi-hidden-features 64 --sbi-num-components 8 \
  --sbi-batch-size 256 --sbi-lr 5e-4 \
  --max-epochs 300 --validation-fraction 0.10 --stop-after-epochs 20 \
  --sbi-seed 54000 \
  --test-pi-values 0.07,0.10,0.30,0.50,0.65,0.68 \
  --test-seeds 100,101,102,103,104,105,106,107,108,109 \
  --posterior-n 5000 --posterior-seed 87000 \
  --grid-size 5000 \
  --device "${DEVICE:-auto}" --data-device cpu \
  --output-dir "${output_dir}"
