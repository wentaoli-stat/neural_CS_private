#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {model1|model2} OUTPUT_ROOT" >&2
  exit 2
fi

model="$1"
output_root="$2"
if [[ "${model}" != "model1" && "${model}" != "model2" ]]; then
  echo "model must be model1 or model2" >&2
  exit 2
fi
if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite output root: ${output_root}" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-auto}"
mkdir -p "${output_root}"

cd "${repo_root}"
"${python_bin}" -u -m common.raw_fsm stage1 \
  --model "${model}" \
  --device "${device}" \
  --output-dir "${output_root}/stage1"

"${python_bin}" -u -m common.raw_fsm stage2 \
  --model "${model}" \
  --stage1-run-dir "${output_root}/stage1" \
  --device "${device}" \
  --pilot-device "${device}" \
  --data-device cpu \
  --output-dir "${output_root}/stage2"
