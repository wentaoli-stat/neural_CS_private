#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root="$1"
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

for model in model1 model2; do
  "${python_bin}" -u -m common.raw_fsm stage1 \
    --model "${model}" \
    --architecture direct_flat_mlp \
    --hidden 64 \
    --depth 2 \
    --device "${device}" \
    --output-dir "${output_root}/${model}/stage1"
done

for model in model1 model2; do
  "${python_bin}" -u -m common.raw_fsm stage2 \
    --model "${model}" \
    --stage1-run-dir "${output_root}/${model}/stage1" \
    --device "${device}" \
    --pilot-device "${device}" \
    --data-device cpu \
    --output-dir "${output_root}/${model}/stage2"
done
