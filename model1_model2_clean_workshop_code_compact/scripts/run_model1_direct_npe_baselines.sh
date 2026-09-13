#!/usr/bin/env bash
# Model 1 direct-NPE baselines for the Stage-2 input comparison: NPE conditioned
# on raw Y (400 dims) and on every local subscore at the data-only pilot
# (401 dims). Neither uses Stage 1, so each is a single run. Seeds, NPE recipe,
# pilot and test datasets match run_model1_input_comparison_seed.sh.
#
# NPE training on a 400-dimensional context is sensitive to seed and platform
# nondeterminism, so SBI_SEEDS replicates each arm over NPE seeds. The 54000 run
# lives in npe_<context>/ and every other seed in npe_<context>_sbi<seed>/.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."
py="${PYTHON_BIN:-python}"
threads="${STAGE2_THREADS:-4}"
root="${RUN_ROOT:-runs/ic}"
pilot_cache="${PILOT_CACHE:-${root}/pilot_cache_sbi20260723.npz}"
mkdir -p "${root}/logs"
for sbi in ${SBI_SEEDS:-54000}; do
for ctx in ${CONTEXTS:-raw subscores_at_pilot}; do
  suffix=""; [[ "${sbi}" == "54000" ]] || suffix="_sbi${sbi}"
  out="${root}/npe_${ctx}${suffix}"
  if [[ -f "${out}/posterior_by_seed.csv" ]]; then continue; fi
  start=$(date +%s)
  OMP_NUM_THREADS="${threads}" "${py}" -u -m common.raw_data_npe \
    --model model1 --context "${ctx}" \
    --pi-prior-min 0.05 --pi-prior-max 0.70 --n-sbi-train 50000 --sbi-train-seed 20260723 \
    --sbi-model mdn --sbi-hidden-features 64 --sbi-num-components 8 \
    --sbi-batch-size 256 --sbi-lr 5e-4 --max-epochs 300 --validation-fraction 0.10 \
    --stop-after-epochs 20 --sbi-seed "${sbi}" \
    --test-pi-values 0.07,0.10,0.30,0.50,0.65,0.68 \
    --test-seeds 100,101,102,103,104,105,106,107,108,109 \
    --posterior-n 5000 --posterior-seed 87000 --grid-size 5000 \
    --pilot-grid-size 201 --pilot-batch-size 512 --pilot-grid-chunk-size 16 \
    --pilot-progress-every 10 --pilot-cache "${pilot_cache}" \
    --device "${DEVICE:-cpu}" --data-device cpu --output-dir "${out}" \
    > "${root}/logs/npe_${ctx}${suffix}.log" 2>&1
  echo "npe_${ctx}${suffix} seconds $(( $(date +%s) - start ))" >> "${root}/logs/timings.txt"
done
done
