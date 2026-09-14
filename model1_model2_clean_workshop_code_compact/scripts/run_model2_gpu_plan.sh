#!/usr/bin/env bash
# Model 2 (40x40) Stage-1 study on a CUDA GPU.
#
#   1. benchmark   time real training steps and project the plan (scripts/benchmark_model2_gpu.py)
#   2. audit       linear + gate (shared_radial) at lr 1e-4 and 1e-3, every rate matched, 5 seeds
#   3. stacked     stacked_shared + stacked_split (m_dim 8) at lr 1e-3, 5 seeds
#
# Runs for one seed share the simulation bank, the minibatch stream and the
# untrained linear readout, so the stacked runs are paired with the audit's
# lr 1e-3 linear and gate runs. Each run ends with the frozen exact-score
# evaluation. A run is skipped only if that evaluation exists; an incomplete
# directory stops that run and is reported, never silently skipped. One failed
# run does not stop the others; the plan exits non-zero if any failed.
#
#   BENCHMARK_ONLY=1 bash scripts/run_model2_gpu_plan.sh     # step 1 only
#   bash scripts/run_model2_gpu_plan.sh                      # everything
#
# Environment: PYTHON_BIN, DEVICE (default cuda), RUN_ROOT (default runs/m2gpu),
# SEEDS, BENCHMARK_ARGS (extra arguments for the benchmark).
set -uo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."
py="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
root="${RUN_ROOT:-runs/m2gpu}"
seeds="${SEEDS:-20260709 20260710 20260711 20260712 20260713}"
mkdir -p "${root}/logs"
status="${root}/logs/status.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "${status}"; }

if [[ "${device}" == cuda* ]] && ! "${py}" -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA is not available to ${py}; set DEVICE or run on a CUDA machine" >&2
  exit 2
fi

log "benchmark on ${device}"
# shellcheck disable=SC2086
"${py}" scripts/benchmark_model2_gpu.py --device "${device}" ${BENCHMARK_ARGS:-} \
  2>&1 | tee "${root}/logs/benchmark.txt"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { log "benchmark FAILED; nothing else run"; exit 1; }
[[ "${BENCHMARK_ONLY:-0}" == "1" ]] && exit 0

failed=0
run_one() {
  local name="$1" methods="$2" lr="$3" seed="$4"
  local out="${root}/${name}"
  if [[ -f "${out}/score_summary_by_pi.csv" ]]; then log "skip complete ${name}"; return 0; fi
  if [[ -e "${out}" ]]; then
    log "INCOMPLETE ${out} exists; inspect and remove it to rerun ${name}"; failed=1; return 0
  fi
  local start; start=$(date +%s)
  log "start ${name}"
  if SEED="${seed}" LR="${lr}" GATE_LR="${lr}" METHODS="${methods}" M_DIM=8 \
       DEVICE="${device}" PYTHON_BIN="${py}" \
       bash scripts/run_model2_stage1_stacked.sh "${out}" > "${root}/logs/${name}.log" 2>&1; then
    log "done ${name} in $(( ($(date +%s) - start) / 60 )) min"
  else
    log "FAILED ${name}; see ${root}/logs/${name}.log"; failed=1
  fi
}

for seed in ${seeds}; do
  for lr in 1e-4 1e-3; do
    run_one "audit_lr${lr}_${seed}" linear,shared_radial "${lr}" "${seed}"
  done
done
for seed in ${seeds}; do
  run_one "stacked_lr1e-3_${seed}" stacked_shared,stacked_split 1e-3 "${seed}"
done

log "plan finished (failures: ${failed})"
exit "${failed}"
