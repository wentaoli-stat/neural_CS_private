#!/usr/bin/env bash
# Model 1 (p=1) Stage-2 input comparison for one Stage-1 seed.
#
# Trains, on one shared Stage-1 simulation bank, three Fisher-score families:
#   raw-input score      common.raw_fsm direct_flat_mlp   (flatten(Y) + anchor)
#   all-subscores score  common.raw_fsm subscore_flat_mlp (flatten(s_kj) + anchor)
#   NLSA (paper)         model1.stage1 linear, radial (gate), stacked m_dim=8
# then feeds each through Stage-2 NPE with context (u_pilot, S(Y, u_pilot)).
# Every arm shares the NPE simulation bank (seed 20260723), the NPE seed (54000),
# the data-only pilot (one cache), and the 60 test datasets.
#
# The unstructured scores have a single learning rate; it is chosen per seed
# from {1e-4, 1e-3} by validation FSM loss only, never by an exact or posterior
# metric. NLSA arms use 1e-3 for every parameter group (readouts matched).
set -euo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 SEED" >&2; exit 2; }
seed="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."
py="${PYTHON_BIN:-python}"
sq="${SIGMA_Q:-1.052}"
device="${DEVICE:-cpu}"
s1_threads="${STAGE1_THREADS:-4}"
s2_threads="${STAGE2_THREADS:-4}"
root="${RUN_ROOT:-runs/ic}"
pilot_cache="${PILOT_CACHE:-${root}/pilot_cache_sbi20260723.npz}"
mkdir -p "${root}/logs"

nlsa_s1="${root}/s1_nlsa_sq${sq}_${seed}"
nlsa_s2="${root}/s2_nlsa_sq${sq}_${seed}"
log() { echo "$(date '+%F %T') $*" | tee -a "${root}/logs/status.txt"; }
timed() {
  local name="$1"; shift
  local start; start=$(date +%s)
  "$@" > "${root}/logs/${name}.log" 2>&1
  echo "${name} seconds $(( $(date +%s) - start ))" >> "${root}/logs/timings.txt"
}

nlsa_stage1() {
  if [[ -f "${nlsa_s1}/score_summary_by_pi.csv" ]]; then return 0; fi
  [[ -e "${nlsa_s1}" ]] && { echo "incomplete run directory: ${nlsa_s1}" >&2; return 1; }
  OMP_NUM_THREADS="${s1_threads}" "${py}" -u -m model1.stage1 \
    --n-train 40000 --n-val 8000 --n-blocks 20 --block-size 20 --tau 0.5 \
    --anchor-pi-min 0.05 --anchor-pi-max 0.70 --sigma-q "${sq}" --seed "${seed}" \
    --methods linear,radial,stacked --hidden 64 --depth 2 --gate-hidden 16 --m-dim 8 \
    --iters 20000 --batch-size 512 --lr 1e-3 --gate-lr 1e-3 --joint-rho-lr 1e-3 \
    --lr-schedule cosine_tail --lr-decay-start-step 10000 --lr-min-ratio 0.1 \
    --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 --checkpoint-selection raw \
    --patience 0 --print-every 100 --device "${device}" --output-dir "${nlsa_s1}"
  OMP_NUM_THREADS="${s1_threads}" "${py}" -u -m model1.evaluate_stage1 \
    --run-dir "${nlsa_s1}" --n-test 5000 --pi-values 0.10,0.30,0.50,0.65 \
    --batch-size 512 --device "${device}"
}

fsm_stage1() {
  local arch="$1" tag="$2" lr="$3"
  local out="${root}/s1_${tag}_lr${lr}_sq${sq}_${seed}"
  if [[ -f "${out}/training_info.json" ]]; then return 0; fi
  OMP_NUM_THREADS="${s1_threads}" "${py}" -u -m common.raw_fsm stage1 \
    --model model1 --architecture "${arch}" --sigma-q "${sq}" --seed "${seed}" \
    --n-train 40000 --n-val 8000 --hidden 64 --depth 2 --iters 20000 --batch-size 512 \
    --lr "${lr}" --lr-schedule cosine_tail --lr-decay-start-step 10000 --lr-min-ratio 0.1 \
    --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 --checkpoint-selection raw \
    --print-every 100 --device "${device}" --output-dir "${out}"
}

pick_lr() {
  "${py}" - "${root}" "$1" "${sq}" "${seed}" <<'PY'
import json, sys
root, tag, sq, seed = sys.argv[1:]
losses = {lr: float(json.load(open(f"{root}/s1_{tag}_lr{lr}_sq{sq}_{seed}/training_info.json"))["best_val_loss"])
          for lr in ("1e-4", "1e-3")}
print(min(losses, key=losses.get))
PY
}

nlsa_stage2() {
  if [[ -f "${nlsa_s2}/posterior_by_seed.csv" ]]; then return 0; fi
  PILOT_CACHE="${pilot_cache}" OMP_NUM_THREADS="${s2_threads}" DEVICE="${device}" PYTHON_BIN="${py}" \
    bash scripts/run_model1_stage2_npe_stacked.sh "${nlsa_s1}" "${nlsa_s2}"
}

fsm_stage2() {
  local tag="$1" lr="$2"
  local s1="${root}/s1_${tag}_lr${lr}_sq${sq}_${seed}" out="${root}/s2_${tag}_sq${sq}_${seed}"
  if [[ -f "${out}/posterior_by_seed.csv" ]]; then return 0; fi
  OMP_NUM_THREADS="${s2_threads}" "${py}" -u -m common.raw_fsm stage2 \
    --model model1 --stage1-run-dir "${s1}" \
    --pi-prior-min 0.05 --pi-prior-max 0.70 --n-sbi-train 50000 --sbi-train-seed 20260723 \
    --sbi-model mdn --sbi-hidden-features 64 --sbi-num-components 8 \
    --sbi-batch-size 256 --sbi-lr 5e-4 --max-epochs 300 --validation-fraction 0.10 \
    --stop-after-epochs 20 --sbi-seed 54000 \
    --test-pi-values 0.07,0.10,0.30,0.50,0.65,0.68 \
    --test-seeds 100,101,102,103,104,105,106,107,108,109 \
    --posterior-n 5000 --posterior-seed 87000 --grid-size 5000 --score-batch-size 256 \
    --pilot-grid-size 201 --pilot-batch-size 512 --pilot-grid-chunk-size 16 \
    --pilot-progress-every 10 --pilot-cache "${pilot_cache}" \
    --device "${device}" --pilot-device "${device}" --data-device cpu --output-dir "${out}"
}

log "seed ${seed}: Stage 1 start (sigma_q=${sq})"
( timed "s1_nlsa_sq${sq}_${seed}" nlsa_stage1 ) & p_nlsa=$!
( for lr in 1e-4 1e-3; do
    timed "s1_raw_lr${lr}_sq${sq}_${seed}" fsm_stage1 direct_flat_mlp raw "${lr}"
    timed "s1_sub_lr${lr}_sq${sq}_${seed}" fsm_stage1 subscore_flat_mlp sub "${lr}"
  done ) & p_fsm=$!
wait "${p_nlsa}"; wait "${p_fsm}"
"${py}" scripts/check_model1_stage1_run.py "${nlsa_s1}" >> "${root}/logs/s1_checks.log" 2>&1 \
  || { log "seed ${seed}: NLSA Stage-1 checks FAILED; stopping"; exit 1; }
raw_lr="$(pick_lr raw)"; sub_lr="$(pick_lr sub)"
log "seed ${seed}: Stage 1 done; lr by validation FSM loss: raw=${raw_lr} sub=${sub_lr}"

timed "s2_nlsa_sq${sq}_${seed}" nlsa_stage2
( timed "s2_raw_sq${sq}_${seed}" fsm_stage2 raw "${raw_lr}" ) & p_raw=$!
( timed "s2_sub_sq${sq}_${seed}" fsm_stage2 sub "${sub_lr}" ) & p_sub=$!
wait "${p_raw}"; wait "${p_sub}"
log "seed ${seed}: Stage 2 done"
