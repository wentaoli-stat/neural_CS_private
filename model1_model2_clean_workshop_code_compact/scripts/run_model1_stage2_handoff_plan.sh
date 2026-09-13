#!/usr/bin/env bash
# 2026-09-13 handoff plan: matched-readout Stage 1 at sigma_q in {0.20, 1.052},
# frozen exact-score evaluation, Stage-1 checks, then Stage-2 NPE with stacked.
# Each seed's Stage 2 runs in the background while the next seed's Stage 1 trains.
#
#   PILOT_CACHE=runs/s2_sq0.20_20260709/pilot_cache.npz \
#   PYTHON_BIN=.venv/Scripts/python.exe \
#   bash scripts/run_model1_stage2_handoff_plan.sh 20260710 20260711 20260712 20260713
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/.."
py="${PYTHON_BIN:-python}"
device="${DEVICE:-cpu}"
s1_threads="${STAGE1_THREADS:-3}"
s2_threads="${STAGE2_THREADS:-4}"
sigmas=(0.20 1.052)
mkdir -p runs/logs
[[ $# -ge 1 ]] || { echo "usage: $0 SEED [SEED...]" >&2; exit 2; }

for seed in "$@"; do
  for sq in "${sigmas[@]}"; do
    for d in "runs/s1_sq${sq}_${seed}" "runs/s2_sq${sq}_${seed}"; do
      [[ -e "$d" ]] && { echo "refusing to overwrite existing directory: $d" >&2; exit 2; }
    done
  done
done

stage1() {
  local sq="$1" seed="$2" out="runs/s1_sq$1_$2" start
  start=$(date +%s)
  OMP_NUM_THREADS="$s1_threads" MKL_NUM_THREADS="$s1_threads" "$py" -u -c \
    "import torch, runpy, sys; torch.set_num_threads(${s1_threads}); sys.argv=['stage1']+sys.argv[1:]; runpy.run_module('model1.stage1', run_name='__main__')" \
    --n-train 40000 --n-val 8000 --n-blocks 20 --block-size 20 --tau 0.5 \
    --anchor-pi-min 0.05 --anchor-pi-max 0.70 --sigma-q "$sq" --seed "$seed" \
    --methods linear,radial,stacked --hidden 64 --depth 2 --gate-hidden 16 --m-dim 8 \
    --iters 20000 --batch-size 512 --lr 1e-3 --gate-lr 1e-3 --joint-rho-lr 1e-3 \
    --lr-schedule cosine_tail --lr-decay-start-step 10000 --lr-min-ratio 0.1 \
    --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 --checkpoint-selection raw \
    --patience 0 --print-every 100 --device "$device" --output-dir "$out" \
    > "runs/logs/s1_sq${sq}_${seed}.log" 2>&1 || return 1
  OMP_NUM_THREADS="$s1_threads" MKL_NUM_THREADS="$s1_threads" "$py" -u -m model1.evaluate_stage1 \
    --run-dir "$out" --n-test 5000 --pi-values 0.10,0.30,0.50,0.65 --batch-size 512 --device "$device" \
    > "runs/logs/s1_sq${sq}_${seed}_eval.log" 2>&1 || return 1
  echo "s1_sq${sq}_${seed} seconds $(( $(date +%s) - start ))" >> runs/logs/timings.txt
}

stage2_pair() {
  local seed="$1" sq start
  for sq in "${sigmas[@]}"; do
    start=$(date +%s)
    OMP_NUM_THREADS="$s2_threads" MKL_NUM_THREADS="$s2_threads" DEVICE="$device" PYTHON_BIN="$py" \
      PILOT_CACHE="${PILOT_CACHE:-}" \
      bash scripts/run_model1_stage2_npe_stacked.sh "runs/s1_sq${sq}_${seed}" "runs/s2_sq${sq}_${seed}" \
      > "runs/logs/s2_sq${sq}_${seed}.log" 2>&1 \
      || { echo "Stage 2 failed: sq=${sq} seed=${seed}" | tee -a runs/logs/plan_status.txt; continue; }
    echo "s2_sq${sq}_${seed} seconds $(( $(date +%s) - start ))" >> runs/logs/timings.txt
    echo "Stage 2 done: sq=${sq} seed=${seed}" | tee -a runs/logs/plan_status.txt
  done
}

stage2_pids=()
for seed in "$@"; do
  echo "$(date) Stage 1 start: seed=${seed}" | tee -a runs/logs/plan_status.txt
  stage1 0.20 "$seed" & p1=$!
  stage1 1.052 "$seed" & p2=$!
  wait "$p1"; r1=$?
  wait "$p2"; r2=$?
  if [[ $r1 -ne 0 || $r2 -ne 0 ]]; then
    echo "$(date) Stage 1 failed: seed=${seed} (0.20:$r1 1.052:$r2)" | tee -a runs/logs/plan_status.txt
    continue
  fi
  if ! "$py" scripts/check_model1_stage1_run.py "runs/s1_sq0.20_${seed}" "runs/s1_sq1.052_${seed}" \
      >> runs/logs/s1_checks.log 2>&1; then
    echo "$(date) Stage-1 checks FAILED: seed=${seed}; Stage 2 skipped" | tee -a runs/logs/plan_status.txt
    continue
  fi
  echo "$(date) Stage 1 checked; Stage 2 launched: seed=${seed}" | tee -a runs/logs/plan_status.txt
  stage2_pair "$seed" & stage2_pids+=($!)
done
for pid in "${stage2_pids[@]}"; do wait "$pid"; done
echo "$(date) plan finished" | tee -a runs/logs/plan_status.txt
