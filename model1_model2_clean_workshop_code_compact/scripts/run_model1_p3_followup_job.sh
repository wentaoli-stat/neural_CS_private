#!/usr/bin/env bash
# Model 1 p=3 follow-up job: matched-readout Stage 1 (readout --lr and local
# branch --branch-lr both 1e-3), frozen exact-score evaluation, then Stage 2
# (model1_p3.stage2_npe) if that module exists.
#
#   bash scripts/run_model1_p3_followup_job.sh SEED
#
# Directories: runs/p3_s1_SEED, runs/p3_s2_SEED. Needs PYTHON_BIN; JOB_THREADS
# (default 3). Completed stages are skipped; partial directories are refused.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${PYTHON_BIN:?PYTHON_BIN is required}"
threads="${JOB_THREADS:-3}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
mkdir -p runs/logs
seed="$1"
s1="runs/p3_s1_${seed}"
s2="runs/p3_s2_${seed}"

if [[ ! -f "$s1/score_summary_by_beta.csv" ]]; then
  [[ -e "$s1" ]] && { echo "partial Stage-1 directory exists: $s1" >&2; exit 2; }
  start=$(date +%s)
  "$py" -u -c "import torch, runpy, sys; torch.set_num_threads(${threads}); sys.argv=['stage1']+sys.argv[1:]; runpy.run_module('model1_p3.stage1', run_name='__main__')" \
    --n-train 40000 --n-val 8000 --n-blocks 20 --block-size 20 \
    --sigma-q 0.20,0.08,0.04 --seed "$seed" \
    --methods linear,gate,stacked --hidden 64 --depth 2 --gate-hidden 16 --m-dim 8 \
    --iters 20000 --batch-size 512 --lr 1e-3 --branch-lr 1e-3 \
    --lr-schedule cosine_tail --lr-decay-start-step 10000 --lr-min-ratio 0.1 \
    --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 --print-every 100 \
    --device cpu --output-dir "$s1" > "runs/logs/p3_s1_${seed}.log" 2>&1 || exit 1
  "$py" -u -m model1_p3.evaluate_stage1 --run-dir "$s1" --device cpu \
    > "runs/logs/p3_s1_${seed}_eval.log" 2>&1 || exit 1
  echo "p3_s1_${seed} seconds $(( $(date +%s) - start ))" >> runs/logs/timings.txt
fi

[[ -f model1_p3/stage2_npe.py ]] || { echo "model1_p3/stage2_npe.py not present; Stage 1 only"; exit 0; }
[[ -f "$s2/posterior_by_seed.csv" ]] && exit 0
[[ -e "$s2" ]] && { echo "partial Stage-2 directory exists: $s2" >&2; exit 2; }
start=$(date +%s)
DEVICE=cpu PYTHON_BIN="$py" bash scripts/run_model1_p3_stage2_npe.sh "$s1" "$s2" \
  > "runs/logs/p3_s2_${seed}.log" 2>&1 || exit 1
echo "p3_s2_${seed} seconds $(( $(date +%s) - start ))" >> runs/logs/timings.txt
