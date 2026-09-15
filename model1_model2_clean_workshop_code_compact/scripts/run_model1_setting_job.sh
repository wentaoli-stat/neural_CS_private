#!/usr/bin/env bash
# One Model 1 (p=1) run at a chosen data-generating setting (K blocks, m per
# block, shift tau): matched-readout Stage 1 for ILSA (1, s), gate and stacked
# m_dim=8, frozen exact-score evaluation, Stage-1 checks, then Stage-2 NPE.
#
#   run_model1_setting_job.sh TAG K M TAU SEED [SIGMA_Q]
#
# Everything except K, M and TAU matches run_model1_followup_job.sh, including
# the prior pi in [0.05, 0.70]. Directories: runs/setting/s1_TAG_sqSIGMA_Q_SEED
# and runs/setting/s2_TAG_sqSIGMA_Q_SEED. Needs PYTHON_BIN; JOB_THREADS
# (default 3) sets the thread count. Completed stages are skipped and partial
# directories are never overwritten.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -ge 5 ]] || { echo "usage: $0 TAG K M TAU SEED [SIGMA_Q]" >&2; exit 2; }
py="${PYTHON_BIN:?PYTHON_BIN is required}"
threads="${JOB_THREADS:-3}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
tag="$1"; k="$2"; m="$3"; tau="$4"; seed="$5"; sq="${6:-0.20}"
root=runs/setting
mkdir -p "$root/logs"
name="${tag}_sq${sq}_${seed}"
s1="$root/s1_${name}"
s2="$root/s2_${name}"

if [[ ! -f "$s1/score_summary_by_pi.csv" ]]; then
  [[ -e "$s1" ]] && { echo "partial Stage-1 directory exists: $s1" >&2; exit 2; }
  start=$(date +%s)
  "$py" -u -c "import torch, runpy, sys; torch.set_num_threads(${threads}); sys.argv=['stage1']+sys.argv[1:]; runpy.run_module('model1.stage1', run_name='__main__')" \
    --n-train 40000 --n-val 8000 --n-blocks "$k" --block-size "$m" --tau "$tau" \
    --anchor-pi-min 0.05 --anchor-pi-max 0.70 --sigma-q "$sq" --seed "$seed" \
    --methods linear,radial,stacked --hidden 64 --depth 2 --gate-hidden 16 --m-dim 8 \
    --linear-constant-channel 1 --include-constant-channel 1 \
    --iters 20000 --batch-size 512 --lr 1e-3 --gate-lr 1e-3 --joint-rho-lr 1e-3 \
    --lr-schedule cosine_tail --lr-decay-start-step 10000 --lr-min-ratio 0.1 \
    --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 --checkpoint-selection raw \
    --patience 0 --print-every 100 --device cpu --output-dir "$s1" \
    > "$root/logs/s1_${name}.log" 2>&1 || exit 1
  "$py" -u -m model1.evaluate_stage1 --run-dir "$s1" --n-test 5000 \
    --pi-values 0.07,0.10,0.30,0.50,0.65 --batch-size 512 --device cpu \
    > "$root/logs/s1_${name}_eval.log" 2>&1 || exit 1
  echo "s1_${name} seconds $(( $(date +%s) - start ))" >> "$root/logs/timings.txt"
fi
"$py" scripts/check_model1_stage1_run.py "$s1" >> "$root/logs/s1_checks.log" 2>&1 \
  || { echo "Stage-1 checks failed: $s1" >&2; exit 1; }
[[ -f "$s2/posterior_by_seed.csv" ]] && exit 0
[[ -e "$s2" ]] && { echo "partial Stage-2 directory exists: $s2" >&2; exit 2; }
start=$(date +%s)
DEVICE=cpu PYTHON_BIN="$py" bash scripts/run_model1_stage2_npe_stacked.sh "$s1" "$s2" \
  > "$root/logs/s2_${name}.log" 2>&1 || exit 1
echo "s2_${name} seconds $(( $(date +%s) - start ))" >> "$root/logs/timings.txt"
