#!/usr/bin/env bash
# One Model 1 (p=1) follow-up job from STAGE2_FINDINGS_20260913.md "Next steps".
#
#   s1s2 SIGMA_Q SEED ITERS DECAY_START
#       Matched-readout Stage 1 (lr = gate_lr = joint_rho_lr = 1e-3), frozen
#       exact-score evaluation, Stage-1 checks, then Stage-2 NPE with stacked.
#       Directories: runs/s1_<tag>_SEED and runs/s2_<tag>_SEED, where
#       <tag> = sq<SIGMA_Q>, plus _it<ITERS> when ITERS != 20000.
#   rep SIGMA_Q SEED REP [S1_TAG]
#       Stage-2 replicate REP >= 1 on an existing Stage-1 directory with a new
#       NPE simulation bank, NPE seed, posterior seed and test bank
#       (scripts/run_model1_stage2_npe_stacked_rep.sh).
#       Directory: runs/s2rep<REP>_<S1_TAG>_SEED.
#
# Needs PYTHON_BIN. JOB_THREADS (default 3) sets the torch/BLAS thread count.
# PILOT_CACHE, if set, is passed to REP-0 Stage 2 (valid only for the default
# NPE simulation bank). Completed stages are skipped; partial directories are
# never overwritten.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
py="${PYTHON_BIN:?PYTHON_BIN is required}"
threads="${JOB_THREADS:-3}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
mkdir -p runs/logs

kind="$1"
sq="$2"
seed="$3"

case "$kind" in
  s1s2)
    iters="$4"
    decay="$5"
    tag="sq${sq}"
    [[ "$iters" != 20000 ]] && tag="${tag}_it${iters}"
    s1="runs/s1_${tag}_${seed}"
    s2="runs/s2_${tag}_${seed}"
    if [[ ! -f "$s1/score_summary_by_pi.csv" ]]; then
      [[ -e "$s1" ]] && { echo "partial Stage-1 directory exists: $s1" >&2; exit 2; }
      start=$(date +%s)
      "$py" -u -c "import torch, runpy, sys; torch.set_num_threads(${threads}); sys.argv=['stage1']+sys.argv[1:]; runpy.run_module('model1.stage1', run_name='__main__')" \
        --n-train 40000 --n-val 8000 --n-blocks 20 --block-size 20 --tau 0.5 \
        --anchor-pi-min 0.05 --anchor-pi-max 0.70 --sigma-q "$sq" --seed "$seed" \
        --methods linear,radial,stacked --hidden 64 --depth 2 --gate-hidden 16 --m-dim 8 \
        --iters "$iters" --batch-size 512 --lr 1e-3 --gate-lr 1e-3 --joint-rho-lr 1e-3 \
        --lr-schedule cosine_tail --lr-decay-start-step "$decay" --lr-min-ratio 0.1 \
        --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 --checkpoint-selection raw \
        --patience 0 --print-every 100 --device cpu --output-dir "$s1" \
        > "runs/logs/s1_${tag}_${seed}.log" 2>&1 || exit 1
      "$py" -u -m model1.evaluate_stage1 --run-dir "$s1" --n-test 5000 \
        --pi-values 0.10,0.30,0.50,0.65 --batch-size 512 --device cpu \
        > "runs/logs/s1_${tag}_${seed}_eval.log" 2>&1 || exit 1
      echo "s1_${tag}_${seed} seconds $(( $(date +%s) - start ))" >> runs/logs/timings.txt
    fi
    "$py" scripts/check_model1_stage1_run.py "$s1" >> runs/logs/s1_checks.log 2>&1 \
      || { echo "Stage-1 checks failed: $s1" >&2; exit 1; }
    [[ -f "$s2/posterior_by_seed.csv" ]] && exit 0
    [[ -e "$s2" ]] && { echo "partial Stage-2 directory exists: $s2" >&2; exit 2; }
    start=$(date +%s)
    DEVICE=cpu PYTHON_BIN="$py" PILOT_CACHE="${PILOT_CACHE:-}" \
      bash scripts/run_model1_stage2_npe_stacked.sh "$s1" "$s2" \
      > "runs/logs/s2_${tag}_${seed}.log" 2>&1 || exit 1
    echo "s2_${tag}_${seed} seconds $(( $(date +%s) - start ))" >> runs/logs/timings.txt
    ;;
  rep)
    rep="$4"
    s1tag="${5:-sq${sq}}"
    s1="runs/s1_${s1tag}_${seed}"
    s2="runs/s2rep${rep}_${s1tag}_${seed}"
    (( rep >= 1 )) || { echo "REP must be >= 1 (REP 0 is the original run)" >&2; exit 2; }
    [[ -f "$s1/score_summary_by_pi.csv" ]] || { echo "missing Stage-1 run: $s1" >&2; exit 2; }
    [[ -f "$s2/posterior_by_seed.csv" ]] && exit 0
    [[ -e "$s2" ]] && { echo "partial Stage-2 directory exists: $s2" >&2; exit 2; }
    start=$(date +%s)
    REP="$rep" DEVICE=cpu PYTHON_BIN="$py" \
      bash scripts/run_model1_stage2_npe_stacked_rep.sh "$s1" "$s2" \
      > "runs/logs/s2rep${rep}_${s1tag}_${seed}.log" 2>&1 || exit 1
    echo "s2rep${rep}_${s1tag}_${seed} seconds $(( $(date +%s) - start ))" >> runs/logs/timings.txt
    ;;
  *)
    echo "unknown job kind: $kind" >&2
    exit 2
    ;;
esac
