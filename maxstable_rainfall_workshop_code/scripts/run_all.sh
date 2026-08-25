#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda}
RUN_ROOT=${1:-$ROOT/runs/rainfall79_10k}

export PYTHONPATH="$ROOT/code/src:$ROOT/code/runners${PYTHONPATH:+:$PYTHONPATH}"
export MAXSTABLE_UPSTREAM="$ROOT/code/upstream/maxstable"

SOURCE="$RUN_ROOT/source_40k"
PREPARED="$RUN_ROOT/prepared_10k_all3081"
STAGE1="$RUN_ROOT/stage1"
STAGE2="$RUN_ROOT/stage2"
mkdir -p "$RUN_ROOT"

# The archived validation bank was generated after a 40k source train bank.
# Only the first 10k rows enter Stage-1; preserving the 40k source replay keeps
# the validation RNG stream identical to the selected artifact.
"$PYTHON_BIN" "$ROOT/code/runners/prepare_stage1.py" \
  --output-dir "$SOURCE" --n-train 40000 --n-val 8000 \
  --sigma-q .20 --seed 20260731

"$PYTHON_BIN" "$ROOT/code/runners/prepare_pair_memmap.py" \
  --source-prepared-dir "$SOURCE" --output-dir "$PREPARED" \
  --train-pair-path "$PREPARED/train_pair.npy" \
  --n-train 10000 --n-val 8000 --n-distance-groups 20 \
  --feature-chunk-size 16

"$PYTHON_BIN" "$ROOT/code/runners/train_stage1.py" \
  --prepared-dir "$PREPARED" --output-dir "$STAGE1" --device "$DEVICE" \
  --iters 10000 --formal-batch-size 128 --pair-microbatch-size 32 \
  --validation-every 1000 --hidden 64 --depth 2 --gate-hidden 16 \
  --lr 1e-4 --weight-decay 1e-3 --grad-clip 5 --ema-decay .995 \
  --seed 20260819

"$PYTHON_BIN" "$ROOT/code/runners/run_stage2.py" \
  --stage1-checkpoint "$STAGE1/stage1_fixed_final_ema.pt" \
  --output-dir "$STAGE2" --device "$DEVICE" \
  --n-train 10000 --n-test 100 --simulation-seed 20260731 \
  --pilot-pairs-per-group 25 --pilot-pair-seed 20260819 \
  --pilot-iterations 12 --pilot-n-starts 15 --pilot-n-refine-starts 3 \
  --pilot-start-radius 2 --pilot-backtracking-steps 6 \
  --feature-chunk-size 4 --score-batch-size 128 \
  --npe-hidden 64 --npe-components 8 \
  --npe-batch-size 256 --npe-lr .0005 --npe-epochs 300 \
  --validation-fraction .10 --stop-after-epochs 20 --npe-seed 54000 \
  --posterior-n 5000 --posterior-seed 87000
