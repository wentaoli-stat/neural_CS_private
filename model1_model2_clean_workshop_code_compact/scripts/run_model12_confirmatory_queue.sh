#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 JOB_ROOT" >&2
  exit 2
fi

job_root="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-auto}"
run_root="${job_root}/runs"
log_root="${job_root}/logs"

if [[ -e "${run_root}" ]]; then
  echo "refusing to overwrite run root: ${run_root}" >&2
  exit 2
fi
mkdir -p "${run_root}" "${log_root}"
cd "${repo_root}"

run_direct_stage1() {
  local model="$1"
  local hidden="$2"
  local lr="$3"
  local output="$4"
  local blind="$5"
  local schedule diagnostics
  local blind_args=()
  local diagnostic_args=()
  local schedule_args=()
  if [[ "${model}" == "model1" ]]; then
    schedule="cosine_tail"
    diagnostics="0.10,0.30,0.50,0.65"
    schedule_args=(--lr-decay-start-step 10000 --lr-min-ratio 0.1)
  else
    schedule="constant"
    diagnostics="0.07,0.10,0.30,0.50,0.65,0.68"
  fi
  if [[ "${blind}" == "1" ]]; then
    blind_args=(--skip-exact-diagnostics)
  else
    diagnostic_args=(--diagnostic-pi-values "${diagnostics}")
  fi
  "${python_bin}" -u -m common.raw_fsm stage1 \
    --model "${model}" --architecture direct_flat_mlp \
    --n-train 40000 --n-val 8000 --n-test 5000 \
    --anchor-pi-min 0.05 --anchor-pi-max 0.70 --seed 20260709 \
    --hidden "${hidden}" --depth 2 --iters 20000 \
    --batch-size 512 --eval-batch-size 512 --lr "${lr}" \
    --lr-schedule "${schedule}" "${schedule_args[@]}" \
    --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 --print-every 100 \
    "${diagnostic_args[@]}" "${blind_args[@]}" \
    --device "${device}" --output-dir "${output}"
}

run_direct_sweep() {
  local sweep_root="${run_root}/direct_raw_sweep"
  local model lr hidden output
  for model in model2 model1; do
    for lr in 0.00003 0.0001 0.0003; do
      for hidden in 32 64 128; do
        output="${sweep_root}/${model}/h${hidden}_lr${lr}/stage1"
        mkdir -p "$(dirname "${output}")"
        run_direct_stage1 "${model}" "${hidden}" "${lr}" "${output}" 1 \
          >"${log_root}/raw_sweep_${model}_h${hidden}_lr${lr}.log" 2>&1 &
      done
      wait
    done
  done
}

run_capacity_model1() {
  local hidden="$1"
  local output="${run_root}/capacity/model1_linear_h${hidden}"
  "${python_bin}" -u -m model1.stage1 \
    --n-train 40000 --n-val 8000 \
    --n-blocks 20 --block-size 20 --tau 0.5 \
    --anchor-pi-min 0.05 --anchor-pi-max 0.70 \
    --sigma-q 0.20 --seed 20260709 --methods linear \
    --hidden "${hidden}" --depth 2 --gate-hidden 16 \
    --iters 20000 --batch-size 512 \
    --lr 1e-4 --gate-lr 1e-4 --joint-rho-lr 1e-4 \
    --lr-schedule cosine_tail --lr-decay-start-step 10000 --lr-min-ratio 0.1 \
    --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 \
    --patience 0 --print-every 100 \
    --device "${device}" --output-dir "${output}"
  "${python_bin}" -u -m model1.evaluate_stage1 \
    --run-dir "${output}" --n-test 5000 \
    --pi-values 0.10,0.30,0.50,0.65 --batch-size 512 --device "${device}"
}

run_capacity_model2() {
  local hidden="$1"
  local output="${run_root}/capacity/model2_linear_h${hidden}"
  OMP_NUM_THREADS=8 "${python_bin}" -u -m model2.stage1 \
    --n-train 40000 --train-cache-replicates 1 --n-val 8000 \
    --n-blocks 40 --block-size 40 --tau 1.0 \
    --anchor-pi-min 0.05 --anchor-pi-max 0.70 \
    --sigma-q 0.15 --seed 20260709 --methods linear \
    --hidden "${hidden}" --depth 2 --gate-hidden 16 \
    --iters 20000 --batch-size 512 \
    --lr 1e-4 --gate-lr 1e-4 --joint-rho-lr 1e-4 \
    --lr-schedule constant \
    --weight-decay 1e-3 --grad-clip 5 --ema-decay 0.995 \
    --milestone-steps 10000,15000,20000 --patience 0 --print-every 100 \
    --save-final-ema-validation --device "${device}" --output-dir "${output}"
  OMP_NUM_THREADS=8 "${python_bin}" -u -m model2.evaluate_stage1 \
    --run-dir "${output}" --n-test 5000 \
    --pi-values 0.07,0.10,0.30,0.50,0.65,0.68 \
    --batch-size 512 --device "${device}"
}

run_capacity_controls() {
  run_capacity_model1 66 >"${log_root}/capacity_model1_h66.log" 2>&1
  run_capacity_model1 67 >"${log_root}/capacity_model1_h67.log" 2>&1
  run_capacity_model2 66 >"${log_root}/capacity_model2_h66.log" 2>&1
  run_capacity_model2 67 >"${log_root}/capacity_model2_h67.log" 2>&1
}

select_direct_winner() {
  local model="$1"
  "${python_bin}" - "${run_root}/direct_raw_sweep/${model}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for info_path in root.glob("h*_lr*/stage1/training_info.json"):
    info = json.loads(info_path.read_text())
    config = json.loads((info_path.parent / "config.json").read_text())
    rows.append((float(info["best_val_loss"]), int(config["hidden"]), float(config["lr"])))
if len(rows) != 9:
    raise SystemExit(f"expected 9 complete candidates under {root}, found {len(rows)}")
rows.sort(key=lambda row: (row[0], row[1]))
loss, hidden, lr = rows[0]
print(hidden, format(lr, ".12g"), format(loss, ".12g"))
PY
}

run_direct_stage2() {
  local model="$1"
  local stage1_dir="$2"
  local output="$3"
  "${python_bin}" -u -m common.raw_fsm stage2 \
    --model "${model}" --stage1-run-dir "${stage1_dir}" \
    --device "${device}" --pilot-device "${device}" --data-device cpu \
    --output-dir "${output}"
}

model1_worker() {
  local hidden lr loss
  read -r hidden lr loss < <(select_direct_winner model1)
  mkdir -p "${run_root}/direct_raw_winner/model1"
  printf 'hidden=%s\nlr=%s\nbest_val_loss=%s\n' "${hidden}" "${lr}" "${loss}" \
    >"${run_root}/direct_raw_winner/model1/selection.txt"
  run_direct_stage1 model1 "${hidden}" "${lr}" \
    "${run_root}/direct_raw_winner/model1/stage1" 0
  run_direct_stage2 model1 "${run_root}/direct_raw_winner/model1/stage1" \
    "${run_root}/direct_raw_winner/model1/stage2"

  local seed stage1_dir stage2_dir
  for seed in 20260710 20260711 20260712 20260713; do
    stage1_dir="${run_root}/multiseed/model1/stage1/seed_${seed}"
    stage2_dir="${run_root}/multiseed/model1/stage2/stage1_seed_${seed}_npe_seed_54000"
    DEVICE="${device}" PYTHON_BIN="${python_bin}" SEED="${seed}" \
      scripts/run_model1_stage1.sh "${stage1_dir}"
    DEVICE="${device}" PYTHON_BIN="${python_bin}" \
      scripts/run_model1_stage2_npe.sh "${stage1_dir}" "${stage2_dir}"
  done
}

model2_worker() {
  local hidden lr loss
  local pilot_cache="${repo_root}/artifacts/model2/k40_m40/stage2/npe_fixed20k_ema_seed_20260709/pilot_cache.npz"
  read -r hidden lr loss < <(select_direct_winner model2)
  mkdir -p "${run_root}/direct_raw_winner/model2"
  printf 'hidden=%s\nlr=%s\nbest_val_loss=%s\n' "${hidden}" "${lr}" "${loss}" \
    >"${run_root}/direct_raw_winner/model2/selection.txt"
  run_direct_stage1 model2 "${hidden}" "${lr}" \
    "${run_root}/direct_raw_winner/model2/stage1" 0
  run_direct_stage2 model2 "${run_root}/direct_raw_winner/model2/stage1" \
    "${run_root}/direct_raw_winner/model2/stage2"

  PILOT_CACHE="${pilot_cache}" DEVICE="${device}" PYTHON_BIN="${python_bin}" \
    scripts/run_model2_stage2_validation_best_npe.sh \
    "${repo_root}/artifacts/model2/k40_m40/stage1/seed_20260710" \
    "${run_root}/multiseed/model2/stage2/stage1_seed_20260710_npe_seed_54000"

  local seed stage1_dir stage2_dir
  for seed in 20260711 20260712 20260713; do
    stage1_dir="${run_root}/multiseed/model2/stage1/seed_${seed}"
    stage2_dir="${run_root}/multiseed/model2/stage2/stage1_seed_${seed}_npe_seed_54000"
    DEVICE="${device}" PYTHON_BIN="${python_bin}" SEED="${seed}" OMP_NUM_THREADS=8 \
      scripts/run_model2_stage1_40x40.sh "${stage1_dir}"
    PILOT_CACHE="${pilot_cache}" DEVICE="${device}" PYTHON_BIN="${python_bin}" \
      scripts/run_model2_stage2_validation_best_npe.sh "${stage1_dir}" "${stage2_dir}"
  done
}

run_direct_sweep >"${log_root}/raw_sweep_driver.log" 2>&1 &
raw_sweep_pid=$!
run_capacity_controls >"${log_root}/capacity_driver.log" 2>&1 &
capacity_pid=$!
wait "${raw_sweep_pid}"
wait "${capacity_pid}"

model1_worker >"${log_root}/model1_worker.log" 2>&1 &
model1_pid=$!
model2_worker >"${log_root}/model2_worker.log" 2>&1 &
model2_pid=$!
wait "${model1_pid}"
wait "${model2_pid}"
