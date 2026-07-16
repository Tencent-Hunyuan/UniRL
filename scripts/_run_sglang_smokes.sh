#!/bin/bash
# Run the three sglang-stack rollout smokes (AR, sglang_diffusion, composed)
# sequentially in one process, logging each to $SMOKE_LOG. A single burn-stop
# window covers all three. Generic: model paths + venv come from env (defaults
# match the pod-local copies). LIN-454 all-modality e2e.
cd "$(dirname "$0")/.." || exit 1
LOG="${SMOKE_LOG:-/root/allsmoke.log}"
: > "$LOG"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN3_PATH="${QWEN3_PATH:-/root/unirl/models/local/Qwen3-4B-Base}"
export PRETRAINED_MODEL="${PRETRAINED_MODEL:-/root/unirl/models/local/stable-diffusion-3.5-medium}"
PY="${SMOKE_PY:-.venv-sglang/bin/python}"

run() {
  local name="$1"; shift
  echo "=== $name START ===" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  echo "=== $name EXIT $? ===" >> "$LOG"
}

run AR "$PY" scripts/rollout_ar_smoke.py
run SGLANG_DIFFUSION "$PY" scripts/rollout_sd3_sglang_smoke.py
run COMPOSED "$PY" scripts/rollout_composed_smoke.py
echo "=== ALLDONE ===" >> "$LOG"
