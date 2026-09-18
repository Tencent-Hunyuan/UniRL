#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mode="${1:?usage: run_sglang_0519_alignment.sh ar|sd3|tp_sync|sd3_sync|ep MODEL_PATH OUTPUT_JSON}"
model_path="${2:?model path is required}"
output_json="${3:?output JSON path is required}"

export PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
export PIP_TRUSTED_HOST=mirrors.cloud.tencent.com

python3 -m pip install --no-deps --force-reinstall "sglang==0.5.19"
if [ "$mode" = "tp_sync" ] || [ "$mode" = "sd3_sync" ]; then
  python3 -m pip install --no-deps "peft==0.21.0"
fi
python3 -m pip install --no-build-isolation --no-deps -e .

exec python3 scripts/sglang_0519_alignment.py \
  "$mode" \
  --model "$model_path" \
  --output "$output_json" \
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.3}"
