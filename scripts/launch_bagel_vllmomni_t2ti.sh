#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROXY_URL="${STAR_PROXY_URL:-http://star-proxy.oa.com:3128}"

export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export no_proxy="${no_proxy:-localhost,127.0.0.1},.woa.com,.oa.com,mirrors.tencent.com"
export NO_PROXY="$no_proxy"

export BAGEL_PATH="${BAGEL_PATH:-$ROOT_DIR/models/local/BAGEL-7B-MoT}"
export PICKSCORE_PROCESSOR_ID="${PICKSCORE_PROCESSOR_ID:-$ROOT_DIR/models/local/CLIP-ViT-H-14-laion2B-s32B-b79K}"
export PICKSCORE_MODEL_ID="${PICKSCORE_MODEL_ID:-$ROOT_DIR/models/local/PickScore_v1}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

for path in "$BAGEL_PATH" "$PICKSCORE_PROCESSOR_ID" "$PICKSCORE_MODEL_ID"; do
    if [[ ! -d "$path" ]]; then
        echo "Required local model directory is missing: $path" >&2
        exit 2
    fi
done

cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" -m unirl.train_unified_model \
    --config-name unified_model/bagel_vllmomni_t2ti "$@"
