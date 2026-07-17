#!/usr/bin/env bash
set -euo pipefail

resolve_bagel_profile_config() {
    case "$1" in
        production)
            printf '%s\n' "unified_model/bagel_vllmomni_t2ti"
            ;;
        smoke)
            printf '%s\n' "unified_model/bagel_vllmomni_t2ti_smoke"
            ;;
        *)
            echo "Unknown BAGEL scale profile '$1'; expected production or smoke" >&2
            return 2
            ;;
    esac
}

main() {
    local profile="${BAGEL_VLLMOMNI_PROFILE:-production}"
    if [[ "${1:-}" == "--profile" ]]; then
        if [[ $# -lt 2 ]]; then
            echo "--profile requires production or smoke" >&2
            return 2
        fi
        profile="$2"
        shift 2
    elif [[ "${1:-}" == --profile=* ]]; then
        profile="${1#*=}"
        shift
    fi

    local config_name
    config_name="$(resolve_bagel_profile_config "$profile")"
    local root_dir
    root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local proxy_url="${STAR_PROXY_URL:-http://star-proxy.oa.com:3128}"

    export http_proxy="$proxy_url"
    export https_proxy="$proxy_url"
    export HTTP_PROXY="$proxy_url"
    export HTTPS_PROXY="$proxy_url"
    export no_proxy="${no_proxy:-localhost,127.0.0.1},.woa.com,.oa.com,mirrors.tencent.com"
    export NO_PROXY="$no_proxy"

    export BAGEL_PATH="${BAGEL_PATH:-$root_dir/models/local/BAGEL-7B-MoT}"
    export PICKSCORE_PROCESSOR_ID="${PICKSCORE_PROCESSOR_ID:-$root_dir/models/local/CLIP-ViT-H-14-laion2B-s32B-b79K}"
    export PICKSCORE_MODEL_ID="${PICKSCORE_MODEL_ID:-$root_dir/models/local/PickScore_v1}"
    # Expandable segments avoid fixed-segment fragmentation. The 90% per-process
    # ceiling leaves nominal headroom for sleeping Omni helpers, NCCL, and other
    # CUDA users without moving FSDP state off the GPU.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.95,per_process_memory_fraction:0.90}"
    export PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}"

    local path
    for path in "$BAGEL_PATH" "$PICKSCORE_PROCESSOR_ID" "$PICKSCORE_MODEL_ID"; do
        if [[ ! -d "$path" ]]; then
            echo "Required local model directory is missing: $path" >&2
            return 2
        fi
    done

    cd "$root_dir"
    exec "$root_dir/.venv/bin/python" -m unirl.train_unified_model \
        --config-name "$config_name" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
