set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

# === Data ===
# Newline-delimited prompts .txt (or {"prompt": ...} .jsonl). VideoAlign is
# text-conditioned; no reference video / first-frame needed.
export DATA_PATH=${DATA_PATH:-/path/to/wan21_prompts.txt}
export EVAL_DATA_PATH=${EVAL_DATA_PATH:-${DATA_PATH}}

# === Models ===
# WAN 2.1 T2V 1.3B base checkpoint
export PRETRAINED_MODEL=${PRETRAINED_MODEL:-/path/to/Wan2.1-T2V-1.3B-Diffusers}
# VideoAlign Qwen2-VL reward checkpoint
export VIDEOALIGN_MODEL_PATH=${VIDEOALIGN_MODEL_PATH:-/path/to/VideoReward}

# === Output / Logging ===
export OUTPUT_DIR=${OUTPUT_DIR:-./outputs/wan21_t2v_videoalign_refl}
export REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
export WANDB_PROJECT=${WANDB_PROJECT:-unirl-refl}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-wan21_t2v_videoalign_refl_recipe_opt}

mkdir -p "${OUTPUT_DIR}" logs

LOG_FILE="logs/wan21_t2v_videoalign_refl_$(date +%Y%m%d_%H%M%S).log"
echo "=== launching wan21 t2v videoalign refl, log → ${LOG_FILE} ==="

RAY_ADDRESS=auto python -u -m recipes.refl.run \
    --config-name=wan21_t2v_videoalign_refl \
    num_devices=8 \
    2>&1 | tee "${LOG_FILE}"
