set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

export PRETRAINED_MODEL=${PRETRAINED_MODEL:-/path/to/Wan2.2-I2V-A14B-Diffusers}
export DATA_PATH=${DATA_PATH:-/path/to/wan22_face_refl_prompts.jsonl}
export EVAL_DATA_PATH=${EVAL_DATA_PATH:-${DATA_PATH}}
export FACE_MODEL_PATH=${FACE_MODEL_PATH:-/path/to/antelodev2}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/wan22_face_refl_recipe_opt}

export REPORT_TO_WANDB=${REPORT_TO_WANDB:-true}
export WANDB_PROJECT=${WANDB_PROJECT:-unirl-refl}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-wan22_face_refl_recipe_opt}

mkdir -p logs

RAY_ADDRESS=auto python -m recipes.refl.run \
  num_devices=8 \
  2>&1 | tee ./logs/wan22_i2v_refl.log
