#!/usr/bin/env bash
#
# Single-node experiment launcher (cluster-agnostic — no platform env needed).
# Starts a local Ray head on this machine and runs the v2 training driver.
#
# The driver is one of the Hydra entrypoints, selected with ENTRY:
#   train_diffusion (default)  examples/diffusion/ (sd3_*, wan2*, qwen_image_*)
#   train_ar                   examples/ar/ (qwen_vl_grpo_*, qwen3_drpo_*)
#   train_pe                   examples/pe/ (prompt-enhancement joint diffusion+AR)
#   train_unified_model                  examples/unified_model/ (HunyuanImage3, unified AR+diffusion)
#
# The first positional arg is the examples/ config name, domain-qualified
# (passed to Hydra as --config-name); any extra args are forwarded verbatim as Hydra overrides.
# The launcher sets num_devices to this node's GPU count so a conf authored for
# a different node count still runs here (an explicit num_devices=... wins).
#
# Run settings come from the conf via ${oc.env:...}: model checkpoint
# (PRETRAINED_MODEL / QWEN_VL_PATH / ...), data (DATA_PATH / EVAL_DATA_PATH —
# read only by the VLM/AR recipes; diffusion recipes use their own data_source),
# and W&B (REPORT_TO_WANDB / WANDB_RUN_NAME / WANDB_ENTITY / WANDB_PROJECT).
# Export any of them before running to override a conf's own default.
#
# Examples:
#   bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
#   REPORT_TO_WANDB=true bash examples/run_experiment_single_node.sh diffusion/qwen_image/qwen_image_trainside
#   ENTRY=train_ar bash examples/run_experiment_single_node.sh ar/qwen_vl_grpo_geo3k_mc_4x8
#   ENTRY=train_pe bash examples/run_experiment_single_node.sh pe/pe_trainside_pickscore
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${EXPERIMENT:-}" ]; then
    if [ "$#" -lt 1 ]; then
        echo "Usage: $0 <config-name> [hydra overrides...]"
        exit 2
    fi
    EXPERIMENT="$1"
    shift
fi

# --- Python env (optional) --------------------------------------------------
if [ -n "${CONDA_ENV:-}" ]; then
    if [ -f "${CONDA_SH:-}" ]; then
        # shellcheck disable=SC1090
        source "${CONDA_SH}"
    elif [ -f "/data/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/data/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/opt/conda/etc/profile.d/conda.sh"
    fi
    conda activate "${CONDA_ENV}"
elif [ -n "${VENV_DIR:-}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi

# --- Run defaults (the conf reads these via ${oc.env:...}) ------------------
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

# --- GPU count -> num_devices -----------------------------------------------
# Override with GPUS_PER_NODE, else autodetect, else assume 8. num_devices is
# the size of the v2 DevicePool, i.e. the GPUs on this node.
if [ -z "${GPUS_PER_NODE:-}" ]; then
    GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
    [ "${GPUS_PER_NODE:-0}" -gt 0 ] 2>/dev/null || GPUS_PER_NODE=8
fi

ENTRY="${ENTRY:-train_diffusion}"
CMD=(
    python -m "unirl.${ENTRY}"
    "--config-name=${EXPERIMENT}"
    "num_devices=${GPUS_PER_NODE}"
)
CMD+=("$@")

echo "Command (ENTRY=${ENTRY}):"
printf '  %q' "${CMD[@]}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

if [ "${INSTALL_EDITABLE:-1}" = "1" ]; then
    pip install --no-deps -e .
fi

# --- Single-node Ray (local head) -------------------------------------------
NODE_IP="${NODE_IP:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-6379}"

ray stop >/dev/null 2>&1 || true
ray start --head \
    --node-ip-address="${NODE_IP}" \
    --port="${RAY_PORT}" \
    --dashboard-host=0.0.0.0 \
    --num-gpus="${GPUS_PER_NODE}"

exec "${CMD[@]}"
