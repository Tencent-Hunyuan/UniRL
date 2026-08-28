#!/usr/bin/env bash
#
# Persistent Taiji reward-curve verification launcher.
#
# Run this script from tmux on an already-allocated pod. It deliberately uses
# the repository recipes unchanged apart from explicit W&B identity overrides.
# Model and dataset paths must be pod-local; never point them at Ceph.
#
# Profiles:
#   sd3-trainside    1x8, .venv,        diffusion/sd3/sd3_trainside
#   sd3-vllm-omni   1x8, .venv,        diffusion/sd3/sd3_vllmomni
#   pe               1x8, .venv,        pe/pe_trainside_pickscore
#   ar-drpo          4x8, .venv-sglang, ar/qwen3_drpo_4b_base_dapo_sglang
#   qwen-omni        1x8, .venv,        ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x8
#
# Example:
#   tmux new-session -d -s unirl-sd3 \
#     'bash examples/run_reward_curve_verification_taiji.sh sd3-trainside'
#
set +u
if [ -r /etc/bashrc ]; then
    # Taiji's non-interactive shell does not otherwise get network credentials.
    # shellcheck disable=SC1091
    source /etc/bashrc
fi
set -euo pipefail

PROXY_URL="${PROXY_URL:-http://star-proxy.oa.com:3128}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
export HTTP_PROXY="${http_proxy}"
export HTTPS_PROXY="${https_proxy}"
export NO_PROXY="${no_proxy}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

WANDB_ENV_FILE="${UNIRL_WANDB_ENV:-/root/.config/unirl/wandb.env}"
if [ -r "${WANDB_ENV_FILE}" ]; then
    # Kept outside git; expected to export WANDB_API_KEY and optionally entity.
    # shellcheck disable=SC1090
    source "${WANDB_ENV_FILE}"
fi

export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-linyuwus}"
export WANDB_PROJECT="${WANDB_PROJECT:-unirl-agentic-main-verification}"
export REPORT_TO_WANDB=true
: "${WANDB_API_KEY:?WANDB_API_KEY must be exported or stored in ${WANDB_ENV_FILE}}"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <sd3-trainside|sd3-vllm-omni|pe|ar-drpo|qwen-omni> [hydra overrides...]" >&2
    exit 2
fi

PROFILE="$1"
shift
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${WANDB_RUN_NAME:-unirl-${PROFILE}-${RUN_STAMP}}"

require_file() {
    if [ ! -f "$1" ]; then
        echo "Required local file is missing: $1" >&2
        exit 2
    fi
}

require_dist_version() {
    local dist_name="$1"
    local expected="$2"
    local actual
    actual="$("${PYTHON}" -c \
        "import importlib.metadata as m; print(m.version('${dist_name}'))" 2>/dev/null || true)"
    if [ "${actual}" != "${expected}" ]; then
        echo "${dist_name} must be ${expected}; found ${actual:-missing} in ${VENV_DIR}" >&2
        exit 2
    fi
}

require_torch_flavor() {
    local expected="$1"
    local actual
    actual="$("${PYTHON}" -c "import torch; print(torch.__version__)")"
    if [ "${actual}" != "${expected}" ]; then
        echo "torch must be ${expected}; found ${actual} in ${VENV_DIR}" >&2
        exit 2
    fi
}

SINGLE_NODE=1
case "${PROFILE}" in
    sd3-trainside)
        ENTRY=train_diffusion
        EXPERIMENT=diffusion/sd3/sd3_trainside
        VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
        export PRETRAINED_MODEL="${PRETRAINED_MODEL:-${REPO_ROOT}/models/local/stable-diffusion-3.5-medium}"
        require_file "${PRETRAINED_MODEL}/model_index.json"
        ;;
    sd3-vllm-omni)
        ENTRY=train_diffusion
        EXPERIMENT=diffusion/sd3/sd3_vllmomni
        VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
        export PRETRAINED_MODEL="${PRETRAINED_MODEL:-${REPO_ROOT}/models/local/stable-diffusion-3.5-medium}"
        require_file "${PRETRAINED_MODEL}/model_index.json"
        ;;
    pe)
        ENTRY=train_pe
        EXPERIMENT=pe/pe_trainside_pickscore
        VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
        # The colocated SD3 + Qwen FSDP stacks run close to the H20 memory
        # ceiling.  Variable AR sequence lengths can otherwise fragment the
        # caching allocator across rollouts until a small unshard allocation
        # fails despite substantial reserved-but-unused memory.
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
        export PRETRAINED_MODEL="${PRETRAINED_MODEL:-${REPO_ROOT}/models/local/stable-diffusion-3.5-medium}"
        export LLM_MODEL="${LLM_MODEL:-${REPO_ROOT}/models/local/Qwen3-0.6B}"
        require_file "${PRETRAINED_MODEL}/model_index.json"
        require_file "${LLM_MODEL}/config.json"
        ;;
    ar-drpo)
        ENTRY=train_ar
        EXPERIMENT=ar/qwen3_drpo_4b_base_dapo_sglang
        VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-sglang}"
        CUDA_TOOLKIT_DIR="${CUDA_TOOLKIT_DIR:-}"
        if [ -z "${CUDA_TOOLKIT_DIR}" ]; then
            for candidate in \
                "${VENV_DIR}"/lib/python*/site-packages/nvidia/cu13; do
                if [ -x "${candidate}/bin/nvcc" ]; then
                    CUDA_TOOLKIT_DIR="${candidate}"
                    break
                fi
            done
        fi
        if [ -z "${CUDA_TOOLKIT_DIR}" ] || [ ! -x "${CUDA_TOOLKIT_DIR}/bin/nvcc" ]; then
            echo "CUDA 13.0 toolkit is missing from ${VENV_DIR}; install the pinned nvidia CUDA compiler wheels." >&2
            exit 2
        fi
        export CUDA_HOME="${CUDA_TOOLKIT_DIR}"
        export CUDA_PATH="${CUDA_TOOLKIT_DIR}"
        export CUDACXX="${CUDA_TOOLKIT_DIR}/bin/nvcc"
        export NVCC="${CUDACXX}"
        export PATH="${CUDA_TOOLKIT_DIR}/bin:${PATH}"
        CUDA_RUNTIME_LIB_DIR=""
        for candidate in "${CUDA_TOOLKIT_DIR}/lib64" "${CUDA_TOOLKIT_DIR}/lib"; do
            if compgen -G "${candidate}/libcudart.so*" >/dev/null; then
                CUDA_RUNTIME_LIB_DIR="${candidate}"
                break
            fi
        done
        if [ -z "${CUDA_RUNTIME_LIB_DIR}" ]; then
            echo "CUDA 13 runtime libraries are missing from ${CUDA_TOOLKIT_DIR}." >&2
            exit 2
        fi
        # NVIDIA's pip toolkit has lib/libcudart.so.13 but no conventional
        # lib64/libcudart.so linker name. The multinode launcher creates this
        # small per-node shim before Ray starts, so SGLang TVM-FFI JIT links.
        export CUDA_RUNTIME_LIB_DIR
        export CUDA_RUNTIME_LINK_DIR="${CUDA_RUNTIME_LINK_DIR:-/tmp/unirl-cuda-runtime-${UID}}"
        CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-}"
        if [ -z "${CUDA_COMPAT_DIR}" ]; then
            for candidate in \
                "${REPO_ROOT}"/.cuda-compat-13/usr/local/cuda-13.*/compat \
                /usr/local/cuda-13.*/compat; do
                if [ -d "${candidate}" ]; then
                    CUDA_COMPAT_DIR="${candidate}"
                    break
                fi
            done
        fi
        if [ -z "${CUDA_COMPAT_DIR}" ] || [ ! -d "${CUDA_COMPAT_DIR}" ]; then
            echo "CUDA 13 forward-compat libraries are missing; set CUDA_COMPAT_DIR." >&2
            exit 2
        fi
        export CUDA_COMPAT_DIR
        export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export QWEN3_PATH="${QWEN3_PATH:-${REPO_ROOT}/models/local/Qwen3-4B-Base}"
        export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/dapo_math/train.jsonl}"
        export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/dapo_math/aime_eval.jsonl}"
        require_file "${QWEN3_PATH}/config.json"
        require_file "${DATA_PATH}"
        require_file "${EVAL_DATA_PATH}"
        SINGLE_NODE=0
        ;;
    qwen-omni)
        ENTRY=train_ar
        EXPERIMENT=ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x8
        VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
        export QWEN3_OMNI_PATH="${QWEN3_OMNI_PATH:-${REPO_ROOT}/models/local/Qwen3-Omni-30B-A3B-Instruct}"
        export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/video_r1_260k/train.jsonl}"
        export EVAL_DATA_PATH="${EVAL_DATA_PATH:-${REPO_ROOT}/data/video_r1_260k/val.jsonl}"
        require_file "${QWEN3_OMNI_PATH}/config.json"
        require_file "${DATA_PATH}"
        require_file "${EVAL_DATA_PATH}"
        ;;
    *)
        echo "Unknown verification profile: ${PROFILE}" >&2
        exit 2
        ;;
esac

PYTHON="${VENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "Required environment is missing: ${VENV_DIR}" >&2
    exit 2
fi
export VENV_DIR
export PATH="${VENV_DIR}/bin:${PATH}"

if [ "${PROFILE}" = "ar-drpo" ]; then
    require_torch_flavor "2.11.0+cu130"
    require_dist_version "sglang" "0.5.12.post1"
    require_dist_version "nvidia-cuda-nvcc" "13.0.88"
    require_dist_version "nvidia-cuda-crt" "13.0.88"
    require_dist_version "nvidia-nvvm" "13.0.88"
    require_dist_version "nvidia-cuda-cccl" "13.0.85"
    require_dist_version "nvidia-cuda-runtime" "13.0.96"
    if ! "${CUDACXX}" --version | grep -q "release 13.0"; then
        echo "SGLang JIT compilation requires CUDA 13.0 nvcc; found $(${CUDACXX} --version | tail -1)." >&2
        exit 2
    fi
else
    require_torch_flavor "2.11.0+cu129"
fi
if [ "${PROFILE}" = "sd3-vllm-omni" ] || [ "${PROFILE}" = "qwen-omni" ]; then
    require_dist_version "vllm" "0.22.0+cu129"
    require_dist_version "vllm-omni" "0.22.0"
fi

WANDB_OVERRIDES=(
    "logging.report_to_wandb=true"
    "logging.project_name=${WANDB_PROJECT}"
    "logging.run_name=${RUN_NAME}"
    "logging.entity=${WANDB_ENTITY}"
)

echo "Verification profile: ${PROFILE}"
echo "W&B run: ${WANDB_ENTITY}/${WANDB_PROJECT}/${RUN_NAME}"
echo "Python environment: ${VENV_DIR}"

if [ "${SINGLE_NODE}" = "1" ]; then
    export ENTRY EXPERIMENT
    exec bash examples/run_experiment_single_node.sh \
        "${WANDB_OVERRIDES[@]}" "$@"
fi

export ENTRY EXPERIMENT
export LAUNCH="${LAUNCH:-ssh}"
exec bash examples/run_experiment_multinode_taiji.sh \
    "${WANDB_OVERRIDES[@]}" "$@"
