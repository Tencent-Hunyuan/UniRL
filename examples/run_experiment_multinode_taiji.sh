#!/usr/bin/env bash
#
# Multi-node experiment launcher for the taiji platform. Two launch modes pick
# how Ray is brought up across nodes (LAUNCH); both end with the head running
# the single training driver and every other node joined to its Ray cluster:
#
#   LAUNCH=spmd (default) — the platform runs this SAME script on every node.
#       rank 0 (INDEX=0) starts the Ray head + driver; every other rank joins
#       Ray and idles. Submit once as the platform's multi-node job entrypoint;
#       taiji fans the script out and sets INDEX + CHIEF_IP per node.
#
#   LAUNCH=ssh — run this script ONCE on the head. It starts the Ray head, then
#       ssh's `ray start` onto every other node in NODE_IP_LIST, then runs the
#       driver. For interactive multi-node sessions where you only have a shell
#       on the head. Prereqs: passwordless ssh head->workers (taiji provides it),
#       the repo at the SAME path on every node (shared mount), and CONDA_ENV set
#       so a non-login ssh shell finds `ray`. taiji sets NODE_IP_LIST (format
#       IP:GPUS,IP:GPUS,...) + CHIEF_IP.
#
# Cluster topology defaults to taiji's job env (see "Cluster topology" below);
# set the explicit vars to run on any other cluster.
#
# The driver is one of the Hydra entrypoints, selected with ENTRY:
#   train_diffusion (default)  examples/diffusion/ (sd3_*, wan2*, qwen_image_*)
#   train_ar                   examples/ar/ (qwen_vl_grpo_*, qwen3_drpo_*)
#   train_pe                   examples/pe/ (prompt-enhancement joint diffusion+AR)
#   train_unified_model                  examples/unified_model/ (HunyuanImage3, unified AR+diffusion)
#
# The first positional arg is the examples/ config name, domain-qualified
# (passed to Hydra as --config-name); any extra args are forwarded verbatim as Hydra overrides. The
# launcher sets num_devices to the whole cluster (NUM_NODES * GPUS_PER_NODE) so a
# conf authored for a different size still runs here (an explicit num_devices=...
# wins). Run settings (PRETRAINED_MODEL / DATA_PATH / WANDB_*) come from the conf
# via ${oc.env:...}; export them to override a conf's own default.
#
# Examples:
#   # SPMD batch (taiji lands this same line on every node):
#   bash examples/run_experiment_multinode_taiji.sh diffusion/sd3/sd3_sglang_rollout_colocate
#   # ssh fan-out (run once on the head only):
#   LAUNCH=ssh bash examples/run_experiment_multinode_taiji.sh diffusion/sd3/sd3_sglang_rollout_colocate
#   # VLM/AR recipe (4x8):
#   ENTRY=train_ar bash examples/run_experiment_multinode_taiji.sh ar/qwen_vl_grpo_geo3k_mc_4x8
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

# --- ssh fan-out worker join (LAUNCH=ssh) -----------------------------------
# In LAUNCH=ssh the head ssh-invokes THIS script with RAY_JOIN_ONLY=1 on each
# worker. The Python env is already activated above; here we just join Ray with
# the head address + this node's IP/GPUs (all passed in over ssh) and return so
# the raylet stays up. Workers do no pip install, no CMD build, no driver.
if [ "${RAY_JOIN_ONLY:-0}" = "1" ]; then
    : "${HEAD_IP:?RAY_JOIN_ONLY requires HEAD_IP (the head passes it over ssh)}"
    : "${NODE_IP:?RAY_JOIN_ONLY requires NODE_IP (the head passes it over ssh)}"
    JOIN_GPUS="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"
    JOIN_PORT="${RAY_PORT:-6379}"
    ray stop >/dev/null 2>&1 || true
    until ray start \
        --address="${HEAD_IP}:${JOIN_PORT}" \
        --node-ip-address="${NODE_IP}" \
        --num-gpus="${JOIN_GPUS}"; do
        echo "[worker ${NODE_IP}] Ray head not ready; retry in 5s..."
        sleep 5
    done
    echo "[worker ${NODE_IP}] joined Ray at ${HEAD_IP}:${JOIN_PORT}."
    exit 0
fi

# --- Run defaults (the conf reads these via ${oc.env:...}) ------------------
export REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXPERIMENT}}"
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"

# --- Cluster topology (taiji platform defaults) -----------------------------
# Defaults come from taiji's multi-node job env (commented per line); the
# explicit vars always win, so this launcher also runs on a non-taiji cluster.
NUM_NODES="${NUM_NODES:-${HOST_NUM:-2}}"               # taiji HOST_NUM:     node count
GPUS_PER_NODE="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"   # taiji HOST_GPU_NUM: GPUs per node
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"                  # taiji INDEX:        this node's rank
RAY_PORT="${RAY_PORT:-6379}"
LAUNCH="${LAUNCH:-spmd}"                               # spmd: platform runs this per node | ssh: head fans out

# This node's IP. Prefer an explicit NODE_IP, else taiji's LOCAL_IP. On multi-NIC
# / container nodes `hostname -I` often returns a container-internal IP that peers
# can't reach, so when CHIEF_IP is known, pick this node's IP on the chief's /16.
all_ips="$(hostname -I 2>/dev/null || true)"
if [ -z "${NODE_IP:-}" ] && [ -n "${LOCAL_IP:-}" ]; then
    NODE_IP="${LOCAL_IP}"
fi
if [ -z "${NODE_IP:-}" ] && [ -n "${CHIEF_IP:-}" ]; then
    chief_subnet="$(echo "${CHIEF_IP}" | cut -d. -f1-2)"
    NODE_IP="$(echo "${all_ips}" | tr ' ' '\n' | grep "^${chief_subnet}\." | head -1 || true)"
fi
if [ -z "${NODE_IP:-}" ]; then
    NODE_IP="$(echo "${all_ips}" | awk '{print $1}')"
fi
NODE_IP="${NODE_IP:-127.0.0.1}"

HEAD_IP="${HEAD_IP:-${CHIEF_IP:-${NODE_IP}}}"          # taiji CHIEF_IP:     head node IP

# --- Driver command ---------------------------------------------------------
# num_devices spans the whole cluster; the conf's own value is overridden so one
# recipe runs across node counts (an explicit num_devices=... in "$@" still wins).
ENTRY="${ENTRY:-train_diffusion}"
CMD=(
    python -m "unirl.${ENTRY}"
    "--config-name=${EXPERIMENT}"
    "num_devices=$((NUM_NODES * GPUS_PER_NODE))"
)
CMD+=("$@")

echo "Command (ENTRY=${ENTRY}, ${NUM_NODES}x${GPUS_PER_NODE}):"
printf '  %q' "${CMD[@]}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

if [ "${INSTALL_EDITABLE:-1}" = "1" ]; then
    pip install --no-deps -e .
fi

# Start the Ray head on this node. Used by both the SPMD rank-0 path and the ssh
# head path.
start_ray_head() {
    ray start --head \
        --node-ip-address="${NODE_IP}" \
        --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 \
        --num-gpus="${GPUS_PER_NODE}"
}

ray stop >/dev/null 2>&1 || true

if [ "${LAUNCH}" = "ssh" ]; then
    # Head-only fan-out: this script runs ONCE on the head. Start the head, then
    # ssh `ray start` onto every other node in NODE_IP_LIST, then fall through to
    # run the driver below. Workers re-enter this script with RAY_JOIN_ONLY=1
    # (early-exit near the top), so they only join Ray and return.
    if [ "${NODE_RANK}" != "0" ]; then
        echo "LAUNCH=ssh must be invoked on the head (rank 0); got NODE_RANK=${NODE_RANK}." >&2
        exit 2
    fi
    if [ -z "${NODE_IP_LIST:-}" ]; then
        echo "LAUNCH=ssh needs NODE_IP_LIST (taiji sets it; format IP:GPUS,IP:GPUS,...)." >&2
        exit 2
    fi
    start_ray_head
    SSH_USER="${SSH_USER:-$(whoami)}"
    # Forward the head's NCCL_* env to each worker's `ray start`. The ssh shell is
    # bare (no .bashrc), so without this the workers miss e.g. NCCL_NET_GDR_LEVEL
    # and crash at multi-node FSDP init. (SPMD mode gets NCCL env per-node from the
    # job env; in ssh mode the head's env is the only source, so propagate it.)
    ssh_nccl_env=""
    while IFS='=' read -r _k _v; do
        [ -n "${_k}" ] && ssh_nccl_env+="${_k}='${_v}' "
    done < <(env | grep -E '^NCCL_|^PYTORCH_CUDA_ALLOC_CONF' || true)
    IFS=',' read -ra _NODE_ENTRIES <<< "${NODE_IP_LIST}"
    worker_n=0
    for entry in "${_NODE_ENTRIES[@]}"; do
        w_ip="${entry%%:*}"
        w_gpu="${entry##*:}"
        if [ "${w_gpu}" = "${entry}" ]; then  # entry had no ":GPUS" suffix
            w_gpu="${GPUS_PER_NODE}"
        fi
        if [ -z "${w_ip}" ] || [ "${w_ip}" = "${HEAD_IP}" ] || [ "${w_ip}" = "${NODE_IP}" ]; then
            continue  # the head runs the driver, not a joined worker
        fi
        worker_n=$((worker_n + 1))
        echo "[head] ssh ray start -> ${SSH_USER}@${w_ip} (gpus=${w_gpu})"
        ssh -f -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o BatchMode=yes \
            "${SSH_USER}@${w_ip}" \
            "cd '${REPO_ROOT}' && \
             RAY_JOIN_ONLY=1 HEAD_IP='${HEAD_IP}' NODE_IP='${w_ip}' \
             GPUS_PER_NODE='${w_gpu}' RAY_PORT='${RAY_PORT}' \
             CONDA_ENV='${CONDA_ENV:-}' CONDA_SH='${CONDA_SH:-}' VENV_DIR='${VENV_DIR:-}' \
             ${ssh_nccl_env}\
             nohup bash examples/run_experiment_multinode_taiji.sh '${EXPERIMENT}' \
             >/tmp/unirl_ray_worker_${worker_n}.log 2>&1 &" \
            || echo "[head] WARNING: ssh to ${w_ip} failed; that node will not join." >&2
    done
    echo "[head] fanned out to ${worker_n} worker(s); they join /tmp/unirl_ray_worker_*.log."
elif [ "${NODE_RANK}" = "0" ]; then
    # SPMD (default): the platform runs this script on every node; rank 0 is head.
    start_ray_head
else
    # SPMD worker: join Ray and idle so the platform keeps this node alive while
    # the head owns the driver.
    until ray start \
        --address="${HEAD_IP}:${RAY_PORT}" \
        --node-ip-address="${NODE_IP}" \
        --num-gpus="${GPUS_PER_NODE}"; do
        echo "Ray head not ready yet; retrying in 5s..."
        sleep 5
    done
    echo "Worker node joined Ray cluster; head node owns the training driver."
    exec tail -f /dev/null  # block forever; only the head runs the driver below
fi

echo "Ray cluster target: ${NUM_NODES} node(s) x ${GPUS_PER_NODE} GPU(s)"
sleep "${RAY_CLUSTER_WAIT_S:-30}"
exec "${CMD[@]}"
