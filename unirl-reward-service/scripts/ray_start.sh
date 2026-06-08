#!/usr/bin/env bash
# Start a multi-host Ray cluster for the reward service.
#
# Usage:
#   export NODE_IP_LIST="10.1.2.3:8 10.1.2.4:8"   # first entry = head
#   scripts/ray_start.sh                           # uses defaults below
#   scripts/ray_start.sh --gpus 8 --port 6379      # override per-node GPU count / GCS port
#
# Environment:
#   NODE_IP_LIST   space-separated "ip:cards" tokens (the ":cards" suffix is stripped).
#                  First token is the head node; the rest are workers.
#   RAY_PORT       overridable GCS port (default 6379).
#   NUM_GPUS       per-node GPU count advertised to Ray (default 8).
#   RAY_TMPDIR     per-host Ray temp-dir (default /tmp/ray-$USER). Must stay
#                  short enough for plasma_store's AF_UNIX socket to fit in
#                  the kernel's 107-byte limit; see docs/DEVELOPMENT_LOG
#                  §12.10 for the math.
#
# Assumptions (per resume-prompt & user confirmation):
#   - Same conda env and repo path on every node; pdsh can reach them over SSH
#     with no cd/activate needed beforehand.
#   - Ray runtime temp-dir is a process-runtime concern (Unix sockets +
#     shm), distinct from the project's "caches in current dir" rule which
#     applies to accumulating caches (.pycache / .pytest_cache / .pip-cache
#     / .install.out). Keeping Ray's temp-dir on local /tmp sidesteps
#     AF_UNIX path-length limits on deep share paths.
#   - Default Ray GCS port 6379; no external override needed.
#
# On exit: prints the `cluster.ray_address` line to paste into your YAML.

set -euo pipefail

# Shared NODE_IP_LIST parsing lives in _ray_lib.sh.
# shellcheck source=./_ray_lib.sh
source "$(dirname "$0")/_ray_lib.sh"

RAY_PORT="${RAY_PORT:-6379}"
NUM_GPUS="${NUM_GPUS:-8}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-$USER}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)  RAY_PORT="$2"; shift 2 ;;
    --gpus)  NUM_GPUS="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

for tool in pdsh ray; do
  command -v "$tool" >/dev/null || { echo "need $tool on PATH" >&2; exit 1; }
done

resolve_cluster_nodes   # exports HEAD_NODE / WORKER_NODES / NODES

echo ">>> ray_start: head=${HEAD_NODE}, workers=[${WORKER_NODES}], port=${RAY_PORT}, gpus/node=${NUM_GPUS}"
echo ">>> ray_tmpdir=${RAY_TMPDIR} (per-host, under /tmp to keep AF_UNIX socket paths short)"

# Head node. `--node-ip-address` is explicit so Ray does not accidentally
# bind to 127.0.0.1 inside containers where hostname resolution is odd.
# Proxy env vars (if set in the launching shell) are forwarded to raylet
# so actors inherit them — lets ImageReward etc. reach HuggingFace from
# nodes without direct internet.
pdsh -R ssh -w "${HEAD_NODE}" "
  set -euo pipefail
  mkdir -p ${RAY_TMPDIR}
  ray stop --force >/dev/null 2>&1 || true
  http_proxy='${http_proxy:-${HTTP_PROXY:-}}' https_proxy='${https_proxy:-${HTTPS_PROXY:-}}' \
  HTTP_PROXY='${http_proxy:-${HTTP_PROXY:-}}' HTTPS_PROXY='${https_proxy:-${HTTPS_PROXY:-}}' \
  ray start --head \
    --node-ip-address=${HEAD_NODE} \
    --port=${RAY_PORT} \
    --num-gpus=${NUM_GPUS} \
    --temp-dir=${RAY_TMPDIR}
"

# Workers: one pdsh call per IP so --node-ip-address can be interpolated
# from NODE_IP_LIST rather than derived on-node with `hostname -i` (which
# frequently returns 127.0.0.1 inside containers with quirky /etc/hosts,
# silently binding Ray to loopback and breaking the cluster). One pdsh
# per host serializes starts but the operation is <5s each; parallelism
# here isn't worth the loopback risk.
if [[ -n "${WORKER_NODES}" ]]; then
  for worker_ip in ${WORKER_NODES}; do
    pdsh -R ssh -w "${worker_ip}" "
      set -euo pipefail
      mkdir -p ${RAY_TMPDIR}
      ray stop --force >/dev/null 2>&1 || true
      http_proxy='${http_proxy:-${HTTP_PROXY:-}}' https_proxy='${https_proxy:-${HTTPS_PROXY:-}}' \
      HTTP_PROXY='${http_proxy:-${HTTP_PROXY:-}}' HTTPS_PROXY='${https_proxy:-${HTTPS_PROXY:-}}' \
      ray start \
        --address=${HEAD_NODE}:${RAY_PORT} \
        --node-ip-address=${worker_ip} \
        --num-gpus=${NUM_GPUS} \
        --temp-dir=${RAY_TMPDIR}
    "
  done
fi

echo ""
echo ">>> Ray cluster is up."
echo ">>> Point the service at it by adding to your YAML:"
echo ""
echo "    cluster:"
echo "      ray_address: ${HEAD_NODE}:${RAY_PORT}"
echo ""
echo ">>> Or use 'auto' if the service runs on the head node."
