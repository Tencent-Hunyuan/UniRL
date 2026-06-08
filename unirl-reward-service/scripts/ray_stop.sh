#!/usr/bin/env bash
# Stop the reward-service stack on every node in NODE_IP_LIST.
#
# pdsh runs these three kills on each node:
#   1. `ray stop --force`           — Ray cluster (raylet / gcs / actors)
#   2. `pkill -9 -f reward_service` — the service python main process
#   3. `pkill -9 -f VLLM::`         — any vLLM engine/worker orphans
#
# Each step `|| true` so a node where the target is absent exits clean.
#
# Usage:
#   export NODE_IP_LIST="10.1.2.3:8 10.1.2.4:8"
#   scripts/ray_stop.sh

set -euo pipefail

# shellcheck source=./_ray_lib.sh
source "$(dirname "$0")/_ray_lib.sh"

command -v pdsh >/dev/null || { echo "need pdsh on PATH" >&2; exit 1; }

resolve_cluster_nodes   # exports HEAD_NODE / WORKER_NODES / NODES
NODE_LIST=$(echo "${NODES}" | tr ' ' ',')

echo ">>> ray_stop: stopping Ray + reward_service + vLLM on [${NODES}]"
pdsh -R ssh -w "${NODE_LIST}" '
    ray stop --force 2>/dev/null || true
    pkill -9 -f "reward[_]service" 2>/dev/null || true
    pkill -9 -f "VLLM[:]:" 2>/dev/null || true
'
echo ">>> done."
