#!/usr/bin/env bash
# Stop the reward-service stack on every node in NODE_IP_LIST.
#
# Shutdown order is service -> actors/Ray -> MPS. The service gets a bounded
# SIGTERM grace period so actor drain/close runs before Ray is forced down.
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

echo ">>> ray_stop: draining reward_service, then stopping Ray on [${NODES}]"
pdsh -R ssh -w "${NODE_LIST}" '
    service_pattern="(python.*-m reward[_]service|unirl-reward-servic[e])"
    pkill -TERM -f "${service_pattern}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      pgrep -f "${service_pattern}" >/dev/null 2>&1 || break
      sleep 1
    done
    pkill -9 -f "${service_pattern}" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    pkill -9 -f "VLLM[:]:" 2>/dev/null || true
'
if [[ "${ENABLE_MPS:-0}" == "1" ]]; then
  "$(dirname "$0")/mps.sh" stop
fi
echo ">>> done."
