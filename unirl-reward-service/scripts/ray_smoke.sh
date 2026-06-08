#!/usr/bin/env bash
# End-to-end smoke + small benchmark for a running multi-host deployment.
#
# Assumes the cluster is already up (scripts/ray_start.sh) and the service
# is running on the head node against a YAML whose `cluster.ray_address`
# points at that cluster. This script does NOT start the service — starting
# it is part of the human operator's checklist because stdout/log capture
# decisions vary per deployment.
#
# Usage:
#   scripts/ray_smoke.sh http://head-node:8080                # defaults
#   scripts/ray_smoke.sh http://head-node:8080 --rewards clip,hpsv2
#   TOTAL=500 CONCURRENCY=100 scripts/ray_smoke.sh http://head-node:8080
#
# Flow:
#   1. smoke_client.py  — one /score per reward to prove correctness
#   2. bench_concurrent.py --sweep  — throughput + p50/p95 latency at
#      increasing concurrency levels to see cross-node scaling
#
# Cache discipline: pytest cache / pycache prefix routed to the repo dir
# per the project-wide "caches in current dir" rule.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '2,22p' "$0" >&2
  exit 2
fi

URL="$1"; shift || true
EXTRA_ARGS=("$@")

TOTAL="${TOTAL:-200}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-50 100 200 400}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${PYTHONPATH:-${REPO_ROOT}}"
export PYTHONPYCACHEPREFIX="${REPO_ROOT}/.pycache"

echo "=========================================="
echo "1. Smoke test (one /score per reward)"
echo "=========================================="
python3 scripts/smoke_client.py --url "${URL}" "${EXTRA_ARGS[@]}"

echo ""
echo "=========================================="
echo "2. Concurrent sweep (${TOTAL} reqs @ {${CONCURRENCY_LEVELS}})"
echo "=========================================="
# shellcheck disable=SC2086  # intentional word-splitting for --sweep args
python3 scripts/bench_concurrent.py \
  --url "${URL}" \
  --total "${TOTAL}" \
  --sweep ${CONCURRENCY_LEVELS} \
  "${EXTRA_ARGS[@]}"

echo ""
echo ">>> ray_smoke done. Logs printed above; compare against the"
echo ">>> single-host baseline to judge cross-node scaling."
