#!/usr/bin/env bash
# Run all Phase 0 grad-context probe configs sequentially, each in a fresh
# process (keeps Ray / NCCL / FSDP default-PG state clean between configs).
# Logs land in /root/phase0_logs/<name>.log; verdicts in summary.log.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
LOGDIR=/root/phase0_logs
mkdir -p "$LOGDIR"
: > "$LOGDIR/summary.log"

run() {  # name policy_gpus reward_gpus fsdp
  local name=$1 P=$2 Q=$3 F=$4
  echo "=== RUN $name: P=$P Q=$Q fsdp=$F ==="
  python scripts/phase0_gradctx_probe.py --policy-gpus "$P" --reward-gpus "$Q" --fsdp "$F" \
    > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  local verdict
  verdict=$(grep -hE "PASS:|FAIL:|ERROR:|SKIP:" "$LOGDIR/$name.log" | tail -1)
  echo "$name rc=$rc | $verdict" | tee -a "$LOGDIR/summary.log"
}

run A_p1q1       1 1 0   # mechanism sanity (no DP, no FSDP)
run B_p2q2       2 2 0   # symmetric DP
run C_p4q1       4 1 0   # asymmetric DP (the sharp seam)
run D_p2q2_fsdp  2 2 1   # FSDP symmetric
run E_p4q1_fsdp  4 1 1   # FSDP asymmetric (full ReFL scenario)

echo "ALLDONE" | tee -a "$LOGDIR/summary.log"
