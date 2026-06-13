#!/usr/bin/env bash
#
# One command to verify the tensor transport layer. Autodetects GPUs + Mooncake
# and runs the maximal subset this host supports — the conftest collection hook
# (tests/transport/conftest.py) auto-skips tiers the host can't run.
#
#   bash scripts/run_transport_tests.sh              # every available tier
#   RUN_SLOW=1 bash scripts/run_transport_tests.sh   # + the 300-iter leak loop
#   uv run pytest tests/transport -m cpu             # just the always-on CPU layer
#
# Tiers: cpu (laptop) -> gpu (1 GPU) -> multigpu (>=2 GPU) -> mooncake (master).
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve an interpreter that can import the project (unirl + ray + torch).
PY="${UNIRL_PY:-}"
if [[ -z "$PY" ]]; then
  if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import unirl, ray, torch' 2>/dev/null; then
    PY=".venv/bin/python"
  else
    PY="uv run python"
  fi
fi
if ! $PY -c 'import unirl, ray, torch' 2>/dev/null; then
  echo "ERROR: cannot import unirl/ray/torch with '$PY'. Activate the project venv or run 'uv sync'." >&2
  exit 1
fi

GPUS=$($PY -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')
echo "transport tests: ${GPUS} GPU(s); Mooncake=${UNIRL_MOONCAKE_MASTER:-<none>}; RUN_SLOW=${RUN_SLOW:-0}"

# Select every tier; the collection hook skips what this host can't support and
# gates `slow` behind RUN_SLOW=1. (Do NOT add `slow` to -m here — that would
# trip the hook's run_slow detection and unconditionally enable the leak loop.)
exec $PY -m pytest tests/transport tests/test_tensorref_spans.py \
  -m "cpu or gpu or multigpu or mooncake" \
  --cov=unirl.distributed.tensor --cov=unirl.distributed.group --cov-report=term-missing \
  "$@"
