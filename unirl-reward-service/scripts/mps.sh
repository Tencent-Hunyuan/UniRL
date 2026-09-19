#!/usr/bin/env bash
# Manage one job-scoped MPS daemon on every node in NODE_IP_LIST.

set -euo pipefail
source "$(dirname "$0")/_ray_lib.sh"

ACTION="${1:-}"
case "$ACTION" in start|status|stop) ;; *) echo "usage: $0 {start|status|stop}" >&2; exit 2 ;; esac
command -v pdsh >/dev/null || { echo "need pdsh on PATH" >&2; exit 1; }

resolve_cluster_nodes
NODE_LIST=${NODES// /,}
PIPE="${MPS_PIPE_DIRECTORY:-/tmp/unirl-mps-$USER}"
LOG="${MPS_LOG_DIRECTORY:-/tmp/unirl-mps-log-$USER}"
case "$PIPE:$LOG" in
  /tmp/unirl-mps-*:/tmp/unirl-mps-log-*) ;;
  *) echo "MPS directories must use /tmp/unirl-mps-* paths" >&2; exit 2 ;;
esac

pdsh -R ssh -w "$NODE_LIST" "
  set -euo pipefail
  export CUDA_MPS_PIPE_DIRECTORY='$PIPE'
  export CUDA_MPS_LOG_DIRECTORY='$LOG'
  case '$ACTION' in
    start)
      command -v nvidia-cuda-mps-control >/dev/null
      install -d -m 700 '$PIPE' '$LOG'
      if ! printf 'get_server_list\n' | nvidia-cuda-mps-control >/dev/null 2>&1; then
        nvidia-cuda-mps-control -d
      fi
      ;;
    status)
      printf 'get_server_list\n' | nvidia-cuda-mps-control
      ;;
    stop)
      if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        clients=''
        for _ in \$(seq 1 30); do
          servers=\$(printf 'get_server_list\n' | nvidia-cuda-mps-control 2>/dev/null || true)
          clients=''
          for pid in \$servers; do
            clients=\"\$clients \$(printf 'get_client_list %s\n' \"\$pid\" |
              nvidia-cuda-mps-control 2>/dev/null || true)\"
          done
          [[ -z \"\${clients//[[:space:]]/}\" ]] && break
          sleep 1
        done
        [[ -z \"\${clients//[[:space:]]/}\" ]] ||
          { echo \"live MPS clients:\$clients\" >&2; exit 1; }
        printf 'quit\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || true
      fi
      rm -rf -- '$PIPE' '$LOG'
      ;;
  esac
"
