#!/usr/bin/env bash
# Start/stop/status the external mooncake_master that the transfer_queue +
# mooncake backend needs. The training config (e.g. conf/sd3_trainside_tq_mooncake.yaml)
# only *points* at the master — it does not launch it — so bring this up on the
# head node BEFORE launching training, and tear it down after.
#
#   bash examples/mooncake_master.sh start     # HTTP metadata + RPC server
#   bash examples/mooncake_master.sh status
#   bash examples/mooncake_master.sh stop
#
# Ports — read from the SAME env vars the training config interpolates into
# transfer_queue.metadata_server / .master_server_address, so master and client
# always agree:
#   TQ_MC_METADATA_SERVER_PORT   HTTP metadata server   (default 50041)
#   TQ_MC_RPC_PORT               RPC server             (default 50051)
#
# NOTE (taiji H20 pods): 50041 is frequently already bound on these images and
# the metadata server then fails to bind silently — export
# TQ_MC_METADATA_SERVER_PORT=8080 for both this script and the training run.
set -uo pipefail

META_PORT="${TQ_MC_METADATA_SERVER_PORT:-50041}"
RPC_PORT="${TQ_MC_RPC_PORT:-50051}"
LOG="${MOONCAKE_MASTER_LOG:-/tmp/mooncake_master.log}"

_port_in_use() {  # 0 = in use, 1 = free
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -qE "[^0-9]${1}\b"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -i ":${1}" >/dev/null 2>&1
  else
    return 1
  fi
}

start() {
  if ! command -v mooncake_master >/dev/null 2>&1; then
    echo "ERROR: mooncake_master not on PATH — bake it into the pod image." >&2
    exit 1
  fi
  if pgrep -x mooncake_master >/dev/null 2>&1; then
    echo "mooncake_master already running (pid $(pgrep -x mooncake_master | tr '\n' ' '))"
    return 0
  fi
  for p in "$META_PORT" "$RPC_PORT"; do
    if _port_in_use "$p"; then
      echo "ERROR: port ${p} already in use. Pick another (export TQ_MC_METADATA_SERVER_PORT / TQ_MC_RPC_PORT)." >&2
      exit 1
    fi
  done
  nohup mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port="$META_PORT" \
    --rpc_port="$RPC_PORT" \
    --rpc_thread_num=64 \
    --default_kv_lease_ttl=100000000 \
    > "$LOG" 2>&1 &
  local pid=$!
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: mooncake_master exited immediately — see $LOG" >&2
    tail -n 5 "$LOG" >&2 2>/dev/null || true
    exit 1
  fi
  echo "mooncake_master started: pid=${pid} metadata=http://\$HEAD_IP:${META_PORT}/metadata rpc=\$HEAD_IP:${RPC_PORT} log=${LOG}"
}

stop() {
  if pkill -x mooncake_master 2>/dev/null; then
    echo "mooncake_master stopped"
  else
    echo "no mooncake_master running"
  fi
}

status() {
  if pgrep -x mooncake_master >/dev/null 2>&1; then
    echo "mooncake_master: running (pid $(pgrep -x mooncake_master | tr '\n' ' '))"
    _port_in_use "$META_PORT" && echo "  metadata :${META_PORT} LISTEN" || echo "  metadata :${META_PORT} NOT listening"
    _port_in_use "$RPC_PORT"  && echo "  rpc      :${RPC_PORT} LISTEN"  || echo "  rpc      :${RPC_PORT} NOT listening"
  else
    echo "mooncake_master: not running"
  fi
}

case "${1:-start}" in
  start)  start ;;
  stop)   stop ;;
  status) status ;;
  restart) stop; sleep 1; start ;;
  *) echo "usage: $0 {start|stop|status|restart}" >&2; exit 2 ;;
esac
