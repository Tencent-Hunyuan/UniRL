#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# stop_videoalign.sh — 停止由 start_videoalign.sh 启动的服务
#
# 用法：
#   bash scripts/stop_videoalign.sh          # 读 PID 文件停止
#   bash scripts/stop_videoalign.sh --force  # 强制 kill
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

PID_FILE="/tmp/videoalign_service.pid"
FORCE=false

[[ "${1:-}" == "--force" ]] && FORCE=true

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[INFO] 停止服务进程 PID=$PID ..."
        if [[ "$FORCE" == "true" ]]; then
            kill -9 "$PID" 2>/dev/null || true
        else
            kill "$PID" 2>/dev/null || true
            # 等待最多 10 秒
            for i in $(seq 1 10); do
                if ! ps -p "$PID" > /dev/null 2>&1; then break; fi
                sleep 1
            done
            # 仍未退出则强杀
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "[WARN] 进程未响应 SIGTERM，强制 kill..."
                kill -9 "$PID" 2>/dev/null || true
            fi
        fi
        echo "[OK] 服务已停止。"
    else
        echo "[INFO] PID=$PID 的进程已不存在。"
    fi
    rm -f "$PID_FILE"
else
    echo "[INFO] 未找到 PID 文件 ($PID_FILE)，尝试查找进程..."
    PIDS=$(pgrep -f "python.*reward_service.*service.yaml\|python.*reward_service.*videoalign" 2>/dev/null || true)
    if [[ -n "$PIDS" ]]; then
        echo "[INFO] 找到进程: $PIDS"
        echo "$PIDS" | xargs kill 2>/dev/null || true
        sleep 2
        echo "[OK] 已发送 SIGTERM。"
    else
        echo "[INFO] 未找到运行中的 reward_service 进程。"
    fi
fi

# 停止 Ray（可选）
if ray status >/dev/null 2>&1; then
    echo "[INFO] 停止 Ray..."
    ray stop --force 2>/dev/null || true
fi

echo "[DONE]"
