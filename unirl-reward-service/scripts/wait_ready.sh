#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# wait_ready.sh — 等待 VideoAlign 服务就绪
#
# 用法：
#   bash scripts/wait_ready.sh                       # 默认 localhost:8080，最多 45 分钟
#   bash scripts/wait_ready.sh --port 8090           # 指定端口
#   bash scripts/wait_ready.sh --timeout 1800        # 自定义超时（秒）
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

PORT=8080
TIMEOUT=2700  # 45 分钟（首次启动含 flash-attn 编译 + 模型下载）
INTERVAL=15

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)    PORT="$2";    shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        *) echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

URL="http://localhost:${PORT}/health"
echo "[INFO] 等待服务就绪: $URL (超时 ${TIMEOUT}s)"

START=$(date +%s)
while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "[ERROR] 超时 (${TIMEOUT}s)。查看日志: tail -50 /tmp/videoalign_service.log" >&2
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$URL" 2>/dev/null || echo "000")

    if [[ "$HTTP_CODE" == "200" ]]; then
        BODY=$(curl -s "$URL")
        echo ""
        echo "[OK] 服务就绪！ (${ELAPSED}s)"
        echo "$BODY" | python -m json.tool 2>/dev/null || echo "$BODY"
        exit 0
    fi

    # 检查进程是否还活着
    PID_FILE="/tmp/videoalign_service.pid"
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo ""
            echo "[ERROR] 服务进程 (PID=$PID) 已退出！" >&2
            echo "最后几行日志:" >&2
            tail -30 /tmp/videoalign_service.log >&2
            exit 1
        fi
    fi

    printf "\r  [%3ds / %ds] 等待中 (HTTP %s) ..." "$ELAPSED" "$TIMEOUT" "$HTTP_CODE"
    sleep "$INTERVAL"
done
