#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# start_videoalign.sh — 一键启动仅含 VideoAlign scorer 的 RewardService
#
# 用法：
#   bash scripts/start_videoalign.sh                  # 默认 GPU 4-7, port 8080
#   bash scripts/start_videoalign.sh --gpus 4,5       # 指定 GPU
#   bash scripts/start_videoalign.sh --port 8090      # 指定端口
#   bash scripts/start_videoalign.sh --replicas 2     # 2 个 actor 副本
#   bash scripts/start_videoalign.sh --weights /path  # 指定权重路径
#   bash scripts/start_videoalign.sh --fg              # 前台运行（不 nohup）
#
# 前置条件：
#   - 已激活包含 ray/fastapi/uvicorn/torch 的 Python 环境
#   - python -m pip 可用（若首次启动报 "No module named pip"，
#     先执行 python -m ensurepip --upgrade）
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 默认值 ──────────────────────────────────────────────────────
GPUS="4,5,6,7"
PORT=8080
REPLICAS=1
NUM_GPUS_PER_REPLICA=1
NUM_CPUS=4
WEIGHTS_PATH="/path/to/VideoAlign/checkpoints/VideoReward"
DTYPE="bfloat16"
USE_NORM="true"
DISABLE_FLASH_ATTN2="false"
FOREGROUND=false
TIMEOUT=300
LOG_FILE="/tmp/videoalign_service.log"

# ── 参数解析 ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)         GPUS="$2";         shift 2 ;;
        --port)         PORT="$2";         shift 2 ;;
        --replicas)     REPLICAS="$2";     shift 2 ;;
        --weights)      WEIGHTS_PATH="$2"; shift 2 ;;
        --dtype)        DTYPE="$2";        shift 2 ;;
        --no-norm)      USE_NORM="false";  shift   ;;
        --sdpa)         DISABLE_FLASH_ATTN2="true"; shift ;;
        --fg)           FOREGROUND=true;   shift   ;;
        --timeout)      TIMEOUT="$2";      shift 2 ;;
        --log)          LOG_FILE="$2";     shift 2 ;;
        -h|--help)
            head -20 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "未知参数: $1" >&2; exit 1 ;;
    esac
done

# ── 前置检查 ──────────────────────────────────────────────────────
command -v python >/dev/null || { echo "python 不在 PATH 中" >&2; exit 1; }

# 确保 pip 可用（解决 venv 中缺 pip 的问题）
if ! python -m pip --version >/dev/null 2>&1; then
    echo "[INFO] 检测到 python -m pip 不可用，执行 ensurepip..."
    python -m ensurepip --upgrade
fi

# 权重路径检查
if [[ ! -f "$WEIGHTS_PATH/model_config.json" ]]; then
    echo "[ERROR] 找不到 $WEIGHTS_PATH/model_config.json" >&2
    echo "  请用 --weights 指定包含 model_config.json 的 VideoReward checkpoint 目录" >&2
    exit 1
fi

# 端口占用检查
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    echo "[ERROR] 端口 $PORT 已被占用" >&2
    ss -tlnp | grep ":${PORT} "
    exit 1
fi

echo "======================================"
echo " VideoAlign RewardService 启动配置"
echo "======================================"
echo "  CUDA_VISIBLE_DEVICES : $GPUS"
echo "  Port                 : $PORT"
echo "  Replicas             : $REPLICAS"
echo "  GPU per replica      : $NUM_GPUS_PER_REPLICA"
echo "  Weights              : $WEIGHTS_PATH"
echo "  dtype                : $DTYPE"
echo "  use_norm             : $USE_NORM"
echo "  flash_attn2          : $([ "$DISABLE_FLASH_ATTN2" = "true" ] && echo "off (sdpa)" || echo "on")"
echo "  score_timeout_s      : $TIMEOUT"
echo "  Log file             : $LOG_FILE"
echo "======================================"

# ── 生成临时 YAML 配置 ───────────────────────────────────────────
CONFIG_FILE=$(mktemp /tmp/videoalign_service_XXXXXX.yaml)
cat > "$CONFIG_FILE" <<YAML
server:
  host: 0.0.0.0
  port: ${PORT}
  score_timeout_s: ${TIMEOUT}.0

rewards:
  - name: videoalign
    scorer: videoalign
    runtime_env: envs/videoalign.txt
    num_replicas: ${REPLICAS}
    num_gpus: ${NUM_GPUS_PER_REPLICA}
    num_cpus: ${NUM_CPUS}
    max_concurrency: 1
    params:
      weights_path: "${WEIGHTS_PATH}"
      checkpoint_step: -1
      dtype: "${DTYPE}"
      use_norm: ${USE_NORM}
      disable_flash_attn2: ${DISABLE_FLASH_ATTN2}
      fps: null
      num_frames: null
      max_pixels: null
YAML

echo "[INFO] 生成配置: $CONFIG_FILE"

# ── 启动 ─────────────────────────────────────────────────────────
cd "$REPO_ROOT"

if [[ "$FOREGROUND" == "true" ]]; then
    echo "[INFO] 前台启动 (Ctrl+C 停止)..."
    CUDA_VISIBLE_DEVICES="$GPUS" exec python -m reward_service --config "$CONFIG_FILE"
else
    echo "[INFO] 后台启动，日志写入 $LOG_FILE ..."
    CUDA_VISIBLE_DEVICES="$GPUS" nohup python -m reward_service --config "$CONFIG_FILE" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > /tmp/videoalign_service.pid
    echo "[INFO] PID=$PID  (写入 /tmp/videoalign_service.pid)"
    echo ""
    echo "查看日志:    tail -f $LOG_FILE"
    echo "停止服务:    bash scripts/stop_videoalign.sh"
    echo "健康检查:    curl http://localhost:${PORT}/health"
    echo "运行测试:    python3 scripts/test_videoalign.py --url http://localhost:${PORT}"
    echo ""
    echo "[INFO] 首次启动需等待 Ray venv 安装 + 模型下载（~10-30 分钟）。"
    echo "       执行 tail -f $LOG_FILE 观察进度。"

    # 等几秒确认进程没立即退出
    sleep 3
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "[OK] 服务进程运行中 (PID=$PID)。"
    else
        echo "[ERROR] 服务进程已退出，查看日志:" >&2
        tail -20 "$LOG_FILE" >&2
        exit 1
    fi
fi
