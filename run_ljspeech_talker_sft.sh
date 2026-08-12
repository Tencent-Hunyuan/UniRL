#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-smoke}"

ARCHIVE="${LJSPEECH_ARCHIVE:-${ROOT}/LJSpeech-1.1.tar.bz2}"
DATA_ROOT="${LJSPEECH_TALKER_DATA_ROOT:-${ROOT}/datasets/ljspeech_talker}"
SOURCE_ROOT="${DATA_ROOT}/source"
LJSPEECH_DIR="${SOURCE_ROOT}/LJSpeech-1.1"
MANIFEST_DIR="${DATA_ROOT}/manifests"
CODE_DIR="${DATA_ROOT}/mimi_codes"
LOG_DIR="${DATA_ROOT}/logs"
MODEL_DIR="${ROOT}/models"

QWEN3_OMNI_PATH="${QWEN3_OMNI_PATH:-/apdcephfs_cq8/share_1611098/bruceszchen/HF_Models/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695}"
MIMI_PATH="${MIMI_PATH:-${MODEL_DIR}/kyutai-mimi}"
GPU_CLEANUP_CMD=(bash /apdcephfs_hldy/share_303576955/bruceszchen/tmp/ft_local/occupy_gpu_ray/run_occupy.sh stop)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4
export HF_HOME="${HF_HOME:-/apdcephfs_cq8/share_1611098/bruceszchen/HF_Models}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${SOURCE_ROOT}" "${MANIFEST_DIR}" "${CODE_DIR}" "${LOG_DIR}" "${MODEL_DIR}"

usage() {
  cat <<'EOF'
Usage: bash run_ljspeech_talker_sft.sh <mode>

Modes:
  prepare-smoke  Extract LJSpeech, build deterministic splits, encode 32+8 Mimi samples.
  smoke          Prepare if needed, then run a 20-step 4-GPU overfit/integration smoke.
  smoke-mtp      Run the 20-step smoke with only MTP LoRA trainable.
  prepare-full   Encode the full 12,844 train + 256 validation rows.
  train          Run the full LoRA SFT (default 3,200 steps) from prepared full data.
  train-mtp      Recommended voice adaptation: freeze layer-0, train MTP LoRA only.
  all            Run smoke, prepare-full, then full training.

Environment overrides:
  QWEN3_OMNI_PATH, MIMI_PATH, LJSPEECH_ARCHIVE, LJSPEECH_TALKER_DATA_ROOT
  TALKER_LR, TALKER_MTP_LR, TALKER_SFT_LAMBDA, FULL_STEPS, FULL_BATCH_SIZE
EOF
}

ensure_gpu_0_3_free() {
  local busy=0
  while IFS=',' read -r index used; do
    index="$(echo "${index}" | tr -d ' ')"
    used="$(echo "${used}" | tr -cd '0-9')"
    if [[ "${index}" =~ ^[0-3]$ ]] && (( used > 512 )); then
      busy=1
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

  if (( busy )); then
    echo "GPU 0-3 are occupied; running the requested cleanup command."
    "${GPU_CLEANUP_CMD[@]}"
    sleep 5
  fi

  while IFS=',' read -r index used; do
    index="$(echo "${index}" | tr -d ' ')"
    used="$(echo "${used}" | tr -cd '0-9')"
    if [[ "${index}" =~ ^[0-3]$ ]] && (( used > 512 )); then
      echo "GPU ${index} still uses ${used} MiB after cleanup; refusing to use another GPU." >&2
      exit 1
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
}

extract_and_split() {
  if [[ ! -f "${ARCHIVE}" ]]; then
    echo "Missing LJSpeech archive: ${ARCHIVE}" >&2
    exit 1
  fi
  if [[ ! -f "${LJSPEECH_DIR}/metadata.csv" ]]; then
    echo "Extracting ${ARCHIVE} ..."
    tar -xjf "${ARCHIVE}" -C "${SOURCE_ROOT}"
  fi
  python "${ROOT}/scripts/prepare_ljspeech_talker_manifests.py" \
    --ljspeech_dir "${LJSPEECH_DIR}" \
    --output_dir "${MANIFEST_DIR}" \
    --speaker Ethan \
    --language en \
    --val_size 256 \
    --smoke_train_size 32 \
    --smoke_val_size 8
}

ensure_mimi() {
  if [[ -f "${MIMI_PATH}/config.json" ]] && compgen -G "${MIMI_PATH}/*.safetensors" >/dev/null; then
    return
  fi
  echo "Downloading kyutai/mimi to ${MIMI_PATH} ..."
  python - "${MIMI_PATH}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="kyutai/mimi",
    local_dir=sys.argv[1],
    local_dir_use_symlinks=False,
)
PY
}

encode_split() {
  local input_jsonl="$1"
  local output_jsonl="$2"
  local log_file="$3"
  if [[ -s "${output_jsonl}" ]]; then
    echo "Reusing encoded manifest ${output_jsonl}"
    return
  fi
  ensure_gpu_0_3_free
  python -m unirl.utils.prepare_talker_tts_data \
    --input_jsonl "${input_jsonl}" \
    --output_jsonl "${output_jsonl}" \
    --talker_model "${QWEN3_OMNI_PATH}" \
    --mimi_model "${MIMI_PATH}" \
    --device cuda:0 2>&1 | tee "${log_file}"
}

write_fingerprint_env() {
  local encoded_jsonl="$1"
  python - "${encoded_jsonl}" "${CODE_DIR}/fingerprints.env" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.loads(next(line for line in handle if line.strip()))
metadata = row["metadata"]
model_sha = metadata["talker_model_fingerprint"]["sha256"]
codec_sha = metadata["codec_fingerprint"]["sha256"]
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    handle.write(f"TALKER_MODEL_FINGERPRINT={shlex.quote(model_sha)}\n")
    handle.write(f"TALKER_CODEC_FINGERPRINT={shlex.quote(codec_sha)}\n")
PY
}

prepare_smoke() {
  extract_and_split
  ensure_mimi
  encode_split \
    "${MANIFEST_DIR}/raw_smoke_train.jsonl" \
    "${CODE_DIR}/smoke_train.jsonl" \
    "${LOG_DIR}/encode_smoke_train.log"
  encode_split \
    "${MANIFEST_DIR}/raw_smoke_val.jsonl" \
    "${CODE_DIR}/smoke_val.jsonl" \
    "${LOG_DIR}/encode_smoke_val.log"
  write_fingerprint_env "${CODE_DIR}/smoke_train.jsonl"
}

prepare_full() {
  extract_and_split
  ensure_mimi
  if [[ ! -s "${CODE_DIR}/train.jsonl" ]]; then
    local shard_dir="${MANIFEST_DIR}/full_train_shards"
    mkdir -p "${shard_dir}"
    python - "${MANIFEST_DIR}/raw_train.jsonl" "${shard_dir}" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
chunk = (len(rows) + 3) // 4
for index in range(4):
    part = rows[index * chunk : (index + 1) * chunk]
    (output / f"raw_train_{index}.jsonl").write_text(
        "".join(f"{line}\n" for line in part),
        encoding="utf-8",
    )
PY
    ensure_gpu_0_3_free
    local pids=()
    for gpu in 0 1 2 3; do
      local shard_output="${CODE_DIR}/train_${gpu}.jsonl"
      if [[ -s "${shard_output}" ]]; then
        echo "Reusing encoded train shard ${shard_output}"
        continue
      fi
      python -m unirl.utils.prepare_talker_tts_data \
        --input_jsonl "${shard_dir}/raw_train_${gpu}.jsonl" \
        --output_jsonl "${shard_output}" \
        --talker_model "${QWEN3_OMNI_PATH}" \
        --mimi_model "${MIMI_PATH}" \
        --device "cuda:${gpu}" \
        --log_every 100 >"${LOG_DIR}/encode_train_${gpu}.log" 2>&1 &
      pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
      wait "${pid}" || failed=1
    done
    if (( failed )); then
      echo "At least one full-data Mimi encoder failed; inspect ${LOG_DIR}/encode_train_*.log" >&2
      exit 1
    fi
    python - "${CODE_DIR}" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = root / "train.jsonl"
tmp = root / f".train.jsonl.tmp.{os.getpid()}"
with tmp.open("w", encoding="utf-8") as output:
    for index in range(4):
        with (root / f"train_{index}.jsonl").open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    output.write(line)
tmp.replace(target)
PY
  fi
  encode_split \
    "${MANIFEST_DIR}/raw_val.jsonl" \
    "${CODE_DIR}/val.jsonl" \
    "${LOG_DIR}/encode_val.log"
  write_fingerprint_env "${CODE_DIR}/train.jsonl"
}

run_training() {
  local train_jsonl="$1"
  local val_jsonl="$2"
  local steps="$3"
  local batch_size="$4"
  local eval_interval="$5"
  local save_interval="$6"
  local save_dir="$7"
  local run_name="$8"
  local scope="${9:-talker_mtp}"
  local warmup_steps=25
  local scope_overrides=()
  if [[ "${scope}" == "mtp_only" ]]; then
    scope_overrides+=("backend.lora_cfg.target_modules=\${mtp_lora_targets}")
  elif [[ "${scope}" != "talker_mtp" ]]; then
    echo "Unknown SFT scope: ${scope}" >&2
    exit 1
  fi
  if (( steps <= warmup_steps )); then
    warmup_steps=$(( steps / 10 ))
    (( warmup_steps > 0 )) || warmup_steps=1
  fi

  if [[ ! -s "${train_jsonl}" || ! -s "${val_jsonl}" ]]; then
    echo "Missing encoded train/validation manifests." >&2
    exit 1
  fi
  if [[ ! -f "${CODE_DIR}/fingerprints.env" ]]; then
    write_fingerprint_env "${train_jsonl}"
  fi
  # shellcheck disable=SC1091
  source "${CODE_DIR}/fingerprints.env"
  export TALKER_MODEL_FINGERPRINT TALKER_CODEC_FINGERPRINT
  export QWEN3_OMNI_PATH
  export SFT_DATA="${train_jsonl}"
  export SFT_EVAL_DATA="${val_jsonl}"
  export TALKER_LR="${TALKER_LR:-1.0e-5}"
  export TALKER_MTP_LR="${TALKER_MTP_LR:-2.0e-5}"
  export TALKER_SFT_LAMBDA="${TALKER_SFT_LAMBDA:-2.0}"

  ensure_gpu_0_3_free
  mkdir -p "${save_dir}"
  INSTALL_EDITABLE=0 \
  ENTRY=train_sft \
  GPUS_PER_NODE=4 \
  bash "${ROOT}/examples/run_experiment_single_node.sh" \
    sft/qwen3_omni_talker_sft \
    "+devices_per_node=4" \
    "num_steps=${steps}" \
    "batch_size=${batch_size}" \
    "eval_interval=${eval_interval}" \
    "eval_batch_size=4" \
    "eval_num_samples=-1" \
    "save_interval=${save_interval}" \
    "+save_dir=${save_dir}" \
    "logging.run_name=${run_name}" \
    "logging.report_to_wandb=false" \
    "+bundle.config.meta_init_transformer=true" \
    "bundle.config.max_prompt_length=512" \
    "pipeline.max_prompt_length=512" \
    "track_builder.max_prompt_length=512" \
    "track_builder.max_response_length=256" \
    "backend.block_class_names=[Qwen3OmniMoeTalkerDecoderLayer,Qwen3OmniMoeTalkerCodePredictorDecoderLayer]" \
    "+backend.fsdp_cfg.root_wrap=false" \
    "backend.scheduler_cfg.warmup_steps=${warmup_steps}" \
    "${scope_overrides[@]}" 2>&1 | tee "${LOG_DIR}/${run_name}.log"
}

run_smoke() {
  prepare_smoke
  run_training \
    "${CODE_DIR}/smoke_train.jsonl" \
    "${CODE_DIR}/smoke_val.jsonl" \
    "${SMOKE_STEPS:-20}" \
    4 \
    5 \
    10 \
    "${ROOT}/outputs/qwen3_omni_talker_ljspeech_smoke" \
    qwen3_omni_talker_ljspeech_smoke
}

run_mtp_smoke() {
  prepare_smoke
  run_training \
    "${CODE_DIR}/smoke_train.jsonl" \
    "${CODE_DIR}/smoke_val.jsonl" \
    "${SMOKE_STEPS:-20}" \
    4 \
    5 \
    10 \
    "${ROOT}/outputs/qwen3_omni_talker_ljspeech_mtp_only_smoke" \
    qwen3_omni_talker_ljspeech_mtp_only_smoke \
    mtp_only
}

run_full() {
  if [[ ! -s "${CODE_DIR}/train.jsonl" || ! -s "${CODE_DIR}/val.jsonl" ]]; then
    echo "Run prepare-full before train." >&2
    exit 1
  fi
  run_training \
    "${CODE_DIR}/train.jsonl" \
    "${CODE_DIR}/val.jsonl" \
    "${FULL_STEPS:-3200}" \
    "${FULL_BATCH_SIZE:-8}" \
    "${FULL_EVAL_INTERVAL:-100}" \
    "${FULL_SAVE_INTERVAL:-400}" \
    "${ROOT}/outputs/qwen3_omni_talker_ljspeech" \
    qwen3_omni_talker_ljspeech \
    talker_mtp
}

run_mtp_only() {
  if [[ ! -s "${CODE_DIR}/train.jsonl" || ! -s "${CODE_DIR}/val.jsonl" ]]; then
    echo "Run prepare-full before train-mtp." >&2
    exit 1
  fi
  run_training \
    "${CODE_DIR}/train.jsonl" \
    "${CODE_DIR}/val.jsonl" \
    "${MTP_ONLY_STEPS:-800}" \
    "${FULL_BATCH_SIZE:-8}" \
    "${MTP_ONLY_EVAL_INTERVAL:-100}" \
    "${MTP_ONLY_SAVE_INTERVAL:-200}" \
    "${ROOT}/outputs/qwen3_omni_talker_ljspeech_mtp_only" \
    qwen3_omni_talker_ljspeech_mtp_only \
    mtp_only
}

case "${MODE}" in
  prepare-smoke) prepare_smoke ;;
  smoke) run_smoke ;;
  smoke-mtp) run_mtp_smoke ;;
  prepare-full) prepare_full ;;
  train) run_full ;;
  train-mtp) run_mtp_only ;;
  all)
    run_smoke
    prepare_full
    run_full
    ;;
  -h|--help|help) usage ;;
  *)
    usage >&2
    exit 2
    ;;
esac
