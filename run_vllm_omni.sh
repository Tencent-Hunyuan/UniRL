#!/usr/bin/env bash
# run_vllm_omni.sh — Launch the Qwen3-Omni Thinker GSPO+LoRA smoke on the UniRL
# vLLM-Omni rollout engine.
#
# Mirrors ``run.sh`` (the trainside-rollout twin) plus the environment bits
# borrowed from ``/root/bruceszchen_cq/MyProjects/verl-omni/run_verl_omni.sh``
# (conda activation, HF offline, CUDA-13 forward-compat via LD_LIBRARY_PATH).
#
# Recipe: examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_a2_1x8.yaml (A2 topology:
# anchor rollout to a single Worker actor, TP=8 across all 8 GPUs, sleep/wake time-share
# with training FSDP DP=8; RemoteLoraWeightSync(copy=True) push from rank 0).
# Engine: unirl.rollout.engine.vllm_omni.VLLMOmniRolloutEngine
#         modality = qwen3_omni_thinker (single AR stage, TP=8, LoRA)
# Sync:   unirl.distributed.weight_sync.lora.LocalLoraWeightSync (copy=True)
#
# The recipe defaults to running the Video-R1 clevrer/perceptiontest jsonl at
# ${DATA_PATH} / ${EVAL_DATA_PATH}. This script overrides them to the same
# paths ``run.sh`` uses so results are directly comparable.

set -uo pipefail

# WANDB — offline by default (matches run.sh); flip via env if desired.
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-./outputs/wandb_runs}

timestamp=$(date +%Y%m%d%H%M%S)

# -----------------------------------------------------------------------------
# Paths (override via env). These mirror run.sh's normal_8card() defaults.
# -----------------------------------------------------------------------------
export QWEN3_OMNI_PATH=${QWEN3_OMNI_PATH:-/dev/shm/Qwen3-Omni-30B-A3B-Instruct}
export DATA_PATH=${DATA_PATH:-/apdcephfs_cq8/share_1611098/bruceszchen/MyProjects/UniRL/datasets/dapo_math_17k/train.jsonl}
#export DATA_PATH=${DATA_PATH:-/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/datasets/video_r1_clevrer_perceptiontest/train.jsonl}
export EVAL_DATA_PATH=${EVAL_DATA_PATH:-/apdcephfs_cq8/share_1611098/bruceszchen/MyProjects/UniRL/datasets/dapo_math_17k/val.jsonl}
#export EVAL_DATA_PATH=${EVAL_DATA_PATH:-/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/datasets/video_r1_clevrer_perceptiontest/val.jsonl}

# Conda env — verl-omni-py313 is the one where vllm-omni is installed and the
# CUDA-13 activate hook fires. run_verl_omni.sh:36-63 is the SSOT.
CONDA_ROOT=${CONDA_ROOT:-/opt/conda}
ENV_NAME=${ENV_NAME:-verl-omni}
#ENV_NAME=${ENV_NAME:-verl-omni-py313}

log() { printf '\033[1;36m[run-vllm-omni]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[run-vllm-omni ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------
[ -d "$QWEN3_OMNI_PATH" ]                                       || die "model dir missing: $QWEN3_OMNI_PATH"
[ -f "$QWEN3_OMNI_PATH/model-00015-of-00015.safetensors" ]       || die "model shards incomplete (need 15/15) at $QWEN3_OMNI_PATH"
[ -f "$DATA_PATH" ]                                              || die "train jsonl missing: $DATA_PATH"
[ -f "$EVAL_DATA_PATH" ]                                         || die "eval jsonl missing: $EVAL_DATA_PATH"
[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]                      || die "conda not found at $CONDA_ROOT"

# -----------------------------------------------------------------------------
# Activate env (fires the CUDA-13 forward-compat activate hook)
# -----------------------------------------------------------------------------
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate "$ENV_NAME"
#
#[ -n "${CONDA_PREFIX:-}" ] || die "CONDA_PREFIX unset after activate"
#[[ "$CONDA_PREFIX" == *"/$ENV_NAME" ]] || die "wrong env: $CONDA_PREFIX (expected /$ENV_NAME)"

log "env:            $CONDA_PREFIX"
log "python:         $(python -V 2>&1)"
log "model:          $QWEN3_OMNI_PATH"
log "train / val:    $DATA_PATH / $EVAL_DATA_PATH"

# HF offline — the model is on disk; no network trips.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

# CUDA sanity — same probe as run_verl_omni.sh:74-78.
python -c "
import torch
torch.zeros(1).cuda()
print('[run] torch.cuda OK — dev0:', torch.cuda.get_device_name(0), '| count:', torch.cuda.device_count())
" || die "torch.cuda smoke failed — driver/compat problem"

cd "$(dirname "${BASH_SOURCE[0]}")"

# -----------------------------------------------------------------------------
# Modes
# -----------------------------------------------------------------------------

smoke() {
  # 3-rollout smoke: no wandb, no eval. Matches run_verl_omni.sh's 1-step
  # ``smoke`` in intent — enough to see boot → generate → LoRA push →
  # sleep/wake → rollout again → GSPO backward.
  #
  # Pre-start a Ray head with ``--num-cpus=${RAY_NUM_CPUS}`` (default 32) so
  # the trainer's ``ray.get(pg.ready())`` reuses this capped context.
  # Rationale: UniRL relies on auto_init, which prestarts one Python worker
  # per detected CPU. On this host (384 CPUs) that OOMs raylet's 30 s
  # registration deadline → connect deadlock (SSOT: run_verl_omni.sh:113-125).
  #local RAY_NUM_CPUS_VAL=${RAY_NUM_CPUS:-32}
  #log "starting Ray head with --num-cpus=${RAY_NUM_CPUS_VAL}"
  #ray stop --force >/dev/null 2>&1 || true
  #ray start --head --num-cpus=${RAY_NUM_CPUS_VAL} --disable-usage-stats \
  #  --temp-dir=/tmp/ray >/dev/null || die "ray start failed"

  # Make the trainer reuse the running head.
  #export RAY_ADDRESS=auto

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
  ROLLOUT_PRINT_N=${ROLLOUT_PRINT_N:-8} \
  ROLLOUT_DUMP_DIR=${ROLLOUT_DUMP_DIR:-./outputs/rollouts_outputs} \
  PYTHONPATH=$(pwd) \
  python -m unirl.train_ar --config-name=ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_a2_1x8 \
    num_rollouts=3 eval_interval=0 eval_before_train=false \
    logging.report_to_wandb=false \
    |& tee outputs/qwen3_omni_vllm_omni_smoke_${timestamp}.log

  local rc=${PIPESTATUS[0]}
  #log "cleaning up Ray head (rc=${rc})"
  #ray stop --force >/dev/null 2>&1 || true
  return $rc
}

full_1x8() {
  # Full training run — 500 steps, periodic eval, wandb enabled.
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
  ROLLOUT_PRINT_N=${ROLLOUT_PRINT_N:-8} \
  PYTHONPATH=$(pwd) \
  python -m unirl.train_ar --config-name=ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_a2_1x8 \
    num_rollouts=500 \
    logging.report_to_wandb=true \
     |& tee outputs/qwen3_omni_vllm_omni_1x8_${timestamp}.log
}

aqua_rat_1x4() {
  # 4-GPU AQuA-RAT run with the LLM-judge reward on 127.0.0.1:8100.
  #
  # Not covered by the top-level pre-flight: that block validates the DEFAULT
  # DATA_PATH / EVAL_DATA_PATH (dapo_math_17k) because it runs before this
  # function overrides them. We re-check the aqua paths here so a missing
  # dataset fails loud instead of blowing up mid-launch.
  local aqua_data=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/datasets/aqua_rat/unirl/train.jsonl
  # NB: point val at the pre-truncated first-100 file, NOT the full 2000-row val.jsonl.
  # UniRL's ARTrainer.evaluate() iterates the ENTIRE eval dataset in batches of
  # `eval_num_prompts`; that field is a batch-size, not a total-cap. So on the
  # full file each val phase would burn 20 batches × ~1min = ~20min. We want
  # "first 100" per your ask, so we serve a 100-row file directly.
  # Regenerate with: `head -100 .../val.jsonl > .../val_first100.jsonl`.
  local aqua_eval=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/datasets/aqua_rat/unirl/val_first100.jsonl
  [ -f "$aqua_data" ] || die "aqua train jsonl missing: $aqua_data"
  [ -f "$aqua_eval" ] || die "aqua val jsonl missing: $aqua_eval"

  # Judge probe: warn (do NOT abort) if :8100 is not reachable. The reward
  # wrapper falls back to math_dapo silently on failure, which would waste
  # the whole run generating rollouts scored by the noisier legacy scorer.
  if ! python -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',8100))==0 else 1)" 2>/dev/null; then
    log "WARNING: judge endpoint http://127.0.0.1:8100 not reachable — reward will silently fall back to math_dapo (~44% false negatives). Start it with scripts/AQuA-RAT-test/serve_judge_qwen35.sh before letting this run finish."
  fi

  local OUTPUT_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/train_outputs/UniRL
  mkdir -p "${OUTPUT_DIR}"

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
  QWEN3_OMNI_PATH=/dev/shm/Qwen3-Omni-30B-A3B-Instruct \
  DATA_PATH="${aqua_data}" \
  EVAL_DATA_PATH="${aqua_eval}" \
  AQUA_JUDGE_URL=${AQUA_JUDGE_URL:-http://127.0.0.1:8100/v1/chat/completions} \
  AQUA_JUDGE_MODEL=${AQUA_JUDGE_MODEL:-aqua-judge} \
  ROLLOUT_PRINT_N=${ROLLOUT_PRINT_N:-8} \
  ROLLOUT_DUMP_DIR=${ROLLOUT_DUMP_DIR:-./outputs/rollouts_outputs_aqua} \
  PYTHONPATH=$(pwd) python -m unirl.train_ar \
    --config-name=ar/run_qwen3_omni_vllm_omni_lora_a2_aqua_rat_1x4 \
    |& tee "${OUTPUT_DIR}/qwen3_omni_vllm_omni_1x4_aqua_rat_${timestamp}.log"
}

video_r1_1x4() {
  # 4-GPU video_r1_clevrer_perceptiontest MCQA run on the vLLM-Omni engine (A2).
  # Same topology as aqua_rat_1x4; the recipe swaps in MCExactMatch (strict) and
  # the real video pipeline (max_prompt_length=12288, fps=1, max_pixels=262144).
  #
  # Recipe: examples/ar/run_qwen3_omni_vllm_omni_lora_a2_video_r1_1x4.yaml.
  # Local pre-flight (same reason as aqua_rat_1x4): the top-level checks validate
  # the DEFAULT dapo paths, so re-check the video paths here so a missing dataset
  # fails loud instead of blowing up mid-launch.
  local video_data=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/datasets/video_r1_clevrer_perceptiontest/for_unirl/train.jsonl
  local video_eval=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/datasets/video_r1_clevrer_perceptiontest/for_unirl/val.jsonl
  [ -f "$video_data" ] || die "video train jsonl missing: $video_data"
  [ -f "$video_eval" ] || die "video val jsonl missing: $video_eval"

  local OUTPUT_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/train_outputs/UniRL
  mkdir -p "${OUTPUT_DIR}"

  # WANDB_DIR is a safety net — the recipe also sets logging.logging_dir which
  # is passed as wandb.init(dir=…) and takes precedence over the env var
  # (unirl/utils/wandb_logger.py:301). Export it anyway so wandb offline-sync
  # tools that read the env find the same directory.
  local WANDB_RUNS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/wandb_runs
  mkdir -p "${WANDB_RUNS_DIR}"

  local ROLLOUT_DUMPS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/rollout_dumps
  mkdir -p "${ROLLOUT_DUMPS_DIR}"

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
  QWEN3_OMNI_PATH=/dev/shm/Qwen3-Omni-30B-A3B-Instruct \
  DATA_PATH="${video_data}" \
  EVAL_DATA_PATH="${video_eval}" \
  WANDB_DIR="${WANDB_RUNS_DIR}" \
  ROLLOUT_PRINT_N=${ROLLOUT_PRINT_N:-8} \
  ROLLOUT_DUMP_DIR=${ROLLOUT_DUMP_DIR:-${ROLLOUT_DUMPS_DIR}} \
  PYTHONPATH=$(pwd) python -m unirl.train_ar \
    --config-name=ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x4 \
    ${EXTRA_OVERRIDES:-} \
    |& tee "${OUTPUT_DIR}/myunirl2_qwen3_omni_vllm_omni_1x4_video_r1_${timestamp}.log"
}

audio_dcase_1x4() {
  # 4-GPU standalone-audio DCASE 2025 MCQA run. Unlike ``omni``, each request
  # carries an ``(audio, prompt)`` MediaRef and use_audio_in_video=false.
  # The train set is variance-filtered: every prompt has 1..7 correct answers out
  # of 8 samples, so no group collapses to a zero-advantage all-right/all-wrong.
  local audio_data=/root/bruceszchen_gy2/datasets/dcase2025_audio_qa_train_mixed/train_mixed_2000.jsonl
  # Deduplicated against train_mixed_2000 by source ID and exact decoded audio.
  local audio_eval=/root/bruceszchen_gy2/datasets/dcase2025_audio_qa_mixed_300/test_mixed_300_no_train_overlap.jsonl
  [ -f "$audio_data" ] || die "DCASE audio train jsonl missing: $audio_data"
  [ -f "$audio_eval" ] || die "DCASE audio val jsonl missing: $audio_eval"

  local OUTPUT_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/train_outputs/UniRL
  local WANDB_RUNS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/wandb_runs
  local ROLLOUT_DUMPS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/rollout_dumps/dcase_audio_${timestamp}
  local VAL_DUMPS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/val_dumps/dcase_audio_${timestamp}
  mkdir -p "${OUTPUT_DIR}" "${WANDB_RUNS_DIR}" "${ROLLOUT_DUMPS_DIR}" "${VAL_DUMPS_DIR}"

  log "audio rollout dumps: ${ROLLOUT_DUMPS_DIR}"
  log "audio val dumps:     ${VAL_DUMPS_DIR}"

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7} \
  QWEN3_OMNI_PATH=/dev/shm/Qwen3-Omni-30B-A3B-Instruct \
  DATA_PATH="${audio_data}" \
  EVAL_DATA_PATH="${audio_eval}" \
  WANDB_DIR="${WANDB_RUNS_DIR}" \
  ROLLOUT_DUMP_DIR="${ROLLOUT_DUMPS_DIR}" \
  ROLLOUT_DUMP_N=${ROLLOUT_DUMP_N:-32} \
  VAL_DUMP_DIR="${VAL_DUMPS_DIR}" \
  PYTHONPATH=$(pwd) python -m unirl.train_ar \
    --config-name=ar/qwen3_omni_audio_dcase_gspo_lora_vllm_omni_1x4 \
    ${EXTRA_OVERRIDES:-} \
    |& tee "${OUTPUT_DIR}/qwen3_omni_vllm_omni_1x4_dcase_audio_${timestamp}.log"

  return ${PIPESTATUS[0]}
}

image_video_r1_1x4() {
  # 4-GPU standalone-image Video-R1 MCQA run. The image recipe inherits all
  # optimizer, rollout, reward, sampling, and evaluation settings from video.
  local image_data=/root/bruceszchen_gy2/datasets/video_r1_image_imaged_mixed_3500_400/train.jsonl
  local image_eval=/root/bruceszchen_gy2/datasets/video_r1_image_imaged_mixed_3500_400/val.jsonl
  local image_devices=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
  [ -f "$image_data" ] || die "Video-R1 image train jsonl missing: $image_data"
  [ -f "$image_eval" ] || die "Video-R1 image val jsonl missing: $image_eval"

  local OUTPUT_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/train_outputs/UniRL
  local WANDB_RUNS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/wandb_runs
  local ROLLOUT_DUMPS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/rollout_dumps/video_r1_image_${timestamp}
  local VAL_DUMPS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/val_dumps/video_r1_image_${timestamp}
  mkdir -p "${OUTPUT_DIR}" "${WANDB_RUNS_DIR}" "${ROLLOUT_DUMPS_DIR}" "${VAL_DUMPS_DIR}"

  log "image rollout dumps: ${ROLLOUT_DUMPS_DIR}"
  log "image val dumps:     ${VAL_DUMPS_DIR}"

  CUDA_VISIBLE_DEVICES="${image_devices}" \
  UNIRL_OMNI_TP_DEVICES="${image_devices}" \
  QWEN3_OMNI_PATH=/dev/shm/Qwen3-Omni-30B-A3B-Instruct \
  DATA_PATH="${image_data}" \
  EVAL_DATA_PATH="${image_eval}" \
  WANDB_DIR="${WANDB_RUNS_DIR}" \
  ROLLOUT_DUMP_DIR="${ROLLOUT_DUMPS_DIR}" \
  ROLLOUT_DUMP_N=${ROLLOUT_DUMP_N:-32} \
  VAL_DUMP_DIR="${VAL_DUMPS_DIR}" \
  PYTHONPATH=$(pwd) python -m unirl.train_ar \
    --config-name=ar/qwen3_omni_image_video_r1_gspo_lora_vllm_omni_1x4 \
    ${EXTRA_OVERRIDES:-} \
    |& tee "${OUTPUT_DIR}/qwen3_omni_vllm_omni_1x4_video_r1_image_${timestamp}.log"

  return ${PIPESTATUS[0]}
}

video_trainside_1x4() {
  # 4-GPU TRAINSIDE-rollout twin of video_r1_1x4 — no vLLM engine, no weight
  # sync: the training FSDP module IS the sampler, so rollout and replay share
  # one set of weights and k3 should sit at ~0. That makes this the regression
  # gate for the trainside path after the rebase onto the vLLM-Omni main.
  #
  # Same Video-R1-260k data as video_r1_1x4 (only CLEVRER + PerceptionTest have
  # their mp4s unzipped locally, which is exactly what
  # scripts/convert_video_r1_260k_to_unirl.py emits on this host — see the
  # ``video_r1_260k:<source>:<id>`` prompt_ids).
  #
  # Recipe: examples/ar/qwen3_omni_video_gspo_lora.yaml — a 2-GPU smoke by
  # default, overridden to DP=4 here. prompts_per_rollout must equal batch_size.
  # Pinned to GPUs 4-7; cards 0-3 are left free for interactive work.
  local video_data=/root/bruceszchen_gy2/datasets/video_r1_clevrer_perceptiontest/for_unirl/train.jsonl
  local video_eval=/root/bruceszchen_gy2/datasets/video_r1_clevrer_perceptiontest/for_unirl/val.jsonl
  [ -f "$video_data" ] || die "video_r1_260k train jsonl missing: $video_data"
  [ -f "$video_eval" ] || die "video_r1_260k val jsonl missing: $video_eval"

  mkdir -p outputs

  # Device placement is Ray's, not CUDA_VISIBLE_DEVICES': placement groups assign
  # GPUs from the cluster's own inventory, so exporting a device list in the
  # launcher does not move the workers. Join the shared head and Ray hands out
  # only the cards no other job has reserved. Do NOT let this auto-init a fresh
  # cluster: auto_init prestarts one worker per detected CPU and 384 of them
  # blow raylet's 30 s registration deadline (same reason smoke() caps --num-cpus).
  local ray_addr=${RAY_ADDRESS:-29.127.48.221:6380}
  ray status --address="${ray_addr}" >/dev/null 2>&1 \
    || die "no Ray cluster reachable at ${ray_addr} — start one or set RAY_ADDRESS"
  log "ray cluster:    ${ray_addr}"
  log "free GPUs:      $(ray status --address="${ray_addr}" 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1)"

  RAY_ADDRESS="${ray_addr}" \
  QWEN3_OMNI_PATH=/dev/shm/Qwen3-Omni-30B-A3B-Instruct \
  DATA_PATH="${video_data}" \
  EVAL_DATA_PATH="${video_eval}" \
  ROLLOUT_PRINT_N=${ROLLOUT_PRINT_N:-4} \
  ROLLOUT_DUMP_DIR=${ROLLOUT_DUMP_DIR:-./outputs/rollouts_trainside} \
  PYTHONPATH=$(pwd) python -m unirl.train_ar \
    --config-name=ar/qwen3_omni_video_gspo_lora \
    num_devices=4 devices_per_node=4 batch_size=4 \
    data_source.args.algorithm.prompts_per_rollout=4 \
    num_rollouts=${NUM_ROLLOUTS:-3} \
    backend.scheduler_cfg.total_steps=${NUM_ROLLOUTS:-3} \
    logging.report_to_wandb=false \
    ${EXTRA_OVERRIDES:-} \
    |& tee outputs/qwen3_omni_trainside_1x4_video_r1_${timestamp}.log

  return ${PIPESTATUS[0]}
}

omni() {
  unset UNIRL_TENSOR_DUMP_DIR

  # 4-GPU AUDIO-IN-VIDEO MCQA run on the vLLM-Omni engine (A2, TP=4), eval disabled.
  # use_audio_in_video=true (mp4 audio track fused via TMRoPE). Dataset: the
  # daily_omni_av variant with the ``<think>...</think> + The answer is [X]``
  # prompt format (see scripts/convert_daily_omni_prompt_think_ans.py) — the
  # base model's SFT-native ``<think>`` special tokens keep CoT tokenization
  # cheap and the answer scorer picks the phrase up cleanly.
  # Val is the mixed-variance subset (255 prompts, 1 ≤ correct/8 ≤ 7 under the
  # base model at fps=2, filtered by scripts/filter_daily_omni_val_by_variance.py).
  # This removes the ~44% all-correct + ~8% all-wrong prompts that dominate the
  # raw val's acc without carrying any GRPO signal.
  #
  # Recipe: examples/ar/run_qwen3_omni_vllm_omni_lora_a2_daily_omni_1x4.yaml
  #   (eval_before_train=true, eval_interval=5, video_fps=2.0). VAL_DUMP_DIR
  #   dumps every eval's (prompt/output/reward/gt) to val_rollout_<id>.jsonl.
  # Previous think-ans dataset (kept for reference; re-enable by swapping the two blocks):
  #local daily_data=/root/bruceszchen_gy2/datasets/daily_omni_av/unirl_think_ans/train.jsonl
  #local daily_eval=/root/bruceszchen_gy2/datasets/daily_omni_av/unirl_think_ans/val.jsonl
  #[ -f "$daily_data" ] || die "daily_omni_av think-ans (UniRL-native) train jsonl missing: $daily_data (run scripts/convert_daily_omni_av_to_unirl_think_ans.py)"
  #[ -f "$daily_eval" ] || die "daily_omni_av think-ans (UniRL-native) val jsonl missing: $daily_eval (run scripts/convert_daily_omni_av_to_unirl_think_ans.py)"

  # Built straight from the official Daily-Omni release (qa.json + Videos.tar) by
  # datasets/daily_omni_av/convert_daily_omni_dataset_format_to_unirl.py --qa-json ...
  # 1085 train / 112 val rows, split by video_id. The converter appends the
  # "The answer is [X]" instruction the recipe's require_answer_phrase=true scorer needs.
  local daily_data=/root/bruceszchen_gy2/datasets/daily_omni_av/unirl_from_qa_json/train.jsonl
  local daily_eval=/root/bruceszchen_gy2/datasets/daily_omni_av/unirl_from_qa_json/val.jsonl
  [ -f "$daily_data" ] || die "daily_omni_av train jsonl missing: $daily_data (run convert_daily_omni_dataset_format_to_unirl.py --qa-json)"
  [ -f "$daily_eval" ] || die "daily_omni_av val jsonl missing: $daily_eval (run convert_daily_omni_dataset_format_to_unirl.py --qa-json)"

  local OUTPUT_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/train_outputs/UniRL
  mkdir -p "${OUTPUT_DIR}"
  local WANDB_RUNS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/wandb_runs
  mkdir -p "${WANDB_RUNS_DIR}"
  local ROLLOUT_DUMPS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/rollout_dumps
  mkdir -p "${ROLLOUT_DUMPS_DIR}"
  local VAL_DUMPS_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/val_dumps
  mkdir -p "${VAL_DUMPS_DIR}"

  # Physical GPUs 4-7 appear inside the process as logical CUDA devices 0-3.
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7} \
  QWEN3_OMNI_PATH=/dev/shm/Qwen3-Omni-30B-A3B-Instruct \
  DATA_PATH="${daily_data}" \
  EVAL_DATA_PATH="${daily_eval}" \
  WANDB_DIR="${WANDB_RUNS_DIR}" \
  ROLLOUT_DUMP_DIR=${ROLLOUT_DUMP_DIR:-${ROLLOUT_DUMPS_DIR}} \
  VAL_DUMP_DIR=${VAL_DUMP_DIR:-${VAL_DUMPS_DIR}} \
  PYTHONPATH=$(pwd) python -m unirl.train_ar \
    --config-name=ar/qwen3_omni_audio_video_gspo_lora_vllm_omni_1x4 \
    eval_interval=0 \
    ${EXTRA_OVERRIDES:-} \
    |& tee "${OUTPUT_DIR}/qwen3_omni_vllm_omni_1x4_daily_omni_${timestamp}.log"

  return ${PIPESTATUS[0]}
}

debug_dump() {
  # Single-sample deterministic run to verify audio/video feature-processing
  # parity between the UniRL HF replay path and vLLM-Omni rollout.
  #
  # - dataset: train_1.jsonl (1 row) converted to UniRL format under debug_dump/
  # - config: examples/ar/debug_qwen3_omni_av_dump.yaml (temp=0, eval off,
  #           num_rollouts=1, seed fixed)
  # - dumps:  UNIRL_TENSOR_DUMP_DIR receives one .pt file per (tag, process)
  #           on rank 0 (HF: torch.distributed rank 0; vLLM: TP rank 0). Tags
  #           are namespaced hf_* (from unirl/models/qwen3_omni/ar.py) and
  #           vllm_* (from vllm_omni site-packages).
  local debug_data=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/datasets/daily_omni_av/debug_dump/train.jsonl
  local debug_eval=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/datasets/daily_omni_av/debug_dump/val.jsonl
  [ -f "$debug_data" ] || die "debug train jsonl missing: $debug_data"
  [ -f "$debug_eval" ] || die "debug eval jsonl missing: $debug_eval"

  local OUTPUT_DIR=/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/train_outputs/UniRL
  mkdir -p "${OUTPUT_DIR}"
  local DUMP_DIR=${UNIRL_TENSOR_DUMP_DIR:-/apdcephfs_gy2/share_303407316/hunyuan/bruceszchen/train_outputs/UniRL/av_dump_${timestamp}}
  mkdir -p "${DUMP_DIR}"
  log "tensor dumps -> ${DUMP_DIR}"

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} \
  QWEN3_OMNI_PATH=/dev/shm/Qwen3-Omni-30B-A3B-Instruct \
  DATA_PATH="${debug_data}" \
  EVAL_DATA_PATH="${debug_eval}" \
  UNIRL_TENSOR_DUMP_DIR="${DUMP_DIR}" \
  UNIRL_TENSOR_DUMP_LAYERS="${UNIRL_TENSOR_DUMP_LAYERS:-all}" \
  WANDB_MODE=disabled \
  PYTHONPATH=$(pwd) python -m unirl.train_ar \
    --config-name=ar/debug_qwen3_omni_av_dump \
    |& tee "${OUTPUT_DIR}/qwen3_omni_av_debug_dump_${timestamp}.log"
}

# -----------------------------------------------------------------------------
# Dispatch — default to smoke; pass "full" as arg to run the real training.
# -----------------------------------------------------------------------------
mode=${1:-smoke}
case "$mode" in
  smoke)      smoke ;;
  full)       full_1x8 ;;
  aqua)       aqua_rat_1x4 ;;
  video)      video_r1_1x4 ;;
  audio)      audio_dcase_1x4 ;;
  image)      image_video_r1_1x4 ;;
  trainside)  video_trainside_1x4 ;;
  omni)       omni ;;
  debug_dump) debug_dump ;;
  *)          die "unknown mode: $mode (expected: smoke | full | aqua | video | audio | image | trainside | omni | debug_dump)" ;;
esac
