#!/usr/bin/env bash
# UniRL side of the aligned SD3.5+FlowGRPO speed pair (see README.md).
# Run from the UniRL repo root, inside a UniRL environment, 1x8 GPUs.
#   SD35=<hf-id-or-local-dir> STEPS=25 bash benchmarks/speed_benchmarks/verl_omni/run_unirl_sd35_aligned.sh
# Then: python benchmarks/speed_benchmarks/parse_perf.py <log> --samples-per-step 768 --gpus 8
set -ex
export PRETRAINED_MODEL=${SD35:-stabilityai/stable-diffusion-3.5-medium}
export REPORT_TO_WANDB=${REPORT_TO_WANDB:-false}

python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_vllmomni \
  batch_size=48 sampling.samples_per_prompt=16 \
  sampling.height=384 sampling.width=384 sampling.eta=0.8 \
  backend.optimizer_cfg.learning_rate=1e-4 backend.optimizer_cfg.weight_decay=1e-4 \
  algorithm.clip_range=1e-5 \
  stack.micro_batch_size=8 \
  +num_rollouts=${STEPS:-25} "$@"
