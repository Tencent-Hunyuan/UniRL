#!/usr/bin/env bash
# verl-omni side of the aligned SD3.5+FlowGRPO speed pair (see README.md).
# Run inside a verl-omni environment (install per upstream/docs/start/install.md
# against the pinned submodule), 1x8 GPUs. Data first:
#   python ../make_pickscore_parquet.py        # or via this repo's root path
# Then:
#   SD35=<hf-id-or-local-dir> STEPS=25 ATTN=sdpa bash run_verlomni_sd35_aligned.sh
# ATTN=sdpa is the backend-aligned row (matches UniRL's SDPA-class kernels);
# ATTN=fa3 is verl-omni's own default/best attention config.
# Parse: python ../parse_verl_timing.py <log> --samples-per-step 768 --gpus 8
set -ex
cd "$(dirname "$0")/upstream"

SD35=${SD35:-stabilityai/stable-diffusion-3.5-medium}
DATA=${DATA:-$HOME/data/pickscore_sd3}
STEPS=${STEPS:-25}
if [ "${ATTN:-sdpa}" = "fa3" ]; then
  ACTOR_ATTN=_flash_3_varlen_hub; ROLLOUT_ATTN=FLASH_ATTN
else
  ACTOR_ATTN=native; ROLLOUT_ATTN=TORCH_SDPA
fi
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$DATA/train.parquet \
    data.val_files=$DATA/test.parquet \
    data.train_batch_size=48 \
    data.val_max_samples=8 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.model.path=$SD35 \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    actor_rollout_ref.model.attn_backend=$ACTOR_ATTN \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.pipeline.height=384 \
    actor_rollout_ref.rollout.pipeline.width=384 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type="cps" \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward.num_workers=1 \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/pickscore_reward.py \
    reward.custom_reward_function.name=compute_score_pickscore \
    trainer.logger='["console"]' \
    trainer.project_name=speed_benchmarks \
    trainer.experiment_name=sd35_flowgrpo_pickscore_aligned \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$STEPS "$@"
