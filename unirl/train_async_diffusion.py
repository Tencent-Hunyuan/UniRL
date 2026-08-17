#!/usr/bin/env python
"""UniRL async diffusion training entry point (Hydra-native)."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.async_diffusion import AsyncDiffusionTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="diffusion/bagel/bagel_vllmomni_async")
def main(cfg: DictConfig) -> None:
    trainer = AsyncDiffusionTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        algorithm_cfg=cfg.algorithm,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        layout="separate",
        train_fraction=cfg.get("train_fraction", 0.5),
        reward_fraction=cfg.get("reward_fraction", 0.0),
        # Forwarded so the trainer can reject it — async scores at reap time outside
        # _reward_phase(), and dropping the key here would silently ignore the policy.
        offload_train_during_reward=cfg.get("offload_train_during_reward", False),
        # Async default False (sync entry defaults True): the dedicated rollout
        # slab stays resident; evaluate() also passes sleep_after=False here.
        rollout_sleep_after_generate=cfg.get("rollout_sleep_after_generate", False),
        adv_use_global_std=cfg.get("adv_use_global_std", False),
        eval_interval=cfg.get("eval_interval", 0),
        eval_num_prompts=cfg.get("eval_num_prompts", 64),
        eval_samples_per_prompt=cfg.get("eval_samples_per_prompt", 4),
        eval_chunk_prompts=cfg.get("eval_chunk_prompts", 16),
        eval_eta=cfg.get("eval_eta", 0.0),
        # Any DiffusionSamplingParams field; everything it omits inherits `sampling`.
        eval_sampling_cfg=cfg.get("eval_sampling"),
        eval_rewards_cfg=cfg.get("eval_rewards"),
        task_config=cfg.get("task_config"),
        max_inflight=int(cfg.get("max_inflight", 1)),
        weight_sync_interval=int(cfg.get("weight_sync_interval", 1)),
    )
    trainer.train(
        num_rollouts=cfg.get("num_rollouts", 100),
        save_interval=cfg.get("save_interval", 0),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=cfg.get("save_mode", "auto"),
    )


if __name__ == "__main__":
    main()
