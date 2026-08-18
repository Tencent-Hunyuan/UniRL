#!/usr/bin/env python
"""UniRL diffusion training entry point (Hydra-native)."""

from __future__ import annotations

import warnings

import hydra
from omegaconf import DictConfig

from unirl.trainer.diffusion import DiffusionTrainer


def _resolve_task_config(cfg: DictConfig):
    if "stage_config" not in cfg:
        return cfg.get("task_config")
    if "task_config" in cfg:
        raise ValueError("Specify only task_config; do not set deprecated stage_config alongside it")
    warnings.warn(
        "`stage_config` is deprecated; rename the recipe key to `task_config`",
        FutureWarning,
        stacklevel=2,
    )
    return cfg.get("stage_config")


@hydra.main(version_base=None, config_path="../examples", config_name="diffusion/sd3/sd3_trainside")
def main(cfg: DictConfig) -> None:
    trainer = DiffusionTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        rollout_cfg=cfg.rollout,
        # Optional for requires_advantages=False algorithms (e.g. DiffusionOPD):
        # no ``reward:`` block ⇒ no reward model is built and scoring is skipped.
        reward_cfg=cfg.get("reward"),
        algorithm_cfg=cfg.algorithm,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        scout_sampling_cfg=cfg.get("scout_sampling"),
        contrastive_rollout_cfg=cfg.get("contrastive_rollout"),
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        layout=cfg.get("layout", "colocate"),
        train_fraction=cfg.get("train_fraction", 0.5),
        reward_fraction=cfg.get("reward_fraction", 0.0),
        enable_fsdp_offload=cfg.get("enable_fsdp_offload", False),
        offload_train_during_reward=cfg.get("offload_train_during_reward", False),
        rollout_sleep_after_generate=cfg.get("rollout_sleep_after_generate", True),
        adv_use_global_std=cfg.get("adv_use_global_std", False),
        accumulate_rollouts=cfg.get("accumulate_rollouts", 1),
        eval_interval=cfg.get("eval_interval", 0),
        eval_num_prompts=cfg.get("eval_num_prompts", 64),
        eval_samples_per_prompt=cfg.get("eval_samples_per_prompt", 4),
        eval_chunk_prompts=cfg.get("eval_chunk_prompts", 16),
        eval_eta=cfg.get("eval_eta", 0.0),
        # Any DiffusionSamplingParams field; everything it omits inherits `sampling`.
        eval_sampling_cfg=cfg.get("eval_sampling"),
        eval_rewards_cfg=cfg.get("eval_rewards"),
        task_config=_resolve_task_config(cfg),
    )
    trainer.train(
        num_rollouts=cfg.get("num_rollouts", 100),
        weight_sync_interval=cfg.get("weight_sync_interval", 1),
        save_interval=cfg.get("save_interval", 0),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=cfg.get("save_mode", "auto"),
    )


if __name__ == "__main__":
    main()
