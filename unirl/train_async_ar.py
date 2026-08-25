#!/usr/bin/env python
"""UniRL async AR training entry point (Hydra-native)."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.async_ar import AsyncARTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="ar/qwen3_grpo_4b_base_dapo_sglang_async")
def main(cfg: DictConfig) -> None:
    trainer = AsyncARTrainer(
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
        adv_normalization_scope=cfg.get("adv_normalization_scope", "group"),
        normalize_adv_by_std=bool(cfg.get("normalize_adv_by_std", True)),
        advantage_mode=cfg.get("advantage_mode", "grpo"),
        balance_shards=bool(cfg.get("balance_shards", False)),
        eval_interval=int(cfg.get("eval_interval", 0)),
        eval_num_prompts=int(cfg.get("eval_num_prompts", -1)),
        eval_batch_size=int(cfg.get("eval_batch_size", 8)),
        eval_samples_per_prompt=int(cfg.get("eval_samples_per_prompt", 16)),
        eval_temperature=float(cfg.get("eval_temperature", 1.0)),
        train_fraction=float(cfg.get("train_fraction", 0.5)),
        max_inflight=int(cfg.get("max_inflight", 1)),
        per_worker_inflight=int(cfg.get("per_worker_inflight", 1)),
        weight_sync_interval=int(cfg.get("weight_sync_interval", 1)),
        buffer_max_staleness=cfg.get("buffer_max_staleness"),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        save_interval=int(cfg.get("save_interval", 0)),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=str(cfg.get("save_mode", "full")),
    )


if __name__ == "__main__":
    main()
