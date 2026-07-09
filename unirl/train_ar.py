#!/usr/bin/env python
"""UniRL v2 AR (autoregressive) training entry point (Hydra-native).

Thin wrapper around :class:`unirl.trainer.ar.ARTrainer` — the AR path's
sibling of ``train_diffusion.py`` (which drives the diffusion ``DiffusionTrainer``).
Kept separate so the AR path never routes through diffusion / SDE code.

Launch (per node, SPMD; rank 0 owns the driver):
  QWEN_VL_PATH=/path/to/Qwen2.5-VL-7B-Instruct DATA_PATH=/path/to/train.jsonl \
  python -m unirl.train_ar --config-name=ar/qwen_vl_grpo_geo3k_mc_4x8

This AR entrypoint serves both vision-language (``qwen_vl``) and pure-LLM
(``qwen3``) recipes under ``examples/ar/``
(e.g. ``--config-name=ar/qwen3_drpo_4b_base_dapo_sglang``).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.ar import ARTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="ar/qwen_vl_grpo_geo3k_mc_4x8")
def main(cfg: DictConfig) -> None:
    trainer = ARTrainer(
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
        normalize_adv_by_std=cfg.get("normalize_adv_by_std", True),
        balance_shards=cfg.get("balance_shards", False),
        eval_interval=cfg.get("eval_interval", 0),
        eval_num_prompts=cfg.get("eval_num_prompts", -1),
        eval_batch_size=cfg.get("eval_batch_size", 8),
        eval_samples_per_prompt=cfg.get("eval_samples_per_prompt", 16),
        eval_temperature=cfg.get("eval_temperature", 1.0),
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
