#!/usr/bin/env python
"""UniRL async diffusion training entry point (Hydra-native).

Sibling of ``train_diffusion.py`` that drives
:class:`unirl.trainer.async_diffusion.AsyncDiffusionTrainer` — the disaggregated,
async variant of the diffusion path (training and rollout on DISJOINT GPU slabs,
generation overlapped with training, reward scored synchronously at reap time
rather than overlapped, weights pushed via cross-slab weight sync). The synchronous
diffusion trainer is unchanged.

Launch (single node):
  BAGEL_PATH=/path/to/BAGEL-7B-MoT \
  python -m unirl.train_async_diffusion \
    --config-name=diffusion/bagel/bagel_vllmomni_async num_devices=8

Extra config knobs vs the synchronous separate recipe:
  * ``max_inflight`` — must be ``1``; other values fail during trainer initialization.
  * ``weight_sync_interval`` — rollout batches one published snapshot serves; max staleness is interval - 1.
``layout`` is forced to ``separate`` (async needs disjoint train/rollout slabs).
"""

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
        adv_use_global_std=cfg.get("adv_use_global_std", False),
        eval_interval=cfg.get("eval_interval", 0),
        eval_num_prompts=cfg.get("eval_num_prompts", 64),
        eval_samples_per_prompt=cfg.get("eval_samples_per_prompt", 4),
        eval_chunk_prompts=cfg.get("eval_chunk_prompts", 16),
        eval_cfg_text_scale=cfg.get("eval_cfg_text_scale", 4.0),
        eval_eta=cfg.get("eval_eta", 0.0),
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
