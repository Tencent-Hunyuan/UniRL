#!/usr/bin/env python
"""UniRL fully-async deep-research (agentic) training entry point (LIN-531).

Drives :class:`unirl.trainer.agentic_async.AsyncAgenticTrainer` — the disaggregated
producer/consumer sibling of ``train_deep_research`` (:class:`AgenticTrainer`).
Training and the agentic rollout engine run on DISJOINT GPU slabs (``train_fraction``);
the engine stays resident and weights cross the slab boundary via ``NCCLWeightSync``
(not the colocate ``TensorWeightSync``). Partial rollout checkpoints the in-flight tail
at a turn boundary on each sync and resumes it under the new weights;
``buffer_max_staleness`` bounds the off-policy gap. The trainer is hard-coded per
entrypoint (the repo pattern), because the async agentic driver differs from both the
colocate agentic loop and the async-AR DP_SCATTER loop.

Launch (per node, SPMD; rank 0 owns the driver + the agentic coordinator on the
rollout slab):
  QWEN3_INSTRUCT_PATH=/path/to/Qwen3-4B-Instruct DATA_PATH=/path/to/train.jsonl \
  python -m unirl.train_async_deep_research \
    --config-name=deep_research/deep_research_calc_mathverify_async num_devices=2
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.agentic_async import AsyncAgenticTrainer


@hydra.main(
    version_base=None,
    config_path="../examples",
    config_name="deep_research/deep_research_calc_mathverify_async",
)
def main(cfg: DictConfig) -> None:
    trainer = AsyncAgenticTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        # Must equal the engine's episode_sampling.samples_per_prompt (the GRPO group
        # size n); the recipe keeps sampling.samples_per_prompt == episode_sampling.
        samples_per_prompt=cfg.sampling.samples_per_prompt,
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
        stop=cfg.get("stop"),
        train_fraction=cfg.get("train_fraction", 0.5),
        oversample_batch_size=cfg.get("oversample_batch_size"),
        buffer_max_staleness=cfg.get("buffer_max_staleness"),
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
