#!/usr/bin/env python
"""UniRL fully-async agentic (answer-graded) training entry point (LIN-531).

Drives :class:`unirl.trainer.agentic_async.AsyncAgenticTrainer` — the disaggregated
producer/consumer sibling of ``train_agentic`` (:class:`AgenticTrainer`).
Training and the agentic rollout engine run on DISJOINT GPU slabs (``train_fraction``);
the engine stays resident and weights cross the slab boundary via ``NCCLWeightSync``
(not the colocate ``TensorWeightSync``). Partial rollout checkpoints the in-flight tail
at a turn boundary on each sync. The search/visit recipe uses ``tail_policy=carry``
because its stateless tool state is fully represented by the carried ``Sample``;
``buffer_max_staleness`` bounds completed groups in the consumer buffer, while the
per-token ratio corrects turns carried across weight versions. Stateful tool sessions
must use ``drop`` until cross-worker stateful resume is implemented. The trainer is
hard-coded per entrypoint (the repo pattern), because the async agentic driver differs
from both the colocate agentic loop and the async-AR DP_SCATTER loop.

Launch (per node, SPMD; rank 0 owns the driver + the agentic coordinator on the
rollout slab):
  QWEN3_INSTRUCT_PATH=/path/to/Qwen3-4B-Instruct DATA_PATH=data/asearcher/train.jsonl \
  SERPER_KEY_ID=... JINA_API_KEYS=... JUDGE_URL=... JUDGE_MODEL=... \
  python -m unirl.train_agentic_async \
    --config-name=deep_research/deep_research_search_judge_async num_devices=2

This entrypoint serves every answer-graded fully-async recipe under
``examples/deep_research/``; ``train_agentic_env_async.py`` is the env-reward sibling.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.agentic_async import AsyncAgenticTrainer


@hydra.main(
    version_base=None,
    config_path="../examples",
    config_name="deep_research/deep_research_search_judge_async",
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
        tail_policy=cfg.get("tail_policy", "carry"),
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
