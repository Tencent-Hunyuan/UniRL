#!/usr/bin/env python
"""UniRL colocate PARTIAL-ROLLOUT agentic (answer-graded) training entry point (LIN-531).

Drives :class:`unirl.trainer.agentic_partial.AgenticPartialTrainer` — the colocate/synchronous
partial-rollout sibling of ``train_agentic`` (barrier `AgenticTrainer`). Train and rollout
still time-share each GPU; the trainer over-samples, commits the freshest `batch_size` complete
GRPO groups, and checkpoints the in-flight tail at a turn boundary (carried and resumed next
round when the tools are stateless, as search/visit are) instead of waiting for the slowest
trajectory.

Launch (single node; rank 0 owns the driver + the agentic coordinator):
  QWEN3_INSTRUCT_PATH=... DATA_PATH=data/asearcher/train.jsonl \
  SERPER_KEY_ID=... JINA_API_KEYS=... JUDGE_URL=... JUDGE_MODEL=... \
  python -m unirl.train_agentic_partial \
    --config-name=deep_research/deep_research_search_judge_partial num_devices=8

This entrypoint serves every answer-graded partial-rollout recipe under
``examples/deep_research/``; ``train_agentic_env_partial.py`` is the env-reward sibling.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.agentic_partial import AgenticPartialTrainer


@hydra.main(
    version_base=None,
    config_path="../examples",
    config_name="deep_research/deep_research_search_judge_partial",
)
def main(cfg: DictConfig) -> None:
    trainer = AgenticPartialTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        # Must equal the engine's episode_sampling.samples_per_prompt (the GRPO group size n).
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
        balance_shards=cfg.get("balance_shards", False),
        eval_interval=cfg.get("eval_interval", 0),
        stop=cfg.get("stop"),
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
