#!/usr/bin/env python
"""UniRL agentic (multi-turn, answer-graded reward) training entry point (LIN-519).

Drives :class:`unirl.trainer.agentic.AgenticTrainer` over the
:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine`. Sibling of
``train_ar.py``; a separate entrypoint because the agentic rollout returns a
``List[Sample]`` of variable-depth trajectories (the trainer is hard-coded per
entrypoint, per the repo pattern).

The reward grades each trajectory's terminal ``<answer>`` through the reward backend;
``train_agentic_env.py`` is the sibling whose reward is the environment's own
per-trajectory return.

Launch (per node, SPMD; rank 0 owns the driver + the agentic coordinator):
  QWEN3_INSTRUCT_PATH=/path/to/Qwen3-4B-Instruct DATA_PATH=data/asearcher/train.jsonl \
  SERPER_KEY_ID=... JINA_API_KEYS=... JUDGE_URL=... JUDGE_MODEL=... \
  python -m unirl.train_agentic --config-name=deep_research/deep_research_search_judge

This entrypoint serves every answer-graded agentic recipe under
``examples/deep_research/``; ``train_agentic_partial.py`` and ``train_agentic_async.py``
are the same reward source under the other two execution topologies.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.agentic import AgenticTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="deep_research/deep_research_search_judge")
def main(cfg: DictConfig) -> None:
    trainer = AgenticTrainer(
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
        stop=cfg.get("stop"),
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
