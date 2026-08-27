#!/usr/bin/env python
"""Service-scored barrier agentic training entry point."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.agentic import AgenticTrainer
from unirl.utils.graceful_shutdown import GracefulShutdown


@hydra.main(version_base=None, config_path="../examples", config_name="deep_research/deep_research_search_judge")
def main(cfg: DictConfig) -> None:
    trainer = None

    def teardown() -> None:
        if trainer is not None:
            trainer.shutdown()

    with GracefulShutdown(teardown, name="agentic-train") as guard:
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
            sync_cfg=cfg.sync,
            logging_cfg=cfg.get("logging"),
            stop=cfg.get("stop"),
            per_worker_inflight=cfg.get("per_worker_inflight", 8),
        )
        guard.claim_signals()
        trainer.train(
            num_rollouts=cfg.get("num_rollouts", 100),
            save_interval=cfg.get("save_interval", 0),
            save_dir=cfg.get("save_dir"),
            load_dir=cfg.get("load_dir"),
            save_mode=cfg.get("save_mode", "auto"),
        )


if __name__ == "__main__":
    main()
