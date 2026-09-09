#!/usr/bin/env python
"""Hydra entry point for the train/inference parity experiment."""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from experimental.train_inference_parity.lifecycle import (
    apply_profile_environment,
    connect_ray_with_parity_environment,
    require_vllm_plugin_installed,
)
from experimental.train_inference_parity.profiles import resolve_profile
from unirl.trainer.ar import ARTrainer
from unirl.utils.graceful_shutdown import GracefulShutdown


@hydra.main(
    version_base=None,
    config_path="examples",
    config_name="qwen3_moe_30b_a3b_fsdp_tp4",
)
def main(cfg: DictConfig) -> None:
    profile = resolve_profile(cfg.get("parity_profile"))
    apply_profile_environment(profile)
    parity = cfg.get("parity", {})
    os.environ["UNIRL_PARITY_TRAIN_TP"] = str(parity.get("train_tp", 1))
    require_vllm_plugin_installed()
    connect_ray_with_parity_environment()
    trainer = None

    def teardown() -> None:
        if trainer is not None:
            trainer.shutdown()

    with GracefulShutdown(teardown, name="train-inference-parity") as guard:
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
            enable_fsdp_offload=cfg.get("enable_fsdp_offload", True),
        )
        guard.claim_signals()
        trainer.train(
            num_rollouts=cfg.get("num_rollouts", 1),
            weight_sync_interval=cfg.get("weight_sync_interval", 1),
        )


if __name__ == "__main__":
    main()
