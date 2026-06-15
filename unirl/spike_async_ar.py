#!/usr/bin/env python
"""Gating spike for AsyncARTrainer — isolates the cross-slab NCCL path.

The one piece the AR path has never exercised is a RESIDENT SGLang engine acting
as a cross-slab NCCL weight receiver (the colocate AR recipe uses TensorWeightSync
on a shared GPU). This builds the trainer (two disjoint slabs + the NCCLWeightSync
handshake), then pushes weights once and generates once — WITHOUT any
reward/advantage/training — so a failure points squarely at the
placement / handshake / broadcast / generate path, not the train stack.

Run it FIRST, before trusting any training run. Smallest layout: 4 GPUs.

  DATA_PATH=data/dapo_math/train.jsonl \
  python -m unirl.spike_async_ar --config-name=ar/qwen3_grpo_4b_base_dapo_sglang_async \
    num_devices=4 batch_size=8 sampling.samples_per_prompt=4 train_fraction=0.5

Validates: integer slab split, pick_master/set_rollout_targets/connect rendezvous
into a LIVE engine, NCCL update_weights_from_distributed without sleep, and that
generation returns a full-batch RolloutTrack carrying per-token log_probs.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from unirl.trainer.async_ar import AsyncARTrainer

logger = logging.getLogger(__name__)


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
        logging_cfg=None,  # spike: no wandb
        train_fraction=float(cfg.get("train_fraction", 0.5)),
        rollout_mode="pipeline",
    )

    expected = int(trainer.batch_size) * int(trainer.sampling_params.samples_per_prompt)

    # 1) Build a generation request from the data stream.
    req = trainer._build_req(trainer.data_source.get_samples(trainer.batch_size), 0)

    # 2) Cross-slab weight push into the live engine (the gating step).
    logger.info("spike: pushing weights via NCCLWeightSync.sync() ...")
    trainer.weight_sync.sync()
    logger.info("spike: sync OK (weight_version bumped on the train slab)")

    # 3) One generation on the resident engine.
    logger.info("spike: generating %d samples ...", expected)
    resp = trainer._gen(req)
    track = resp.tracks["ar"]

    assert track.batch_size == expected, f"got {track.batch_size} samples, expected {expected}"
    assert track.segment is not None, "track.segment is None (no generated tokens)"
    assert track.segment.log_probs is not None, "segment.log_probs is None (rollout logprobs not captured)"
    logger.info(
        "spike PASS: %d samples, segment carries tokens + log_probs. Cross-slab NCCL path is sound.",
        track.batch_size,
    )


if __name__ == "__main__":
    main()
