#!/usr/bin/env python
"""UniRL v2 HunyuanImage3 training entry point (Hydra-native).

Thin wrapper around :class:`unirl.trainer.unified_model.UnifiedModelTrainer`. The trainer
owns the placement scope, sibling Remote wiring, and the ``train_step → train``
loop; this module just maps the loaded Hydra config blocks to constructor
kwargs.

Pairs with ``examples/unified_model/hi3_vllmomni.yaml``::

    python -m unirl.train_unified_model --config-name unified_model/hi3_vllmomni
"""

from __future__ import annotations

import math
import os

import hydra
from omegaconf import DictConfig

from unirl.trainer.unified_model import UnifiedModelTrainer

_CUDA_ALLOCATOR_ENV_PRECEDENCE = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
_MAX_BAGEL_FLOW_MANY_MEMORY_FRACTION = 0.90


def _nested_config_get(config, *keys):
    value = config
    for key in keys:
        if value is None or not hasattr(value, "get"):
            return None
        value = value.get(key)
    return value


def _requires_bagel_flow_many_allocator_cap(cfg: DictConfig) -> bool:
    """Return whether this run uses the GPU-resident BAGEL flow-many path."""
    return (
        _nested_config_get(cfg, "rollout", "config", "modality") == "bagel_t2ti"
        and bool(_nested_config_get(cfg, "pipeline", "t2ti_flow_many_enabled"))
        and not bool(cfg.get("enable_fsdp_offload", True))
    )


def _effective_cuda_allocator_conf() -> tuple[str | None, str | None]:
    """Resolve allocator aliases with PyTorch's precedence, including empty values."""
    for name in _CUDA_ALLOCATOR_ENV_PRECEDENCE:
        if name in os.environ:
            return name, os.environ[name]
    return None, None


def _allocator_setting(config: str, name: str) -> str | None:
    """Return the last value for a comma-delimited PyTorch allocator setting."""
    value = None
    for item in config.split(","):
        key, separator, candidate = item.partition(":")
        if separator and key.strip() == name:
            value = candidate.strip()
    return value


def _validate_bagel_flow_many_allocator_cap(source: str | None, config: str | None) -> None:
    setting = _allocator_setting(config or "", "per_process_memory_fraction")
    try:
        fraction = float(setting) if setting is not None else None
    except ValueError:
        fraction = None

    if fraction is not None and math.isfinite(fraction) and 0.0 < fraction <= _MAX_BAGEL_FLOW_MANY_MEMORY_FRACTION:
        return

    effective = f"{source}={config!r}" if source is not None else "no allocator configuration"
    raise ValueError(
        "GPU-resident BAGEL T2TI flow-many requires the effective allocator configuration "
        f"to include a finite per_process_memory_fraction in (0, "
        f"{_MAX_BAGEL_FLOW_MANY_MEMORY_FRACTION:.2f}]; got {effective}. "
        "PYTORCH_CUDA_ALLOC_CONF takes precedence over PYTORCH_ALLOC_CONF. "
        "Unset the override to use cuda_allocator_conf, or add a safe cap to the effective alias."
    )


def _configure_cuda_allocator(cfg: DictConfig) -> None:
    """Apply and validate allocator policy before the trainer creates Ray workers."""
    configured = cfg.get("cuda_allocator_conf")
    source, effective = _effective_cuda_allocator_conf()
    if configured and source is None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(configured)
        source, effective = "PYTORCH_CUDA_ALLOC_CONF", str(configured)

    if _requires_bagel_flow_many_allocator_cap(cfg):
        _validate_bagel_flow_many_allocator_cap(source, effective)


@hydra.main(version_base=None, config_path="../examples", config_name="unified_model/hi3_vllmomni")
def main(cfg: DictConfig) -> None:
    _configure_cuda_allocator(cfg)
    trainer = UnifiedModelTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        ar_rollout_cfg=cfg.get("ar_rollout"),
        dit_rollout_cfg=cfg.get("dit_rollout"),
        rollout_cfg=cfg.get("rollout"),
        reward_cfg=cfg.reward,
        ar_algorithm_cfg=cfg.algorithm.ar,
        image_algorithm_cfg=cfg.algorithm.image,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        dump_dir=cfg.get("dump_dir"),
        logging_cfg=cfg.get("logging"),
        enable_fsdp_offload=cfg.get("enable_fsdp_offload", True),
        park_optimizer_state_during_rollout=cfg.get("park_optimizer_state_during_rollout", False),
        park_optimizer_state_during_train=cfg.get("park_optimizer_state_during_train", False),
        eval_interval=cfg.get("eval_interval", 0),
        eval_num_prompts=cfg.get("eval_num_prompts", cfg.batch_size),
        eval_cfg_text_scale=float(cfg.get("eval_cfg_text_scale", 4.0)),
        eval_eta=float(cfg.get("eval_eta", 0.0)),
        eval_rewards_cfg=cfg.get("eval_rewards"),
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
