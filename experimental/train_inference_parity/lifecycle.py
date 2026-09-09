"""Explicit environment propagation and plugin discovery checks."""

from __future__ import annotations

import importlib.metadata
import os
from typing import Mapping

from .runtime_contract import (
    PLUGIN_DISTRIBUTION,
    PLUGIN_ENTRYPOINT,
    required_environment,
)


def apply_parity_environment(environment: Mapping[str, str]) -> None:
    """Install the complete, explicit environment before validating or starting Ray."""
    expected_names = set(required_environment())
    missing = sorted(expected_names - set(environment))
    if missing:
        raise ValueError(f"incomplete parity environment; missing {missing}")
    for name, value in environment.items():
        os.environ[str(name)] = str(value)


def require_vllm_plugin_installed() -> None:
    entries = importlib.metadata.entry_points().select(group="vllm.general_plugins")
    names = {entry.name for entry in entries}
    if PLUGIN_ENTRYPOINT not in names:
        raise RuntimeError(
            f"missing vLLM plugin entry point {PLUGIN_ENTRYPOINT!r}; install it with "
            "`pip install -e experimental/train_inference_parity/vllm_plugin`"
        )


def connect_ray_with_parity_environment() -> None:
    """Connect the driver and propagate parity variables to every Ray role."""
    import ray

    if ray.is_initialized():
        raise RuntimeError("Ray was initialized before the parity runtime_env could be installed")
    explicit = set(required_environment())
    optional_passthrough = {
        "CUDA_VISIBLE_DEVICES",
        "DATA_PATH",
        "EVAL_DATA_PATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONPATH",
        "PYTORCH_CUDA_ALLOC_CONF",
        "QWEN3_MOE_PATH",
        "QWEN3_MOE_REVISION",
        "UNIRL_PARITY_ARTIFACT_PATH",
        "UNIRL_PARITY_STAGING_DIR",
    }
    env_vars = {name: os.environ[name] for name in sorted(explicit | optional_passthrough) if name in os.environ}
    ray.init(
        address=os.environ.get("RAY_ADDRESS", "auto"),
        runtime_env={"env_vars": env_vars},
    )


__all__ = [
    "PLUGIN_DISTRIBUTION",
    "PLUGIN_ENTRYPOINT",
    "apply_parity_environment",
    "connect_ray_with_parity_environment",
    "require_vllm_plugin_installed",
]
