"""Runtime manifest and plugin discovery checks."""

from __future__ import annotations

import importlib.metadata
import os

from .profiles import ProfileConfig

PLUGIN_ENTRYPOINT = "unirl_train_inference_parity"
PLUGIN_DISTRIBUTION = "unirl-train-inference-parity-vllm"


def apply_profile_environment(profile: ProfileConfig) -> None:
    os.environ["UNIRL_PARITY_PROFILE"] = profile.name.value
    os.environ["UNIRL_PARITY_MODEL"] = "qwen3_moe_30b_a3b"
    os.environ["UNIRL_PARITY_STRICT"] = "1"
    os.environ["VLLM_PLUGINS"] = PLUGIN_ENTRYPOINT
    os.environ["UNIRL_PARITY_PATCHES"] = "common,qwen3_moe_30b_a3b"
    for name, value in profile.providers.items():
        os.environ[f"UNIRL_PARITY_{name.upper()}_PROVIDER"] = value


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
        return
    prefixes = ("UNIRL_PARITY_",)
    explicit = {
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "FLASH_ATTENTION_DETERMINISTIC",
        "HF_HUB_OFFLINE",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTORCH_CUDA_ALLOC_CONF",
        "QWEN3_MOE_PATH",
        "TRANSFORMERS_OFFLINE",
        "VLLM_BATCH_INVARIANT",
        "VLLM_PLUGINS",
        "VLLM_WORKER_MULTIPROC_METHOD",
    }
    env_vars = {key: value for key, value in os.environ.items() if key in explicit or key.startswith(prefixes)}
    ray.init(
        address=os.environ.get("RAY_ADDRESS", "auto"),
        runtime_env={"env_vars": env_vars},
    )


__all__ = [
    "PLUGIN_DISTRIBUTION",
    "PLUGIN_ENTRYPOINT",
    "apply_profile_environment",
    "connect_ray_with_parity_environment",
    "require_vllm_plugin_installed",
]
