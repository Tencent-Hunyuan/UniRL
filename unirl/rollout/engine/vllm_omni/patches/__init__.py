"""Quarantined vllm / vllm-omni monkeypatches — one idempotent ``install()``."""

from __future__ import annotations


def install() -> None:
    """Install the full vllm/vllm-omni patch bundle (idempotent)."""
    from unirl.rollout.engine.vllm_omni.patches.runtime import VLLMOmniHijack

    VLLMOmniHijack.hijack()


def __getattr__(name: str):
    if name in ("VLLMOmniHijack", "OmniTensorLoRARequest"):
        from unirl.rollout.engine.vllm_omni.patches import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["install", "OmniTensorLoRARequest", "VLLMOmniHijack"]
