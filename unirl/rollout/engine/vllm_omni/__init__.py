"""vLLM-Omni rollout engine for unirl.

Modality-keyed wrapper around vllm-omni's ``Omni`` orchestrator for
HunyuanImage 3.0. Supports all four upstream modalities: t2i, it2i,
i2t, t2t.

Imports are lazy to avoid a circular dependency: the trainer-side IPC
weight-sync handler needs ``weight_sync.bucketed_transfer`` which lives
under this package. Eager-importing ``engine`` here would pull
``rollout.engine.base`` which is still initializing when the import
chain starts from ``base → types → distributed → weight_sync → full.ipc``.
"""


def __getattr__(name: str):
    if name == "VLLMOmniEngineConfig":
        from unirl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig

        return VLLMOmniEngineConfig
    if name == "VLLMOmniRolloutEngine":
        from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine

        return VLLMOmniRolloutEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VLLMOmniEngineConfig", "VLLMOmniRolloutEngine"]
