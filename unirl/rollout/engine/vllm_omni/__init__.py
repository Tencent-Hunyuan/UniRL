"""vLLM-Omni rollout engine."""


def __getattr__(name: str):
    if name == "VLLMOmniEngineConfig":
        from unirl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig

        return VLLMOmniEngineConfig
    if name == "VLLMOmniPorts":
        from unirl.rollout.engine.vllm_omni.config import VLLMOmniPorts

        return VLLMOmniPorts
    if name == "VLLMOmniRolloutEngine":
        from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine

        return VLLMOmniRolloutEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VLLMOmniPorts", "VLLMOmniEngineConfig", "VLLMOmniRolloutEngine"]
