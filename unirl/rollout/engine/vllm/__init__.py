"""Direct text-only vLLM rollout integration."""


def __getattr__(name: str):
    if name == "VLLMEngineConfig":
        from .config import VLLMEngineConfig

        return VLLMEngineConfig
    if name == "VLLMRolloutEngine":
        from .engine import VLLMRolloutEngine

        return VLLMRolloutEngine
    raise AttributeError(name)


__all__ = ["VLLMEngineConfig", "VLLMRolloutEngine"]
