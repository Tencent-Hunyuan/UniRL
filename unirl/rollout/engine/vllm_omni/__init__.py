"""vLLM-Omni rollout engine.

A thin engine core over one backend seam (``backends/`` — the only code that
imports the vllm-omni runtime, boot included), with per-modality adapters
(``adapters/`` — registry keyed on ``config.modality``, per-output-shape base
adapters holding the conversion), a pure ``utils/`` bag, a ``WeightSync``
component, typed self-reserved ports, and the worker-side role packages
(``worker/`` / ``pipelines/`` / ``patches/``). Recipes select it by pointing
their rollout ``_target_`` lines here.

Imports are lazy: engine modules pull ``rollout.engine.base`` whose import
chain is still initializing when reached from ``base → types → distributed``.
"""


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
