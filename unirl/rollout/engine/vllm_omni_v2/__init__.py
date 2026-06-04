"""vLLM-Omni rollout engine, v2 layout (strangler replacement for ``vllm_omni``).

A thin engine core over one backend seam (``backends/`` — the only code that
imports the vllm-omni runtime, boot included), with per-modality adapters
(``adapters/`` — registry keyed on ``config.modality``, per-output-shape base
adapters holding the conversion), a pure ``utils/`` bag, a ``WeightSync``
component, typed self-reserved ports, and the worker-side role-8 packages
(``worker/`` / ``pipelines/`` / ``patches/``). Coexists with v1 under the
distinct ``vllm_omni_v2`` name until the parity gate retires v1; recipes
opt in by pointing the rollout ``_target_`` lines here.

Imports are lazy for the same reason as v1: engine modules pull
``rollout.engine.base`` whose import chain is still initializing when reached
from ``base → types → distributed``.
"""


def __getattr__(name: str):
    if name == "VLLMOmniV2EngineConfig":
        from unirl.rollout.engine.vllm_omni_v2.config import VLLMOmniV2EngineConfig

        return VLLMOmniV2EngineConfig
    if name == "VLLMOmniPorts":
        from unirl.rollout.engine.vllm_omni_v2.config import VLLMOmniPorts

        return VLLMOmniPorts
    if name == "VLLMOmniV2RolloutEngine":
        from unirl.rollout.engine.vllm_omni_v2.engine import VLLMOmniV2RolloutEngine

        return VLLMOmniV2RolloutEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["VLLMOmniPorts", "VLLMOmniV2EngineConfig", "VLLMOmniV2RolloutEngine"]
