"""SGLang LLM/VLM rollout engine (v2 — role-decomposed rewrite of ``sglang_llm/``).

A thin core over one runtime seam (``backends`` — the only sglang import, boot
included), with ``adapters`` holding the ``RolloutReq``↔``RolloutResp``
conversion (``text`` base + ``vlm`` override, keyed by ``model_family`` /
derived from ``image_token``), a small ``utils`` helper bag, and a
``WeightSync`` component owning the sync ops + LoRA lifecycle (the offload
flags live on the engine itself). The engine reserves its own
:class:`SGLangV2Ports` at boot and ``config.server_intent`` spells them into
ServerArgs intent. Coexists with the legacy ``sglang_llm`` engine; recipes opt
in via the two rollout ``_target_`` lines.

Importing this package populates the adapter registry (the ``adapters`` import
fires the ``@register_adapter`` side-effects).
"""

# Import adapters first so the registry is populated before config validation.
from unirl.rollout.engine.sglang_v2 import adapters  # noqa: F401
from unirl.rollout.engine.sglang_v2.config import SGLangV2EngineConfig, SGLangV2Ports
from unirl.rollout.engine.sglang_v2.engine import SGLangV2RolloutEngine

__all__ = [
    "SGLangV2RolloutEngine",
    "SGLangV2EngineConfig",
    "SGLangV2Ports",
]
