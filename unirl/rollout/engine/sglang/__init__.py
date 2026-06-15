"""SGLang LLM/VLM rollout engine — role-decomposed.

A thin core over one runtime seam (``backends`` — the only sglang import, boot
included), with ``adapters`` holding the ``RolloutReq``↔``RolloutResp``
conversion (``text`` base + ``vlm`` override, keyed by ``model_family`` /
derived from ``image_token``), a small ``utils`` helper bag, and a
``WeightSync`` component owning the sync ops + LoRA lifecycle (the offload
flags live on the engine itself). The engine reserves its own
:class:`SGLangPorts` at boot and ``config.server_intent`` spells them into
ServerArgs intent. Recipes select it via the two rollout ``_target_`` lines
(engine + config).

Importing this package populates the adapter registry (the ``adapters`` import
fires the ``@register_adapter`` side-effects).
"""

# Import adapters first so the registry is populated before config validation.
from unirl.rollout.engine.sglang import adapters  # noqa: F401
from unirl.rollout.engine.sglang.config import SGLangEngineConfig, SGLangPorts
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine

__all__ = [
    "SGLangRolloutEngine",
    "SGLangEngineConfig",
    "SGLangPorts",
]
