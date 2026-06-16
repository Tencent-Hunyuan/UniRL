"""SGLang diffusion rollout engine (v2 — role-decomposed rewrite of ``sglang/``).

A thin core over one runtime seam (``backends``), with per-model ``adapters``
holding the ``RolloutReq``↔``RolloutResp`` conversion, a small ``utils`` helper bag,
and a ``WeightSync`` component owning the sync ops + LoRA lifecycle (the offload
flag lives on the engine itself). In local mode the engine reserves its own
``SGLangDiffusionPorts`` at boot and ``config.server_intent`` spells them into
ServerArgs intent. Coexists with the legacy ``sglang`` engine; recipes opt in by
pointing the rollout ``_target_``s at this engine + config (wired by ``_target_``
only; the actor constructs the engine via ``config.make_engine``).

Importing this package populates the adapter registry (the ``adapters`` import
fires the ``@register_adapter`` side-effects).
"""

# Import adapters first so the registry is populated before config validation.
from unirl.rollout.engine.sglang_diffusion import adapters  # noqa: F401
from unirl.rollout.engine.sglang_diffusion.config import (
    SGLangDiffusionEngineConfig,
    SGLangDiffusionPorts,
)
from unirl.rollout.engine.sglang_diffusion.engine import SGLangDiffusionRolloutEngine

__all__ = [
    "SGLangDiffusionRolloutEngine",
    "SGLangDiffusionEngineConfig",
    "SGLangDiffusionPorts",
]
