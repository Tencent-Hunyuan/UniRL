"""Bagel-7B (ByteDance BAGEL-MoT) model package for UniRL — official-repo migration.

Unlike the working ``UniRL/`` integration (which vendors flow_grpo's *fork* and
monkeypatches RL hacks into the model), this package vendors the **pristine
official** ``ByteDance-Seed/Bagel`` modeling under ``vendor/`` (no logic edits;
import roots rewritten only) and adds the RL primitives as standalone functions
in ``rl_ops.py`` over ``model._forward_flow``. See ``docs/BAGEL_MIGRATION_PLAN.md``.

Bagel is a unified Mixture-of-Transformers model: one transformer runs both the
text/understanding path and the image-generation (rectified-flow) path. The
integration mirrors ``unirl.models.hunyuan_image3`` — the diffusion "transformer"
is the LLM backbone itself and text conditioning flows through a KV cache rather
than a separate text encoder.

Five task modes, two stages (rollout + replay each): image-out ``t2i`` / ``it2i``
(editing) ride :class:`BagelDiffusionStage`; text-out ``t2t`` / ``i2t`` / ``it2t``
ride :class:`BagelARStage` (the und path, ``ar.py`` + the AR adapters in
``rl_ops.py``). ``BagelPipeline.generate`` routes via ``stage_config["task"]``
or infers from the sampling-params type + image-input presence.

Kept flash-attn-free at import time: ``import unirl.models.bagel`` must not pull
the vendored modeling (which hard-imports ``flash_attn``). The adapter wrappers
(config / conditions / diffusion / vae / pipeline) are flash-free; ``BagelBundle``
is intentionally NOT re-exported (it pulls the vendored modeling + flash_attn at
import time) — load it explicitly via ``from unirl.models.bagel.bundle import
BagelBundle`` only where a GPU model is constructed. ``rl_ops`` (the RL primitives)
is flash-free too (torch + diffusers only).
"""

from .ar import BagelARParams, BagelARStage, BagelARStep
from .conditions import BagelARConditions, BagelDiffusionConditions
from .config import BAGEL_MOE_GEN_LORA_TARGETS, BAGEL_UND_LORA_TARGETS, BagelPipelineConfig
from .diffusion import BagelDiffusionParams, BagelDiffusionStage, BagelDiffusionStep
from .pipeline import BagelPipeline
from .vae import BagelVAEDecodeStage, bagel_latent_geometry, bagel_latent_shape, unpatchify_latent

__all__ = [
    "BAGEL_MOE_GEN_LORA_TARGETS",
    "BAGEL_UND_LORA_TARGETS",
    "BagelARConditions",
    "BagelARParams",
    "BagelARStage",
    "BagelARStep",
    "BagelDiffusionConditions",
    "BagelDiffusionParams",
    "BagelDiffusionStage",
    "BagelDiffusionStep",
    "BagelPipeline",
    "BagelPipelineConfig",
    "BagelVAEDecodeStage",
    "bagel_latent_geometry",
    "bagel_latent_shape",
    "unpatchify_latent",
]
