"""Bagel-7B (ByteDance BAGEL-MoT) model package for UniRL — official-repo migration.

Vendors the pristine official ``ByteDance-Seed/Bagel`` modeling under ``vendor/`` and
adds RL primitives over ``model._forward_flow`` in ``rl_ops.py``. Kept flash-attn-free
at import: ``BagelBundle`` is NOT re-exported (load it from ``.bundle`` on GPU).
"""

from .conditions import BagelDiffusionConditions
from .config import BAGEL_MOE_GEN_LORA_TARGETS, BagelPipelineConfig
from .diffusion import BagelDiffusionParams, BagelDiffusionStage, BagelDiffusionStep
from .pipeline import BagelPipeline
from .vae import BagelVAEDecodeStage, bagel_latent_geometry, bagel_latent_shape, unpatchify_latent

__all__ = [
    "BAGEL_MOE_GEN_LORA_TARGETS",
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
