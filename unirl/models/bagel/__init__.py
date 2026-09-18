"""Bagel-7B (ByteDance BAGEL-MoT) model package — vendors the pristine official modeling."""

from .ar import BagelARParams, BagelARStage, BagelARStep
from .chat_template import BagelChatTemplateStage
from .conditions import BagelARConditions, BagelDiffusionConditions
from .config import BAGEL_MOE_GEN_LORA_TARGETS, BAGEL_UND_LORA_TARGETS, BagelPipelineConfig
from .diffusion import BagelDiffusionParams, BagelDiffusionStage, BagelDiffusionStep
from .pipeline import BagelPipeline, BagelUniPipeline
from .vae import BagelVAEDecodeStage, bagel_latent_geometry, bagel_latent_shape, unpatchify_latent

__all__ = [
    "BAGEL_MOE_GEN_LORA_TARGETS",
    "BAGEL_UND_LORA_TARGETS",
    "BagelARConditions",
    "BagelARParams",
    "BagelARStage",
    "BagelARStep",
    "BagelChatTemplateStage",
    "BagelDiffusionConditions",
    "BagelDiffusionParams",
    "BagelDiffusionStage",
    "BagelDiffusionStep",
    "BagelPipeline",
    "BagelUniPipeline",
    "BagelPipelineConfig",
    "BagelVAEDecodeStage",
    "bagel_latent_geometry",
    "bagel_latent_shape",
    "unpatchify_latent",
]
