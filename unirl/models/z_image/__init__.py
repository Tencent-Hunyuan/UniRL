"""Z-Image pipeline on the typed four-tier architecture.

Implements the typed ``Bundle`` / ``Pipeline`` / ``EmbedStage`` /
``DiffusionStage`` / ``DecodeStage`` protocols. Sibling of
:mod:`unirl.models.sd3` and :mod:`unirl.models.qwen_image`.

Z-Image (Tongyi-MAI) is a 6B Scalable Single-Stream DiT (S3-DiT) image
diffusion model: text tokens, (optional) visual semantic tokens, and image
VAE tokens are concatenated into one unified sequence. This package wires
the text-to-image path (``Z-Image`` / ``Z-Image-Turbo``); Z-Image-Turbo is
the RL-friendly distilled checkpoint (few-step, no CFG).

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.z_image.bundle import ZImageBundle
from unirl.models.z_image.conditions import ZImageConditions
from unirl.models.z_image.config import ZImagePipelineConfig
from unirl.models.z_image.pipeline import ZImagePipeline

__all__ = [
    "ZImageBundle",
    "ZImageConditions",
    "ZImagePipeline",
    "ZImagePipelineConfig",
]
