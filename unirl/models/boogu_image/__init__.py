"""Boogu-Image model package — Base (T2I) RL support.

Recipes wire the classes by ``_target_`` dotpath:

- ``unirl.models.boogu_image.BooguImageBundle.from_config`` (+ nested
  ``unirl.models.boogu_image.BooguImagePipelineConfig``)
- ``unirl.models.boogu_image.BooguImagePipeline``
- ``unirl.models.boogu_image.conditions.BooguImageConditions``
  (``algorithm.conditions_cls``)

The DiT architecture is vendored under ``vendor/`` (pinned upstream commit —
see ``vendor/VENDOR_COMMIT.txt``); the Edit (TI2I) and Turbo (DMD few-step)
variants are follow-ups.
"""

from .bundle import BooguImageBundle
from .conditions import BooguImageConditions
from .config import BOOGU_IMAGE_BASE_STATIC_SHIFT, BooguImagePipelineConfig
from .diffusion import BooguImageDiffusionStage, BooguImageDiffusionStep
from .pipeline import BooguImagePipeline
from .text_embed import BooguImageTextEmbedStage
from .vae import BooguImageVAEDecodeStage

__all__ = [
    "BOOGU_IMAGE_BASE_STATIC_SHIFT",
    "BooguImageBundle",
    "BooguImageConditions",
    "BooguImagePipelineConfig",
    "BooguImagePipeline",
    "BooguImageDiffusionStage",
    "BooguImageDiffusionStep",
    "BooguImageTextEmbedStage",
    "BooguImageVAEDecodeStage",
]
