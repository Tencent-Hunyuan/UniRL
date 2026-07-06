"""Qwen-Image-Edit-Plus pipeline on the typed four-tier architecture.

Text+image → image editing variant of :mod:`unirl.models.qwen_image`. The
Edit-Plus transformer (:class:`diffusers.QwenImageTransformer2DModel` with
``in_channels=64``) concatenates VAE-encoded source-image latents onto the
packed noise latent along the token dimension, then slices the prediction
back to the noise segment. Sibling of :mod:`unirl.models.qwen_image` and
:mod:`unirl.models.flux2_klein` (which uses the same token-concat pattern).

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.qwen_image_edit_plus.bundle import QwenImageEditPlusBundle
from unirl.models.qwen_image_edit_plus.conditions import QwenImageEditPlusConditions
from unirl.models.qwen_image_edit_plus.config import QwenImageEditPlusPipelineConfig
from unirl.models.qwen_image_edit_plus.pipeline import QwenImageEditPlusPipeline

__all__ = [
    "QwenImageEditPlusBundle",
    "QwenImageEditPlusConditions",
    "QwenImageEditPlusPipeline",
    "QwenImageEditPlusPipelineConfig",
]
