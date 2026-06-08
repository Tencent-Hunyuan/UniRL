"""WAN 2.1 T2V / I2V pipeline on the typed four-tier architecture.

Implements the ``Bundle`` / ``Pipeline`` / ``EmbedStage`` /
``DiffusionStage`` / ``DecodeStage`` protocols. Sibling of
:mod:`unirl.models.sd3` and :mod:`unirl.models.wan22` (text-to-video peer).

I2V is wired through two optional ``EncodeStage`` siblings:
:class:`WAN21ImageLatentEncodeStage` (``Images`` → 20-channel mask + VAE
latent payload, channel-concatted onto noise before the transformer)
and :class:`WAN21CLIPVisionEncodeStage` (``Images`` → CLIP penultimate
patch embeddings, forwarded as ``encoder_hidden_states_image``). Both
fire only when the I2V checkpoint declares ``transformer.config.image_dim
> 0``; T2V bundles skip them and the pipeline is unchanged.

Importing this package re-exports its bundle / pipeline / config classes;
recipes wire them by ``_target_`` dotpath.
"""

from unirl.models.wan21.bundle import WAN21Bundle
from unirl.models.wan21.clip_vision_encode import WAN21CLIPVisionEncodeStage
from unirl.models.wan21.conditions import WAN21Conditions
from unirl.models.wan21.config import WAN21PipelineConfig
from unirl.models.wan21.diffusion import (
    WAN21DiffusionStage,
    WAN21DiffusionStep,
)
from unirl.models.wan21.image_encode import WAN21ImageLatentEncodeStage
from unirl.models.wan21.pipeline import WAN21Pipeline
from unirl.models.wan21.text_embed import WAN21TextEmbedStage
from unirl.models.wan21.vae import WAN21VAEDecodeStage

__all__ = [
    "WAN21Bundle",
    "WAN21CLIPVisionEncodeStage",
    "WAN21Conditions",
    "WAN21DiffusionStage",
    "WAN21DiffusionStep",
    "WAN21ImageLatentEncodeStage",
    "WAN21Pipeline",
    "WAN21PipelineConfig",
    "WAN21TextEmbedStage",
    "WAN21VAEDecodeStage",
]
