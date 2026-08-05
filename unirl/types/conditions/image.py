"""Image conditioning types.

``ImageLatentCondition`` carries VAE latents (img2img, first-frame, etc.).
``ImageEmbedCondition`` carries ViT-style patch embeddings (SigLIP / CLIP
vision tower output, AR-emitted-image-token re-embeddings). Other roles
(``ImageMaskedLatentCondition``, ``ImageTokenCondition``) remain deferred
to first consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch

from unirl.distributed.tensor.batch import FieldKind, field, shared_field
from unirl.types.conditions.base import Condition, Modality


@dataclass
class ImageLatentCondition(Condition):
    """Image conditioning carried as VAE latents (img2img, first-frame, etc.)."""

    modality: ClassVar[Modality] = Modality.IMAGE

    latents: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)


@dataclass
class ImageEmbedCondition(Condition):
    """Image conditioning carried as ViT-style patch embeddings.

    First consumer is HunyuanImage 3.0 (SigLIP2 ViT for i2t/it2i comprehension,
    plus AR-emitted image-vocab token re-embeddings on the diffusion side).
    Same shape as ``TextEmbedCondition.embeds`` but tagged ``Modality.IMAGE``.

    ``spatial_shapes`` is a per-sample list of ``(H, W)`` patch grid sizes,
    used by ViT encoders that do dynamic positional encoding (e.g. SigLIP2).
    Optional — cross-attention models that don't need it leave it ``None``.
    """

    modality: ClassVar[Modality] = Modality.IMAGE

    embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    attn_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    spatial_shapes: Optional[List[Tuple[int, int]]] = shared_field(default=None)


__all__ = ["ImageEmbedCondition", "ImageLatentCondition"]
