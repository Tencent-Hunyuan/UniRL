"""Image conditioning types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional

import torch

from unirl.distributed.tensor.batch import FieldKind, field
from unirl.types.conditions.base import Condition, Modality


@dataclass
class ImageLatentCondition(Condition):
    """Image conditioning carried as VAE latents (img2img, first-frame, etc.)."""

    modality: ClassVar[Modality] = Modality.IMAGE

    latents: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    def __post_init__(self) -> None:
        if isinstance(self.latents, (list, tuple)):
            raise TypeError(
                "ImageLatentCondition.latents must be a dense tensor; use the owning model's ragged condition type"
            )


@dataclass
class ImageEmbedCondition(Condition):
    """Image conditioning carried as ViT-style patch embeddings."""

    modality: ClassVar[Modality] = Modality.IMAGE

    embeds: Optional[Any] = field(kind=FieldKind.CONCAT, default=None)
    attn_mask: Optional[Any] = field(kind=FieldKind.CONCAT, default=None)
    spatial_shapes: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)


__all__ = ["ImageEmbedCondition", "ImageLatentCondition"]
