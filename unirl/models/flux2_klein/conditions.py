"""Flux2KleinConditions — typed conditions container for the Klein diffusion stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import Condition, ImageLatentCondition, TextEmbedCondition


@dataclass
class Flux2KleinConditions(Batch):
    """Typed conditions container for FLUX.2-klein-9B diffusion."""

    text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    image_latent: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    image_latent_ids: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "Flux2KleinConditions":
        """Build from the generic ``Conditions`` dict shape."""
        text = d.get("text")
        if not isinstance(text, TextEmbedCondition):
            raise TypeError(
                f"Flux2KleinConditions.from_dict: expected d['text'] to be a "
                f"TextEmbedCondition, got "
                f"{type(text).__name__ if text is not None else 'None'}"
            )
        negative_text = d.get("negative_text")
        if negative_text is not None and not isinstance(negative_text, TextEmbedCondition):
            raise TypeError(
                f"Flux2KleinConditions.from_dict: expected d['negative_text'] to be a "
                f"TextEmbedCondition or absent, got {type(negative_text).__name__}"
            )
        image_cond = d.get("image_latent")
        image_latent = None
        image_latent_ids = None
        if image_cond is not None:
            if not isinstance(image_cond, ImageLatentCondition):
                raise TypeError(
                    f"Flux2KleinConditions.from_dict: expected d['image_latent'] to be an "
                    f"ImageLatentCondition or absent, got {type(image_cond).__name__}"
                )
            image_latent = image_cond.latents
            ids_cond = d.get("image_latent_ids")
            image_latent_ids = ids_cond.latents if isinstance(ids_cond, ImageLatentCondition) else None
        return cls(
            text=text,
            negative_text=negative_text,
            image_latent=image_latent,
            image_latent_ids=image_latent_ids,
        )

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for packing into ``Part.conditions``."""
        if self.text is None:
            raise ValueError("Flux2KleinConditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        if self.image_latent is not None:
            out["image_latent"] = ImageLatentCondition(latents=self.image_latent)
            if self.image_latent_ids is not None:
                out["image_latent_ids"] = ImageLatentCondition(latents=self.image_latent_ids)
        return out


__all__ = ["Flux2KleinConditions"]
