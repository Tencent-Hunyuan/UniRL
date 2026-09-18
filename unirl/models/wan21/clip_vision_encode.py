"""WAN21CLIPVisionEncodeStage — packed Images → CLIP vision condition."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from unirl.models.types.codec import EncodeStage
from unirl.types.conditions import ImageEmbedCondition
from unirl.types.primitives import Images


@runtime_checkable
class _VisionBundle(Protocol):
    """Structural Protocol for bundles that own a CLIP vision tower."""

    vision_encoder: Any
    image_processor: Any
    device: torch.device
    dtype: torch.dtype


class WAN21CLIPVisionEncodeStage(EncodeStage[Images, ImageEmbedCondition]):
    """Encode reference images through CLIP ViT into patch-token embeds."""

    def __init__(self, bundle: _VisionBundle) -> None:
        if bundle.vision_encoder is None or bundle.image_processor is None:
            raise ValueError(
                "WAN21CLIPVisionEncodeStage: bundle.vision_encoder / image_processor "
                "is None — this stage requires an I2V bundle "
                "(transformer.config.image_dim > 0). Check `bundle.uses_clip_vision` "
                "before constructing this stage."
            )
        self.bundle = bundle

    def encode(self, p: Images) -> ImageEmbedCondition:
        if not isinstance(p, Images):
            raise TypeError(f"WAN21CLIPVisionEncodeStage.encode: expected Images, got {type(p).__name__}")

        pils = p.to_pils()
        processed = self.bundle.image_processor(images=pils, return_tensors="pt").pixel_values
        processed = processed.to(device=self.bundle.device, dtype=self.bundle.dtype)

        with torch.no_grad():
            out = self.bundle.vision_encoder(processed, output_hidden_states=True)

        embeds = out.hidden_states[-2]
        attn_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        return ImageEmbedCondition(embeds=embeds, attn_mask=attn_mask)


__all__ = ["WAN21CLIPVisionEncodeStage"]
