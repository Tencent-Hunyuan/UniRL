"""WAN21CLIPVisionEncodeStage — Images → ImageEmbedCondition via CLIP vision.

Mirrors diffusers ``pipelines/wan/pipeline_wan_i2v.py`` CLIP encoding:
preprocess via ``CLIPImageProcessor``, run ``CLIPVisionModel`` with
``output_hidden_states=True``, then read the **penultimate** hidden
state (``hidden_states[-2]``) as the patch-token embedding stream.

This stage is constructed only when ``bundle.uses_clip_vision`` is
``True`` (i.e. the transformer checkpoint declares ``image_dim > 0``).
WAN 2.2's mainstream checkpoints set ``image_dim == 0`` and skip the
stage entirely; the WAN 2.1 I2V 14B/720P family declares ``image_dim``
and triggers it.

The output ``ImageEmbedCondition`` carries:

- ``embeds: [B, num_patches, dim]`` — penultimate hidden state.
- ``attn_mask: [B, num_patches]`` — all-ones long tensor (CLIP ViT
  encodes the full patch grid without padding).

The diffusion stage forwards ``embeds`` to the transformer as
``encoder_hidden_states_image=…`` (with CFG batch-doubling when
``guidance_scale > 1.0``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from unirl.models.types.codec import EncodeStage
from unirl.types.conditions import ImageEmbedCondition
from unirl.types.primitives import Images


@runtime_checkable
class _VisionBundle(Protocol):
    """Structural Protocol for bundles that own a CLIP vision tower.

    Both :class:`WAN21Bundle` (when ``uses_clip_vision`` is true) and
    structurally-compatible bundles satisfy this. WAN 2.2 does NOT load
    CLIP by default, but the same stage could be reused if a 2.2 I2V
    variant with ``image_dim > 0`` ever ships.
    """

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

        # Penultimate hidden state — matches diffusers WAN I2V upstream
        # (``pipeline_wan_i2v.py::encode_image``); using ``last_hidden_state``
        # would shift the embedding distribution the transformer was
        # trained against.
        embeds = out.hidden_states[-2]
        attn_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
        return ImageEmbedCondition(embeds=embeds, attn_mask=attn_mask)


__all__ = ["WAN21CLIPVisionEncodeStage"]
