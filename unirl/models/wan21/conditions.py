"""WAN21Conditions — typed conditions container for WAN 2.1 T2V / I2V.

Concrete instantiation of the ``DiffusionStage[C]`` type parameter. WAN
2.1 consumes a text-conditioning slot with an explicit CFG negative
branch, plus an optional ``image_latent`` slot carrying the
mask+first-frame VAE latents for I2V (20-channel payload produced by
``WAN21ImageLatentEncodeStage``).

The CFG negative branch is split into a sibling ``negative_text`` field
(rather than nested under ``text.negative``) so the schema is honest
about which slots travel on the wire — a reader of
``RolloutResp.tracks["video"].conditions`` sees ``"text"`` and ``"negative_text"`` as
two equal-status entries. This matches the SD3 convention exactly.

WAN uses UMT5 (single encoder) so ``TextEmbedCondition.pooled`` is
unused (always ``None``); ``TextEmbedCondition.attn_mask`` is also
unused at the diffusion stage because ``WAN21TextEmbedStage`` already
zeros out padded positions before storing ``embeds``.

Pairs ``from_dict`` / ``to_dict`` for round-tripping between the typed
form (used inside the pipeline at stage call sites) and the generic
``Conditions = Dict[str, Condition]`` shape on ``RolloutResp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import (
    Condition,
    ImageEmbedCondition,
    ImageLatentCondition,
    TextEmbedCondition,
)


@dataclass
class WAN21Conditions(Batch):
    """Typed conditions container for WAN 2.1 T2V / I2V diffusion."""

    text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    image_latent: Optional[ImageLatentCondition] = field(kind=FieldKind.CONCAT, default=None)
    image_embed: Optional[ImageEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "WAN21Conditions":
        """Build from the generic ``Conditions`` dict shape.

        Validates that the ``"text"`` slot is present and is a
        ``TextEmbedCondition``. The ``"negative_text"``,
        ``"image_latent"`` and ``"image_embed"`` slots are optional.
        """
        text = d.get("text")
        if not isinstance(text, TextEmbedCondition):
            raise TypeError(
                f"WAN21Conditions.from_dict: expected d['text'] to be a TextEmbedCondition, "
                f"got {type(text).__name__ if text is not None else 'None'}"
            )
        negative_text = d.get("negative_text")
        if negative_text is not None and not isinstance(negative_text, TextEmbedCondition):
            raise TypeError(
                f"WAN21Conditions.from_dict: expected d['negative_text'] to be a "
                f"TextEmbedCondition or absent, got {type(negative_text).__name__}"
            )
        image_latent = d.get("image_latent")
        if image_latent is not None and not isinstance(image_latent, ImageLatentCondition):
            raise TypeError(
                f"WAN21Conditions.from_dict: expected d['image_latent'] to be an "
                f"ImageLatentCondition or absent, got {type(image_latent).__name__}"
            )
        image_embed = d.get("image_embed")
        if image_embed is not None and not isinstance(image_embed, ImageEmbedCondition):
            raise TypeError(
                f"WAN21Conditions.from_dict: expected d['image_embed'] to be an "
                f"ImageEmbedCondition or absent, got {type(image_embed).__name__}"
            )
        return cls(
            text=text,
            negative_text=negative_text,
            image_latent=image_latent,
            image_embed=image_embed,
        )

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for
        packing into ``RolloutResp.tracks["video"].conditions``.

        Emits optional slots only when non-``None`` so the dict stays
        minimal for T2V / CFG-off rollouts.
        """
        if self.text is None:
            raise ValueError("WAN21Conditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        if self.image_latent is not None:
            out["image_latent"] = self.image_latent
        if self.image_embed is not None:
            out["image_embed"] = self.image_embed
        return out


__all__ = ["WAN21Conditions"]
