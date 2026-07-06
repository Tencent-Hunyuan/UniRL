"""QwenImageEditPlusConditions — typed conditions for Qwen-Image-Edit-Plus.

Mirrors :class:`unirl.models.qwen_image.QwenImageConditions` (same ``text`` +
``negative_text`` slots) and adds one slot: ``image_latent: ImageLatentCondition``
carrying the VAE-encoded source image. The diffusion step packs it (2×2
channel-pack, same as the noise latent) and concatenates along the token
dimension before the transformer call, then slices the prediction back to the
noise segment — mirrors ``vde_editplus.py:232,246`` and the FLUX.2-Klein
image-edit pattern (``flux2_klein/diffusion.py:160-183``).

``image_latent`` is optional so the conditions container round-trips cleanly
for the T2I degenerate path (no source image); the Edit-Plus pipeline itself
requires a source image and raises at ``generate(req)`` time if absent
(fail-fast, constraint #27).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import Condition, ImageLatentCondition, TextEmbedCondition


@dataclass
class QwenImageEditPlusConditions(Batch):
    """Typed conditions container for Qwen-Image-Edit-Plus diffusion."""

    text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    image_latent: Optional[ImageLatentCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "QwenImageEditPlusConditions":
        """Build from the generic ``Conditions`` dict shape.

        ``"text"`` is required and must be a ``TextEmbedCondition``.
        ``"negative_text"`` and ``"image_latent"`` are optional; when absent
        the corresponding field is ``None`` (CFG-off / T2I degenerate path).
        """
        text = d.get("text")
        if not isinstance(text, TextEmbedCondition):
            raise TypeError(
                f"QwenImageEditPlusConditions.from_dict: expected d['text'] to be a "
                f"TextEmbedCondition, got "
                f"{type(text).__name__ if text is not None else 'None'}"
            )
        negative_text = d.get("negative_text")
        if negative_text is not None and not isinstance(negative_text, TextEmbedCondition):
            raise TypeError(
                f"QwenImageEditPlusConditions.from_dict: expected d['negative_text'] to be a "
                f"TextEmbedCondition or absent, got {type(negative_text).__name__}"
            )
        image_latent = d.get("image_latent")
        if image_latent is not None and not isinstance(image_latent, ImageLatentCondition):
            raise TypeError(
                f"QwenImageEditPlusConditions.from_dict: expected d['image_latent'] to be an "
                f"ImageLatentCondition or absent, got {type(image_latent).__name__}"
            )
        return cls(text=text, negative_text=negative_text, image_latent=image_latent)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape.

        Emits ``"negative_text"`` / ``"image_latent"`` only when not ``None``
        so the dict shape stays minimal for CFG-off / T2I-degenerate rollouts.
        """
        if self.text is None:
            raise ValueError("QwenImageEditPlusConditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        if self.image_latent is not None:
            out["image_latent"] = self.image_latent
        return out


__all__ = ["QwenImageEditPlusConditions"]
