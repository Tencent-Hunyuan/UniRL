"""ZImageConditions — typed conditions container for Z-Image diffusion.

Concrete instantiation of the ``DiffusionStage[C]`` type parameter.
Mirrors :class:`unirl.models.sd3.SD3Conditions` /
:class:`unirl.models.qwen_image.QwenImageConditions`: text + optional
negative_text, both as :class:`TextEmbedCondition` instances. Z-Image
does not emit a ``pooled`` text vector, so ``TextEmbedCondition.pooled``
is always ``None``; the ``attn_mask`` field carries the per-prompt valid
token mask used to rebuild the variable-length caption list the
single-stream transformer consumes.

The CFG negative branch is split into a sibling ``negative_text`` field
(rather than nested under ``text.negative``) so the schema is honest
about which slots travel on the wire — a reader of
``RolloutResp.tracks["image"].conditions`` sees ``"text"`` and
``"negative_text"`` as two equal-status entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import Condition, TextEmbedCondition


@dataclass
class ZImageConditions(Batch):
    """Typed conditions container for Z-Image diffusion."""

    text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "ZImageConditions":
        """Build from the generic ``Conditions`` dict shape.

        Validates that the ``"text"`` slot is present and is a
        ``TextEmbedCondition``. The ``"negative_text"`` slot is optional;
        when absent the result has ``negative_text=None`` (CFG-off).
        """
        text = d.get("text")
        if not isinstance(text, TextEmbedCondition):
            raise TypeError(
                f"ZImageConditions.from_dict: expected d['text'] to be a "
                f"TextEmbedCondition, got "
                f"{type(text).__name__ if text is not None else 'None'}"
            )
        negative_text = d.get("negative_text")
        if negative_text is not None and not isinstance(negative_text, TextEmbedCondition):
            raise TypeError(
                f"ZImageConditions.from_dict: expected d['negative_text'] to be a "
                f"TextEmbedCondition or absent, got {type(negative_text).__name__}"
            )
        return cls(text=text, negative_text=negative_text)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for
        packing into ``RolloutResp.tracks["image"].conditions``.

        Emits ``"negative_text"`` only when ``negative_text is not None``
        so the dict shape stays minimal for CFG-off rollouts.
        """
        if self.text is None:
            raise ValueError("ZImageConditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        return out


__all__ = ["ZImageConditions"]
