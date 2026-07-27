"""BooguImageConditions — typed conditions container for Boogu-Image diffusion.

Concrete instantiation of the ``DiffusionStage[C]`` type parameter.
Mirrors :class:`unirl.models.z_image.ZImageConditions`: text + optional
negative_text, both as :class:`TextEmbedCondition` instances. Boogu-Image
does not emit a ``pooled`` text vector, so ``TextEmbedCondition.pooled`` is
always ``None``; the ``attn_mask`` field carries the right-padded
chat-template token mask that the transformer's variable-length attention
consumes directly (no unpad/repack — see ``BooguImageTextEmbedStage``).

The CFG negative branch is a sibling ``negative_text`` field so the schema
is honest about which slots travel on the wire — a reader of
``Part.conditions`` sees ``"text"`` and ``"negative_text"`` as two
equal-status entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import Condition, TextEmbedCondition


@dataclass
class BooguImageConditions(Batch):
    """Typed conditions container for Boogu-Image diffusion."""

    text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "BooguImageConditions":
        """Build from the generic ``Conditions`` dict shape.

        Validates that the ``"text"`` slot is present and is a
        ``TextEmbedCondition``. The ``"negative_text"`` slot is optional;
        when absent the result has ``negative_text=None`` (CFG-off).
        """
        text = d.get("text")
        if not isinstance(text, TextEmbedCondition):
            raise TypeError(
                f"BooguImageConditions.from_dict: expected d['text'] to be a "
                f"TextEmbedCondition, got "
                f"{type(text).__name__ if text is not None else 'None'}"
            )
        negative_text = d.get("negative_text")
        if negative_text is not None and not isinstance(negative_text, TextEmbedCondition):
            raise TypeError(
                f"BooguImageConditions.from_dict: expected d['negative_text'] to be a "
                f"TextEmbedCondition or absent, got {type(negative_text).__name__}"
            )
        return cls(text=text, negative_text=negative_text)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for
        packing into ``Part.conditions``.

        Emits ``"negative_text"`` only when ``negative_text is not None``
        so the dict shape stays minimal for CFG-off rollouts.
        """
        if self.text is None:
            raise ValueError("BooguImageConditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        return out


__all__ = ["BooguImageConditions"]
