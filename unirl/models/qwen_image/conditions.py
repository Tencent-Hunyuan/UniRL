"""QwenImageConditions — typed conditions container for Qwen-Image diffusion.

Concrete instantiation of the ``DiffusionStage[C]`` type parameter.
Mirrors :class:`unirl.models.sd3.SD3Conditions` deliberately:
text + optional negative_text, both as :class:`TextEmbedCondition`
instances. Qwen-Image does not emit a ``pooled`` text vector (unlike
SD3), so its ``TextEmbedCondition.pooled`` is always ``None``; the
``attn_mask`` field carries the Qwen-VL prompt mask
(``prompt_embeds_mask`` in the legacy sampler).

The CFG negative branch is split into a sibling ``negative_text`` field
(rather than nested under ``text.negative``) so the schema is honest
about which slots travel on the wire — a reader of
``RolloutResp.tracks["image"].conditions`` sees ``"text"`` and ``"negative_text"`` as
two equal-status entries.

Pairs ``from_dict`` / ``to_dict`` for round-tripping between the typed
form (used inside the pipeline at stage call sites) and the generic
``Conditions = Dict[str, Condition]`` shape on ``RolloutResp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import Condition, TextEmbedCondition


@dataclass
class QwenImageConditions(Batch):
    """Typed conditions container for Qwen-Image diffusion."""

    text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "QwenImageConditions":
        """Build from the generic ``Conditions`` dict shape.

        Validates that the ``"text"`` slot is present and is a
        ``TextEmbedCondition``. The ``"negative_text"`` slot is optional;
        when absent the result has ``negative_text=None`` (CFG-off).
        """
        text = d.get("text")
        if not isinstance(text, TextEmbedCondition):
            raise TypeError(
                f"QwenImageConditions.from_dict: expected d['text'] to be a "
                f"TextEmbedCondition, got "
                f"{type(text).__name__ if text is not None else 'None'}"
            )
        negative_text = d.get("negative_text")
        if negative_text is not None and not isinstance(negative_text, TextEmbedCondition):
            raise TypeError(
                f"QwenImageConditions.from_dict: expected d['negative_text'] to be a "
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
            raise ValueError("QwenImageConditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        return out


__all__ = ["QwenImageConditions"]
