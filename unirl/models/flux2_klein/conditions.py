"""Flux2KleinConditions — typed conditions container for the Klein diffusion stage.

Concrete instantiation of the ``DiffusionStage[C]`` type parameter.
Mirrors :class:`unirl.models.sd3.SD3Conditions` and
:class:`unirl.models.qwen_image.QwenImageConditions`: text +
optional ``negative_text``, both as :class:`TextEmbedCondition`
instances. FLUX.2-klein's text encoder is the single Qwen3 LLM (no
CLIP-style pooled output is consumed by the transformer, but the
encoder still produces a pooled vector for API symmetry with FLUX.2-dev;
the Klein transformer ignores ``pooled_projections`` entirely).

The CFG negative branch is split into a sibling ``negative_text``
field (rather than nested under ``text.negative``) so the schema is
honest about which slots travel on the wire — a reader of
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
class Flux2KleinConditions(Batch):
    """Typed conditions container for FLUX.2-klein-9B diffusion."""

    text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "Flux2KleinConditions":
        """Build from the generic ``Conditions`` dict shape.

        Validates that the ``"text"`` slot is present and is a
        :class:`TextEmbedCondition`. The ``"negative_text"`` slot is
        optional; when absent the result has ``negative_text=None``
        (CFG-off, the canonical Klein recipe with
        ``guidance_scale=1.0``).
        """
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
        return cls(text=text, negative_text=negative_text)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for
        packing into ``RolloutResp.tracks["image"].conditions``.

        Emits ``"negative_text"`` only when ``negative_text is not None``
        so the dict shape stays minimal for CFG-off rollouts.
        """
        if self.text is None:
            raise ValueError("Flux2KleinConditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        return out


__all__ = ["Flux2KleinConditions"]
