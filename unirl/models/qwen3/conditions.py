"""Qwen3ARConditions — typed conditions container for the Qwen3 AR stage.

Concrete instantiation of the ``ARStage[C]`` type parameter. Mirrors
:class:`unirl.models.qwen_image.QwenImageConditions` in shape:
a single typed slot (``prompt``) carrying a :class:`TextTokenCondition`
with the chat-template-built ``input_ids`` + ``attention_mask``.

The ``TextTokenCondition`` (declared in
:mod:`unirl.types.conditions.text`) is the canonical
pre-encoder-text condition for unified-vocab models — Qwen3's transformer
owns its own embedding table and consumes ``input_ids`` directly, so this
is the right wire format. The chat-template stage produces it; the AR
stage's ``autoregress`` / ``replay`` read it.

Pairs ``from_dict`` / ``to_dict`` for round-tripping between the typed
form (used inside the pipeline at stage call sites) and the generic
``Conditions = Dict[str, Condition]`` shape on ``RolloutResp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import Condition, TextTokenCondition


@dataclass
class Qwen3ARConditions(Batch):
    """Typed conditions container for the Qwen3 AR stage."""

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "Qwen3ARConditions":
        """Build from the generic ``Conditions`` dict shape.

        Validates that the ``"prompt"`` slot is present and is a
        ``TextTokenCondition``.
        """
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                f"Qwen3ARConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got "
                f"{type(prompt).__name__ if prompt is not None else 'None'}"
            )
        return cls(prompt=prompt)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for
        packing into ``RolloutResp.tracks["ar"].conditions``.
        """
        if self.prompt is None:
            raise ValueError("Qwen3ARConditions.to_dict: prompt field is None")
        return {"prompt": self.prompt}


__all__ = ["Qwen3ARConditions"]
