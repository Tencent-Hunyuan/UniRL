"""HunyuanVideoConditions -- typed conditions container for HunyuanVideo-1.0.

Concrete instantiation of the ``DiffusionStage[C]`` type parameter.
Diverges from HunyuanVideo-1.5 because the 1.0 model uses:

- ``text_llama``: LLaMA encoder output, 3D ``[B, seq, 4096]`` + attention mask.
- ``pooled_clip``: CLIP pooled output, stored as embeds ``[B, 768]`` (no
  attention mask needed; the transformer reads it as ``pooled_projections``).

No negative variants are needed because HunyuanVideo-1.0 uses guidance
embedding (``guidance_embeds=True``) instead of classifier-free guidance.
The guidance scale is passed directly as a tensor kwarg to the transformer,
not via cond/uncond stacking.

Pairs ``from_dict`` / ``to_dict`` mirroring HV15 / SD3 for round-tripping
between the typed form (used inside the pipeline at stage call sites) and
the generic ``Conditions = Dict[str, Condition]`` shape on ``RolloutResp``.
Keys emitted: ``text_llama`` / ``pooled_clip``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import (
    Condition,
    TextEmbedCondition,
)


@dataclass
class HunyuanVideoConditions(Batch):
    """Typed conditions container for HunyuanVideo-1.0 diffusion."""

    # LLaMA encoder: 3D [B, seq, 4096] + attn_mask [B, seq].
    text_llama: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, transport=True, default=None)
    # CLIP pooled: embeds [B, 768] (attn_mask not used by transformer).
    pooled_clip: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, transport=True, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "HunyuanVideoConditions":
        """Build from the generic ``Conditions`` dict shape.

        Validates that BOTH text streams are present (HunyuanVideo-1.0's
        transformer requires both ``encoder_hidden_states`` from LLaMA
        and ``pooled_projections`` from CLIP).
        """
        text_llama = d.get("text_llama")
        pooled_clip = d.get("pooled_clip")
        if not isinstance(text_llama, TextEmbedCondition):
            raise TypeError(
                f"HunyuanVideoConditions.from_dict: expected d['text_llama'] "
                f"to be a TextEmbedCondition, got "
                f"{type(text_llama).__name__ if text_llama is not None else 'None'}"
            )
        if not isinstance(pooled_clip, TextEmbedCondition):
            raise TypeError(
                f"HunyuanVideoConditions.from_dict: expected d['pooled_clip'] "
                f"to be a TextEmbedCondition, got "
                f"{type(pooled_clip).__name__ if pooled_clip is not None else 'None'}"
            )
        return cls(
            text_llama=text_llama,
            pooled_clip=pooled_clip,
        )

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for
        packing into ``RolloutResp.conditions``.
        """
        if self.text_llama is None or self.pooled_clip is None:
            raise ValueError(
                "HunyuanVideoConditions.to_dict: both text_llama and pooled_clip "
                "must be set (the transformer requires both encoder streams)."
            )
        out: Dict[str, Condition] = {
            "text_llama": self.text_llama,
            "pooled_clip": self.pooled_clip,
        }
        return out


__all__ = ["HunyuanVideoConditions"]
