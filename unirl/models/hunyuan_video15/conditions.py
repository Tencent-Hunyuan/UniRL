"""HunyuanVideo15Conditions — typed conditions container for HunyuanVideo-1.5.

Concrete instantiation of the ``DiffusionStage[C]`` type parameter.
Diverges from the SD3 / WAN21 / Qwen-Image conditions shape because
HunyuanVideo-1.5 uses **two parallel text encoders** — a Qwen2.5-VL
MLLM stream (``text_mllm``) and a ByT5 glyph stream (``text_glyph``)
— that the transformer cross-attends to in separate attention heads
(``encoder_hidden_states`` vs ``encoder_hidden_states_2``). The CFG
negative branch carries its own pair (``negative_text_*``).

A future I2V port will add the SigLIP ``vision`` slot (already a
type-level supported :class:`ImageEmbedCondition`); v1 leaves it
``None`` and the diffusion stage emits a zero placeholder of shape
``[B, vision_num_semantic_tokens, vision_states_dim]`` per the upstream
contract.

Each ``TextEmbedCondition`` field carries the encoder's output
``embeds`` and ``attn_mask``. ``pooled`` is always ``None`` because
neither Qwen-VL nor ByT5 emits a pooled vector — the transformer reads
token-level hidden states only.

Pairs ``from_dict`` / ``to_dict`` mirroring SD3 / Qwen-Image for
round-tripping between the typed form (used inside the pipeline at
stage call sites) and the generic ``Conditions = Dict[str, Condition]``
shape on ``RolloutResp``. Keys emitted: ``text_mllm`` /
``text_glyph`` / (optional) ``negative_text_mllm`` /
``negative_text_glyph`` / (optional) ``vision``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import (
    Condition,
    ImageEmbedCondition,
    TextEmbedCondition,
)


@dataclass
class HunyuanVideo15Conditions(Batch):
    """Typed conditions container for HunyuanVideo-1.5 diffusion."""

    text_mllm: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    text_glyph: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text_mllm: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text_glyph: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    # I2V slot — v1 always None, T2V uses the zero-placeholder path inside
    # the diffusion stage.
    vision: Optional[ImageEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "HunyuanVideo15Conditions":
        """Build from the generic ``Conditions`` dict shape.

        Validates that BOTH text streams are present (HunyuanVideo-1.5's
        transformer is dual-stream by contract; a single-stream call
        would crash at the cross-attention KV pack). The negative
        branches are optional (CFG-off) but must be present together
        when CFG is on (the diffusion stage checks at call time).
        Vision is optional (T2V-only branch).
        """
        text_mllm = d.get("text_mllm")
        text_glyph = d.get("text_glyph")
        if not isinstance(text_mllm, TextEmbedCondition):
            raise TypeError(
                f"HunyuanVideo15Conditions.from_dict: expected d['text_mllm'] "
                f"to be a TextEmbedCondition, got "
                f"{type(text_mllm).__name__ if text_mllm is not None else 'None'}"
            )
        if not isinstance(text_glyph, TextEmbedCondition):
            raise TypeError(
                f"HunyuanVideo15Conditions.from_dict: expected d['text_glyph'] "
                f"to be a TextEmbedCondition, got "
                f"{type(text_glyph).__name__ if text_glyph is not None else 'None'}"
            )
        negative_text_mllm = d.get("negative_text_mllm")
        negative_text_glyph = d.get("negative_text_glyph")
        for name, val in (
            ("negative_text_mllm", negative_text_mllm),
            ("negative_text_glyph", negative_text_glyph),
        ):
            if val is not None and not isinstance(val, TextEmbedCondition):
                raise TypeError(
                    f"HunyuanVideo15Conditions.from_dict: expected d[{name!r}] "
                    f"to be a TextEmbedCondition or absent, got {type(val).__name__}"
                )
        vision = d.get("vision")
        if vision is not None and not isinstance(vision, ImageEmbedCondition):
            raise TypeError(
                f"HunyuanVideo15Conditions.from_dict: expected d['vision'] to be "
                f"an ImageEmbedCondition or absent, got {type(vision).__name__}"
            )
        return cls(
            text_mllm=text_mllm,
            text_glyph=text_glyph,
            negative_text_mllm=negative_text_mllm,
            negative_text_glyph=negative_text_glyph,
            vision=vision,
        )

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for
        packing into ``RolloutResp.tracks["video"].conditions``.

        Emits the optional negative_* / vision keys only when populated
        so the dict shape stays minimal for CFG-off / T2V rollouts.
        """
        if self.text_mllm is None or self.text_glyph is None:
            raise ValueError(
                "HunyuanVideo15Conditions.to_dict: both text_mllm and text_glyph "
                "must be set (the transformer is dual-stream by contract)."
            )
        out: Dict[str, Condition] = {
            "text_mllm": self.text_mllm,
            "text_glyph": self.text_glyph,
        }
        if self.negative_text_mllm is not None:
            out["negative_text_mllm"] = self.negative_text_mllm
        if self.negative_text_glyph is not None:
            out["negative_text_glyph"] = self.negative_text_glyph
        if self.vision is not None:
            out["vision"] = self.vision
        return out


__all__ = ["HunyuanVideo15Conditions"]
