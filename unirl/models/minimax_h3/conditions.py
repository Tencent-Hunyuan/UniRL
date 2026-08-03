"""MiniMax-H3 conditions -- typed container for the diffusion stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.conditions import TextEmbedCondition


@dataclass
class MiniMaxH3Conditions(Batch):
    """Conditions passed to the MiniMax-H3 diffusion stage.

    Deliberately one slot. MiniMax-H3's checkpoint is guidance-distilled: there
    is no CFG, no unconditional pass and no negative prompt, so -- unlike every
    other video model in this repo -- there is no ``negative_text`` twin here.
    A recipe that sets ``guidance_scale`` is not doing anything; the stage
    ignores it rather than silently batching a second forward.

    Slots:
        text: Qwen3-VL hidden states from layer 50, unnormalized.
    """

    text: Optional[TextEmbedCondition] = concat_field(default=None)

    @classmethod
    def from_dict(cls, d: dict) -> "MiniMaxH3Conditions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {name: value for name in self.__dataclass_fields__ if (value := getattr(self, name)) is not None}


__all__ = ["MiniMaxH3Conditions"]
