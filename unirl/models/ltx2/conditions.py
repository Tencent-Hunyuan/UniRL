"""LTX2 conditions — typed container for diffusion stage conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.conditions import TextEmbedCondition


@dataclass
class LTX2Conditions(Batch):
    """Conditions passed to the LTX2 diffusion stage.

    Slots:
        text: Gemma3 text embeddings (via connector projection for video).
        audio_text: Gemma3 text embeddings projected for audio (T2AV only).
        negative_text: Negative prompt embeddings for CFG.
        negative_audio_text: Negative audio embeddings for CFG.
        image_latent: VAE-encoded first frame for I2V conditioning.
    """

    # Video text conditioning (connector output → video hidden dim)
    text: Optional[TextEmbedCondition] = concat_field(default=None)

    # Audio text conditioning (connector output → audio hidden dim, T2AV only)
    audio_text: Optional[TextEmbedCondition] = concat_field(default=None)

    # Negative prompts for CFG
    negative_text: Optional[TextEmbedCondition] = concat_field(default=None)
    negative_audio_text: Optional[TextEmbedCondition] = concat_field(default=None)

    # I2V conditioning: VAE-encoded first frame latents
    image_latent: Optional[torch.Tensor] = concat_field(default=None)

    @classmethod
    def from_dict(cls, d: dict) -> "LTX2Conditions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        result = {}
        for name in self.__dataclass_fields__:
            val = getattr(self, name)
            if val is not None:
                result[name] = val
        return result


__all__ = ["LTX2Conditions"]
