"""TextSegment — SoA token container; every packed field is ``[total_tokens]`` along dim 0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

import torch

from unirl.distributed.tensor.batch import packed_field
from unirl.types.conditions.base import Condition, Modality
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.segments.base import Segment


@dataclass
class TextSegment(Segment):
    """AR text segment with packed varlen tokens."""

    modality: ClassVar[Modality] = Modality.TEXT

    tokens: Optional[torch.Tensor] = packed_field(default=None)
    log_probs: Optional[torch.Tensor] = packed_field(default=None)
    rollout_log_probs: Optional[torch.Tensor] = packed_field(default=None)
    loss_mask: Optional[torch.Tensor] = packed_field(default=None)
    rollout_log_probs: Optional[torch.Tensor] = packed_field(default=None)
    values: Optional[torch.Tensor] = packed_field(default=None)
    returns: Optional[torch.Tensor] = packed_field(default=None)
    token_advantages: Optional[torch.Tensor] = packed_field(default=None)

    def as_condition_with(self, encoder: Callable[..., Any]) -> Condition:
        """Re-embed packed tokens via the supplied encoder into a TextEmbedCondition."""
        if self.tokens is None:
            raise ValueError("TextSegment.as_condition_with: tokens is None")
        out = encoder(self.tokens)
        embeds = getattr(out, "embeds", out)
        return TextEmbedCondition(embeds=embeds)


__all__ = ["TextSegment"]
