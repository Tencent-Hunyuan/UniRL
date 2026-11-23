"""TextSegment — SoA token container; every packed field is ``[total_tokens]`` along dim 0."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Callable, ClassVar, Optional

import torch

from unirl.distributed.tensor.batch import FieldKind, packed_field
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
    teacher_log_probs: Optional[torch.Tensor] = packed_field(default=None)
    loss_mask: Optional[torch.Tensor] = packed_field(default=None)
    values: Optional[torch.Tensor] = packed_field(default=None)
    returns: Optional[torch.Tensor] = packed_field(default=None)
    token_advantages: Optional[torch.Tensor] = packed_field(default=None)

    def pad_empty_rows(self) -> None:
        """Insert one masked token per empty row for FSDP lockstep; see ``../../train/readme.md``."""
        cu = self.cu_seqlens
        if self.tokens is None or cu is None:
            return
        lengths = (cu[1:] - cu[:-1]).tolist()
        if 0 not in lengths:
            return
        if self.loss_mask is None:
            self.loss_mask = torch.ones_like(self.tokens, dtype=torch.float32)
        for field in fields(self):
            if field.metadata.get("kind") is not FieldKind.PACKED:
                continue
            value = getattr(self, field.name)
            if value is not None:
                rows = value.split(lengths)
                setattr(
                    self,
                    field.name,
                    torch.cat([row if n else value.new_zeros((1, *value.shape[1:])) for row, n in zip(rows, lengths)]),
                )
        padded_lengths = cu.new_tensor([max(n, 1) for n in lengths])
        self._packed_cu_seqlens = torch.nn.functional.pad(padded_lengths.cumsum(0, dtype=cu.dtype), (1, 0))

    def as_condition_with(self, encoder: Callable[..., Any]) -> Condition:
        """Re-embed packed tokens via the supplied encoder into a TextEmbedCondition."""
        if self.tokens is None:
            raise ValueError("TextSegment.as_condition_with: tokens is None")
        out = encoder(self.tokens)
        embeds = getattr(out, "embeds", out)
        return TextEmbedCondition(embeds=embeds)


__all__ = ["TextSegment"]
