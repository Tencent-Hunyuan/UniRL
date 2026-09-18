"""FusedMultimodalCondition — token sequence over a unified text+image vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch

from unirl.distributed.tensor.batch import FieldKind, field, shared_field
from unirl.types.conditions.base import Condition, Modality


@dataclass
class FusedMultimodalCondition(Condition):
    """Generic fused-multimodal-sequence input — ``input_ids [B, L]``, ``attention_mask [B, 1, L, L]``."""

    modality: ClassVar[Modality] = Modality.MULTIMODAL

    input_ids: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    attention_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    position_ids: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = shared_field(default=None)


__all__ = ["FusedMultimodalCondition"]
