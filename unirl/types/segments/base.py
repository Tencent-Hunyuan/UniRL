"""Segment base class and SegmentStatus enum."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ClassVar, Optional

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions.base import Condition, Modality


class SegmentStatus(int, Enum):
    PENDING = 0
    COMPLETED = 1
    TRUNCATED = 2
    ABORTED = 3


@dataclass
class Segment(Batch):
    """SoA container for one modality's outputs; row ``k`` is track sample ``k``, per-step fields ``[N_segs, S]``."""

    modality: ClassVar[Modality]

    status: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    def as_condition(self) -> Optional[Condition]:
        """Encoder-free promotion. Override on subclasses where it makes sense."""
        return None

    def as_condition_with(self, encoder: Callable[..., Any]) -> Condition:
        """Encoder-mediated promotion (e.g. token → embedding)."""
        raise NotImplementedError(f"{type(self).__name__}.as_condition_with(encoder) is not implemented")


__all__ = ["Segment", "SegmentStatus"]
