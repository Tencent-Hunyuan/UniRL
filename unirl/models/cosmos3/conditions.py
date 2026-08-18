"""Typed supervised conditions for Cosmos3's packed omnimodal stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from unirl.distributed.tensor.batch import FieldKind, field, packed_field
from unirl.types.conditions import Condition, Modality


@dataclass
class Cosmos3SFTCondition(Condition):
    """Per-sample packed text tokens + optional action targets beside a clean video latent (README.md)."""

    modality: ClassVar[Modality] = Modality.MULTIMODAL

    input_ids: Optional[torch.Tensor] = packed_field(default=None)
    fps: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    flow_shifts: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    actions: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)


__all__ = ["Cosmos3SFTCondition"]
