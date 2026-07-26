"""Typed supervised conditions for Cosmos3's packed omnimodal stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from unirl.distributed.tensor.batch import FieldKind, field, packed_field
from unirl.types.conditions import Condition, Modality


@dataclass
class Cosmos3SFTCondition(Condition):
    """Per-sample text/action inputs that accompany a clean video latent.

    Token IDs are packed because Cosmos3 prompts are variable length. The clean
    video targets live in the track's ``LatentSegment``; optional action targets
    stay here because the base segment has only one latent stream.
    """

    modality: ClassVar[Modality] = Modality.MULTIMODAL

    input_ids: Optional[torch.Tensor] = packed_field(default=None)
    fps: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    flow_shifts: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    actions: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)


__all__ = ["Cosmos3SFTCondition"]
