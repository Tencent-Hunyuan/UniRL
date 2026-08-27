"""Condition base class and Modality enum."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from unirl.distributed.tensor.batch import Batch


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class Condition(Batch):
    """Marker base for conditioning inputs."""

    modality: ClassVar[Modality]


__all__ = ["Condition", "Modality"]
