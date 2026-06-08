"""Condition base class and Modality enum.

A ``Condition`` is the modality-tagged, encoded input that an architecture's
diffusion / AR stage consumes. The base contributes only ``Batch``
round-trip semantics and the ``modality`` ClassVar used by generic dispatch
(e.g. ``LatentSegment.as_condition`` for promotion). Concrete subclasses
(``TextEmbedCondition``, ``ImageLatentCondition``, …) declare their payload
tensors and set the modality.

Encoder versioning, attention masks, position IDs, CFG branches — all
model-specific, all live on subclasses (or on the typed model-conditions
container, e.g. ``SD3Conditions``).
"""

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
    """Marker base for conditioning inputs.

    Subclasses declare their payload tensors and set the ``modality``
    ClassVar. The base contributes only ``Batch`` round-trip
    semantics and the modality discriminator.
    """

    modality: ClassVar[Modality]


__all__ = ["Condition", "Modality"]
