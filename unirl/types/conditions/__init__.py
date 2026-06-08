"""Composable conditioning types for the four-tier pipeline.

A ``Condition`` is the typed encoded input that an architecture's diffusion
or AR stage consumes. Concrete subclasses (``TextEmbedCondition``,
``ImageLatentCondition``, …) cover one modality + role each. Architectures
declare which condition slots they accept in their stage signature.

The aggregate input type is::

    Conditions = Dict[str, Condition]

Free-form string keys for slot names (``"text"``, ``"image_grid"``, …);
typed wrapper deferred until first omni-bundle consumer.
"""

from __future__ import annotations

from typing import Dict

from unirl.types.conditions.base import Condition, Modality
from unirl.types.conditions.fused_multimodal import FusedMultimodalCondition
from unirl.types.conditions.image import ImageEmbedCondition, ImageLatentCondition
from unirl.types.conditions.text import TextEmbedCondition, TextTokenCondition

Conditions = Dict[str, Condition]


__all__ = [
    "Condition",
    "Conditions",
    "FusedMultimodalCondition",
    "ImageEmbedCondition",
    "ImageLatentCondition",
    "Modality",
    "TextEmbedCondition",
    "TextTokenCondition",
]
