"""Composable conditioning types for the four-tier pipeline."""

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
