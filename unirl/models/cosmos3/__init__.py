"""Cosmos3 (NVIDIA omnimodal world model) SFT support.

CPU-importable package top: the diffusers>=0.39 runtime imports live inside
``bundle.py``/``pipeline.py`` bodies.
"""

from unirl.models.cosmos3.conditions import Cosmos3SFTCondition
from unirl.models.cosmos3.config import Cosmos3SFTConfig

__all__ = ["Cosmos3SFTCondition", "Cosmos3SFTConfig"]
