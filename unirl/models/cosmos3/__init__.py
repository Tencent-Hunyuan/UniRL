"""Cosmos3 (NVIDIA omnimodal world model) SFT support.

CPU-importable package top: the diffusers>=0.39 runtime imports live inside
``bundle.py``/``sft_task.py`` bodies.
"""

from unirl.models.cosmos3.config import Cosmos3SFTConfig

__all__ = ["Cosmos3SFTConfig"]
