"""Stable scalar constants for the Qwen3-Omni direct-TTS contract.

Phase-1 RL treats only layer-0 codec tokens, including codec EOS, as policy
actions. Frozen MTP residual codes are replay state and Code2Wav output is a
reward observation; neither contributes an independently weighted policy
log-probability.
"""

from __future__ import annotations

# Mimi / Qwen3-Omni talker codebook groups (layer0 + 15 residual).
NUM_CODE_GROUPS: int = 16

# Code2Wav / Mimi operating rate.
AUDIO_SAMPLE_RATE: int = 24000

__all__ = [
    "AUDIO_SAMPLE_RATE",
    "NUM_CODE_GROUPS",
]
