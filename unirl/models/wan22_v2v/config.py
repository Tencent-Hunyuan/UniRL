"""Construction config for the WAN 2.2 video-to-video (V2V) pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from unirl.models.wan22.config import WAN22PipelineConfig

DEFAULT_V2V_STRENGTH: float = 0.8


@dataclass
class WAN22V2VPipelineConfig(WAN22PipelineConfig):
    """WAN 2.2 config plus the V2V denoising strength default."""

    strength: float = DEFAULT_V2V_STRENGTH


__all__ = ["DEFAULT_V2V_STRENGTH", "WAN22V2VPipelineConfig"]
