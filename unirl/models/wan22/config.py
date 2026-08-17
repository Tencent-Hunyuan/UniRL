"""Construction config for the new typed WAN 2.2 T2V dual-transformer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from unirl.models.wan21.config import WAN21PipelineConfig

DEFAULT_BOUNDARY_RATIO: float = 0.875


@dataclass
class WAN22PipelineConfig(WAN21PipelineConfig):
    """Construction args for ``WAN22Pipeline.from_config``."""

    boundary_ratio: float = DEFAULT_BOUNDARY_RATIO

    guidance_scale_2: Optional[float] = None

    num_train_timesteps: int = 1000

    transformer_2_pretrained_path: Optional[str] = None

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None


__all__ = ["DEFAULT_BOUNDARY_RATIO", "WAN22PipelineConfig"]
