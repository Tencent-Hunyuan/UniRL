"""Construction config for the new typed WAN 2.2 T2V dual-transformer pipeline.

WAN 2.2 uses two ``WanTransformer3DModel`` instances —

- ``high_noise`` for ``sigma >= boundary_ratio`` (coarse structure)
- ``low_noise`` for ``sigma < boundary_ratio`` (detail refinement)

— exposed through a single ``WanDualTransformer`` composite. The
composite is the trainable-module surface used by LoRA injection and
FSDPPolicy block discovery. FSDPPolicy does block-only wrapping: it
recurses into both branches and fully-shards each ``WanTransformerBlock``;
the composite root remains unwrapped.

Inherits from :class:`WAN21PipelineConfig` to reuse the precision /
schedule / text-encoder / weight-sync conventions. The new fields are
WAN 2.2-specific: ``boundary_ratio`` (the sigma threshold for routing),
``guidance_scale_2`` (per-stage CFG scale for the low-noise branch),
``num_train_timesteps`` (used by future training-time helpers), and
``transformer_2_pretrained_path`` (override for the low-noise weights
when not co-located under the main checkpoint).

The :class:`WAN21PipelineConfig` schedule fields (``shift``,
``max_sequence_length``, precision, etc.) carry over unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from unirl.models.wan21.config import WAN21PipelineConfig

DEFAULT_BOUNDARY_RATIO: float = 0.875


@dataclass
class WAN22PipelineConfig(WAN21PipelineConfig):
    """Construction args for ``WAN22Pipeline.from_config``.

    Extends :class:`WAN21PipelineConfig` with the dual-transformer
    knobs. ``device`` may be runtime-injected by the actor; the other
    fields are set at compose time.
    """

    boundary_ratio: float = DEFAULT_BOUNDARY_RATIO

    guidance_scale_2: Optional[float] = None

    num_train_timesteps: int = 1000

    transformer_2_pretrained_path: Optional[str] = None

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None


__all__ = ["DEFAULT_BOUNDARY_RATIO", "WAN22PipelineConfig"]
