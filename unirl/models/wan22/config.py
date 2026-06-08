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
from typing import Optional

from unirl.models.wan21.config import WAN21PipelineConfig

# WAN 2.2 reference default: 0.875. Matches
# ``models/wan22.py::DEFAULT_BOUNDARY_RATIO``.
DEFAULT_BOUNDARY_RATIO: float = 0.875


@dataclass
class WAN22PipelineConfig(WAN21PipelineConfig):
    """Construction args for ``WAN22Pipeline.from_config``.

    Extends :class:`WAN21PipelineConfig` with the dual-transformer
    knobs. ``device`` may be runtime-injected by the actor; the other
    fields are set at compose time.
    """

    # Sigma threshold for high-noise → low-noise transformer switching.
    # When sigma >= boundary_ratio: route to ``high_noise``;
    # else: route to ``low_noise``.
    boundary_ratio: float = DEFAULT_BOUNDARY_RATIO

    # Optional per-stage CFG scale for the low-noise branch. When None,
    # the low-noise branch reuses ``DiffusionParams.guidance_scale``.
    guidance_scale_2: Optional[float] = None

    # Used by training-time helpers / scheduler initialization.
    num_train_timesteps: int = 1000

    # Override path for the low-noise transformer when it is NOT
    # co-located under the main pretrained path's ``transformer_2``
    # subfolder. Defaults to ``None`` → fall back to
    # ``pretrained_model_ckpt_path/transformer_2``.
    transformer_2_pretrained_path: Optional[str] = None


__all__ = ["DEFAULT_BOUNDARY_RATIO", "WAN22PipelineConfig"]
