"""SD3 / SDXL-family flow-match image adapter — the default path, end to end.

Generic schedule policy (``model_config.shift``), ``transformer.`` LoRA prefix,
SDE label from the strategy, 5-D passthrough trajectory, ``Images`` decode, and
``text`` + ``negative_text`` condition fusion all come from ``ImageAdapter``;
nothing here needs overriding.
"""

from __future__ import annotations

from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter


@register_adapter("sd3")
class SD3Adapter(ImageAdapter):
    pass


__all__ = ["SD3Adapter"]
