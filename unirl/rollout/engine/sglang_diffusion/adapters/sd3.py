"""SD3 / SDXL-family flow-match image adapter — the default path, end to end."""

from __future__ import annotations

from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter


@register_adapter("sd3")
class SD3Adapter(ImageAdapter):
    pass


__all__ = ["SD3Adapter"]
