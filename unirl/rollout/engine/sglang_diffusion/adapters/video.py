"""Video-family adapters: Mochi + HunyuanVideo.

These families still follow the legacy SGLang image path in this PR: build an
image-form ``LatentSegment`` and drop 4-D decoded samples. Do not squeeze
``[C, T=1, H, W]`` here — it can be a real single-frame video.
"""

from __future__ import annotations

from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter


@register_adapter("mochi")
class MochiAdapter(ImageAdapter):
    """Mochi — legacy image-path parity until verified video output lands."""

    squeeze_single_frame_4d = False


@register_adapter("hunyuan_video")
class HunyuanVideoAdapter(ImageAdapter):
    """HunyuanVideo — legacy image-path parity until verified video output lands."""

    squeeze_single_frame_4d = False


__all__ = ["MochiAdapter", "HunyuanVideoAdapter"]
