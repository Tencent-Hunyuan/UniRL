"""Video-family adapters: Mochi + HunyuanVideo.

PARITY NOTE: the legacy ``sglang`` engine treats every family — including the
video ones — through the image path: it builds an image-form ``LatentSegment``
(``make_image_segment``) and *drops* 4-D decoded video with a warning (there is no
video reward consumer yet). These adapters reproduce that behavior exactly so the
per-family parity gate holds.

Proper video output (a ``video`` track, ``make_video_segment``, and ``Videos``
decoded via ``from_list``) is a deliberate follow-up — it would diverge from the
current engine, needs the packed/ragged ``Videos`` wiring, and has no parity
baseline until a video reward consumer lands. When that happens, give these a
``VideoAdapter`` base that overrides ``segment_factory`` / ``build_decoded`` /
``track_name``.
"""

from __future__ import annotations

from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter


@register_adapter("mochi")
class MochiAdapter(ImageAdapter):
    """Mochi — image-path parity (see module note); proper video output is a follow-up."""

    pass


@register_adapter("hunyuan_video")
class HunyuanVideoAdapter(ImageAdapter):
    """HunyuanVideo — image-path parity (see module note); proper video output is a follow-up."""

    pass


__all__ = ["MochiAdapter", "HunyuanVideoAdapter"]
