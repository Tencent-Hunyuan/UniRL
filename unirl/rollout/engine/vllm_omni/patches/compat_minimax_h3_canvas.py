"""Let MiniMax-H3 serve a canvas below its released 768 short edge, as the trainside path already does."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SENTINEL = "_unirl_h3_canvas_short_edge"


def patch_minimax_h3_canvas_short_edge() -> None:
    """Drop the short-edge equality in ``_resolve_output_canvas``, keeping its ratio and area policy."""
    try:
        from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as module
    except ImportError:
        return

    if getattr(module, _SENTINEL, False):
        return

    align = module._align_multiple
    max_pixels = module.MINIMAX_H3_OUTPUT_MAX_PIXELS
    error_cls = module.OmniClientError

    def _resolve_output_canvas(aspect_ratio: float, short_edge: int) -> tuple[int, int]:
        """Resolve the official H3 ratio/area policy to a 32-pixel canvas at any short edge."""
        ratio = float(aspect_ratio)
        if ratio != ratio or ratio in (float("inf"), float("-inf")) or ratio <= 0:
            raise error_cls(f"MiniMax H3 canvas aspect ratio must be positive, got {aspect_ratio!r}")
        edge = int(short_edge)
        if edge <= 0 or edge % 32:
            raise error_cls(f"MiniMax H3 short_edge must be a positive multiple of 32, got {short_edge!r}")
        if ratio >= 1.0:
            width, height = float(edge) * ratio, float(edge)
        else:
            width, height = float(edge), float(edge) / ratio
        area = width * height
        if area > max_pixels:
            scale = (max_pixels / area) ** 0.5
            width *= scale
            height *= scale
        return align(height, 32), align(width, 32)

    module._resolve_output_canvas = _resolve_output_canvas
    setattr(module, _SENTINEL, True)
    logger.info(
        "Patched MiniMax-H3 _resolve_output_canvas: short edge no longer pinned to %d",
        module.MINIMAX_H3_OUTPUT_SHORT_EDGE,
    )


__all__ = ["patch_minimax_h3_canvas_short_edge"]
