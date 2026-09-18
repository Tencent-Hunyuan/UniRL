"""Shared FLUX.2-Klein condition-image preprocessing."""

from typing import Any, List


def resize_condition_pils(pils: List[Any], *, height: int, width: int) -> List[Any]:
    """Resize condition PILs to the generation canvas with one canonical filter."""
    from PIL import Image

    target_size = (int(width), int(height))
    return [pil if pil.size == target_size else pil.resize(target_size, Image.Resampling.LANCZOS) for pil in pils]


__all__ = ["resize_condition_pils"]
