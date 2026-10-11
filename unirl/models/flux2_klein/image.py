"""Shared FLUX.2-Klein condition-image preprocessing."""

from typing import Any, List


def resize_condition_pils(
    pils: List[Any],
    *,
    height: int,
    width: int,
    mode: str = "stretch",
) -> List[Any]:
    """Resize condition PILs to one dense canvas using stretch or center-crop."""
    from PIL import Image

    target_size = (int(width), int(height))
    mode = str(mode).strip().lower()
    if mode not in {"stretch", "crop"}:
        raise ValueError(f"resize_condition_pils: mode must be 'stretch' or 'crop', got {mode!r}.")
    output = []
    for pil in pils:
        if pil.size == target_size:
            output.append(pil)
            continue
        if mode == "stretch":
            output.append(pil.resize(target_size, Image.Resampling.LANCZOS))
            continue
        scale = max(target_size[0] / pil.width, target_size[1] / pil.height)
        resized = pil.resize(
            (max(target_size[0], round(pil.width * scale)), max(target_size[1], round(pil.height * scale))),
            Image.Resampling.LANCZOS,
        )
        left = (resized.width - target_size[0]) // 2
        top = (resized.height - target_size[1]) // 2
        output.append(resized.crop((left, top, left + target_size[0], top + target_size[1])))
    return output


__all__ = ["resize_condition_pils"]
