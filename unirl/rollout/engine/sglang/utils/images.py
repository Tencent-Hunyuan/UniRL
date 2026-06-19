"""Image encoding for the SRT wire payload."""

from __future__ import annotations

from typing import Any


def pil_to_base64(image: Any) -> str:
    """Encode a PIL image as a ``data:image/png;base64,...`` URI for SRT."""
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


__all__ = ["pil_to_base64"]
