"""QwenImageEditPlusBundle — thin subclass of :class:`QwenImageBundle`."""

from __future__ import annotations

from unirl.models.qwen_image.bundle import QwenImageBundle


class QwenImageEditPlusBundle(QwenImageBundle):
    """Qwen-Image-Edit-Plus bundle: transformer (in_channels=64) + VAE + Qwen-VL text encoder + scheduler."""


__all__ = ["QwenImageEditPlusBundle"]
