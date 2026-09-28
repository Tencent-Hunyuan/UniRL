"""Packed-pixel geometry and decode stage for SenseNova-U1.5."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Tuple

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments.latent import LatentSegment

if TYPE_CHECKING:
    from .bundle import SenseNovaU1Bundle


def packed_pixel_shape(
    image_shape: Tuple[int, int],
    *,
    patch_size: int,
) -> Tuple[int, int]:
    """Return packed ``(sequence, channels)`` for an ``(H, W)`` RGB image."""
    height, width = (int(v) for v in image_shape)
    patch = int(patch_size)
    if height <= 0 or width <= 0 or height % patch or width % patch:
        raise ValueError(
            f"SenseNova-U1 image shape {height}x{width} must be positive and divisible by pixel patch {patch}."
        )
    return (height // patch) * (width // patch), 3 * patch * patch


def unpatchify_pixels(
    packed: torch.Tensor,
    *,
    image_shape: Tuple[int, int],
    patch_size: int,
) -> torch.Tensor:
    """Convert ``[B, L, 3*p*p]`` packed pixels to ``[B, 3, H, W]``."""
    height, width = (int(v) for v in image_shape)
    patch = int(patch_size)
    expected_sequence, expected_channels = packed_pixel_shape(image_shape, patch_size=patch)
    if packed.ndim != 3 or tuple(packed.shape[1:]) != (expected_sequence, expected_channels):
        raise ValueError(
            "unpatchify_pixels expected packed shape "
            f"[B, {expected_sequence}, {expected_channels}], got {tuple(packed.shape)}."
        )
    batch = int(packed.shape[0])
    token_h, token_w = height // patch, width // patch
    blocks = packed.reshape(batch, token_h, token_w, patch, patch, 3)
    blocks = torch.einsum("nhwpqc->nchpwq", blocks)
    return blocks.reshape(batch, 3, height, width)


def patchify_pixels(
    pixels: torch.Tensor,
    *,
    patch_size: int,
) -> torch.Tensor:
    """Convert ``[B, 3, H, W]`` normalized pixels to ``[B, L, 3*p*p]``."""
    if pixels.ndim != 4 or int(pixels.shape[1]) != 3:
        raise ValueError(f"patchify_pixels expected [B, 3, H, W], got {tuple(pixels.shape)}.")
    batch, _, height, width = pixels.shape
    patch = int(patch_size)
    packed_pixel_shape((int(height), int(width)), patch_size=patch)
    token_h, token_w = int(height) // patch, int(width) // patch
    blocks = pixels.reshape(batch, 3, token_h, patch, token_w, patch)
    blocks = torch.einsum("nchpwq->nhwpqc", blocks)
    return blocks.reshape(batch, token_h * token_w, 3 * patch * patch)


class SenseNovaU1PixelDecodeStage(DecodeStage[LatentSegment, Images]):
    """Decode the final packed state; SenseNova-U1.5 diffuses pixels directly."""

    def __init__(self, bundle: "SenseNovaU1Bundle") -> None:
        self.bundle = bundle

    @property
    def pixel_patch_size(self) -> int:
        model = self.bundle.model
        return int(model.patch_size) * int(1 / float(model.downsample_ratio))

    def decode(
        self,
        segment: LatentSegment,
        *,
        image_shape: Tuple[int, int],
        grad: bool = False,
        activation_checkpoint: bool = False,
    ) -> Images:
        """Decode final normalized pixels to an ``Images`` batch in ``[0, 1]``."""
        del activation_checkpoint
        if segment.latents is None or segment.latents.ndim != 4:
            got = None if segment.latents is None else tuple(segment.latents.shape)
            raise ValueError(f"SenseNovaU1PixelDecodeStage.decode expected latents [B, K, L, C], got {got}.")
        with nullcontext() if grad else torch.no_grad():
            normalized = unpatchify_pixels(
                segment.latents[:, -1].float(),
                image_shape=image_shape,
                patch_size=self.pixel_patch_size,
            )
            pixels = (normalized * 0.5 + 0.5).clamp(0.0, 1.0)
        return Images.from_dense(pixels.cpu())


__all__ = [
    "SenseNovaU1PixelDecodeStage",
    "packed_pixel_shape",
    "patchify_pixels",
    "unpatchify_pixels",
]
