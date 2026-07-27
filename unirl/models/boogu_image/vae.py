"""BooguImageVAEDecodeStage — LatentSegment → Images via VAE decode.

Implements ``DecodeStage[LatentSegment, Images]``. Reads the final stored
position from ``LatentSegment.latents[:, -1]`` (``BooguImageDiffusionStage``
always stores position ``T``, the clean latent), runs the ``AutoencoderKL``
decode in fp32, and normalizes the output from ``[-1, 1]`` to ``[0, 1]``
before wrapping in ``Images``.

Boogu-Image uses the FLUX.1 16-channel ``AutoencoderKL`` with both
``scaling_factor`` (0.3611) and ``shift_factor`` (0.1159) — identical VAE
family to z_image/SD3. The latent un-normalization mirrors the reference
``processing`` tail (pipeline_boogu.py:3681-3686):
``x = latent / scaling_factor + shift_factor``.

Documented divergence from the reference: no bilinear resize of the decoded
image back to the pre-floor request size (pipeline_boogu.py:2939) — recipes
must use ``height``/``width`` ≡ 0 (mod 16), which makes the resize a no-op.

No ``BooguImageVAEEncodeStage`` here — Base is text-to-image only; the
encoder path lands with the Edit (TI2I) variant.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import BooguImageBundle


class BooguImageVAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """Boogu-Image VAE decode stage."""

    def __init__(self, bundle: BooguImageBundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Images:
        """Decode the final-step latents in *s* into pixel images.

        Reads ``s.latents[:, -1]`` (the final stored position, which is
        ``T`` — the clean latent ``x_0`` in spatial shape ``[B, C, H, W]``).
        VAE forward runs in fp32 (the FLUX.1 VAE declares ``force_upcast``);
        output is clamped to ``[0, 1]`` before being wrapped in ``Images``.

        ``grad=False`` (default) keeps the rollout path under ``torch.no_grad()``.
        ``grad=True`` (ReFL direct-reward backprop) runs the decode WITH grad so it
        flows from the reward through the frozen VAE into ``clean``; the VAE has no
        trainable params, so only ``clean``'s graph is extended. ``activation_checkpoint``
        (grad only) recomputes the decode in backward to trade compute for memory.
        """
        if self.bundle.vae is None:
            raise RuntimeError(
                "BooguImageVAEDecodeStage.decode: no VAE loaded (load_vae=False). "
                "The trainer-side pipeline cannot decode latents in this "
                "configuration — separate-engine recipes decode in the "
                "rollout engine; trainside rollout requires load_vae=True."
            )
        if s.latents is None:
            raise ValueError("BooguImageVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim < 5:
            raise ValueError(
                f"BooguImageVAEDecodeStage.decode: expected latents shape [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]  # [B, C, H, W]

        scaling_factor = self.bundle.vae.config.scaling_factor
        shift_factor = getattr(self.bundle.vae.config, "shift_factor", None)

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32) / scaling_factor
            if shift_factor is not None:
                latents_f32 = latents_f32 + float(shift_factor)
            return self.bundle.vae.to(torch.float32).decode(latents_f32).sample

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)
        pixels = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images(pixels=pixels)


__all__ = ["BooguImageVAEDecodeStage"]
