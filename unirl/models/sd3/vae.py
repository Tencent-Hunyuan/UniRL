"""SD3 VAE codec stages — LatentSegment ↔ Images.

``SD3VAEDecodeStage`` implements ``DecodeStage[LatentSegment, Images]``:
reads the final stored position from ``LatentSegment.latents[:, -1]``
(``SD3DiffusionStage`` always stores position ``T``, the clean latent), runs
VAE decode in fp32 (bf16 is unsupported by most VAE implementations), and
normalizes the output from ``[-1, 1]`` to ``[0, 1]`` before wrapping in
``Images``.

``SD3VAEEncodeStage`` implements ``EncodeStage[Images, ImageLatentCondition]``
as the STRICT INVERSE of the decode math (deterministic ``.mode()``, then
``(z - shift_factor) * scaling_factor``) — first consumer is diffusion SFT's
target-image encoding; img2img / SDEdit conditioning can reuse it.

Decode math copied from ``models/sd3.py:533-550`` and
``samplers/fsdp/base_sampler.py:162-180`` (do NOT import legacy code).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage, EncodeStage
from unirl.types.conditions.image import ImageLatentCondition
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import SD3Bundle


class SD3VAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """SD3 VAE decode stage."""

    def __init__(self, bundle: SD3Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Images:
        """Decode the final-step latents in *s* into pixel images.

        Reads ``s.latents[:, -1]`` (the final stored position, which is ``T`` —
        the clean latent ``x_0``). VAE forward runs in fp32; output is clamped to
        ``[0, 1]`` before being wrapped in ``Images``.

        ``grad=False`` (default) keeps the rollout path under ``torch.no_grad()``.
        ``grad=True`` (ReFL direct-reward backprop) runs the decode WITH grad so it
        flows from the reward through the frozen VAE into ``clean``; the VAE has no
        trainable params, so only ``clean``'s graph is extended. ``activation_checkpoint``
        (grad only) recomputes the decode in backward to trade compute for memory.
        """
        if s.latents is None:
            raise ValueError("SD3VAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim < 5:
            raise ValueError(
                f"SD3VAEDecodeStage.decode: expected latents shape [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]
        scaling_factor = self.bundle.vae.config.scaling_factor
        shift_factor = getattr(self.bundle.vae.config, "shift_factor", None)
        vae = self.bundle.vae.to(torch.float32)

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32) / scaling_factor
            if shift_factor is not None:
                latents_f32 = latents_f32 + float(shift_factor)
            return vae.decode(latents_f32).sample

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)
        pixels = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images(pixels=pixels)


class SD3VAEEncodeStage(EncodeStage[Images, ImageLatentCondition]):
    """SD3 VAE encode stage — the strict inverse of :class:`SD3VAEDecodeStage`.

    ``pixels ∈ [0, 1] → [-1, 1] → vae.encode(·).latent_dist.mode() →
    (z - shift_factor) * scaling_factor``. Deterministic ``.mode()`` (not
    ``.sample()`` — posterior sampling would make the stored latent drift from
    what decode reproduces); normalization constants read from ``vae.config``
    at call time (SD3.5 ships ≈1.5305 / ≈0.0609 — never hardcode). Runs in
    fp32 under ``no_grad``, matching decode's precision policy.
    """

    def __init__(self, bundle: SD3Bundle) -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, p: Images) -> ImageLatentCondition:
        pixels = p.pixels
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(
                f"SD3VAEEncodeStage.encode: expected pixels [B, 3, H, W], got "
                f"{tuple(pixels.shape) if isinstance(pixels, torch.Tensor) else type(pixels).__name__}"
            )
        scaling_factor = self.bundle.vae.config.scaling_factor
        shift_factor = getattr(self.bundle.vae.config, "shift_factor", None)
        vae = self.bundle.vae.to(torch.float32)
        x = pixels.to(device=self.bundle.device, dtype=torch.float32) * 2.0 - 1.0
        z = vae.encode(x).latent_dist.mode()
        if shift_factor is not None:
            z = z - float(shift_factor)
        z = z * scaling_factor
        # Keep the clean latent fp32: it becomes the supervised target
        # (``ε - x0``); a bf16 round-trip here is needless precision loss.
        return ImageLatentCondition(latents=z.to(dtype=torch.float32))


__all__ = ["SD3VAEDecodeStage", "SD3VAEEncodeStage"]
