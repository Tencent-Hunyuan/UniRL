"""HunyuanImage3 VAE encode + decode stages.

``HunyuanImage3VAEDecodeStage`` implements ``DecodeStage[LatentSegment, Images]``
— reads ``LatentSegment.latents[:, -1]`` (the clean latent at position T,
which ``HunyuanImage3DiffusionStage.diffuse`` always stores) and runs the
3D-VAE decode in fp32 (bf16 is unsupported by most VAE implementations),
then normalizes ``[-1, 1] → [0, 1]`` before wrapping in ``Images``.

``HunyuanImage3VAEEncodeStage`` implements ``EncodeStage[Images, ImageLatentCondition]``
for the it2i original-image conditioning branch (lands fully in PR 5;
included here so the import surface is stable).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage, EncodeStage
from unirl.types.conditions import ImageLatentCondition
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import HunyuanImage3Bundle


class HunyuanImage3VAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """HunyuanImage3 3D-VAE decode stage."""

    def __init__(self, bundle: HunyuanImage3Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Images:
        """Decode the final-step latents in *s* into pixel images.

        Reads ``s.latents[:, -1]`` (the final stored position, which is
        ``T`` — the clean latent ``x_0``). VAE forward runs in fp32; output
        is clamped to ``[0, 1]`` before being wrapped in ``Images``.

        HunyuanImage 3.0's VAE is a 3D-VAE shared with HunyuanVideo: its
        ``decode`` expects ``[B, C, T, H, W]`` (5D) input. For still
        images we add a singleton time dim before decode and squeeze it
        out of the decoded ``[B, 3, T_out, H_out, W_out]`` output.

        ``grad=False`` (default) keeps the rollout path under ``torch.no_grad()``.
        ``grad=True`` (ReFL direct-reward backprop) runs the decode WITH grad so it
        flows from the reward through the frozen VAE into ``clean``; the VAE has no
        trainable params, so only ``clean``'s graph is extended. ``activation_checkpoint``
        (grad only) recomputes the decode in backward to trade compute for memory.
        """
        if s.latents is None:
            raise ValueError("HunyuanImage3VAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim < 5:
            raise ValueError(
                f"HunyuanImage3VAEDecodeStage.decode: expected latents shape [N, K, ...], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]

        scaling_factor = getattr(self.bundle.vae.config, "scaling_factor", 1.0)

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32) / scaling_factor  # [B, C, H, W]
            latents_f32 = latents_f32.unsqueeze(2)  # [B, C, 1, H, W]
            # Force per-rank decode; the distributed path returns data only on rank 0.
            import torch.distributed as _dist

            _orig_is_init = _dist.is_initialized
            _dist.is_initialized = lambda: False
            try:
                out = self.bundle.vae.to(torch.float32).decode(latents_f32)
            finally:
                _dist.is_initialized = _orig_is_init
            # Accept both DecoderOutput and bare-tensor decode results.
            decoded = out.sample if hasattr(out, "sample") else out
            # Decoded shape: [B, 3, T, H, W].
            if decoded.dim() == 5:
                decoded = decoded.squeeze(2)
            return decoded

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)
        pixels = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images.from_dense(pixels)


class HunyuanImage3VAEEncodeStage(EncodeStage[Images, ImageLatentCondition]):
    """HunyuanImage3 3D-VAE encode stage (it2i edit conditioning).

    Encodes uniform ``Images`` into dense VAE latents and packages them as
    ``ImageLatentCondition.latents``. Used by the it2i path in PR 5 to carry
    the original image into the DiT stage's conditioning.
    """

    def __init__(self, bundle: HunyuanImage3Bundle) -> None:
        self.bundle = bundle

    def encode(self, p: Images) -> ImageLatentCondition:
        """Encode pixel images into VAE latents.

        Adds a singleton time axis on the way in (3D-VAE expects
        ``[B, C, T, H, W]``) and squeezes it back out of the resulting
        latent ``[B, C_lat, T_lat, H_lat, W_lat]`` so callers see a flat
        ``[B, C_lat, H_lat, W_lat]`` consistent with the rest of the
        unirl image pipeline.
        """
        try:
            pixels = p.to_dense()
        except ValueError as exc:
            raise ValueError(
                "HunyuanImage3VAEEncodeStage.encode requires uniform image shapes; "
                "the canonical i2t/it2i path must use the upstream per-sample image processor"
            ) from exc
        scaling_factor = getattr(self.bundle.vae.config, "scaling_factor", 1.0)
        with torch.no_grad():
            # pixels: [B, 3, H, W] in [0, 1] → [B, 3, 1, H, W] in [-1, 1]
            x = (pixels.to(dtype=torch.float32) * 2.0 - 1.0).unsqueeze(2)
            latents = self.bundle.vae.to(torch.float32).encode(x).latent_dist.sample()
            if latents.dim() == 5:
                latents = latents.squeeze(2)
            latents = latents * scaling_factor
        return ImageLatentCondition(latents=latents)


__all__ = ["HunyuanImage3VAEDecodeStage", "HunyuanImage3VAEEncodeStage"]
