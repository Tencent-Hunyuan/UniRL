"""WAN21VAEDecodeStage — LatentSegment → Videos via 3D VAE decode.

Framework-level ``DecodeStage`` implementation (no REFL-only knobs). Both
GRPO and REFL recipes route their final ``LatentSegment`` through this stage
— the memory / autograd optimizations in ``WanVideoVAE`` (below) are what
make BPTT decode feasible, so pushing this into a recipe would fork the
decode contract across recipes. Kept in the framework.

Implements ``DecodeStage[LatentSegment, Videos]``. Reads the final stored
position from ``LatentSegment.latents[:, -1]`` (the clean latent at
``T``, which ``WAN21DiffusionStage`` always stores), then dispatches to
the bundle's ``WanVideoVAE.decode`` . All un-normalization
(per-channel ``mean`` / ``std``), spatial tiling, nested gradient
checkpointing inside the decoder, and ``Conv3dActGradOnly`` are owned
by ``WanVideoVAE`` itself; this stage is just the
``LatentSegment → Videos`` adapter.

VAE encode for I2V's image-condition latent lives in
``WAN21ImageLatentEncodeStage`` (``image_encode.py``) — it goes through
the same ``WanVideoVAE.encode(x).latent_dist.mode()`` diffusers-style
contract; there is intentionally no generic ``WAN21VAEEncodeStage``.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Video, Videos
from unirl.types.segments import LatentSegment

from .bundle import WAN21Bundle


class WAN21VAEDecodeStage(DecodeStage[LatentSegment, Videos]):
    """WAN 2.1 3D VAE decode stage."""

    def __init__(self, bundle: WAN21Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Videos:
        """Decode the final-step latents in *s* into a packed ``Videos`` payload.

        Reads ``s.latents[:, -1]`` (the final stored position, which is
        ``T`` — the clean latent ``x_0``) as a 5D channel-first tensor
        ``[B, C, T_lat, H_lat, W_lat]``. VAE forward runs in fp32; output
        is normalized from ``[-1, 1]`` to ``[0, 1]`` and clamped before
        being packed sample-by-sample into a ``Videos`` primitive.

        ``grad=False`` (default) keeps the rollout path under ``torch.no_grad()``.
        ``grad=True`` (ReFL direct-reward backprop) runs the decode WITH grad so it
        flows from the reward through the frozen VAE into ``clean``; the VAE has no
        trainable params, so only ``clean``'s graph is extended. ``activation_checkpoint``
        (grad only) recomputes the decode in backward to trade compute for memory.
        """
        if self.bundle.vae is None:
            raise RuntimeError(
                "WAN21VAEDecodeStage.decode: no VAE loaded (load_vae=False). "
                "The trainer-side pipeline cannot decode latents in this "
                "configuration — separate-engine recipes decode in the "
                "rollout engine; trainside rollout requires load_vae=True."
            )
        if s.latents is None:
            raise ValueError("WAN21VAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 6:
            raise ValueError(
                f"WAN21VAEDecodeStage.decode: expected latents shape "
                f"[N, K, C, T_lat, H_lat, W_lat], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]
        if clean.ndim != 5:
            raise ValueError(
                f"WAN21VAEDecodeStage.decode: expected 5D clean latents "
                f"[B, C, T_lat, H_lat, W_lat], got {tuple(clean.shape)}"
            )

        with nullcontext() if grad else torch.no_grad():
            decoded = self._vae_decode(clean)

        # Decoded layout is [B, C, T_dec, H_dec, W_dec] in [-1, 1].
        # Normalize to [0, 1] and clamp before packing.
        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)

        return self._pack_videos(decoded)

    # ------------------------------------------------------------------
    # BPTT path (REFL): differentiable decode of a live grad latent.
    # ------------------------------------------------------------------

    def decode_with_grad(self, z_final: torch.Tensor) -> torch.Tensor:
        """Differentiable VAE decode: ``z_final → pixels`` with autograd alive."""
        if z_final.ndim != 5:
            raise ValueError(
                f"WAN21VAEDecodeStage.decode_with_grad: expected 5D z_final "
                f"[B, C, T_lat, H_lat, W_lat], got {tuple(z_final.shape)}"
            )
        return self._vae_decode(z_final)

    # ------------------------------------------------------------------
    # Shared decode kernel (used by both `decode` and `decode_with_grad`).
    # ------------------------------------------------------------------

    def _vae_decode(self, clean: torch.Tensor) -> torch.Tensor:
        """Run ``WanVideoVAE.decode`` and return pixels in the VAE's native ``[-1, 1]``."""
        vae = self.bundle.vae
        vae_dtype = vae.dtype
        latents = clean.to(dtype=vae_dtype)
        decoded = vae.decode(latents, device=latents.device, tiled=True)
        return decoded

    def _pack_videos(self, decoded: torch.Tensor) -> Videos:
        """Pack a dense ``[B, C, T_dec, H_dec, W_dec]`` pixel tensor into ``Videos``."""
        videos = [Video(frames=decoded[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(decoded.shape[0]))]
        return Videos.from_list(videos)


__all__ = ["WAN21VAEDecodeStage"]
