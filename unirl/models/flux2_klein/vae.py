"""Flux2KleinVAEDecodeStage — LatentSegment → Images via VAE decode.

Implements ``DecodeStage[LatentSegment, Images]``. The Klein diffusion
stage stores **patchified** latents ``[B, K, 128, H_pat, W_pat]``
(128 = 32 × 4 packed channels in the post-VAE / 2×2 patchified grid),
so the decode pipeline runs the FLUX.2-klein VAE chain in reverse:

1. Read the final stored position from ``LatentSegment.latents[:, -1]``
   (``Flux2KleinDiffusionStage`` always stores position ``T`` —
   the clean latent ``x_0``).
2. **Denormalize** patchified latents via the VAE's ``BatchNorm`` head
   running stats (``vae.bn.running_mean`` / ``running_var`` +
   ``vae.config.batch_norm_eps``). Klein's VAE was trained with a
   final BN layer at the encoder side, so sampling-time latents are
   normalized; we have to invert that before decoding. See
   :func:`flux2_klein_utils.denormalize_patchified_latents`.
3. **Unpatchify** ``[B, 128, H_pat, W_pat] → [B, 32, 2*H_pat, 2*W_pat]``
   (the inverse of the 2×2 channel-pack the VAE encoder applies).
4. **Decode** via ``AutoencoderKLFlux2.decode`` in fp32.
5. **Normalize** pixels from ``[-1, 1]`` to ``[0, 1]`` and clamp.

No ``Flux2KleinVAEEncodeStage`` here — the Klein training script is
t2i only (no img2img / SDEdit). Add when image conditioning lands.

Math mirrors the Klein branch of
``main_flux_bundle/unirl/models/flux2.py::decode_latents`` and
the sampling-side BN denormalize in
``main_flux_bundle/unirl/samplers/fsdp/flux2_sampler.py``. The
new-design path does NOT import legacy code; spec sync is via review
and tests.
"""

from __future__ import annotations

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import Flux2KleinBundle
from .flux2_klein_utils import denormalize_patchified_latents, unpatchify_latents


class Flux2KleinVAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """FLUX.2-klein VAE decode stage."""

    def __init__(self, bundle: Flux2KleinBundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment) -> Images:
        """Decode the final-step patchified latents in *s* into pixel images.

        Reads ``s.latents[:, -1]`` (``[B, 128, H_pat, W_pat]`` — the
        clean patchified latent ``x_0`` at position ``T``), runs the
        Klein VAE chain (BN denormalize → unpatchify → decode), and
        normalizes pixels to ``[0, 1]``.
        """
        if s.latents is None:
            raise ValueError("Flux2KleinVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 5:
            raise ValueError(
                f"Flux2KleinVAEDecodeStage.decode: expected latents [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )

        clean = s.latents[:, -1]  # [B, 128, H_pat, W_pat]

        vae = self.bundle.vae
        with torch.no_grad():
            latents_f32 = clean.to(dtype=torch.float32)
            vae_f32 = vae.to(torch.float32)
            denorm = denormalize_patchified_latents(latents_f32, vae_f32)
            unpatched = unpatchify_latents(denorm)
            decoded = vae_f32.decode(unpatched, return_dict=False)[0]

        pixels = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images(pixels=pixels)


__all__ = ["Flux2KleinVAEDecodeStage"]
