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

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import Flux2KleinBundle
from .flux2_klein_utils import (
    denormalize_patchified_latents,
    normalize_patchified_latents,
    pack_latents,
    patchify_latents,
    unpatchify_latents,
)


class Flux2KleinVAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """FLUX.2-klein VAE decode stage."""

    def __init__(self, bundle: Flux2KleinBundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Images:
        """Decode the final-step patchified latents in *s* into pixel images.

        Reads ``s.latents[:, -1]`` (``[B, 128, H_pat, W_pat]`` — the
        clean patchified latent ``x_0`` at position ``T``), runs the
        Klein VAE chain (BN denormalize → unpatchify → decode), and
        normalizes pixels to ``[0, 1]``.

        ``grad=False`` (default) keeps the rollout path under ``torch.no_grad()``.
        ``grad=True`` (ReFL direct-reward backprop) runs the decode WITH grad so it
        flows from the reward through the frozen VAE into ``clean``; the VAE has no
        trainable params, so only ``clean``'s graph is extended. ``activation_checkpoint``
        (grad only) recomputes the decode in backward to trade compute for memory.
        """
        if s.latents is None:
            raise ValueError("Flux2KleinVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 5:
            raise ValueError(
                f"Flux2KleinVAEDecodeStage.decode: expected latents [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )

        clean = s.latents[:, -1]  # [B, 128, H_pat, W_pat]

        vae = self.bundle.vae

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32)
            vae_f32 = vae.to(torch.float32)
            denorm = denormalize_patchified_latents(latents_f32, vae_f32)
            unpatched = unpatchify_latents(denorm)
            return vae_f32.decode(unpatched, return_dict=False)[0]

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)

        pixels = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images(pixels=pixels)


class Flux2KleinVAEEncodeStage:
    """Encode a source/reference image into packed condition tokens + ids.

    Image-edit conditioning path (mirrors diffusers
    ``Flux2KleinPipeline.prepare_image_latents`` + ``_prepare_image_ids``):

    1. pixels ``[B, 3, H, W]`` in ``[0, 1]`` → ``[-1, 1]`` (VAE input convention).
    2. ``vae.encode(x).latent_dist.mode()`` (deterministic — matches diffusers'
       ``retrieve_latents(sample_mode="argmax")`` so rollout/replay don't drift).
    3. patchify ``[B, 32, H/8, W/8] → [B, 128, H/16, W/16]``.
    4. BN-normalize the patchified latents (the Klein VAE's BatchNorm head).
    5. pack ``[B, 128, h, w] → [B, h*w, 128]`` condition tokens.
    6. build 4-axis RoPE ids ``[B, h*w, 4]`` with a time offset (T=``scale``)
       so the transformer distinguishes condition tokens from noise tokens
       (whose latent ids use T=0..; see ``prepare_latent_ids``).

    Returns ``(image_tokens [B, N, 128], image_ids [B, N, 4])``.
    """

    # Time-axis offset for the single reference image (diffusers uses
    # ``scale + scale * t``; for one image t=0 → T-coord = scale = 10).
    REFERENCE_TIME_SCALE: int = 10

    def __init__(self, bundle: Flux2KleinBundle) -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, images: Images, *, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = images.pixels
        if pixels is None or pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(
                f"Flux2KleinVAEEncodeStage.encode: expected pixels [B, 3, H, W] in [0,1], "
                f"got shape {None if pixels is None else tuple(pixels.shape)}"
            )

        vae = self.bundle.vae
        device = self.bundle.device
        vae_f32 = vae.to(torch.float32)

        # Resize the source image to the generation size. The data source loads
        # condition images at native resolution (arbitrary H×W), but the VAE
        # patchify requires H,W divisible by 16 (8× VAE + 2× patch), and a
        # consistent token count across a GRPO group needs a fixed size. Using
        # the generation (height, width) satisfies both (recipe sizes are
        # multiples of 16) and matches the edited-image resolution.
        pixels = pixels.to(device=device, dtype=torch.float32)
        if int(pixels.shape[-2]) != int(height) or int(pixels.shape[-1]) != int(width):
            pixels = torch.nn.functional.interpolate(
                pixels, size=(int(height), int(width)), mode="bilinear", align_corners=False
            )

        # [0, 1] → [-1, 1] (VAE input convention).
        scaled = pixels * 2.0 - 1.0

        # Deterministic latents (mode), patchify, BN-normalize.
        image_latents = vae_f32.encode(scaled).latent_dist.mode()  # [B, 32, H/8, W/8]
        image_latents = patchify_latents(image_latents)  # [B, 128, H/16, W/16]
        image_latents = normalize_patchified_latents(image_latents, vae_f32)

        batch_size, _, h_pat, w_pat = image_latents.shape

        # Pack to tokens [B, h*w, 128].
        image_tokens = pack_latents(image_latents)

        # 4-axis ids (T, H, W, L) with the reference time offset on T.
        t = torch.full((1,), self.REFERENCE_TIME_SCALE, device=device, dtype=torch.long)
        h = torch.arange(h_pat, device=device)
        w = torch.arange(w_pat, device=device)
        s = torch.arange(1, device=device)
        coords = torch.cartesian_prod(t, h, w, s)  # [h*w, 4]
        image_ids = coords.unsqueeze(0).expand(batch_size, -1, -1)

        return image_tokens.to(dtype=self.bundle.dtype), image_ids


__all__ = ["Flux2KleinVAEDecodeStage", "Flux2KleinVAEEncodeStage"]
