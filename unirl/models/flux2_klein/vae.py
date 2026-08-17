"""Flux2KleinVAEDecodeStage — LatentSegment → Images via VAE decode."""

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
from .image import resize_condition_pils


class Flux2KleinVAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """FLUX.2-klein VAE decode stage."""

    def __init__(self, bundle: Flux2KleinBundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Images:
        """Decode the final-step patchified latents in *s* into pixel images."""
        if self.bundle.vae is None:
            raise RuntimeError(
                "Flux2KleinVAEDecodeStage.decode: no VAE loaded "
                "(load_vae=False). The trainer-side pipeline cannot decode "
                "latents in this configuration — separate-engine recipes "
                "decode in the rollout engine; trainside rollout requires "
                "load_vae=True."
            )
        if s.latents is None:
            raise ValueError("Flux2KleinVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 5:
            raise ValueError(
                f"Flux2KleinVAEDecodeStage.decode: expected latents [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )

        clean = s.latents[:, -1]

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
        return Images.from_dense(pixels)


class Flux2KleinVAEEncodeStage:
    """Encode a source/reference image into packed condition tokens + ids."""

    REFERENCE_TIME_SCALE: int = 10

    def __init__(self, bundle: Flux2KleinBundle) -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, images: Images, *, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bundle.vae is None:
            raise RuntimeError(
                "Flux2KleinVAEEncodeStage.encode: no VAE loaded "
                "(load_vae=False). The trainer-side pipeline cannot encode "
                "source images in this configuration — separate-engine "
                "recipes encode in the rollout engine; trainside rollout "
                "requires load_vae=True."
            )
        if not isinstance(images, Images):
            raise TypeError(f"Flux2KleinVAEEncodeStage.encode: expected Images, got {type(images).__name__}")
        pixels_list = [image.pixels for image in images.to_list()]
        if not pixels_list or any(pixels is None or pixels.ndim != 3 or pixels.shape[0] != 3 for pixels in pixels_list):
            raise ValueError(
                "Flux2KleinVAEEncodeStage.encode: expected per-sample pixels [3, H, W] in [0,1], "
                f"got {[None if pixels is None else tuple(pixels.shape) for pixels in pixels_list]}"
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
        from torchvision.transforms.functional import pil_to_tensor

        condition_pils = resize_condition_pils(images.to_pils(), height=height, width=width)
        pixels = torch.stack(
            [pil_to_tensor(pil).to(dtype=torch.float32).div_(255.0) for pil in condition_pils],
            dim=0,
        ).to(device=device)

        scaled = pixels * 2.0 - 1.0

        image_latents = vae_f32.encode(scaled).latent_dist.mode()
        image_latents = patchify_latents(image_latents)
        image_latents = normalize_patchified_latents(image_latents, vae_f32)

        batch_size, _, h_pat, w_pat = image_latents.shape

        image_tokens = pack_latents(image_latents)

        t = torch.full((1,), self.REFERENCE_TIME_SCALE, device=device, dtype=torch.long)
        h = torch.arange(h_pat, device=device)
        w = torch.arange(w_pat, device=device)
        s = torch.arange(1, device=device)
        coords = torch.cartesian_prod(t, h, w, s)
        image_ids = coords.unsqueeze(0).expand(batch_size, -1, -1)

        return image_tokens.to(dtype=self.bundle.dtype), image_ids


__all__ = ["Flux2KleinVAEDecodeStage", "Flux2KleinVAEEncodeStage"]
