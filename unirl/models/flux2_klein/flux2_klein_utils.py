"""FLUX.2 Klein helpers aligned with the official diffusers pipeline.

Pure-math helpers for the FLUX.2-klein-9B pipeline: empirical-mu schedule
construction, 2x2 patchify/unpatchify, packing utilities for the
transformer's token-level input layout, RoPE position-id construction
matching diffusers' 4-axis ``(T, H, W, L)`` form, and the
``vae.bn`` patchified-latent (de)normalization required by the
official Klein VAE.

Verbatim port of ``main_flux_bundle/unirl/models/flux2_klein_utils.py``.
The new-design path does NOT import legacy code, so these helpers live
inside the typed pipeline package and must stay in spec sync with the
diffusers reference via review/tests.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Mirror ``Flux2KleinPipeline.compute_empirical_mu`` from diffusers.

    Klein's official inference path derives a per-request mu for
    FlowMatch shifting from the packed image-token count and the number
    of inference steps. The mapping is piecewise linear with a knee at
    ``image_seq_len > 4300``; below that, an additional linear blend
    between the 10-step and 200-step lines is applied.
    """

    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def prepare_text_ids(prompt_embeds: torch.Tensor) -> torch.Tensor:
    """Build official FLUX.2 text RoPE ids ``(T=0, H=0, W=0, L=token_idx)``.

    Shape ``[B, L, 4]``: matches diffusers' ``Flux2KleinPipeline``
    text-token id layout. ``Flux2Transformer2DModel`` configures
    ``axes_dims_rope=[32, 32, 32, 32]`` so feeding ``[L, 3]`` (FLUX.1
    layout) crashes inside ``FluxPosEmbed`` with
    ``IndexError: index 3 is out of bounds``.
    """

    if prompt_embeds.dim() != 3:
        raise ValueError(f"prompt_embeds must be [B, L, D], got {tuple(prompt_embeds.shape)}")
    batch_size, seq_len, _ = prompt_embeds.shape
    t = torch.arange(1, device=prompt_embeds.device)
    h = torch.arange(1, device=prompt_embeds.device)
    w = torch.arange(1, device=prompt_embeds.device)
    s = torch.arange(seq_len, device=prompt_embeds.device)
    coords = torch.cartesian_prod(t, h, w, s)
    return coords.unsqueeze(0).expand(batch_size, -1, -1)


def prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
    """Build official FLUX.2 latent RoPE ids for patchified ``[B, C, H, W]``.

    Returns ``[B, H*W, 4]`` with ``(T=0, h_idx, w_idx, L=0)``. The
    ``H, W`` here are the **patchified** spatial dims, i.e. after the
    ``[B, 32, H_pix/8, W_pix/8] -> [B, 128, H_pix/16, W_pix/16]``
    transform.
    """

    if latents.dim() != 4:
        raise ValueError(f"latents must be [B, C, H, W], got {tuple(latents.shape)}")
    batch_size, _, height, width = latents.shape
    t = torch.arange(1, device=latents.device)
    h = torch.arange(height, device=latents.device)
    w = torch.arange(width, device=latents.device)
    s = torch.arange(1, device=latents.device)
    coords = torch.cartesian_prod(t, h, w, s)
    return coords.unsqueeze(0).expand(batch_size, -1, -1)


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Official FLUX.2 patchify: ``[B, 32, H, W] -> [B, 128, H/2, W/2]``."""

    batch_size, num_channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"FLUX.2 latents height/width must be even, got {height}x{width}")
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Official FLUX.2 unpatchify: ``[B, 128, H, W] -> [B, 32, 2H, 2W]``."""

    batch_size, num_channels, height, width = latents.shape
    if num_channels % 4 != 0:
        raise ValueError(f"patchified channel count must be divisible by 4, got {num_channels}")
    latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Official FLUX.2 pack: ``[B, C, H, W] -> [B, H*W, C]``."""

    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)


def unpack_latents(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Inverse of :func:`pack_latents` when the patchified spatial size is known."""

    batch_size, _, num_channels = tokens.shape
    return tokens.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)


def unpack_latents_with_ids(tokens: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Scatter packed latent tokens back to ``[B, C, H, W]`` using official ids.

    Used when only the ids are available and the patchified ``(H, W)``
    isn't carried on the segment.
    """

    if ids.dim() == 2:
        ids = ids.unsqueeze(0).expand(tokens.shape[0], -1, -1)
    outputs = []
    for data, pos in zip(tokens, ids):
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)
        height = int(torch.max(h_ids).item()) + 1
        width = int(torch.max(w_ids).item()) + 1
        channels = data.shape[-1]
        flat_ids = h_ids * width + w_ids
        out = torch.zeros((height * width, channels), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, channels), data)
        outputs.append(out.view(height, width, channels).permute(2, 0, 1))
    return torch.stack(outputs, dim=0)


def vae_bn_stats(vae: Any, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return FLUX.2 VAE BN mean/std, or ``None`` when the VAE lacks BN stats.

    The official Klein VAE (``AutoencoderKLFlux2``) carries a final
    ``BatchNorm`` head; patchified latents from sampling must be
    normalized by the BN mean/var before being passed back through the
    decoder (and de-normalized in the inverse direction for sampling).
    """

    bn = getattr(vae, "bn", None)
    if bn is None or not hasattr(bn, "running_mean") or not hasattr(bn, "running_var"):
        return None
    eps = float(getattr(getattr(vae, "config", None), "batch_norm_eps", 1e-4))
    mean = bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1).to(device=device, dtype=dtype) + eps)
    return mean, std


def normalize_patchified_latents(latents: torch.Tensor, vae: Any) -> torch.Tensor:
    """Apply official FLUX.2 VAE BN normalization to patchified latents."""

    stats = vae_bn_stats(vae, device=latents.device, dtype=latents.dtype)
    if stats is None:
        logger.warning("FLUX.2 VAE lacks BN stats; leaving patchified latents unnormalized.")
        return latents
    mean, std = stats
    return (latents - mean) / std


def denormalize_patchified_latents(latents: torch.Tensor, vae: Any) -> torch.Tensor:
    """Undo official FLUX.2 VAE BN normalization for patchified latents."""

    stats = vae_bn_stats(vae, device=latents.device, dtype=latents.dtype)
    if stats is None:
        logger.warning("FLUX.2 VAE lacks BN stats; leaving patchified latents unchanged.")
        return latents
    mean, std = stats
    return latents * std + mean


__all__ = [
    "compute_empirical_mu",
    "denormalize_patchified_latents",
    "normalize_patchified_latents",
    "pack_latents",
    "patchify_latents",
    "prepare_latent_ids",
    "prepare_text_ids",
    "unpack_latents",
    "unpack_latents_with_ids",
    "unpatchify_latents",
    "vae_bn_stats",
]
