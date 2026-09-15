"""MiniMax-H3 decode stages -- packed rows -> Videos / stereo Audios."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from unirl.types.primitives import Audio, Audios, Video, Videos

from .config import MINIMAX_H3_LATENT_CHANNELS, MINIMAX_H3_PATCH_SIZE
from .packing import MiniMaxH3Geometry
from .vendor import unpack_audio_tokens, unpatchify_video_tokens

if TYPE_CHECKING:
    from .bundle import MiniMaxH3Bundle

# Fallback when an audio VAE component is unavailable, matching Diffusers.
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32_000

# The video VAE's pixel convention: ImageNet-normalized RGB over a [0, 1] base.
_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)


def _sp_decode_group():
    """Return the Ulysses SP process group when sharded decode is usable, else None."""
    import torch.distributed as dist

    if os.environ.get("UNIRL_H3_DECODE_SHARD") != "1":
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    state = get_parallel_state()
    sp_size = int(state.sp_size)
    if sp_size <= 1:
        return None
    group = state.sp_group
    if group is None:
        raise RuntimeError(f"MiniMax-H3 sharded decode requires an SP process group for sp_size={sp_size}")
    if dist.get_world_size(group) != sp_size or dist.get_rank(group) != int(state.sp_rank):
        raise RuntimeError("MiniMax-H3 sharded decode found an inconsistent VeOmni parallel state")
    return group


def _sharded_decode_clips(vae, z: torch.Tensor, num_chunks: int, group) -> list[torch.Tensor]:
    """Run ``_decode_clip`` for ``num_chunks`` clips spread over the SP group, then all-gather."""
    import torch.distributed as dist

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    # Round-robin, so ranks share the same per-chunk cost and the gather order is chunk order.
    local = [
        vae._decode_clip(
            z[:, :, i * vae.tokens_chunk_size : i * vae.tokens_chunk_size + vae.tokens_chunk_size + vae.token_overlap]
        )
        for i in range(rank, num_chunks, world)
    ]
    # Every rank must contribute the same shape, so agree on it with a fixed-size collective
    # before anyone allocates. Clips are [B, C, T, H, W]; rank 0 always owns chunk 0.
    probe = torch.zeros(6, dtype=torch.int64, device=z.device)
    if local:
        if local[0].ndim != 5:
            raise RuntimeError(f"MiniMax-H3 sharded decode expected 5-D clips, got {local[0].ndim}-D")
        probe[0] = 1
        probe[1:] = torch.tensor(local[0].shape, dtype=torch.int64, device=z.device)
    probes = [torch.empty_like(probe) for _ in range(world)]
    dist.all_gather(probes, probe, group=group)
    ref = next(tuple(int(v) for v in p[1:]) for p in probes if int(p[0]))
    if any(tuple(clip.shape) != ref or clip.dtype != z.dtype for clip in local):
        raise RuntimeError("MiniMax-H3 sharded decode produced ragged clips; this geometry is unsupported")
    per_rank = (num_chunks + world - 1) // world
    slab = torch.zeros((per_rank, *ref), dtype=z.dtype, device=z.device)
    for index, clip in enumerate(local):
        slab[index] = clip
    gathered = [torch.empty_like(slab) for _ in range(world)]
    dist.all_gather(gathered, slab.contiguous(), group=group)
    # Undo the round-robin: rank r held chunks r, r+world, r+2*world, ...
    clips: list[torch.Tensor | None] = [None] * num_chunks
    for r in range(world):
        for slot, i in enumerate(range(r, num_chunks, world)):
            clips[i] = gathered[r][slot]
    if any(clip is None for clip in clips):
        raise RuntimeError("MiniMax-H3 sharded decode did not recover every clip")
    return clips


def _decode_video_latents(vae, z: torch.Tensor) -> torch.Tensor:
    """``AutoencoderKLMiniMaxH3._decode`` with the clip loop optionally sharded across SP ranks."""
    group = _sp_decode_group()
    if group is None:
        return vae.decode(z, return_dict=False)[0]

    tokens_chunk_size = vae.tokens_chunk_size
    token_drop = vae.config.token_drop
    temporal_ratio = vae.temporal_compression_ratio
    chunk_num_frames = tokens_chunk_size * temporal_ratio

    num_tokens = z.shape[2] + token_drop
    pad_tokens = (-num_tokens) % tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
    if pad_tokens > 0:
        z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

    clips = _sharded_decode_clips(vae, z, num_chunks, group)

    # Identical to the vendor post-loop: only the clip forwards were distributed.
    decoded_chunks = []
    overlap = None
    for clip in clips:
        for j in range(int(token_drop > 0) + 1):
            frame_start = j * chunk_num_frames
            chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
            chunk = chunk[:, :, vae.frame_pre_padding :]
            if j == 0:
                if overlap is not None:
                    chunk = vae._blend(overlap, chunk, vae.frame_overlap, dim=-3)
                decoded_chunks.append(chunk)
            else:
                overlap = chunk
    if overlap is not None:
        decoded_chunks.append(overlap)

    dec = torch.cat(decoded_chunks, dim=2)
    if pad_tokens > 0:
        intra_tail = vae.config.clip_length % temporal_ratio
        num_tokens_before_pad = z.shape[2] - pad_tokens
        pad_frames = sum(
            intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
            for k in range(pad_tokens)
        )
        dec = dec[:, :, :-pad_frames]
    return dec


class MiniMaxH3VideoDecodeStage:
    """Packed video rows -> ``Videos``."""

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.vae = bundle.vae

    @torch.no_grad()
    def decode(self, rows: torch.Tensor, geometry: MiniMaxH3Geometry) -> Videos:
        device = self.vae.device
        dtype = next(self.vae.parameters()).dtype
        latents = unpatchify_video_tokens(
            rows.to(device=device, dtype=dtype),
            num_latent_frames=geometry.num_latent_frames,
            latent_height=geometry.latent_height,
            latent_width=geometry.latent_width,
            channels=MINIMAX_H3_LATENT_CHANNELS,
            patch_size=MINIMAX_H3_PATCH_SIZE,
        )
        mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = latents * std + mean

        video = _decode_video_latents(self.vae, latents.to(dtype))
        pixel_mean = torch.tensor(_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)

        # [B, C, T, H, W] -> per-sample [T, C, H, W], the Videos frame layout.
        return Videos.from_list([Video(frames=v.permute(1, 0, 2, 3).contiguous().cpu()) for v in video])


class MiniMaxH3AudioDecodeStage:
    """Packed audio rows -> stereo ``Audios``."""

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.audio_vae = bundle.audio_vae

    @torch.no_grad()
    def decode(self, rows: torch.Tensor, geometry: MiniMaxH3Geometry) -> Audios:
        device = self.audio_vae.device
        dtype = next(self.audio_vae.parameters()).dtype
        latents = unpack_audio_tokens(rows.to(device=device, dtype=dtype), num_audio_latents=geometry.num_audio_latents)
        mean = torch.tensor(self.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        std = torch.tensor(self.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        latents = latents * std + mean

        audio = self.audio_vae.decode(latents.to(dtype), return_dict=False)[0]
        # [channels-as-batch, 1, L] -> [1, channels, L], matching the reference.
        audio = audio.float().permute(1, 0, 2)
        # -> per-sample length-first [L, C] for the varlen pack.
        return Audios.from_list([Audio(waveform=a.transpose(0, 1).contiguous().cpu()) for a in audio])


__all__ = [
    "MINIMAX_H3_AUDIO_SAMPLE_RATE",
    "MiniMaxH3AudioDecodeStage",
    "MiniMaxH3VideoDecodeStage",
]
