"""Expand driver ``initial_noise`` from ``[1, ...]`` to per-sample ``[batch_size, ...]`` and pack the latents."""

from __future__ import annotations

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("UNIRL_DEBUG_LATENT_SHAPE") == "1"


def _expand_initial_noise(latents: torch.Tensor, batch_size: int, num_outputs_per_prompt: int) -> torch.Tensor:
    """Expand provided initial noise to ``batch_size`` (fork's rule)."""
    n = int(latents.shape[0])
    if n == batch_size:
        return latents
    nopp = max(1, int(num_outputs_per_prompt))
    num_prompts = batch_size // nopp
    if n == 1:
        return latents.expand(batch_size, *latents.shape[1:]).contiguous()
    if n == num_prompts and nopp > 1:
        return latents.repeat_interleave(nopp, dim=0)
    raise ValueError(
        f"initial_noise batch dim {n} does not match batch_size={batch_size}, "
        f"num_prompts={num_prompts}, or 1. Expected one of: 1, {num_prompts}, "
        f"or {batch_size}."
    )


def patch_latent_prep() -> None:
    from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
    from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
        LatentPreparationStage,
    )

    orig = LatentPreparationStage.forward
    if getattr(orig, "_unirl_initial_noise_expand", False):
        return

    def forward(self, batch, server_args):
        latents = getattr(batch, "latents", None)

        if latents is None or not torch.is_tensor(latents):
            return orig(self, batch, server_args)

        device = get_local_torch_device()
        batch_size = int(batch.batch_size)
        nopp = int(getattr(batch, "num_outputs_per_prompt", 1) or 1)
        pcfg = server_args.pipeline_config

        latents = _expand_initial_noise(latents, batch_size, nopp).to(device=device)

        try:
            num_frames = int(getattr(batch, "num_frames", 1) or 1)
            expected = tuple(int(d) for d in pcfg.prepare_latent_shape(batch, batch_size, num_frames))
            if (
                latents.ndim == 4
                and len(expected) == 5
                and expected[1] == int(latents.shape[1])
                and expected[2] == 1
                and latents.numel() == math.prod(expected)
            ):
                latents = latents.reshape(expected)
        except Exception as exc:  # pragma: no cover - defensive; preserve prior behavior on any failure
            logger.warning("latent_prep frame-axis reconcile skipped: %s: %s", type(exc).__name__, exc)

        latent_ids = pcfg.maybe_prepare_latent_ids(latents)
        if latent_ids is not None:
            batch.latent_ids = latent_ids.to(device=device)

        latents = pcfg.maybe_pack_latents(latents, batch_size, batch)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        if _DEBUG:
            lid_shape = getattr(batch.latent_ids, "shape", None) if hasattr(batch, "latent_ids") else None
            print(
                f"[UNIRL latent_prep] batch_size={batch_size} latents={tuple(latents.shape)} "
                f"latent_ids={tuple(lid_shape) if lid_shape is not None else None} "
                f"dtype={latents.dtype} contig={latents.is_contiguous()}",
                flush=True,
            )
        return batch

    forward._unirl_initial_noise_expand = True  # type: ignore[attr-defined]
    LatentPreparationStage.forward = forward

    _patch_grouped_initial_noise_slice()


def _patch_grouped_initial_noise_slice() -> None:
    """Slice the full driver noise ``[K, ...]`` to each per-output Req's own ``[i:i+1]`` in the grouped forward."""
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

    orig = GPUWorker.__dict__.get("_execute_forward_batch")
    if orig is None or getattr(orig, "_unirl_noise_slice", False):
        return

    def _execute_forward_batch(self, batch):
        n = len(batch)
        if n > 1:
            for i, req in enumerate(batch):
                lat = getattr(req, "latents", None)
                shape = getattr(lat, "shape", None)
                if lat is not None and shape is not None and len(shape) >= 1 and shape[0] == n:
                    req.latents = lat[i : i + 1]
                audio_lat = getattr(req, "audio_latents", None)
                audio_shape = getattr(audio_lat, "shape", None)
                if audio_lat is not None and audio_shape is not None and len(audio_shape) >= 1 and audio_shape[0] == n:
                    req.audio_latents = audio_lat[i : i + 1]
                seeds = getattr(req, "denoise_seeds", None)
                if isinstance(seeds, (list, tuple)) and len(seeds) == n:
                    req.denoise_seeds = [seeds[i]]
        return orig(self, batch)

    _execute_forward_batch._unirl_noise_slice = True  # type: ignore[attr-defined]
    GPUWorker._execute_forward_batch = _execute_forward_batch
