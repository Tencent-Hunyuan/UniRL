"""Driver-authoritative ``initial_noise`` batch expansion + packed-latents
parity for the provided-latents branch (gap #3 / packed-model gap, LIN-365).

UniRL's NoiseRecipe (#208) makes x_T driver-authoritative; the engine
ships it as ``batch.latents`` (via ``patch_sampling_io`` -> ``Req.latents``).
Stock upstream ``LatentPreparationStage.forward`` handles provided latents with
just ``latents = latents.to(device)`` -- it does NOT (a) expand a ``[1, ...]``
or ``[num_prompts, ...]`` tensor to the per-sample ``[batch_size, ...]``, NOR
(b) run the packing / latent-ids prep that the randn branch performs via
``pipeline_config.maybe_pack_latents`` and ``maybe_prepare_latent_ids``.

The fork did both. Without (a), a group-shared or single-seed initial noise
crashes or mis-shapes the rollout. Without (b), packed models (FLUX.2-Klein /
M3) leave ``batch.latent_ids = None``, and ``DenoisingStage._prepare_denoising_loop
-> Flux2PipelineConfig.prepare_pos_cond_kwargs -> get_freqs_cis`` then dies on
``batch.latent_ids.ndim`` (AttributeError: NoneType). The randn branch in
upstream does exactly the right thing: order is
``maybe_prepare_latent_ids(unpacked) -> set batch.latent_ids -> maybe_pack_latents``,
and we mirror that here for the provided-latents path.

This REPLACES ``forward``'s provided-latents branch and delegates the
``latents is None`` (randn) branch back to upstream's ``forward``. Pure SD3
remains a no-op because SD3's ``maybe_prepare_latent_ids`` returns ``None`` and
``maybe_pack_latents`` returns the latents unchanged. The grouped path
(``run_grouped_requests``) already delegates to ``forward`` per-batch whenever
any ``batch.latents is not None``, so wrapping ``forward`` alone covers
rollout-with-initial_noise on both the single and grouped paths.
"""

from __future__ import annotations

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)

_DEBUG = os.environ.get("UNIRL_DEBUG_LATENT_SHAPE") == "1"


def _expand_initial_noise(latents: torch.Tensor, batch_size: int, num_outputs_per_prompt: int) -> torch.Tensor:
    """Expand provided initial noise to ``batch_size`` (fork's rule).

    shape[0] == 1            -> broadcast to all samples
    shape[0] == num_prompts  -> repeat_interleave per num_outputs_per_prompt
    shape[0] == batch_size   -> use as-is
    """
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
    """Slice driver ``initial_noise`` per output index in the grouped forward.

    The scheduler expands a ``num_outputs_per_prompt=K`` request into K per-output
    Reqs (worker grouped path, ``GPUWorker._execute_forward_batch(batch: list[Req])``),
    but each carries the FULL driver noise ``[K, ...]`` rather than its own
    ``[i:i+1]`` slice -- so ``LatentPreparationStage`` sees ``batch_size=1`` with
    ``initial_noise`` dim K and raises. (The fork ran the whole group as one
    ``batch_size=K`` forward, so ``[K]`` matched.) Slice each per-output Req's
    latents to its index right before the grouped forward; the i-th expanded Req
    is the i-th output (scheduler expands in ``range(nopp)`` order), and the driver
    ships noise in sample order, so position i -> ``noise[i]``.
    """
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
