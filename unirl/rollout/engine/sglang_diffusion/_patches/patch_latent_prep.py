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

        # Pure randn branch: upstream handles shape/ids/packing correctly.
        if latents is None or not torch.is_tensor(latents):
            return orig(self, batch, server_args)

        # Provided-latents branch: drive the same sequence the randn branch
        # runs, so packed pipelines (FLUX.2-Klein etc.) get batch.latent_ids
        # populated and batch.latents arrives at DenoisingStage in packed form.
        device = get_local_torch_device()
        batch_size = int(batch.batch_size)
        nopp = int(getattr(batch, "num_outputs_per_prompt", 1) or 1)
        pcfg = server_args.pipeline_config

        # 1) Expand driver-shipped noise to batch_size (fork's rule).
        latents = _expand_initial_noise(latents, batch_size, nopp).to(device=device)

        # 1b) Insert the singleton frame axis for image-form 5-D DiTs (Z-Image).
        # The driver ships frame-less image-form noise [B, C, H, W] (from the model's
        # latent_shape), but Z-Image's S3-DiT denoises in 5-D [B, C, 1, H, W]:
        # ``ZImageTransformer2DModel._as_image_list`` treats a 4-D tensor as a SINGLE
        # [C, F, H, W] image, folding the 16 channels into the token count so the
        # x_embedder matmul fails. The randn branch already gets 5-D via
        # ``prepare_latent_shape``; mirror that for provided latents. Gated TIGHTLY to
        # the Z-Image signature (4-D driver -> 5-D expected with channels preserved at
        # dim 1 and a size-1 frame at dim 2) so it is a strict no-op for every other
        # family: SD3/Flux/ernie (4-D expected), Qwen-Image ([B, 1, C, H, W] — dim 1 is
        # the frame, not channels, so expected[1] != C), and video (frame > 1).
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
            # Surface at normal log level (not _DEBUG-gated): a swallowed reconcile
            # leaves un-reshaped latents that fail later inside the DiT, far from here.
            logger.warning("latent_prep frame-axis reconcile skipped: %s: %s", type(exc).__name__, exc)

        # 2) Compute latent_ids on the UNPACKED shape (mirror randn branch order).
        latent_ids = pcfg.maybe_prepare_latent_ids(latents)
        if latent_ids is not None:
            batch.latent_ids = latent_ids.to(device=device)

        # 3) Pack latents per pipeline (no-op for SD3; real pack for FLUX.2).
        latents = pcfg.maybe_pack_latents(latents, batch_size, batch)

        # 4) init_noise_sigma scaling (same as upstream forward end).
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
                # initial_noise [K, ...] -> this output's [i:i+1].
                lat = getattr(req, "latents", None)
                shape = getattr(lat, "shape", None)
                if lat is not None and shape is not None and len(shape) >= 1 and shape[0] == n:
                    req.latents = lat[i : i + 1]
                # denoise_seeds [K] -> [this output's seed] (the per-step generator
                # list is built from it; len must equal the per-output batch=1).
                # This per-Req slice is what makes each sample's SDE noise
                # independent (with sample_ids keys) -- the missing-rename of this
                # block is what crashed the B=1 grouped forward (LIN-365).
                seeds = getattr(req, "denoise_seeds", None)
                if isinstance(seeds, (list, tuple)) and len(seeds) == n:
                    req.denoise_seeds = [seeds[i]]
        return orig(self, batch)

    _execute_forward_batch._unirl_noise_slice = True  # type: ignore[attr-defined]
    GPUWorker._execute_forward_batch = _execute_forward_batch
