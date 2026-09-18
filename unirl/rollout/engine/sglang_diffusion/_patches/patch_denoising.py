"""Per-sample SDE noise via ``denoise_seeds`` + per-request fallback."""

from __future__ import annotations

import hashlib
import os

import torch

from unirl.sde.noise import MAX_TORCH_SEED, make_denoise_step_generators


def _make_step_generators(
    base_seed: int,
    step_index: int,
    device: torch.device,
    denoise_seeds: list[str],
) -> list[torch.Generator]:
    """Per-sample deterministic CPU generators for one SDE step."""
    del device
    return make_denoise_step_generators(
        base_seed=int(base_seed),
        step_index=int(step_index),
        sample_ids=[str(seed_key) for seed_key in denoise_seeds],
    )


def _resolve_base_seed(batch) -> int | None:
    seed = getattr(batch, "seed", None)
    if seed is None:
        seed = getattr(getattr(batch, "sampling_params", None), "seed", None)
    return int(seed) if seed is not None else None


def _resolve_fallback_seed(batch) -> int:
    """Deterministic per-request seed for the single-``torch.Generator`` fallback."""
    base_seed = _resolve_base_seed(batch)
    denoise_seeds = getattr(batch, "denoise_seeds", None)
    sample_key = str(denoise_seeds[0]) if denoise_seeds else None
    if base_seed is not None and sample_key is not None:
        payload = (f"{int(base_seed)}::fallback::sample::{sample_key}").encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) % MAX_TORCH_SEED
    return int.from_bytes(os.urandom(8), byteorder="big") % MAX_TORCH_SEED


def patch_denoising() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    orig = DenoisingStage._run_denoising_step
    if getattr(orig, "_unirl_denoise_seeds", False):
        return

    def _run_denoising_step(self, ctx, step, batch, server_args):
        denoise_seeds = getattr(batch, "denoise_seeds", None)
        if getattr(batch, "rollout", False) and denoise_seeds is not None:
            base_seed = _resolve_base_seed(batch)
            if base_seed is not None:
                ctx.extra_step_kwargs["generator"] = _make_step_generators(
                    base_seed,
                    int(step.step_index),
                    ctx.latents.device,
                    list(denoise_seeds),
                )
        return orig(self, ctx, step, batch, server_args)

    _run_denoising_step._unirl_denoise_seeds = True  # type: ignore[attr-defined]
    DenoisingStage._run_denoising_step = _run_denoising_step

    _patch_rollout_variance_noise_device()


def _patch_rollout_variance_noise_device() -> None:
    """Make ``SchedulerRLMixin._rollout_variance_noise`` tolerate CPU generators."""
    from sglang.multimodal_gen.runtime.post_training.scheduler_rl_mixin import (
        SchedulerRLMixin,
    )

    if getattr(SchedulerRLMixin._rollout_variance_noise, "_unirl_dev", False):
        return

    def _rollout_variance_noise(self, batch, model_output, generator):
        assert generator is not None, "Generator must be provided"
        rsd = self._get_rollout_session_data(batch)
        device = model_output.device
        dtype = model_output.dtype
        local_shape = tuple(model_output.shape)
        B = local_shape[0]
        if isinstance(generator, torch.Generator):
            # Seed fallback generators per request when a subclass bypasses the denoising wrapper.
            assert B == 1, "Generator must be a list if batch size is not 1"
            gen = getattr(batch, "_unirl_noise_gen", None)
            if gen is None:
                gen = torch.Generator(device=device)
                gen.manual_seed(_resolve_fallback_seed(batch))
                try:
                    batch._unirl_noise_gen = gen  # type: ignore[attr-defined]
                except AttributeError:
                    pass  # immutable batch — generator is still valid for this step
            generator = [gen]
        else:
            assert len(generator) == B, "Generator list must have the same length as batch size"
        buffer = self._get_or_create_rollout_noise_buffer(rsd, rsd.latents_shape, device, dtype)
        for i in range(B):
            g = generator[i]
            if g is not None and getattr(g, "device", None) is not None and g.device.type != buffer.device.type:
                tmp = torch.randn(rsd.latents_shape, generator=g, dtype=dtype, device=g.device)
                buffer[i : i + 1].copy_(tmp)
            else:
                torch.randn(rsd.latents_shape, out=buffer[i : i + 1], generator=g)
        sharded_noise, _ = rsd.pipeline_config.shard_latents_for_sp(batch=batch, latents=buffer)
        if tuple(sharded_noise.shape) != local_shape:
            raise ValueError(
                "Rollout SP noise shape mismatch after shard. "
                f"Expected local_shape={local_shape}, got {tuple(sharded_noise.shape)}."
            )
        return sharded_noise

    _rollout_variance_noise._unirl_dev = True  # type: ignore[attr-defined]
    SchedulerRLMixin._rollout_variance_noise = _rollout_variance_noise
