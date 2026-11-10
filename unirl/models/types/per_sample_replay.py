"""Per-row replay for segments whose rows each trained one SDE step (``segment.sde_index_per_sample``)."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import torch

from unirl.models.types.replay_result import ReplayResult
from unirl.types.sampling import DiffusionSamplingParams

if TYPE_CHECKING:
    from unirl.types.sample import Part
    from unirl.types.segments.latent import LatentSegment


def stamp_sde_index_per_sample(part: "Part", sde_index: int) -> None:
    """Normalize both streams to [before, after, final]; see models/README.md Gotchas."""
    segment = part.segment
    if segment is None:
        raise ValueError("stamp_sde_index_per_sample: the rollout returned no segment to stamp.")
    segment.sde_index_per_sample = torch.full((part.batch_size,), sde_index, dtype=torch.long)
    if segment.indices is not None and segment.latents is not None:
        before = segment.latents_at(sde_index)
        after = segment.latents_at(sde_index + 1)
        final = segment.latents[:, -1]
        if segment.aux_latents is not None:
            audio_before = segment.aux_latents_at(sde_index)
            audio_after = segment.aux_latents_at(sde_index + 1)
            segment.aux_latents = torch.stack([audio_before, audio_after, segment.aux_latents[:, -1]], dim=1)
        segment.latents = torch.stack([before, after, final], dim=1)
        segment.indices = torch.arange(3, dtype=torch.long, device=segment.indices.device)
    if segment.sde_indices is not None:
        segment.sde_indices = torch.zeros(1, dtype=torch.long, device=segment.sde_indices.device)
    if isinstance(part.sampling_params, DiffusionSamplingParams):
        part.sampling_params = replace(part.sampling_params, sde_indices=None)


class PerSampleStepReplayMixin:
    """Replay each row's single SDE transition at its own sigma; needs ``model/step/strategy`` on the stage."""

    supports_per_sample_sde_index: ClassVar[bool] = True

    def replay_per_sample(
        self,
        conditions: Any,
        *,
        segment: "LatentSegment",
        params: Any,
        device: Optional[torch.device] = None,
    ) -> ReplayResult:
        """``log_probs [N, 1]``, ``prev_sample_means [N, 1, ...]`` from latent slots ``[before, after]``."""
        name = type(self).__name__
        if segment.sigmas is None or segment.latents is None:
            raise ValueError(f"{name}.replay_per_sample: segment.sigmas / latents missing")
        num_samples = segment.latents.shape[0]
        if segment.latents.shape[1] < 2:
            raise ValueError(
                f"{name}.replay_per_sample: expected latents with a [before, after, ...] slot layout "
                f"(K >= 2); got K={segment.latents.shape[1]}."
            )
        if segment.sigmas.shape[0] < 2:
            raise ValueError(f"{name}.replay_per_sample: expected sigmas length >= 2; got {segment.sigmas.shape[0]}.")
        raw_idx = segment.sde_index_per_sample
        if raw_idx.ndim != 1 or raw_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"{name}.replay_per_sample: sde_index_per_sample must be a 1-D integer tensor; "
                f"got dtype={raw_idx.dtype}, shape={tuple(raw_idx.shape)}."
            )
        if raw_idx.shape[0] != num_samples:
            raise ValueError(
                f"{name}.replay_per_sample: sde_index_per_sample length {raw_idx.shape[0]} "
                f"!= sample count {num_samples}."
            )
        if device is None:
            device = segment.latents.device
        sigmas = segment.sigmas.to(device=device, dtype=torch.float32)
        step_idx = raw_idx.to(device=device, dtype=torch.long)
        # Pin the strategy's schedule here: only the generation path calls init_schedule,
        # so a remote-engine run that replays from a segment never generates on this rank.
        self.strategy.init_schedule(sigmas)
        # index_select(0, ...) rejects negative and out-of-bounds indices without host sync.
        sigma = sigmas.index_select(0, step_idx)
        sigma_next = sigmas.index_select(0, step_idx + 1)
        sigma_max = sigmas[1]
        # Fixed slots, not latents_at(): after the per-step groups merge, the shared
        # ``indices`` map holds one group's steps and is not per-sample valid.
        sample = segment.latents[:, 0].to(device)
        prev_sample = segment.latents[:, 1].to(device)

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast_ctx:
            _, log_prob, prev_mean = self.step.step_with_logp(
                self.model,
                conditions,
                strategy=self.strategy,
                sample=sample,
                prev_sample=prev_sample,
                sigma=sigma,
                sigma_next=sigma_next,
                guidance_scale=params.guidance_scale,
                eta=params.eta,
                sigma_max=sigma_max,
                step_index=0,
            )
        if log_prob is None:
            raise RuntimeError(
                f"{name}.replay_per_sample: strategy returned None log-prob (deterministic mode); "
                "per-sample replay requires a stochastic SDE strategy."
            )
        log_probs_t = log_prob.reshape(num_samples, 1).to(dtype=self.logprob_dtype)
        means_t = prev_mean.reshape(num_samples, 1, *prev_mean.shape[1:]) if prev_mean is not None else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)


__all__ = ["PerSampleStepReplayMixin", "stamp_sde_index_per_sample"]
