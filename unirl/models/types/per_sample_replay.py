"""Per-row replay for segments whose rows each trained one SDE step (``segment.sde_index_per_sample``)."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from unirl.models.types.replay_result import ReplayResult

if TYPE_CHECKING:
    from unirl.types.segments.latent import LatentSegment


class PerSampleStepReplayMixin:
    """Replay each row's single SDE transition at its own sigma; needs ``model/step/strategy`` on the stage."""

    supports_per_sample_sde_index: ClassVar[bool] = True

    def _replay_per_sample(
        self,
        conditions: Any,
        *,
        segment: "LatentSegment",
        params: Any,
        device: torch.device,
    ) -> ReplayResult:
        """``log_probs [N, 1]``, ``prev_sample_means [N, 1, ...]`` from latent slots ``[before, after]``."""
        name = type(self).__name__
        if segment.sigmas is None or segment.latents is None:
            raise ValueError(f"{name}._replay_per_sample: segment.sigmas / latents missing")
        num_samples = int(segment.latents.shape[0])
        if int(segment.latents.shape[1]) < 2:
            raise ValueError(
                f"{name}._replay_per_sample: expected latents with a [before, after, ...] slot layout "
                f"(K >= 2); got K={int(segment.latents.shape[1])}."
            )
        sigmas = segment.sigmas.to(device)
        step_idx = segment.sde_index_per_sample.to(device=device, dtype=torch.long)
        if int(step_idx.shape[0]) != num_samples:
            raise ValueError(
                f"{name}._replay_per_sample: sde_index_per_sample length {int(step_idx.shape[0])} "
                f"!= sample count {num_samples}."
            )
        # Pin the strategy's schedule here: only the generation path calls init_schedule,
        # so a remote-engine run that replays from a segment never generates on this rank.
        self.strategy.init_schedule(sigmas)
        sigma = sigmas[step_idx].to(dtype=torch.float32)
        sigma_next = sigmas[step_idx + 1].to(dtype=torch.float32)
        sigma_max = float(sigmas[1].item()) if int(sigmas.shape[0]) > 1 else 0.99
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
                guidance_scale=float(params.guidance_scale),
                eta=float(params.eta),
                sigma_max=sigma_max,
                step_index=0,
            )
        if log_prob is None:
            raise RuntimeError(
                f"{name}._replay_per_sample: strategy returned None log-prob (deterministic mode); "
                "per-sample replay requires a stochastic SDE strategy."
            )
        log_probs_t = log_prob.reshape(num_samples, 1).to(dtype=self.logprob_dtype)
        means_t = prev_mean.reshape(num_samples, 1, *prev_mean.shape[1:]) if prev_mean is not None else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)


__all__ = ["PerSampleStepReplayMixin"]
