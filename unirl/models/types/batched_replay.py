"""Batched replay for stateless SDE stages — S steps stacked step-major as ``[S*B, ...]``, replay-anchor only."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import torch

from unirl.models.types.replay_result import ReplayResult

if TYPE_CHECKING:
    from unirl.types.segments.latent import LatentSegment


class BatchedStepReplayMixin:
    """Model-agnostic batched replay implementation."""

    def _tile_conditions(self, conditions: Any, repeats: int) -> Any:
        """Repeat EVERY conditioning field ``repeats``× along the batch dim."""
        raise NotImplementedError(f"{type(self).__name__} must implement _tile_conditions for batched replay")

    def _batched_step_kwargs(self, segment: "LatentSegment", params: Any) -> Dict[str, Any]:
        """Return model-specific arguments for the batched step call."""
        return {}

    def _batched_replay_target_chunks(
        self,
        *,
        target: List[int],
        sigmas: torch.Tensor,
        params: Any,
    ) -> List[List[int]]:
        """Split targets into bounded step-major batches; subclasses may add routing boundaries."""
        del sigmas, params
        configured_limit = getattr(self, "replay_step_batch_size", None)
        limit = len(target) if configured_limit is None else int(configured_limit)
        if limit < 1:
            raise ValueError(f"{type(self).__name__}.replay_step_batch_size must be >= 1, got {limit}")
        return [target[start : start + limit] for start in range(0, len(target), limit)]

    def _replay_batched_step_chunk(
        self,
        conditions: Any,
        *,
        segment: "LatentSegment",
        params: Any,
        target: List[int],
        sigmas: torch.Tensor,
        sigma_max: torch.Tensor,
        device: torch.device,
    ) -> ReplayResult:
        """Replay one target chunk as a step-major batch; output slot ``s`` is ``target[s]``."""
        S = len(target)
        if S < 1:
            raise ValueError(f"{type(self).__name__} received an empty batched replay chunk")
        sample_all = torch.cat([segment.latents_at(i).to(device) for i in target], dim=0)
        prev_all = torch.cat([segment.latents_at(i + 1).to(device) for i in target], dim=0)
        B = sample_all.shape[0] // S
        sigma_all = torch.cat([sigmas[i].to(torch.float32).expand(B) for i in target], dim=0)
        sigma_next_all = torch.cat([sigmas[i + 1].to(torch.float32).expand(B) for i in target], dim=0)
        tiled = self._tile_conditions(conditions, S)

        _, log_prob_all, prev_mean_all = self.step.step_with_logp(
            self.model,
            tiled,
            strategy=self.strategy,
            sample=sample_all,
            prev_sample=prev_all,
            sigma=sigma_all,
            sigma_next=sigma_next_all,
            guidance_scale=float(params.guidance_scale),
            eta=float(params.eta),
            sigma_max=sigma_max,
            step_index=int(target[0]),
            **self._batched_step_kwargs(segment, params),
        )
        if log_prob_all is None:
            raise RuntimeError(
                f"{type(self).__name__}._replay_batched_steps: strategy returned None "
                f"log-prob (deterministic mode); batched replay requires a stochastic "
                f"SDE strategy."
            )
        log_probs_t = log_prob_all.view(S, B).transpose(0, 1).contiguous().to(dtype=self.logprob_dtype)
        means_t = None
        if prev_mean_all is not None:
            tail = prev_mean_all.shape[1:]
            means_t = prev_mean_all.view(S, B, *tail).transpose(0, 1).contiguous().to(dtype=self.trajectory_dtype)
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    def _replay_batched_steps(
        self,
        conditions: Any,
        *,
        segment: "LatentSegment",
        params: Any,
        target: List[int],
        sigmas: torch.Tensor,
        sigma_max: torch.Tensor,
        device: torch.device,
    ) -> ReplayResult:
        """Replay targets in bounded step-major batches and restore the requested target order."""
        chunks = self._batched_replay_target_chunks(target=target, sigmas=sigmas, params=params)
        flattened = [step_idx for chunk in chunks for step_idx in chunk]
        if flattened != target:
            raise RuntimeError(
                f"{type(self).__name__} batched replay chunks changed target order: "
                f"target={target}, flattened={flattened}"
            )
        if len(chunks) == 1:
            return self._replay_batched_step_chunk(
                conditions,
                segment=segment,
                params=params,
                target=chunks[0],
                sigmas=sigmas,
                sigma_max=sigma_max,
                device=device,
            )

        results = [
            self._replay_batched_step_chunk(
                conditions,
                segment=segment,
                params=params,
                target=chunk,
                sigmas=sigmas,
                sigma_max=sigma_max,
                device=device,
            )
            for chunk in chunks
        ]
        log_probs = torch.cat([result.log_probs for result in results], dim=1)
        means = [result.prev_sample_means for result in results]
        if all(mean is None for mean in means):
            prev_sample_means = None
        elif any(mean is None for mean in means):
            raise RuntimeError(f"{type(self).__name__} returned means for only some replay chunks")
        else:
            prev_sample_means = torch.cat([mean for mean in means if mean is not None], dim=1)
        return ReplayResult(log_probs=log_probs, prev_sample_means=prev_sample_means)


__all__ = ["BatchedStepReplayMixin"]
