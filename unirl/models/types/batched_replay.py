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
        """Replay ``target`` steps in one step-major batch; log-probs ``[B, S]``, slot ``s`` is ``target[s]``."""
        S = len(target)
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


__all__ = ["BatchedStepReplayMixin"]
