"""Shared batched replay for stateless diffusion SDE stages.

The ``S`` replay steps are stacked step-major on the batch dim (``[S*B, ...]``)
so one batched call replaces the serial loop: ONE transformer forward, i.e. one
FSDP all-gather, instead of ``S``. It costs ~``S``× the replay activation
footprint, so recipes without activation checkpointing can OOM where the serial
loop fit. Gated by callers on ``batch_replay_steps``, ``S > 1``, and a stateless
SDE strategy (Flow / Dance / CPS ignore ``step_index``; ODE ``DPM2Strategy``
cannot reach here).

Requires ``old_logp_source='replay'``: a ``[S*B]`` forward is numerically
equivalent to the rollout's ``[B]`` forward but not bit-identical (batch shape
changes GEMM tiling and reduction order), and by more on models that restructure
with the batch — z_image's per-sample lists, flux2_klein's rebuilt RoPE ids —
than on sd3 / qwen_image. So the ratio is exact only when the π_old anchor and
the train pass share this path at the same micro-geometry, which is what a
replay anchor gives (``TrainStack.prepare_segment`` replays per micro-slice).
``_require_replay_anchor_for_batched_replay`` rejects the ``'rollout'`` pairing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import torch

from unirl.models.types.replay_result import ReplayResult

if TYPE_CHECKING:
    from unirl.types.segments.latent import LatentSegment


class BatchedStepReplayMixin:
    """Model-agnostic batched replay implementation.

    The host stage must provide ``model``, ``step``, ``strategy``,
    ``logprob_dtype``, ``trajectory_dtype``, and :meth:`_tile_conditions`.
    """

    def _tile_conditions(self, conditions: Any, repeats: int) -> Any:
        """Repeat EVERY conditioning field ``repeats``× along the batch dim.

        All ``S`` blocks replay the same ``B`` trajectories, so each block
        reuses the same per-sample conditioning. A field left untiled is
        silently dropped from the batched forward.
        """
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
        """Replay ``target`` steps in one step-major batch.

        Returns log-probs as ``[B, S]`` with slot ``s`` aligned to
        ``target[s]``, matching what the serial loop stacks — so callers can
        swap paths without touching ``segment.sde_logp`` ordering.

        Callers restrict this path to stateless SDE strategies, so
        ``target[0]`` can be used as the shared ``step_index``.
        """
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
