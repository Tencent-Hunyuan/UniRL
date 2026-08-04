"""Shared batched replay for stateless diffusion SDE stages.

The ``S`` replay steps of a stored trajectory are stacked step-major on the
batch dim (``[S*B, ...]``), so one batched step call replaces the serial
per-step loop: ONE transformer forward — i.e. one FSDP all-gather of the
sharded transformer — instead of ``S``. Concrete stages supply the per-model
pieces via :meth:`BatchedStepReplayMixin._tile_conditions` and
:meth:`BatchedStepReplayMixin._batched_step_kwargs`.

Callers gate this path on all three of: the stage's ``batch_replay_steps``
flag, ``S > 1``, and a stateless :class:`~unirl.sde.kernels.SDEStrategy`
(Flow / Dance / CPS, which ignore ``step_index``). Anything else takes the
serial loop; the stateful ``DPM2Strategy`` is an ODE strategy and so can
never enter here.

Requires ``old_logp_source='replay'``
-------------------------------------
A ``[S*B]`` forward is numerically equivalent to the ``[B]`` forward the
rollout ran, but **not** bit-identical: the batch shape changes GEMM tiling
and reduction order, so per-sample log-probs move at the rounding level —
and by more on models whose forward restructures with the batch (z_image's
per-sample lists, flux2_klein's rebuilt RoPE ids) than on sd3 / qwen_image.

The PPO ratio therefore stays exact only when the π_old anchor and the train
pass take the *same* batched path at the *same* micro-geometry — which is
what ``old_logp_source='replay'`` gives, since ``TrainStack.prepare_segment``
replays the anchor per micro-slice. Pairing the flag with the default
``'rollout'`` anchor is rejected at algorithm construction
(``_require_replay_anchor_for_batched_replay``).
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

        Each block of ``B`` rows reuses the same per-sample conditioning,
        since all ``S`` steps replay the SAME ``B`` trajectories at different
        timesteps. A field left untiled here is silently dropped from the
        batched forward, so cover the whole conditions container.
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
        # Each step-major block contains all B samples for one target step.
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
        # Restore the [B, S] ordering used by ReplayResult.
        log_probs_t = log_prob_all.view(S, B).transpose(0, 1).contiguous().to(dtype=self.logprob_dtype)
        means_t = None
        if prev_mean_all is not None:
            tail = prev_mean_all.shape[1:]
            means_t = prev_mean_all.view(S, B, *tail).transpose(0, 1).contiguous().to(dtype=self.trajectory_dtype)
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)


__all__ = ["BatchedStepReplayMixin"]
