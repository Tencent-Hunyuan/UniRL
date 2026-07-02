"""Shared batched-step replay for diffusion stages.

Batched-step replay (shipped for SD3 in
`PR #144 <https://github.com/Tencent-Hunyuan/UniRL/pull/144>`_ and extended to
``qwen_image`` / ``z_image`` / ``flux2_klein``) is an **algorithm-level**
optimization, not a per-model one. GRPO/FlowGRPO/FlowDPPO-style diffusion RL
replays a stored trajectory to recompute per-SDE-step log-probs, and every
``DiffusionStage.replay`` does this with a serial S-step loop — S transformer
forwards ⇒ **S FSDP all-gathers** of the full sharded transformer per
``replay()`` call. The optimization stacks all S steps on the batch dim
(``sample``/``prev_sample`` → ``[S*B, ...]``, step-major; per-step sigmas ride
as ``[S*B]`` vectors; conditioning tiled S×) and runs **one** forward + one
vectorized SDE transition — **S all-gathers ⇒ 1**.

The stacking, sigma vectorization, single ``step_with_logp`` call, and the
``[S*B] → [B, S]`` reshape are **identical** across every diffusion model: the
transformer has no cross-sample interaction, so per-sample results match the
serial path up to bf16 batch-shape rounding, and — because the π_old anchor is
replayed through this same path under ``old_logp_source='replay'`` — the
on-policy ratio stays exactly 1. Only two things are genuinely model-specific,
so they are the mixin's extension points:

* ``_tile_conditions(conditions, repeats)`` — repeat this model's conditioning
  ``repeats``× along the batch dim, and
* ``_batched_step_kwargs(segment, params)`` — any extra per-model kwargs for
  ``step.step_with_logp`` (default: none; ``qwen_image`` supplies
  ``latent_h`` / ``latent_w`` / ``distilled_guidance_scale``).

Correctness is gated by the caller (``replay``): the batched path is entered
only for stateless, step-index-independent SDE strategies (``SDEStrategy``:
Flow / Dance / CPS) and only when S > 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import torch

from unirl.models.types.replay_result import ReplayResult

if TYPE_CHECKING:
    from unirl.types.segments.latent import LatentSegment


class BatchedStepReplayMixin:
    """Model-agnostic ``_replay_batched_steps`` for diffusion stages.

    Mix in **before** the ``DiffusionStage[...]`` protocol base so the shared
    ``_replay_batched_steps`` is inherited (concrete stages then delete their
    copy). The host stage must expose ``model``, ``step``, ``strategy``,
    ``logprob_dtype`` and ``trajectory_dtype`` (all four current diffusion
    stages do) and provide ``_tile_conditions``. See the module docstring for
    the algorithm and its correctness invariant.
    """

    def _tile_conditions(self, conditions: Any, repeats: int) -> Any:
        """Repeat this model's conditioning ``repeats``× along the batch dim.

        Each block of ``B`` rows reuses the same per-sample conditioning, since
        all ``S`` replayed steps share the SAME ``B`` trajectories at different
        timesteps. Concrete stages override this (typically as a
        ``@staticmethod``); the tiling is the only inherently model-specific
        part of the batched path.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _tile_conditions for batched replay")

    def _batched_step_kwargs(self, segment: "LatentSegment", params: Any) -> Dict[str, Any]:
        """Extra per-model kwargs for the batched ``step.step_with_logp`` call.

        Default: none — the common signature (sample / prev_sample / sigma /
        sigma_next / guidance_scale / eta / sigma_max / step_index) covers most
        models. Override when the kernel needs more, e.g. ``qwen_image`` derives
        ``latent_h`` / ``latent_w`` from the stored latent grid and forwards
        ``distilled_guidance_scale`` from ``params``.
        """
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
        """Replay all ``target`` SDE steps in a single batched forward.

        Stacks the ``S`` steps step-major on the batch dim (``[S*B, ...]``),
        tiles the conditioning ``S``× (:meth:`_tile_conditions`), carries the
        per-step ``sigma`` / ``sigma_next`` as ``[S*B]`` vectors, runs ONE
        ``step.step_with_logp`` (one transformer forward + one vectorized SDE
        transition), then reshapes the per-step log-probs / means back to
        ``[B, S]`` so slot ``s`` aligns with ``segment.sde_logp`` ordering. See
        the module docstring for why this is bit-identical to the serial loop
        (ratio ≡ 1).

        ``step_index`` is passed as ``target[0]`` for signature parity; the
        guarded stateless SDE strategies ignore it.
        """
        S = len(target)
        # Step-major stack: rows [k*B:(k+1)*B] are all B samples at step target[k].
        sample_all = torch.cat([segment.latents_at(i).to(device) for i in target], dim=0)
        prev_all = torch.cat([segment.latents_at(i + 1).to(device) for i in target], dim=0)
        B = sample_all.shape[0] // S
        # Per-sample sigma vectors aligned with the step-major stack.
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
        # [S*B] -> [S, B] -> [B, S] so slot s aligns with segment.sde_logp ordering.
        log_probs_t = log_prob_all.view(S, B).transpose(0, 1).contiguous().to(dtype=self.logprob_dtype)
        means_t = None
        if prev_mean_all is not None:
            tail = prev_mean_all.shape[1:]
            means_t = prev_mean_all.view(S, B, *tail).transpose(0, 1).contiguous().to(dtype=self.trajectory_dtype)
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)


__all__ = ["BatchedStepReplayMixin"]
