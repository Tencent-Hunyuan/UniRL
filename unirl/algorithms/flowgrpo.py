"""Stage-driven ``FlowGRPO`` over a ``LatentSegment``.

Implements :class:`StageAlgorithm` and shares the module-level
``_grpo_clip_loss`` / ``_resolve_clip_range_from_schedule`` helpers (in
:mod:`unirl.algorithms.base`) with :class:`GRPO` so their loss math
stays identical. CFG batching, predict_noise, SDE math, autocast, and per-step
iteration are owned by ``stage.replay(...)``; the algorithm is ~20 lines of
ratio-clip math.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.config.require import require
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _reference_kl_loss,
    _reference_replay_means,
    _resolve_clip_range_from_schedule,
    _resolve_reference_model,
    _transition_sigma,
    gather_sde_field,
    typed_conditions,
)


@dataclass
class FlowGRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    beta: float = 0.0
    old_logp_source: str = "rollout"
    params: Any = dc_field(default=None)


class FlowGRPO(StageAlgorithm):
    """GRPO over a diffusion ``LatentSegment`` via ``DiffusionStage.replay``.

    The whole forward path (CFG batching, noise prediction, SDE math, autocast,
    per-step iteration) is owned by :meth:`DiffusionStage.replay`; this class
    is pure ratio-clip math against ``segment.sde_logp``.

    Args:
        stage: The :class:`DiffusionStage` whose ``replay`` produces new
            log-probs aligned with ``segment.sde_logp[:, slot_for_steps]``.
        params: The per-call params object the stage's ``replay`` consumes
            (e.g. ``SD3DiffusionParams``). Held as algorithm state so the
            dispatcher doesn't need to know it.
        clip_range: PPO clip range epsilon.
        clip_schedule: ``"constant"``, ``"linear_decay"``, or
            ``"cosine_decay"`` — applied via ``training_progress``.
        old_logp_source: ``"rollout"`` (default) trusts the rollout engine's
            emitted ``segment.sde_logp``; ``"replay"`` recomputes it via
            ``stage.replay`` at pre-update weights. See :meth:`prepare_segment`.
        beta: Reference-policy KL coefficient (Flow-GRPO eq.5). ``> 0`` adds
            ``beta * KL(pi_theta || pi_ref)`` to the clipped loss, where ``pi_ref``
            is the base model with its LoRA adapter disabled (a per-update no_grad
            reference replay). ``0`` (default) disables the term and skips that
            replay. Requires a LoRA recipe + the injected ``backend``.
            The KL is normalized by the full per-step transition std
            (``std_dev_t*sqrt(-dt)`` for Flow/Dance) — the exact Gaussian KL.
            The reference flow_grpo code divides by ``std_dev_t**2`` only, so at
            equal ``beta`` this term is ~``1/|dt|`` stronger (≈10x at 10 sampling
            steps): don't port ``beta`` values 1:1 from flow_grpo configs.
        backend: FSDP backend sibling (injected by the v2 trainer). Only used when
            ``beta > 0`` to reach the trainable model for the adapter-disabled
            reference replay.
        conditions_cls: Stage-typed conditions container with a
            ``from_dict(Mapping[str, Condition])`` classmethod. ``None``
            forwards the dict verbatim (unit-test path).
    """

    # prepare_segment freezes segment.sde_logp once, so the PPO ratio stays
    # anchored across every num_updates_per_batch optimizer step.
    supports_multi_update = True
    # beta>0 disables the LoRA adapter for a reference-policy replay, so the v2
    # trainer must inject the FSDP backend (the trainable model lives on it).
    requires_backend = True
    anchor_fields = ("sde_logp",)

    def recomputes_anchor(self) -> bool:
        # Only ``replay`` re-derives sde_logp; ``rollout`` keeps the engine's emission.
        return self.old_logp_source == "replay"

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        beta: float = 0.0,
        old_logp_source: str = "rollout",
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("FlowGRPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.params = params
        self.clip_range = float(clip_range)
        self.clip_schedule = str(clip_schedule)
        self.beta = float(beta)
        self._ref_model = _resolve_reference_model(backend, beta=self.beta, algo="FlowGRPO")
        self.old_logp_source = str(old_logp_source).strip().lower()
        require(
            self.old_logp_source in ("rollout", "replay"),
            f"FlowGRPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}",
        )
        self.conditions_cls = conditions_cls

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "LatentSegment",
    ) -> None:
        """Establish the frozen π_old anchor (``segment.sde_logp``) before the
        ``num_updates_per_batch`` loop. The source is chosen by ``old_logp_source``:

        - ``"rollout"`` (default): trust the rollout engine's emitted
          ``segment.sde_logp``. Raises if the engine emitted nothing
          (``sde_logp is None``) — pin a rollout build that emits per-step
          log-probs, or set ``old_logp_source='replay'``.
        - ``"replay"``: recompute via a ``torch.no_grad``
          :meth:`DiffusionStage.replay` at the **pre-update** weights and
          **overwrite** ``sde_logp`` (ignoring any engine value). Frozen for
          all N micro-updates that follow.

        No-op if the segment has no SDE-gated steps to train on.
        """
        if segment.sde_indices is None:
            return
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        if self.old_logp_source == "rollout":
            if segment.sde_logp is None:
                raise RuntimeError(
                    "FlowGRPO.prepare_segment: old_logp_source='rollout' but the "
                    "rollout engine emitted no per-step log-probs (segment.sde_logp is "
                    "None). Pin a rollout build that emits trajectory log-probs, or set "
                    "old_logp_source='replay'."
                )
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        segment.sde_logp = result.log_probs.detach().cpu()

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)

        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        new_logp = replay_result.log_probs  # [B, S']
        new_means = replay_result.prev_sample_means  # [B, S', ...]; used only when beta>0

        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_b = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)

        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_b,
            clip_range=clip_range,
        )
        policy_loss = loss_per_elem.mean()
        loss = policy_loss
        metrics: Dict[str, Any] = {
            "policy_loss": float(policy_loss.detach().item()),
            "clip_range": float(clip_range),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }

        # Optional reference-policy KL penalty (Flow-GRPO eq.5): pull pi_theta toward
        # pi_ref (LoRA-disabled base model).
        if self.beta > 0.0:
            if new_means is None:
                raise RuntimeError(
                    "FlowGRPO: beta>0 requires stage.replay() to return prev_sample_means, "
                    "but got None. Ensure the stage's replay method produces means."
                )
            sigma_t = _transition_sigma(
                self.stage,
                segment=segment,
                target_steps=target_steps,
                eta=float(self.params.eta),
                device=new_logp.device,
                add_coefficient=True,
            )
            ref_means = _reference_replay_means(
                self.stage,
                self._ref_model,
                conditions=typed_conds,
                segment=segment,
                params=self.params,
                target_steps=target_steps,
            ).to(dtype=new_means.dtype, device=new_means.device)
            kl_ref = _reference_kl_loss(new_means, ref_means, sigma_t)
            loss = loss + self.beta * kl_ref
            metrics["beta"] = float(self.beta)
            metrics["kl_ref_mean"] = float(kl_ref.detach().item())

        (loss * loss_scale).backward()

        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    # -- helpers --------------------------------------------------------

    def _resolve_target_steps(self, segment: "LatentSegment") -> List[int]:
        """All SDE-recorded step indices on the segment.

        Subclasses can override to apply skip-last / skip-initial filtering or
        to honor a training-indices schedule; the default trains every step
        the rollout recorded.
        """
        if segment.sde_indices is None:
            return []
        return [int(i) for i in segment.sde_indices.tolist()]


__all__ = ["FlowGRPO", "FlowGRPOConfig"]
