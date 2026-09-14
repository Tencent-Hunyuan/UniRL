"""Stage-driven ``FlowGRPO`` over a ``LatentSegment``."""

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
    _require_replay_anchor_for_batched_replay,
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
    timestep_chunk_size: Optional[int] = None
    params: Any = dc_field(default=None)


class FlowGRPO(StageAlgorithm):
    """GRPO over a diffusion ``LatentSegment`` via ``DiffusionStage.replay``."""

    supports_multi_update = True
    requires_backend = True
    anchor_fields = ("sde_logp",)

    @property
    def recomputes_anchor(self) -> bool:
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
        timestep_chunk_size: Optional[int] = None,
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
        self.timestep_chunk_size = None if timestep_chunk_size is None else int(timestep_chunk_size)
        require(
            self.timestep_chunk_size is None or self.timestep_chunk_size >= 1,
            f"FlowGRPO: timestep_chunk_size must be >= 1; got {timestep_chunk_size!r}",
        )
        _require_replay_anchor_for_batched_replay(self.stage, self.old_logp_source, algo="FlowGRPO")
        self.conditions_cls = conditions_cls

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "LatentSegment",
    ) -> None:
        """Establish the frozen π_old anchor (``segment.sde_logp``) before the ``num_updates_per_batch`` loop."""
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
        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)

        if self.timestep_chunk_size is not None:
            return self._chunked_loss_and_backward(
                typed_conds=typed_conds,
                segment=segment,
                advantages=advantages,
                loss_scale=loss_scale,
                clip_range=clip_range,
                target_steps=target_steps,
            )

        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        new_logp = replay_result.log_probs
        new_means = replay_result.prev_sample_means

        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )

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

    def _chunked_loss_and_backward(
        self,
        *,
        typed_conds: Any,
        segment: "LatentSegment",
        advantages: torch.Tensor,
        loss_scale: float,
        clip_range: float,
        target_steps: List[int],
    ) -> AlgorithmStepResult:
        """Replay + backward one ``timestep_chunk_size`` slice at a time, dropping each slice's graph."""
        num_steps = len(target_steps)
        chunks = [target_steps[i : i + self.timestep_chunk_size] for i in range(0, num_steps, self.timestep_chunk_size)]
        total_loss = 0.0
        kl_ref_total = 0.0
        new_logp_parts: List[torch.Tensor] = []
        old_logp_parts: List[torch.Tensor] = []
        for chunk in chunks:
            replay_result = self.stage.replay(
                typed_conds,
                segment=segment,
                params=self.params,
                step_indices=chunk,
            )
            new_logp = replay_result.log_probs
            old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, chunk, field_name="sde_logp").to(
                dtype=new_logp.dtype, device=new_logp.device
            )
            adv_b = (
                advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)
            )
            loss_per_elem, _ = _grpo_clip_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantages=adv_b,
                clip_range=clip_range,
            )
            chunk_loss = loss_per_elem.mean()
            if self.beta > 0.0:
                new_means = replay_result.prev_sample_means
                if new_means is None:
                    raise RuntimeError(
                        "FlowGRPO: beta>0 requires stage.replay() to return prev_sample_means, "
                        "but got None. Ensure the stage's replay method produces means."
                    )
                sigma_t = _transition_sigma(
                    self.stage,
                    segment=segment,
                    target_steps=chunk,
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
                    target_steps=chunk,
                ).to(dtype=new_means.dtype, device=new_means.device)
                kl_ref = _reference_kl_loss(new_means, ref_means, sigma_t)
                chunk_loss = chunk_loss + self.beta * kl_ref
                kl_ref_total += float(kl_ref.detach().item()) * len(chunk)
            # len(chunk)/num_steps makes the chunk backwards sum to the full [B, S] mean backward.
            (chunk_loss * loss_scale * (len(chunk) / num_steps)).backward()
            total_loss += float(chunk_loss.detach().item()) * len(chunk)
            new_logp_parts.append(new_logp.detach())
            old_logp_parts.append(old_logp.detach())
        total_loss /= num_steps

        # Ratio metrics compose only over the full [B, S] set; per-chunk std/min/max would not.
        new_logp_all = torch.cat(new_logp_parts, dim=1)
        old_logp_all = torch.cat(old_logp_parts, dim=1)
        adv_all = (
            advantages.detach()
            .to(dtype=new_logp_all.dtype, device=new_logp_all.device)
            .reshape(-1, 1)
            .expand_as(new_logp_all)
        )
        _, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp_all,
            old_logp=old_logp_all,
            advantages=adv_all,
            clip_range=clip_range,
        )
        metrics: Dict[str, Any] = {
            "policy_loss": total_loss,
            "clip_range": float(clip_range),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        if self.beta > 0.0:
            metrics["beta"] = float(self.beta)
            metrics["kl_ref_mean"] = kl_ref_total / num_steps
        return AlgorithmStepResult(
            loss=total_loss,
            metrics=metrics,
            num_steps_or_tokens=num_steps,
            has_backward=True,
        )

    def _resolve_target_steps(self, segment: "LatentSegment") -> List[int]:
        """All SDE-recorded step indices on the segment."""
        if segment.sde_indices is None:
            return []
        return [int(i) for i in segment.sde_indices.tolist()]


__all__ = ["FlowGRPO", "FlowGRPOConfig"]
