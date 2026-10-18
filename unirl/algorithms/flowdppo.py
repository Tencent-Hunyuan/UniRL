"""FlowDPPO: KL-divergence-based masking for diffusion RL."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

import torch

from unirl.config.require import require
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _gaussian_kl_div,
    _reference_kl_loss,
    _reference_replay_means,
    _require_replay_anchor_for_batched_replay,
    _resolve_reference_model,
    _transition_sigma,
    gather_sde_field,
    require_timestep_chunk_size,
    slice_step_chunks,
    typed_conditions,
)


@dataclass
class FlowDPPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    kl_mask_threshold: float = 1e-5
    add_kl_coefficient: bool = True
    beta: float = 0.0
    old_logp_source: str = "rollout"
    timestep_chunk_size: Optional[int] = None
    params: Any = dc_field(default=None)


def _flowdppo_kl_adv_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    new_means: torch.Tensor,
    old_means: torch.Tensor,
    advantages: torch.Tensor,
    sigma_t: torch.Tensor,
    kl_mask_threshold: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """FlowDPPO KL-ADV masking loss; log-probs and advantages ``[B, S']``, means ``[B, S', *latent_shape]``."""
    log_diff = new_logp - old_logp
    ratio = torch.exp(log_diff)
    adv = advantages.detach()
    unclipped_loss = -adv * ratio

    kl_per_elem = _gaussian_kl_div(new_means, old_means, sigma_t)
    kl_per_sample = kl_per_elem.mean(dim=tuple(range(2, kl_per_elem.ndim)))

    # KL mask: keep samples where KL < threshold (low divergence → safe to update)
    kl_mask = kl_per_sample < kl_mask_threshold

    pos_rm_mask = (~kl_mask) & (ratio > 1.0) & (adv > 0)
    neg_rm_mask = (~kl_mask) & (ratio < 1.0) & (adv < 0)
    rm_mask = pos_rm_mask | neg_rm_mask
    keep_adv_mask = (~rm_mask).detach()

    # Use torch.where for numerical safety: avoids inf * 0 = nan when ratio overflows
    zero = torch.zeros((), dtype=unclipped_loss.dtype, device=unclipped_loss.device)
    loss_per_elem = torch.where(keep_adv_mask, unclipped_loss, zero)

    if ratio.numel() > 1:
        ratio_std = ratio.std()
    else:
        ratio_std = torch.zeros((), dtype=ratio.dtype, device=ratio.device)
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": (0.5 * log_diff.pow(2)).mean().detach(),
        "kl_new_old_mean": kl_per_sample.mean().detach(),
        "kl_new_old_max": kl_per_sample.max().detach(),
        "kl_mask_fraction": (~kl_mask).float().mean().detach(),
        "pos_rm_fraction": pos_rm_mask.float().mean().detach(),
        "neg_rm_fraction": neg_rm_mask.float().mean().detach(),
        "masked_fraction": rm_mask.float().mean().detach(),
        "unmasked_fraction": keep_adv_mask.float().mean().detach(),
    }
    return loss_per_elem, metrics


class FlowDPPO(StageAlgorithm):
    """FlowDPPO: KL-divergence-based masking for diffusion RL."""

    supports_multi_update = True
    requires_backend = True
    recomputes_anchor = True
    anchor_fields = ("sde_logp", "sde_means")

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        kl_mask_threshold: float = 1e-5,
        add_kl_coefficient: bool = True,
        beta: float = 0.0,
        old_logp_source: str = "rollout",
        timestep_chunk_size: Optional[int] = None,
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        if stage is None and pipeline is not None:
            stage = getattr(pipeline, stage_attr)
        if stage is None:
            raise ValueError("FlowDPPO: either `stage` or `pipeline` must be provided")
        self.stage = stage
        self.params = params
        self.kl_mask_threshold = float(kl_mask_threshold)
        self.add_kl_coefficient = bool(add_kl_coefficient)
        self.beta = float(beta)
        self._ref_model = _resolve_reference_model(backend, beta=self.beta, algo="FlowDPPO")
        self.old_logp_source = str(old_logp_source).strip().lower()
        require(
            self.old_logp_source in ("rollout", "replay"),
            f"FlowDPPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}",
        )
        require_timestep_chunk_size(timestep_chunk_size, algo="FlowDPPO")
        self.timestep_chunk_size = timestep_chunk_size
        _require_replay_anchor_for_batched_replay(self.stage, self.old_logp_source, algo="FlowDPPO")
        self.conditions_cls = conditions_cls

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "LatentSegment",
    ) -> None:
        """Freeze the π_old anchor and means at pre-update weights, before the ``num_updates_per_batch`` loop."""
        if segment.sde_indices is None:
            return
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        if self.old_logp_source == "rollout" and segment.sde_logp is None:
            raise RuntimeError(
                "FlowDPPO.prepare_segment: old_logp_source='rollout' but the "
                "rollout engine emitted no per-step log-probs (segment.sde_logp is "
                "None). Pin a rollout build that emits trajectory log-probs, or set "
                "old_logp_source='replay'."
            )
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        logp_parts: List[torch.Tensor] = []
        mean_parts: List[torch.Tensor] = []
        with torch.no_grad():
            for chunk in slice_step_chunks(target_steps, self.timestep_chunk_size):
                result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=chunk)
                if result.prev_sample_means is None:
                    raise RuntimeError(
                        "FlowDPPO.prepare_segment: stage.replay() returned "
                        "prev_sample_means=None. Ensure the stage's replay method "
                        "produces means (required for KL-ADV masking)."
                    )
                if self.old_logp_source == "replay":
                    logp_parts.append(result.log_probs.detach().cpu())
                mean_parts.append(result.prev_sample_means.detach().cpu())
        if self.old_logp_source == "replay":
            segment.sde_logp = torch.cat(logp_parts, dim=1)
        segment.sde_means = torch.cat(mean_parts, dim=1)

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
        num_steps = len(target_steps)
        chunks = slice_step_chunks(target_steps, self.timestep_chunk_size)
        policy_loss_sum: Optional[torch.Tensor] = None
        total_loss_sum: Optional[torch.Tensor] = None
        kl_ref_sum: Optional[torch.Tensor] = None
        weighted_metrics: Dict[str, torch.Tensor] = {}
        ratio_mins: List[torch.Tensor] = []
        ratio_maxs: List[torch.Tensor] = []
        kl_maxs: List[torch.Tensor] = []
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
            new_means = replay_result.prev_sample_means

            if new_means is None:
                raise RuntimeError(
                    "FlowDPPO requires stage.replay() to return prev_sample_means, "
                    "but got None. Ensure the stage's replay method produces means."
                )

            old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, chunk, field_name="sde_logp").to(
                dtype=new_logp.dtype, device=new_logp.device
            )
            old_means = gather_sde_field(segment.sde_means, segment.sde_indices, chunk, field_name="sde_means").to(
                dtype=new_means.dtype, device=new_means.device
            )

            sigma_t = _transition_sigma(
                self.stage,
                segment=segment,
                target_steps=chunk,
                eta=self.params.eta,
                device=new_logp.device,
                add_coefficient=self.add_kl_coefficient,
            )
            adv_b = (
                advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)
            )

            loss_per_elem, chunk_metrics = _flowdppo_kl_adv_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                new_means=new_means,
                old_means=old_means,
                advantages=adv_b,
                sigma_t=sigma_t,
                kl_mask_threshold=self.kl_mask_threshold,
            )

            chunk_policy_loss = loss_per_elem.mean()
            chunk_loss = chunk_policy_loss
            p_term = chunk_policy_loss.detach() * len(chunk)
            policy_loss_sum = p_term if policy_loss_sum is None else policy_loss_sum + p_term
            if self.beta > 0.0:
                ref_means = _reference_replay_means(
                    self.stage,
                    self._ref_model,
                    conditions=typed_conds,
                    segment=segment,
                    params=self.params,
                    target_steps=chunk,
                ).to(dtype=new_means.dtype, device=new_means.device)
                kl_sigma_t = _transition_sigma(
                    self.stage,
                    segment=segment,
                    target_steps=chunk,
                    eta=self.params.eta,
                    device=new_logp.device,
                    add_coefficient=True,
                )
                kl_ref = _reference_kl_loss(new_means, ref_means, kl_sigma_t)
                chunk_loss = chunk_loss + self.beta * kl_ref
                k_term = kl_ref.detach() * len(chunk)
                kl_ref_sum = k_term if kl_ref_sum is None else kl_ref_sum + k_term

            chunk_weight = len(chunk) / num_steps
            (chunk_loss * (loss_scale * chunk_weight)).backward()
            t_term = chunk_loss.detach() * len(chunk)
            total_loss_sum = t_term if total_loss_sum is None else total_loss_sum + t_term
            if len(chunks) > 1:
                new_logp_parts.append(new_logp.detach())
                old_logp_parts.append(old_logp.detach())
                for k, v in chunk_metrics.items():
                    if k == "ratio_min":
                        ratio_mins.append(v)
                    elif k == "ratio_max":
                        ratio_maxs.append(v)
                    elif k == "kl_new_old_max":
                        kl_maxs.append(v)
                    elif k != "ratio_std":
                        m_term = v * len(chunk)
                        weighted_metrics[k] = m_term if k not in weighted_metrics else weighted_metrics[k] + m_term

        policy_loss_val = float((policy_loss_sum / num_steps).item())
        total_loss_val = float((total_loss_sum / num_steps).item())
        if len(chunks) == 1:
            resolved_metrics = {k: float(v.item()) for k, v in chunk_metrics.items()}
        else:
            new_logp_all = torch.cat(new_logp_parts, dim=1)
            old_logp_all = torch.cat(old_logp_parts, dim=1)
            ratio_all = torch.exp(new_logp_all - old_logp_all)
            resolved_metrics = {
                **{k: float((v / num_steps).item()) for k, v in weighted_metrics.items()},
                "ratio_std": float(ratio_all.std().item()),
                "ratio_min": float(torch.stack(ratio_mins).min().item()),
                "ratio_max": float(torch.stack(ratio_maxs).max().item()),
                "kl_new_old_max": float(torch.stack(kl_maxs).max().item()),
            }

        metrics: Dict[str, Any] = {
            "policy_loss": policy_loss_val,
            "kl_mask_threshold": float(self.kl_mask_threshold),
            **resolved_metrics,
        }

        if self.beta > 0.0:
            metrics["beta"] = float(self.beta)
            metrics["kl_ref_mean"] = float((kl_ref_sum / num_steps).item())

        return AlgorithmStepResult(
            loss=total_loss_val,
            metrics=metrics,
            num_steps_or_tokens=num_steps,
            has_backward=True,
        )

    def _resolve_target_steps(self, segment: "LatentSegment") -> List[int]:
        """All SDE-recorded step indices on the segment."""
        if segment.sde_indices is None:
            return []
        return segment.sde_indices.tolist()


__all__ = ["FlowDPPO", "FlowDPPOConfig"]
