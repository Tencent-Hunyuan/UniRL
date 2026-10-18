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
    gather_sde_field,
    gaussian_kl_div,
    reference_kl_loss,
    require_replay_anchor_for_batched_replay,
    require_timestep_chunk_size,
    resolve_reference_model,
    slice_step_chunks,
    transition_sigma,
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


def flowdppo_kl_adv_loss(
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

    kl_per_elem = gaussian_kl_div(new_means, old_means, sigma_t)
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
        self.ref_model = resolve_reference_model(backend, beta=self.beta, algo="FlowDPPO")
        self.old_logp_source = str(old_logp_source).strip().lower()
        require(
            self.old_logp_source in ("rollout", "replay"),
            f"FlowDPPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}",
        )
        require_timestep_chunk_size(timestep_chunk_size, algo="FlowDPPO")
        self.timestep_chunk_size = timestep_chunk_size
        require_replay_anchor_for_batched_replay(self.stage, self.old_logp_source, algo="FlowDPPO")
        self.conditions_cls = conditions_cls

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "LatentSegment",
    ) -> None:
        """Freeze the π_old anchor and means at pre-update weights, before the ``num_updates_per_batch`` loop."""
        if segment.sde_indices is None or segment.sde_indices.numel() == 0:
            return
        if self.old_logp_source == "rollout" and segment.sde_logp is None:
            raise RuntimeError(
                "FlowDPPO.prepare_segment: old_logp_source='rollout' but the "
                "rollout engine emitted no per-step log-probs (segment.sde_logp is "
                "None). Pin a rollout build that emits trajectory log-probs, or set "
                "old_logp_source='replay'."
            )
        target_steps = segment.sde_indices.tolist()
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
        if segment.sde_indices is None or segment.sde_indices.numel() == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        target_steps = segment.sde_indices.tolist()
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        num_steps = len(target_steps)
        chunks = slice_step_chunks(target_steps, self.timestep_chunk_size)
        policy_terms: List[torch.Tensor] = []
        total_terms: List[torch.Tensor] = []
        kl_ref_terms: List[torch.Tensor] = []
        chunk_metrics_list: List[Dict[str, torch.Tensor]] = []
        new_logp_parts: List[torch.Tensor] = []
        old_logp_parts: List[torch.Tensor] = []

        for chunk in chunks:
            chunk_weight = len(chunk) / num_steps
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

            sigma_t = transition_sigma(
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

            loss_per_elem, chunk_metrics = flowdppo_kl_adv_loss(
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
            policy_terms.append(chunk_policy_loss.detach() * chunk_weight)
            if self.beta > 0.0:
                kl_ref = reference_kl_loss(
                    self.stage,
                    self.ref_model,
                    conditions=typed_conds,
                    segment=segment,
                    params=self.params,
                    target_steps=chunk,
                    new_means=new_means,
                    algo="FlowDPPO",
                )
                chunk_loss = chunk_loss + self.beta * kl_ref
                kl_ref_terms.append(kl_ref.detach() * chunk_weight)

            (chunk_loss * (loss_scale * chunk_weight)).backward()
            total_terms.append(chunk_loss.detach() * chunk_weight)
            if len(chunks) > 1:
                chunk_metrics_list.append(chunk_metrics)
                new_logp_parts.append(new_logp.detach())
                old_logp_parts.append(old_logp.detach())

        if len(chunks) == 1:
            resolved_metrics = {k: v.item() for k, v in chunk_metrics.items()}
        else:
            weights = torch.tensor(
                [len(c) / num_steps for c in chunks], dtype=torch.float32, device=new_logp_parts[0].device
            )
            mean_keys = [
                k for k in chunk_metrics_list[0] if k not in ("ratio_std", "ratio_min", "ratio_max", "kl_new_old_max")
            ]
            ratio_all = torch.exp(torch.cat(new_logp_parts, dim=1) - torch.cat(old_logp_parts, dim=1))
            resolved_metrics = {
                **{k: (torch.stack([m[k] for m in chunk_metrics_list]) * weights).sum().item() for k in mean_keys},
                "ratio_std": ratio_all.std().item(),
                "ratio_min": torch.stack([m["ratio_min"] for m in chunk_metrics_list]).min().item(),
                "ratio_max": torch.stack([m["ratio_max"] for m in chunk_metrics_list]).max().item(),
                "kl_new_old_max": torch.stack([m["kl_new_old_max"] for m in chunk_metrics_list]).max().item(),
            }

        total_loss_val = torch.stack(total_terms).sum().item()
        metrics: Dict[str, Any] = {
            "policy_loss": torch.stack(policy_terms).sum().item(),
            "kl_mask_threshold": self.kl_mask_threshold,
            **resolved_metrics,
        }

        if self.beta > 0.0:
            metrics["beta"] = self.beta
            metrics["kl_ref_mean"] = torch.stack(kl_ref_terms).sum().item()

        return AlgorithmStepResult(
            loss=total_loss_val,
            metrics=metrics,
            num_steps_or_tokens=num_steps,
            has_backward=True,
        )


__all__ = ["FlowDPPO", "FlowDPPOConfig"]
