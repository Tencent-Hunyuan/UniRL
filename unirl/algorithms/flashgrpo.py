"""FlashGRPO: FlowGRPO with Flash-GRPO temporal-gradient rectification."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Union

import torch

from unirl.sde.kernels import FlashSDEStrategy
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    typed_conditions,
)
from .flowgrpo import FlowGRPO


@dataclass
class FlashGRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    clip_range: float = 1e-3
    clip_schedule: str = "constant"
    beta: float = 0.0
    old_logp_source: str = "rollout"
    adv_clip_max: float = 5.0
    params: Any = dc_field(default=None)
    rectification_indices: Optional[List[int]] = None


class FlashGRPO(FlowGRPO):
    """FlowGRPO whose loss is scaled by the Flash-GRPO temporal-gradient-rectification coefficient."""

    # The trainer gates its per-sample SDE-index orchestration on this flag: stratify
    # one SDE step per prompt, stamp ``segment.sde_index_per_sample``, then merge the
    # groups into one training track. Without it every sample shares the one step
    # ``get_sde_indices`` draws and the optimizer step sees a single sigma.
    requires_per_sample_sde_index = True

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        clip_range: float = 1e-3,
        clip_schedule: str = "constant",
        beta: float = 0.0,
        old_logp_source: str = "rollout",
        adv_clip_max: float = 5.0,
        rectification_indices: Optional[Sequence[int]] = None,
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        # Checked before super() so a beta>0 recipe never loads the reference model.
        if float(beta) != 0.0:
            raise ValueError(
                f"FlashGRPO: reference-KL is not supported on the per-sample SDE-index path; got beta={beta!r}."
            )
        super().__init__(
            params=params,
            stage=stage,
            pipeline=pipeline,
            stage_attr=stage_attr,
            clip_range=clip_range,
            clip_schedule=clip_schedule,
            beta=beta,
            old_logp_source=old_logp_source,
            backend=backend,
            conditions_cls=conditions_cls,
        )
        self.rectification_indices = None if rectification_indices is None else [int(i) for i in rectification_indices]
        if not float(adv_clip_max) > 0.0:
            raise ValueError(f"FlashGRPO: adv_clip_max must be > 0; got {adv_clip_max!r}.")
        self.adv_clip_max = float(adv_clip_max)
        # The trainer stamps sde_index_per_sample for every FlashGRPO recipe, so a
        # stage whose replay ignores it would score every row at one group's sigma.
        if not getattr(self.stage, "supports_per_sample_sde_index", False):
            raise ValueError(
                f"FlashGRPO requires a stage that honours segment.sde_index_per_sample; "
                f"{type(self.stage).__name__} does not. Use FlowGRPO, or implement the "
                "per-sample replay on that stage and declare supports_per_sample_sde_index."
            )
        # _rectification_coefficients hard-codes the Flash std_dev_t, so the sampler must match.
        if not isinstance(getattr(self.stage, "strategy", None), FlashSDEStrategy):
            raise ValueError(
                f"FlashGRPO requires the stage strategy to be FlashSDEStrategy; "
                f"got {type(getattr(self.stage, 'strategy', None)).__name__}."
            )

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        # The trainer stamps every FlashGRPO rollout; a missing stamp means it never stratified.
        if segment.sde_index_per_sample is None:
            raise ValueError(
                "FlashGRPO requires segment.sde_index_per_sample; the rollout was not stratified per prompt."
            )
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
        new_logp = replay_result.log_probs
        # Each sample took its one stochastic step at its own index, so slot 0 holds its logp.
        old_logp = segment.sde_logp[:, :1].to(dtype=new_logp.dtype, device=new_logp.device)

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_raw = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device)
        adv_clipped = adv_raw.clamp(-self.adv_clip_max, self.adv_clip_max)
        adv_b = adv_clipped.reshape(-1, 1).expand_as(new_logp)

        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_b,
            clip_range=clip_range,
        )

        tgr = self._rectification_weights_per_sample(segment=segment, device=new_logp.device)
        tgr = tgr.to(dtype=loss_per_elem.dtype, device=loss_per_elem.device)
        loss_per_elem = loss_per_elem * tgr
        loss = loss_per_elem.mean()

        (loss * loss_scale).backward()
        metric_tensors = {
            "policy_loss": loss.detach(),
            "flash_tgr_mean": tgr.detach().mean(),
            "flash_tgr_min": tgr.detach().min(),
            "flash_tgr_max": tgr.detach().max(),
            "adv_clip_fraction": (adv_raw.abs() > self.adv_clip_max).float().mean(),
            "adv_abs_max": adv_raw.abs().max(),
            **ratio_metrics,
        }
        metric_values = torch.stack([value.float() for value in metric_tensors.values()]).cpu().tolist()
        metrics: Dict[str, Any] = dict(zip(metric_tensors, metric_values))
        metrics["clip_range"] = float(clip_range)

        return AlgorithmStepResult(
            loss=metrics["policy_loss"],
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    def _rectification_weights_per_sample(
        self,
        *,
        segment: LatentSegment,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-sample rectification weights ``[N, 1]``, normalized over the required candidate pool."""
        if segment.sigmas is None:
            raise ValueError("FlashGRPO requires segment.sigmas to compute temporal rectification weights.")
        if self.rectification_indices is None:
            raise ValueError(
                "FlashGRPO per-sample rectification requires rectification_indices (the shared "
                "candidate timestep pool) to normalize against; got None."
            )
        sigmas = segment.sigmas.to(device=device, dtype=torch.float32)
        weights = self._rectification_coefficients(
            sigmas=sigmas, steps=segment.sde_index_per_sample.to(device=device, dtype=torch.long), device=device
        )
        norm = self._rectification_coefficients(sigmas=sigmas, steps=list(self.rectification_indices), device=device)
        weights = weights / norm.detach().mean().clamp_min(torch.finfo(torch.float32).eps)
        return weights.reshape(-1, 1)

    def _rectification_coefficients(
        self,
        *,
        sigmas: torch.Tensor,
        steps: Union[Sequence[int], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Reciprocal of the upstream per-step temporal-gradient denominator at ``steps``; a tensor stays on device."""
        idx = steps if isinstance(steps, torch.Tensor) else torch.tensor(list(steps), dtype=torch.long)
        idx = idx.to(device=device, dtype=torch.long)
        if idx.numel() == 0:
            raise ValueError("FlashGRPO rectification requires at least one timestep index.")
        eta = float(self.params.eta)
        if eta <= 0.0:
            raise ValueError("FlashGRPO requires params.eta > 0 on trained SDE steps.")
        sigma = sigmas[:-1].index_select(0, idx)
        sigma_next = sigmas[1:].index_select(0, idx)
        sigma_max = sigmas[1] if int(sigmas.shape[0]) > 1 else torch.tensor(0.99, device=device, dtype=sigmas.dtype)
        transition_std = self.stage.strategy.transition_std(
            sigma=sigma, sigma_next=sigma_next, eta=eta, sigma_max=sigma_max
        ).clamp_min(torch.finfo(torch.float32).eps)
        sigma_safe = sigma.clamp_min(torch.finfo(torch.float32).eps)
        denom = (sigma - sigma_next) / transition_std + transition_std * (1.0 - sigma) / (2.0 * sigma_safe)
        return denom.reciprocal()


__all__ = ["FlashGRPO", "FlashGRPOConfig"]
