"""Flash-GRPO for diffusion/video one-step policy optimization.

Flash-GRPO (``Shredded-Pork/Flash-GRPO`` commit
``bd6051f68e1ab444e5ec7c6ffe0a1f7eaf559a0d``) is FlowGRPO with two
training-side changes:

* the rollout records a sparse, typically single, SDE transition (configured via
  ``DiffusionSamplingParams.scheduler`` / ``sde_indices``), while all other
  diffusion steps are deterministic; and
* the PPO/GRPO objective is multiplied by the temporal-gradient-rectification
  coefficient

  ``1 / (sqrt(-dt)/std_dev_t + std_dev_t*sqrt(-dt)*(1-sigma)/(2*sigma))``

  normalized by the configured candidate timestep pool (or by the trained
  steps when no pool is configured).

The transition coefficient ``std_dev_t`` itself lives in
``unirl.sde.kernels.FlashSDEStrategy`` so rollout and replay share one source of
truth. This class owns only the rectified ratio-clip loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    _grpo_clip_loss,
    _reference_kl_loss,
    _reference_replay_means,
    _resolve_clip_range_from_schedule,
    _transition_sigma,
    gather_sde_field,
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
    """FlowGRPO plus Flash-GRPO temporal-gradient rectification.

    ``prepare_segment`` / old-log-prob anchoring / optional reference KL are
    inherited from :class:`FlowGRPO`. Configure one-step policy optimization by
    selecting exactly one SDE index in ``params.sde_indices`` (or a scheduler
    that resolves to one index). If more than one step is present, the same
    rectification formula is applied per step and normalized over those steps.
    """

    # The v2 DiffusionTrainer gates its per-sample SDE-index orchestration on this
    # flag: draw ONE i.i.d. SDE step per prompt (grouping prompts into <=|pool|
    # generate calls), stamp ``segment.sde_index_per_sample``, then merge the
    # groups into one training track. Without it every sample in a rollout shares
    # the single step ``resolve_sde_indices`` draws, so an optimizer step sees one
    # sigma and the policy gradient vanishes (the FlashGRPO collapse this fixes).
    # FlowGRPO and every other shared-schedule algorithm leave it False (the base
    # class has no such attribute → ``getattr(..., False)`` default).
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
        # Upstream clamps advantages to +-adv_clip_max before the ratio (train
        # script:1056, default 5 in config/base.py:91). Without it a batch whose
        # reward spread collapses drives |adv| arbitrarily high through the
        # global-std denominator, and one gradient spike undoes training.
        if not float(adv_clip_max) > 0.0:
            raise ValueError(f"FlashGRPO: adv_clip_max must be > 0; got {adv_clip_max!r}.")
        self.adv_clip_max = float(adv_clip_max)

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
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
        new_logp = replay_result.log_probs
        new_means = replay_result.prev_sample_means

        # FlashGRPO's per-sample path: each sample took exactly one stochastic
        # step (S == 1) at its OWN index (segment.sde_index_per_sample), so the
        # shared sde_indices / gather_sde_field / searchsorted lookup does not
        # apply — old_logp is just the single stored column.
        per_sample = segment.sde_index_per_sample is not None
        if per_sample:
            old_logp = segment.sde_logp[:, :1].to(dtype=new_logp.dtype, device=new_logp.device)
        else:
            old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
                dtype=new_logp.dtype, device=new_logp.device
            )

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_raw = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device)
        # Upstream: torch.clamp(sample["advantages"][:, 0], -adv_clip_max, +adv_clip_max).
        adv_clipped = adv_raw.clamp(-self.adv_clip_max, self.adv_clip_max)
        adv_b = adv_clipped.reshape(-1, 1).expand_as(new_logp)

        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_b,
            clip_range=clip_range,
        )

        if per_sample:
            tgr = self._rectification_weights_per_sample(segment=segment, device=new_logp.device)
        else:
            tgr = self._rectification_weights(segment=segment, target_steps=target_steps, device=new_logp.device)
        tgr = tgr.to(dtype=loss_per_elem.dtype, device=loss_per_elem.device)
        loss_per_elem = loss_per_elem * tgr
        policy_loss = loss_per_elem.mean()
        loss = policy_loss

        metrics: Dict[str, Any] = {
            "policy_loss": float(policy_loss.detach().item()),
            "clip_range": float(clip_range),
            "flash_tgr_mean": float(tgr.detach().mean().item()),
            "flash_tgr_min": float(tgr.detach().min().item()),
            "flash_tgr_max": float(tgr.detach().max().item()),
            # Fraction of samples whose advantage hit +-adv_clip_max. A run that
            # starts regressing while this climbs is being driven by the reward
            # outliers the clamp exists to bound.
            "adv_clip_fraction": float((adv_raw.abs() > self.adv_clip_max).float().mean().item()),
            "adv_abs_max": float(adv_raw.abs().max().item()),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }

        if self.beta > 0.0:
            if per_sample:
                raise NotImplementedError(
                    "FlashGRPO: reference-KL (beta>0) is not supported on the per-sample "
                    "SDE-index path; the WAN2.1 recipe runs beta=0. Set beta=0, or route "
                    "through the shared sde_indices path for KL."
                )
            if new_means is None:
                raise RuntimeError(
                    "FlashGRPO: beta>0 requires stage.replay() to return prev_sample_means, "
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

    def _rectification_weights(
        self,
        *,
        segment: LatentSegment,
        target_steps: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Return normalized Flash-GRPO temporal rectification weights ``[1, S]``.

        Upstream normalizes the per-step coefficient by the mean coefficient over
        the gradient-accumulation window. UniRL's rollout segment stores one
        shared ``sde_indices`` vector, so a single-step rollout would otherwise
        normalize itself to exactly 1. ``rectification_indices`` supplies the
        candidate timestep pool to average over (for the WAN2.1 recipe: the first
        10 of 20 denoising steps). When omitted, the target steps themselves are
        used, preserving FlowGRPO-like behaviour for ad-hoc configs.
        """
        if segment.sigmas is None:
            raise ValueError("FlashGRPO requires segment.sigmas to compute temporal rectification weights.")
        sigmas = segment.sigmas.to(device=device, dtype=torch.float32)
        norm_steps = self.rectification_indices if self.rectification_indices is not None else target_steps
        weights = self._rectification_coefficients(sigmas=sigmas, steps=target_steps, device=device)
        norm = self._rectification_coefficients(sigmas=sigmas, steps=list(norm_steps), device=device)
        weights = weights / norm.detach().mean().clamp_min(torch.finfo(torch.float32).eps)
        return weights.reshape(1, -1)

    def _rectification_weights_per_sample(
        self,
        *,
        segment: LatentSegment,
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-sample Flash-GRPO rectification weights ``[N, 1]``.

        The S==1 flash path: each sample took its single stochastic step at
        ``segment.sde_index_per_sample[n]``, so its coefficient is computed at
        that own index and normalized by the mean coefficient over the shared
        candidate pool (``rectification_indices``). The pool is REQUIRED here:
        without it each sample would normalize by its own single coefficient and
        collapse to ~1 — exactly the zero-diversity failure this path fixes.
        """
        if segment.sigmas is None:
            raise ValueError("FlashGRPO requires segment.sigmas to compute temporal rectification weights.")
        if self.rectification_indices is None:
            raise ValueError(
                "FlashGRPO per-sample rectification requires rectification_indices (the shared "
                "candidate timestep pool) to normalize against; got None."
            )
        if segment.sde_index_per_sample is None:
            raise ValueError("FlashGRPO per-sample rectification requires segment.sde_index_per_sample; got None.")
        sigmas = segment.sigmas.to(device=device, dtype=torch.float32)
        per_sample_steps = [int(i) for i in segment.sde_index_per_sample.tolist()]
        weights = self._rectification_coefficients(sigmas=sigmas, steps=per_sample_steps, device=device)
        norm = self._rectification_coefficients(sigmas=sigmas, steps=list(self.rectification_indices), device=device)
        weights = weights / norm.detach().mean().clamp_min(torch.finfo(torch.float32).eps)
        return weights.reshape(-1, 1)

    def _rectification_coefficients(
        self,
        *,
        sigmas: torch.Tensor,
        steps: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        if not steps:
            raise ValueError("FlashGRPO rectification requires at least one timestep index.")
        eta = float(getattr(self.params, "eta", 1.0))
        if eta <= 0.0:
            raise ValueError("FlashGRPO requires params.eta > 0 on trained SDE steps.")
        T = int(sigmas.shape[0]) - 1
        bad = [int(i) for i in steps if int(i) < 0 or int(i) >= T]
        if bad:
            raise ValueError(f"FlashGRPO rectification indices out of range [0, {T}): {bad}")

        idx = torch.tensor([int(i) for i in steps], dtype=torch.long, device=device)
        sigma = sigmas[idx]
        sigma_next = sigmas[idx + 1]
        sqrt_neg_dt = torch.sqrt((sigma - sigma_next).clamp_min(torch.finfo(torch.float32).eps))
        sigma_max = sigmas[1] if int(sigmas.shape[0]) > 1 else torch.tensor(0.99, device=device, dtype=sigmas.dtype)
        sigma_min = sigmas[-1]
        std_dev_t = (sigma_min + (sigma_max - sigma_min) * sigma) * eta
        std_dev_t = std_dev_t.clamp_min(torch.finfo(torch.float32).eps)
        sigma_safe = sigma.clamp_min(torch.finfo(torch.float32).eps)
        denom = sqrt_neg_dt / std_dev_t + std_dev_t * sqrt_neg_dt * (1.0 - sigma) / (2.0 * sigma_safe)
        return denom.reciprocal()


__all__ = ["FlashGRPO", "FlashGRPOConfig"]
