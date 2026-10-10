"""FlashGRPO: FlowGRPO with Flash-GRPO temporal-gradient rectification."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, List, Optional, Sequence, Type, Union

import torch

from unirl.sde.kernels import FlashSDEStrategy
from unirl.types.segments.latent import LatentSegment

from .base import BaseAlgorithmConfig
from .per_sample_grpo import PerSampleStepGRPO


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


class FlashGRPO(PerSampleStepGRPO):
    """FlowGRPO whose loss is scaled by the Flash-GRPO temporal-gradient-rectification coefficient."""

    # Stratify one SDE step per prompt; the rectification hard-codes the Flash std_dev_t.
    per_sample_sde_layout = "stratified"
    required_strategy = FlashSDEStrategy
    weight_metric = "flash_tgr"

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
            adv_clip_max=adv_clip_max,
            backend=backend,
            conditions_cls=conditions_cls,
        )
        self.rectification_indices = None if rectification_indices is None else [int(i) for i in rectification_indices]

    def _row_weights(
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
        sqrt_neg_dt = torch.sqrt((sigma - sigma_next).clamp_min(torch.finfo(torch.float32).eps))
        sigma_max = sigmas[1] if int(sigmas.shape[0]) > 1 else torch.tensor(0.99, device=device, dtype=sigmas.dtype)
        sigma_min = sigmas[-1]
        std_dev_t = (sigma_min + (sigma_max - sigma_min) * sigma) * eta
        std_dev_t = std_dev_t.clamp_min(torch.finfo(torch.float32).eps)
        sigma_safe = sigma.clamp_min(torch.finfo(torch.float32).eps)
        denom = sqrt_neg_dt / std_dev_t + std_dev_t * sqrt_neg_dt * (1.0 - sigma) / (2.0 * sigma_safe)
        return denom.reciprocal()


__all__ = ["FlashGRPO", "FlashGRPOConfig"]
