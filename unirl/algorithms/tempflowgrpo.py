"""TempFlowGRPO: FlowGRPO over per-step trajectory branches with noise-aware loss weights."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Optional, Type

import torch

from unirl.sde.kernels import FlowSDEStrategy
from unirl.types.segments.latent import LatentSegment

from .base import BaseAlgorithmConfig
from .per_sample_grpo import PerSampleStepGRPO


@dataclass
class TempFlowGRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    beta: float = 0.0
    old_logp_source: str = "rollout"
    adv_clip_max: float = 5.0
    params: Any = dc_field(default=None)
    branches_per_seed: int = 6
    advantage_group: str = "prompt"
    branch_steps_per_rollout: Optional[int] = None
    weight_scale: float = 2.25


class TempFlowGRPO(PerSampleStepGRPO):
    """Clipped loss scaled by ``weight_scale * transition_std`` at each row's branch step (README)."""

    # Every prompt branches at every chosen step; the trainer reads the branch knobs below.
    per_sample_sde_layout = "branched"
    required_strategy = FlowSDEStrategy
    weight_metric = "tempflow_weight"

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
        adv_clip_max: float = 5.0,
        branches_per_seed: int = 6,
        advantage_group: str = "prompt",
        branch_steps_per_rollout: Optional[int] = None,
        weight_scale: float = 2.25,
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
        if type(branches_per_seed) is not int or branches_per_seed < 1:
            raise ValueError(f"TempFlowGRPO: branches_per_seed must be a positive integer; got {branches_per_seed!r}.")
        if advantage_group not in ("prompt", "seed"):
            raise ValueError(f"TempFlowGRPO: advantage_group must be 'prompt' or 'seed'; got {advantage_group!r}.")
        if branch_steps_per_rollout is not None and (
            type(branch_steps_per_rollout) is not int or branch_steps_per_rollout < 1
        ):
            raise ValueError(
                f"TempFlowGRPO: branch_steps_per_rollout must be a positive integer or None; got {branch_steps_per_rollout!r}."
            )
        self.weight_scale = float(weight_scale)
        if not self.weight_scale > 0.0:
            raise ValueError(f"TempFlowGRPO: weight_scale must be > 0; got {weight_scale!r}.")
        if not params.eta > 0.0:
            raise ValueError("TempFlowGRPO requires params.eta > 0 on branch steps.")

    def row_weights(self, *, segment: LatentSegment, device: torch.device) -> torch.Tensor:
        """``weight_scale * transition_std`` (``std_dev_t * sqrt(-dt)``) at each row's branch step ``[N, 1]``."""
        if segment.sigmas is None:
            raise ValueError("TempFlowGRPO requires segment.sigmas to compute the noise-aware weights.")
        if segment.sigmas.shape[0] < 2:
            raise ValueError(f"TempFlowGRPO requires sigmas length >= 2; got {segment.sigmas.shape[0]}.")
        sigmas = segment.sigmas.to(device=device, dtype=torch.float32)
        steps = segment.sde_index_per_sample.to(device=device, dtype=torch.long)
        std = self.stage.strategy.transition_std(
            sigma=sigmas[steps],
            sigma_next=sigmas[steps + 1],
            eta=self.params.eta,
            sigma_max=sigmas[1],
        )
        return (self.weight_scale * std).reshape(-1, 1)


__all__ = ["TempFlowGRPO", "TempFlowGRPOConfig"]
