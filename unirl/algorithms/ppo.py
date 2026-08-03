"""PPO with per-token GAE and a clipped value objective for AR training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type

import torch

from unirl.models.types.replay_result import ReplayResult
from unirl.types.conditions import Condition
from unirl.types.sample import Part
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    rollout_replay_logp_absdiff,
    typed_conditions,
)

_LOSS_AGG_MODES = frozenset({"token-mean", "seq-mean-token-sum-norm", "seq-mean-token-mean"})


@dataclass
class PPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    clip_range: float = 0.2
    clip_range_high: Optional[float] = None
    clip_schedule: str = "constant"
    cliprange_value: float = 0.2
    vf_coef: float = 0.5
    gae_gamma: float = 1.0
    gae_lambda: float = 0.95
    loss_agg_mode: str = "token-mean"
    horizon: int = 8192


def _ppo_clipped_value_loss(
    *,
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    clip_range: float,
) -> torch.Tensor:
    """Elementwise ``0.5 * max((V-R)^2, (V_clipped-R)^2)``."""
    values_f = values.float()
    old_f = old_values.detach().float()
    returns_f = returns.detach().float()
    clipped = old_f + (values_f - old_f).clamp(-clip_range, clip_range)
    return 0.5 * torch.maximum((values_f - returns_f).square(), (clipped - returns_f).square())


def _aggregate_token_loss(
    loss_per_token: torch.Tensor,
    *,
    active: torch.Tensor,
    segment: TextSegment,
    loss_agg_mode: str,
    horizon: int,
) -> torch.Tensor:
    """Reduce only trainable tokens while preserving the configured sequence weighting."""
    if loss_agg_mode == "token-mean":
        return loss_per_token[active].mean()
    if segment.lengths is None:
        raise ValueError(f"PPO: loss_agg_mode={loss_agg_mode!r} requires packed segment lengths")

    loss_chunks = torch.split(loss_per_token, segment.lengths.tolist())
    mask_chunks = torch.split(active, segment.lengths.tolist())
    sequence_losses = []
    for losses, mask in zip(loss_chunks, mask_chunks):
        selected = losses[mask]
        if loss_agg_mode == "seq-mean-token-sum-norm":
            sequence_losses.append(selected.sum() / float(horizon))
        else:
            sequence_losses.append(selected.mean() if selected.numel() else losses.new_zeros(()))
    return torch.stack(sequence_losses).mean()


class PPO(StageAlgorithm):
    """PPO over an AR ``TextSegment`` with a train-side value head."""

    supports_multi_update = True
    anchor_fields = ("values",)

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        clip_range: float = 0.2,
        clip_schedule: str = "constant",
        clip_range_high: Optional[float] = None,
        cliprange_value: float = 0.2,
        vf_coef: float = 0.5,
        gae_gamma: float = 1.0,
        gae_lambda: float = 0.95,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("PPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        if float(clip_range) < 0.0 or float(cliprange_value) < 0.0:
            raise ValueError("PPO: policy and value clip ranges must be non-negative")
        if float(vf_coef) < 0.0:
            raise ValueError("PPO: vf_coef must be non-negative")
        if not (0.0 <= float(gae_gamma) <= 1.0 and 0.0 <= float(gae_lambda) <= 1.0):
            raise ValueError("PPO: gae_gamma and gae_lambda must be in [0, 1]")
        if str(loss_agg_mode) not in _LOSS_AGG_MODES:
            raise ValueError(f"PPO: unsupported loss_agg_mode={loss_agg_mode!r}")
        if int(horizon) <= 0:
            raise ValueError("PPO: horizon must be positive")

        self.stage = stage
        self.clip_range = float(clip_range)
        self.clip_range_high = None if clip_range_high is None else float(clip_range_high)
        self.clip_schedule = str(clip_schedule)
        self.cliprange_value = float(cliprange_value)
        self.vf_coef = float(vf_coef)
        self.gae_gamma = float(gae_gamma)
        self.gae_lambda = float(gae_lambda)
        self.loss_agg_mode = str(loss_agg_mode)
        self.loss_weighting = "token" if self.loss_agg_mode == "token-mean" else "sample"
        self.horizon = int(horizon)
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)

    def recomputes_anchor(self) -> bool:
        """Critic anchors must use the exact micro geometry of the train forward."""
        return True

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
    ) -> None:
        """Freeze pre-update critic values for one planned replay micro."""
        if segment.tokens is None or segment.log_probs is None:
            raise ValueError("PPO.prepare_segment: segment requires tokens and rollout log_probs")
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            replay = self.stage.replay(
                typed_conds,
                segment=segment,
                temperature=self.sampling_temperature,
                return_values=True,
            )
        values = _replay_values(replay)
        if values.shape != segment.tokens.shape:
            raise ValueError(
                f"PPO.prepare_segment: values shape {tuple(values.shape)} "
                f"!= packed tokens shape {tuple(segment.tokens.shape)}"
            )
        segment.values = values.detach()

    def prepare_part(self, part: Part) -> Part:
        """Compute GAE after all per-micro critic anchors have been reassembled."""
        return part.compute_gae_advantages(gamma=self.gae_gamma, gae_lambda=self.gae_lambda)

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del advantages  # PPO consumes packed GAE, not the compatibility summary on Part.
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.numel()) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if segment.token_advantages is None or segment.returns is None or segment.values is None:
            raise ValueError(
                "PPO.compute_loss_and_backward: token_advantages, returns, and frozen values "
                "must be prepared before training"
            )

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        replay = self.stage.replay(
            typed_conds,
            segment=segment,
            temperature=self.sampling_temperature,
            return_values=True,
        )
        if not isinstance(replay, ReplayResult):
            raise TypeError("PPO: replay with return_values=True must return ReplayResult")
        new_logp = replay.log_probs
        new_values = _replay_values(replay)

        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)
        old_values = segment.values.to(dtype=new_values.dtype, device=new_values.device)
        returns = segment.returns.to(dtype=new_values.dtype, device=new_values.device)
        token_advantages = segment.token_advantages.to(dtype=new_logp.dtype, device=new_logp.device).detach()
        expected = new_logp.shape
        named_tensors = {
            "new_values": new_values,
            "old_logp": old_logp,
            "old_values": old_values,
            "returns": returns,
            "token_advantages": token_advantages,
        }
        mismatched = {name: tuple(tensor.shape) for name, tensor in named_tensors.items() if tensor.shape != expected}
        if mismatched:
            raise ValueError(
                f"PPO.compute_loss_and_backward: packed tensor shape mismatch; expected {expected}, got {mismatched}"
            )

        active = torch.ones(expected, dtype=torch.bool, device=new_logp.device)
        if segment.loss_mask is not None:
            active = segment.loss_mask.to(device=new_logp.device, dtype=torch.bool)
            if active.shape != expected:
                raise ValueError(
                    f"PPO.compute_loss_and_backward: loss_mask shape {tuple(active.shape)} != {tuple(expected)}"
                )
        active_count = int(active.sum().item())
        if active_count == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(self.clip_range_high, self.clip_schedule, training_progress)
        )
        policy_active, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp[active],
            old_logp=old_logp[active],
            advantages=token_advantages[active],
            clip_range=clip_range,
            clip_range_high=clip_high,
        )
        value_active = _ppo_clipped_value_loss(
            values=new_values[active],
            old_values=old_values[active],
            returns=returns[active],
            clip_range=self.cliprange_value,
        )

        policy_per_token = torch.zeros_like(new_logp).masked_scatter(active, policy_active)
        value_per_token = torch.zeros_like(new_values, dtype=torch.float32).masked_scatter(active, value_active)
        policy_loss = _aggregate_token_loss(
            policy_per_token,
            active=active,
            segment=segment,
            loss_agg_mode=self.loss_agg_mode,
            horizon=self.horizon,
        )
        value_loss = _aggregate_token_loss(
            value_per_token,
            active=active,
            segment=segment,
            loss_agg_mode=self.loss_agg_mode,
            horizon=self.horizon,
        )
        loss = policy_loss + self.vf_coef * value_loss
        (loss * loss_scale).backward()

        active_returns = returns[active].float()
        active_values = new_values[active].float()
        return_var = active_returns.var(unbiased=False)
        explained_variance = (
            1.0 - (active_returns - active_values).var(unbiased=False) / return_var
            if float(return_var) > 0.0
            else return_var.new_zeros(())
        )
        value_clip_fraction = ((active_values - old_values[active].float()).abs() > self.cliprange_value).float().mean()
        metrics: Dict[str, Any] = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "value_mean": float(active_values.detach().mean()),
            "return_mean": float(active_returns.detach().mean()),
            "explained_variance": float(explained_variance.detach()),
            "value_clip_fraction": float(value_clip_fraction.detach()),
            "clip_range": float(clip_range),
            "cliprange_value": self.cliprange_value,
            **rollout_replay_logp_absdiff(new_logp[active], old_logp[active]),
            **{name: float(value) for name, value in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach()),
            metrics=metrics,
            num_steps_or_tokens=active_count,
            has_backward=True,
        )


def _replay_values(replay: Any) -> torch.Tensor:
    if not isinstance(replay, ReplayResult) or replay.values is None:
        raise TypeError("PPO: replay with return_values=True must return ReplayResult.values")
    return replay.values


__all__ = ["PPO", "PPOConfig"]
