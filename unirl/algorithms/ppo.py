"""PPO with GAE for autoregressive ``TextSegment`` training.

Uses per-token GAE advantages from ``segment.token_advantages`` (populated in
:meth:`prepare_rollout_track`) and clipped value loss against ``segment.returns``.
Policy clip math is shared with :class:`GRPO` via :func:`_grpo_clip_loss`;
value clip math via :func:`_ppo_clipped_value_loss` in the same module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Type

import torch

from unirl.models.types.replay_result import ReplayResult
from unirl.types.conditions import Condition
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _ppo_clipped_value_loss,
    _resolve_clip_range_from_schedule,
    rollout_replay_logp_absdiff,
    typed_conditions,
)


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


def _aggregate_token_loss(
    loss_per_elem: torch.Tensor,
    *,
    segment: TextSegment,
    loss_agg_mode: str,
    horizon: int,
) -> torch.Tensor:
    if loss_agg_mode in ("seq-mean-token-sum-norm", "seq-mean-token-mean") and segment.lengths is not None:
        parts = torch.split(loss_per_elem, segment.lengths.tolist())
        if loss_agg_mode == "seq-mean-token-sum-norm":
            return torch.stack([p.sum() for p in parts]).mean() / float(horizon)
        return torch.stack([p.mean() if p.numel() else p.new_zeros(()) for p in parts]).mean()
    return loss_per_elem.mean()


class PPO(StageAlgorithm):
    """PPO with GAE over an AR ``TextSegment``.

    :meth:`prepare_rollout_track` runs a no-grad critic replay (``return_values=True``),
    stores ``segment.values`` as the frozen value anchor, then calls
    :meth:`RolloutTrack.compute_gae_advantages`. Each train micro-batch replays
    for fresh log-probs and value predictions for the policy and value losses.
    """

    supports_multi_update = True

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
        self.stage = stage
        self.clip_range = float(clip_range)
        self.clip_range_high = None if clip_range_high is None else float(clip_range_high)
        self.clip_schedule = str(clip_schedule)
        self.cliprange_value = float(cliprange_value)
        self.vf_coef = float(vf_coef)
        self.gae_gamma = float(gae_gamma)
        self.gae_lambda = float(gae_lambda)
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)

    def prepare_rollout_track(self, track: RolloutTrack) -> None:
        """Replay critic values and compute GAE on the worker shard."""
        if track.rewards is None:
            raise ValueError("PPO.prepare_rollout_track: track has no rewards")
        if track.segment is None or not isinstance(track.segment, TextSegment):
            raise ValueError("PPO.prepare_rollout_track: requires a TextSegment")
        segment = track.segment
        if segment.log_probs is None:
            raise ValueError("PPO.prepare_rollout_track: segment.log_probs is None")

        typed_conds = typed_conditions(track.conditions, self.conditions_cls)
        with torch.no_grad():
            replay_out = self.stage.replay(
                typed_conds,
                segment=segment,
                temperature=self.sampling_temperature,
                return_values=True,
            )
        values = _replay_values(replay_out)
        track.segment = replace(segment, values=values)
        updated = track.compute_gae_advantages(gamma=self.gae_gamma, gae_lambda=self.gae_lambda)
        track.segment = updated.segment
        track.advantages = updated.advantages

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: TextSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del advantages  # GAE uses segment.token_advantages instead of track-level GRPO scalars.
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if segment.token_advantages is None or segment.returns is None or segment.values is None:
            raise ValueError(
                "PPO.compute_loss_and_backward: segment requires token_advantages, returns, and values "
                "(call prepare_rollout_track first)."
            )

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        replay_out = self.stage.replay(
            typed_conds,
            segment=segment,
            temperature=self.sampling_temperature,
            return_values=True,
        )
        new_logp = _replay_log_probs(replay_out)
        new_values = _replay_values(replay_out)

        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)
        old_values = segment.values.to(dtype=new_values.dtype, device=new_values.device)
        returns = segment.returns.to(dtype=new_values.dtype, device=new_values.device)
        adv_per_token = segment.token_advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device)

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(self.clip_range_high, self.clip_schedule, training_progress)
        )
        policy_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )
        value_per_elem = _ppo_clipped_value_loss(
            values=new_values,
            old_values=old_values,
            returns=returns,
            clip_range=self.cliprange_value,
        )
        total_per_elem = policy_per_elem + self.vf_coef * value_per_elem
        loss = _aggregate_token_loss(
            total_per_elem,
            segment=segment,
            loss_agg_mode=self.loss_agg_mode,
            horizon=self.horizon,
        )
        (loss * loss_scale).backward()

        policy_loss = _aggregate_token_loss(
            policy_per_elem,
            segment=segment,
            loss_agg_mode=self.loss_agg_mode,
            horizon=self.horizon,
        )
        value_loss = _aggregate_token_loss(
            value_per_elem,
            segment=segment,
            loss_agg_mode=self.loss_agg_mode,
            horizon=self.horizon,
        )
        metrics: Dict[str, Any] = {
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "clip_range": float(clip_range),
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


def _replay_log_probs(replay_out: ReplayResult) -> torch.Tensor:
    return replay_out.log_probs


def _replay_values(replay_out: ReplayResult) -> torch.Tensor:
    if replay_out.values is None:
        raise ValueError("PPO: replay with return_values=True must return ReplayResult.values")
    return replay_out.values


__all__ = ["PPO", "PPOConfig"]
