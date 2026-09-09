"""Stage-driven ``GRPO`` over a ``TextSegment``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    rollout_replay_k3,
    rollout_replay_logp_absdiff,
    typed_conditions,
)


@dataclass
class GRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    alignment_require_exact: bool = False


class GRPO(StageAlgorithm):
    """GRPO over an AR ``TextSegment`` via ``ARStage.replay``."""

    supports_multi_update = True

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        clip_range_high: Optional[float] = None,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
        alignment_require_exact: bool = False,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("GRPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.clip_range = float(clip_range)
        self.clip_range_high = None if clip_range_high is None else float(clip_range_high)
        self.clip_schedule = str(clip_schedule)
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)
        self.alignment_require_exact = bool(alignment_require_exact)

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
        if self.alignment_require_exact and segment.rollout_log_probs is None:
            raise RuntimeError("GRPO exact replay/rollout alignment requires rollout_log_probs")
        rollout_anchor = segment.rollout_log_probs if segment.rollout_log_probs is not None else segment.log_probs
        rollout_anchor_logp = rollout_anchor.to(
            dtype=new_logp.dtype,
            device=new_logp.device,
        )
        if self.alignment_require_exact:
            gradient_reference = new_logp.float()
            rollout_reference = rollout_anchor_logp.float()
            if gradient_reference.shape != rollout_reference.shape:
                raise RuntimeError(
                    "GRPO exact replay/rollout shape mismatch: "
                    f"replay={tuple(gradient_reference.shape)} "
                    f"rollout={tuple(rollout_reference.shape)}"
                )
            finite = bool(torch.isfinite(gradient_reference).all() and torch.isfinite(rollout_reference).all())
            equal = torch.equal(gradient_reference, rollout_reference)
            if not finite or not equal:
                mismatch = gradient_reference != rollout_reference
                mismatch_indices = torch.nonzero(
                    mismatch,
                    as_tuple=False,
                ).reshape(-1)
                max_abs = float((gradient_reference - rollout_reference).abs().max().item())
                raise RuntimeError(
                    "GRPO exact gradient-replay/rollout alignment gate failed before backward: "
                    f"finite={finite} mismatch_count={int(mismatch.sum().item())} "
                    f"first_mismatch="
                    f"{int(mismatch_indices[0].item()) if mismatch_indices.numel() else None} "
                    f"max_absdiff_fp32={max_abs!r}"
                )
        adv_per_token = self._expand_advantages_to_tokens(
            advantages, segment.lengths, dtype=new_logp.dtype, device=new_logp.device
        )

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(self.clip_range_high, self.clip_schedule, training_progress)
        )
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=rollout_anchor_logp,
            advantages=adv_per_token,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )

        if self.loss_agg_mode in ("seq-mean-token-sum-norm", "seq-mean-token-mean") and segment.lengths is not None:
            parts = torch.split(loss_per_elem, segment.lengths.tolist())
            if self.loss_agg_mode == "seq-mean-token-sum-norm":
                loss = torch.stack([p.sum() for p in parts]).mean() / float(self.horizon)
            else:  # seq-mean-token-mean — guard 0-length responses (mean of empty = NaN)
                loss = torch.stack([p.mean() if p.numel() else p.new_zeros(()) for p in parts]).mean()
        else:
            loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        gradient_k3 = rollout_replay_k3(new_logp, rollout_anchor_logp)
        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **rollout_replay_logp_absdiff(new_logp, rollout_anchor_logp),
            "rollout_replay_k3_mean": gradient_k3["k3_mean"],
            "rollout_replay_k3_max": gradient_k3["k3_max"],
            "k3_mean": gradient_k3["k3_mean"],
            "k3_max": gradient_k3["k3_max"],
            "rollout_replay_exact_match_fp32": float(
                torch.equal(
                    new_logp.detach().float(),
                    rollout_anchor_logp.detach().float(),
                )
            ),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )

    @staticmethod
    def _expand_advantages_to_tokens(
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand per-sample ``advantages [B]`` to per-token ``[total_tokens]``."""
        bs = int(advantages.shape[0])
        if int(lengths.shape[0]) != bs:
            raise ValueError(f"GRPO advantage expansion: advantages batch={bs} != lengths={int(lengths.shape[0])}")
        chunks: List[torch.Tensor] = []
        adv_cast = advantages.detach().to(dtype=dtype, device=device)
        for k in range(bs):
            n = int(lengths[k].item())
            if n > 0:
                chunks.append(adv_cast[k].expand(n))
        if not chunks:
            return torch.zeros(0, dtype=dtype, device=device)
        return torch.cat(chunks, dim=0)


__all__ = ["GRPO", "GRPOConfig"]
