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
    alignment_probe: bool = False
    alignment_gate_only: bool = False
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
        alignment_probe: bool = False,
        alignment_gate_only: bool = False,
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
        self.alignment_probe = bool(alignment_probe)
        self.alignment_gate_only = bool(alignment_gate_only)
        self.alignment_require_exact = bool(alignment_require_exact)

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        """Capture full-sequence no-grad old-policy logprobs for the rollout tokens."""
        if (
            not self.alignment_probe
            or segment.tokens is None
            or segment.log_probs is None
            or int(segment.tokens.shape[0]) == 0
        ):
            return
        if segment.rollout_log_probs is None:
            segment.rollout_log_probs = segment.log_probs.detach().cpu().clone()
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            old_policy_replay = getattr(self.stage, "old_policy_replay", None)
            replay = old_policy_replay if callable(old_policy_replay) else self.stage.replay
            old_logp = replay(
                typed_conds,
                segment=segment,
                temperature=self.sampling_temperature,
            )
        old_logp_cpu = old_logp.detach().cpu()
        segment.old_log_probs = old_logp_cpu
        # Backward-compatible alias for existing metrics and serialized segments.
        segment.actor_log_probs = old_logp_cpu

        rollout = segment.rollout_log_probs
        if old_logp.shape != rollout.shape:
            raise RuntimeError(
                f"GRPO old-policy/rollout shape mismatch: old={tuple(old_logp.shape)} rollout={tuple(rollout.shape)}"
            )
        old_fp32 = old_logp.detach().float()
        rollout_fp32 = rollout.to(device=old_logp.device).float()
        finite = bool(torch.isfinite(old_fp32).all().item() and torch.isfinite(rollout_fp32).all().item())
        mismatch = torch.nonzero(
            old_fp32 != rollout_fp32,
            as_tuple=False,
        ).reshape(-1)
        absdiff = rollout_replay_logp_absdiff(old_fp32, rollout_fp32)
        k3 = rollout_replay_k3(old_fp32, rollout_fp32)
        equal = torch.equal(old_fp32, rollout_fp32)
        if self.alignment_require_exact and (not finite or not equal):
            max_abs = absdiff["rollout_replay_logp_absdiff_max"]
            raise RuntimeError(
                "GRPO exact full-sequence old-policy alignment gate failed: "
                f"finite={finite} mismatch_count={int(mismatch.numel())} "
                f"first_mismatch={int(mismatch[0]) if mismatch.numel() else None} "
                f"max_absdiff_fp32={max_abs!r} "
                f"k3_mean={k3['k3_mean']!r} k3_max={k3['k3_max']!r}"
            )

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

        if self.alignment_gate_only:
            cached_old_logp = segment.old_log_probs if segment.old_log_probs is not None else segment.actor_log_probs
            if cached_old_logp is None:
                raise RuntimeError("GRPO alignment_gate_only requires alignment_probe old-policy logprobs")
            rollout_logp = segment.rollout_log_probs if segment.rollout_log_probs is not None else segment.log_probs
            old_logp = cached_old_logp.to(device=rollout_logp.device).float()
            rollout_logp = rollout_logp.float()
            old_absdiff = rollout_replay_logp_absdiff(old_logp, rollout_logp)
            old_k3 = rollout_replay_k3(old_logp, rollout_logp)
            metrics = {
                "old_rollout_logp_absdiff_mean": old_absdiff["rollout_replay_logp_absdiff_mean"],
                "old_rollout_logp_absdiff_max": old_absdiff["rollout_replay_logp_absdiff_max"],
                "old_rollout_k3_mean": old_k3["k3_mean"],
                "old_rollout_k3_max": old_k3["k3_max"],
                "k3_mean": old_k3["k3_mean"],
                "k3_max": old_k3["k3_max"],
                # Compatibility aliases; actor_log_probs historically meant
                # the no-grad recomputed old-policy logprobs.
                "actor_rollout_logp_absdiff_mean": old_absdiff["rollout_replay_logp_absdiff_mean"],
                "actor_rollout_logp_absdiff_max": old_absdiff["rollout_replay_logp_absdiff_max"],
                **{f"actor_rollout_{key}": value for key, value in old_k3.items()},
            }
            if self.alignment_require_exact and not torch.equal(
                old_logp,
                rollout_logp,
            ):
                mismatch = old_logp != rollout_logp
                first = int(torch.nonzero(mismatch, as_tuple=False).reshape(-1)[0].item())
                max_abs = float((old_logp - rollout_logp).abs().max().item())
                raise RuntimeError(
                    "GRPO exact alignment gate failed: "
                    f"mismatch_count={int(mismatch.sum().item())} "
                    f"first_mismatch={first} max_abs={max_abs}"
                )
            return AlgorithmStepResult(
                loss=0.0,
                metrics=metrics,
                num_steps_or_tokens=int(old_logp.shape[0]),
                has_backward=False,
            )

        cached_old_logp = segment.old_log_probs if segment.old_log_probs is not None else segment.actor_log_probs
        if self.alignment_require_exact and cached_old_logp is not None:
            rollout_reference = (
                segment.rollout_log_probs if segment.rollout_log_probs is not None else segment.log_probs
            ).float()
            old_reference = cached_old_logp.to(
                device=rollout_reference.device,
            ).float()
            if not torch.equal(old_reference, rollout_reference):
                mismatch = old_reference != rollout_reference
                first = int(torch.nonzero(mismatch, as_tuple=False).reshape(-1)[0].item())
                max_abs = float((old_reference - rollout_reference).abs().max().item())
                raise RuntimeError(
                    "GRPO exact full-sequence old-policy alignment gate failed before backward: "
                    f"mismatch_count={int(mismatch.sum().item())} "
                    f"first_mismatch={first} max_absdiff_fp32={max_abs!r}"
                )

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
        rollout_anchor = segment.rollout_log_probs if segment.rollout_log_probs is not None else segment.log_probs
        rollout_anchor_logp = rollout_anchor.to(
            dtype=new_logp.dtype,
            device=new_logp.device,
        )
        if self.alignment_require_exact:
            gradient_reference = new_logp.float()
            rollout_reference = rollout_anchor_logp.float()
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
            "rollout_replay_exact_match_fp32": float(
                torch.equal(
                    new_logp.detach().float(),
                    rollout_anchor_logp.detach().float(),
                )
            ),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        cached_old_logp = segment.old_log_probs if segment.old_log_probs is not None else segment.actor_log_probs
        if cached_old_logp is not None:
            rollout_logp = (
                (segment.rollout_log_probs if segment.rollout_log_probs is not None else segment.log_probs)
                .to(device=new_logp.device)
                .float()
            )
            old_logp = cached_old_logp.to(device=new_logp.device).float()
            old_absdiff = rollout_replay_logp_absdiff(old_logp, rollout_logp)
            old_k3 = rollout_replay_k3(old_logp, rollout_logp)
            metrics.update(
                {
                    "old_rollout_logp_absdiff_mean": old_absdiff["rollout_replay_logp_absdiff_mean"],
                    "old_rollout_logp_absdiff_max": old_absdiff["rollout_replay_logp_absdiff_max"],
                    "old_rollout_k3_mean": old_k3["k3_mean"],
                    "old_rollout_k3_max": old_k3["k3_max"],
                    "k3_mean": old_k3["k3_mean"],
                    "k3_max": old_k3["k3_max"],
                    "actor_rollout_logp_absdiff_mean": old_absdiff["rollout_replay_logp_absdiff_mean"],
                    "actor_rollout_logp_absdiff_max": old_absdiff["rollout_replay_logp_absdiff_max"],
                    **{f"actor_rollout_{key}": value for key, value in old_k3.items()},
                }
            )
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
