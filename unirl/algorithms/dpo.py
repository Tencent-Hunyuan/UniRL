"""Stage-driven offline ``DPO`` over a ``TextSegment`` — not ``DPPO``, which is Divergence-PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

import torch
import torch.nn.functional as F

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _resolve_reference_model,
    typed_conditions,
)

_LOSS_TYPES = ("sigmoid", "ipo")


@dataclass
class DPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    average_log_prob: bool = False


class DPO(StageAlgorithm):
    """Bradley-Terry DPO over adjacent chosen/rejected rows of an AR ``TextSegment``."""

    requires_advantages = False
    requires_backend = True
    # The reference is the adapter-disabled base, recomputed inline per micro-batch, so it
    # cannot drift as the policy updates — see ``README.md`` Gotchas.
    supports_multi_update = True

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        backend: Any = None,
        stage_attr: str = "ar",
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        average_log_prob: bool = False,
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("DPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        if loss_type not in _LOSS_TYPES:
            raise ValueError(f"DPO: loss_type must be one of {_LOSS_TYPES}; got {loss_type!r}.")
        if not 0.0 <= float(label_smoothing) < 0.5:
            raise ValueError(f"DPO: label_smoothing must be in [0, 0.5); got {label_smoothing!r}.")
        self.stage = stage
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.loss_type = str(loss_type)
        self.average_log_prob = bool(average_log_prob)
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)
        # beta>0 is the DPO temperature, so the reference policy is never optional here.
        self._ref_model = _resolve_reference_model(backend, beta=self.beta, algo="DPO")
        if self._ref_model is None:
            raise ValueError("DPO: beta must be > 0 — the loss is defined against a reference policy.")

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del advantages, training_progress
        if segment is None or segment.tokens is None or segment.lengths is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        loss, metrics, num_pairs = self._pair_loss(conditions, segment)
        (loss * loss_scale).backward()
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=num_pairs,
            has_backward=True,
        )

    @torch.no_grad()
    def evaluate_loss(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        sample_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[float, float]:
        """Forward-only ``(loss_sum, num_pairs)`` for validation."""
        del sample_ids
        if segment is None or segment.tokens is None or int(segment.tokens.shape[0]) == 0:
            return 0.0, 0.0
        loss, _, num_pairs = self._pair_loss(conditions, segment)
        return float(loss.detach().item()) * num_pairs, float(num_pairs)

    def _pair_loss(
        self,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> Tuple[torch.Tensor, Dict[str, Any], int]:
        """Policy and adapter-disabled reference replays combined into the pairwise DPO loss."""
        from unirl.train.lora import adapters_disabled

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        policy_logp = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
        with torch.no_grad(), adapters_disabled(self._ref_model):
            ref_logp = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)

        policy_seq = self._reduce_to_sequences(policy_logp, segment)
        ref_seq = self._reduce_to_sequences(ref_logp.detach(), segment)
        policy_chosen, policy_rejected = self._split_adjacent(policy_seq)
        ref_chosen, ref_rejected = self._split_adjacent(ref_seq)

        logits = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
        if self.loss_type == "ipo":
            losses = (logits - 1.0 / (2.0 * self.beta)) ** 2
        else:
            losses = (
                -F.logsigmoid(self.beta * logits) * (1.0 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        loss = losses.mean()

        chosen_rewards = (self.beta * (policy_chosen - ref_chosen)).detach()
        rejected_rewards = (self.beta * (policy_rejected - ref_rejected)).detach()
        metrics: Dict[str, Any] = {
            "dpo_loss": float(loss.detach().item()),
            "chosen_rewards": float(chosen_rewards.mean().item()),
            "rejected_rewards": float(rejected_rewards.mean().item()),
            "reward_accuracy": float((chosen_rewards > rejected_rewards).float().mean().item()),
            "reward_margin": float((chosen_rewards - rejected_rewards).mean().item()),
            "policy_chosen_logp": float(policy_chosen.detach().mean().item()),
            "policy_rejected_logp": float(policy_rejected.detach().mean().item()),
            "logits_mean": float(logits.detach().mean().item()),
        }
        return loss, metrics, int(policy_chosen.shape[0])

    def _reduce_to_sequences(self, packed_logp: torch.Tensor, segment: "TextSegment") -> torch.Tensor:
        """Segment-sum packed per-token log-probs to one value per sequence; mean only when configured."""
        device = packed_logp.device
        lengths = segment.lengths.to(device)
        num_seqs = int(lengths.shape[0])
        seg_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lengths)
        if segment.loss_mask is not None:
            mask = segment.loss_mask.to(dtype=packed_logp.dtype, device=device)
            packed_logp = packed_logp * mask
            denom = packed_logp.new_zeros(num_seqs).index_add(0, seg_ids, mask)
        else:
            denom = lengths.to(packed_logp.dtype)
        seq_logp = packed_logp.new_zeros(num_seqs).index_add(0, seg_ids, packed_logp)
        if self.average_log_prob:
            seq_logp = seq_logp / denom.clamp(min=1)
        return seq_logp

    @staticmethod
    def _split_adjacent(seq_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split ``[2P]`` adjacent rows into chosen (even) and rejected (odd); see ``README.md`` Gotchas."""
        if int(seq_values.shape[0]) % 2 != 0:
            raise ValueError(
                f"DPO: expected adjacent chosen/rejected rows, got an odd sequence count "
                f"{int(seq_values.shape[0])}. Every micro-batch must hold whole preference pairs — "
                f"set stack.micro_batch_size to an even value (2), and keep batch_size divisible "
                f"by the DP size so no pair is split across ranks."
            )
        return seq_values[0::2], seq_values[1::2]


__all__ = ["DPO", "DPOConfig"]
