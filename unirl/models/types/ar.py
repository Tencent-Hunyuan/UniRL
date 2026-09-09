"""Autoregressive model interfaces."""

from __future__ import annotations

from typing import Any, Protocol, Tuple, TypeVar, runtime_checkable

import torch

from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment

B = TypeVar("B")
C = TypeVar("C")
S = TypeVar("S")


@runtime_checkable
class ARStage(Protocol[C]):
    """Rollout-level AR stage ``C → TextSegment``; ``replay`` returns packed varlen ``[total_tokens]``."""

    def autoregress(
        self,
        conditions: C,
        *,
        sampling_params: ARSamplingParams,
        **kwargs: Any,
    ) -> TextSegment: ...

    def replay(
        self,
        conditions: C,
        *,
        segment: TextSegment,
    ) -> torch.Tensor: ...


@runtime_checkable
class ARStep(Protocol[B, C, S]):
    """Per-token AR kernel: ``sample(logits [B, vocab])`` to ``(token_id [B], log_prob [B])``, then a state advance."""

    def sample(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def init_state(self, model: B, conditions: C, *, max_new_tokens: int) -> S: ...

    def step(self, model: B, conditions: C, state: S) -> Tuple[torch.Tensor, torch.Tensor, S]: ...


def left_pad_prompt(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-pad a right-padded prompt batch to LEFT-padding for batched decode."""
    real_lens = attention_mask.long().sum(dim=1)
    if real_lens.numel() == 0:
        return input_ids, attention_mask
    max_real = int(real_lens.max().item())
    if max_real == 0:
        return input_ids, attention_mask

    batch = int(input_ids.shape[0])
    device = input_ids.device
    lp_ids = torch.full((batch, max_real), int(pad_id), dtype=input_ids.dtype, device=device)
    lp_mask = torch.zeros((batch, max_real), dtype=attention_mask.dtype, device=device)
    bool_mask = attention_mask.bool()
    for b in range(batch):
        n = int(real_lens[b].item())
        if n == 0:
            continue
        real_tokens = input_ids[b][bool_mask[b]][:max_real]
        lp_ids[b, max_real - n :] = real_tokens
        lp_mask[b, max_real - n :] = 1
    return lp_ids, lp_mask


__all__ = ["ARSamplingParams", "ARStage", "ARStep", "left_pad_prompt"]
