"""Autoregressive model interfaces.

Pipeline-level: ``ARStage[C]`` — ``C → TextSegment``, iterates an ``ARStep``
token-by-token. Parameterized on the conditions container type ``C`` for
parity with ``DiffusionStage[C]`` so each bundle declares its own typed
container.

Step-level kernel: ``ARStep`` — per-token transition kernel: ``sample`` is
the logits→token math kernel; ``init_state`` / ``step`` own the per-token
model forward (mirroring ``DiffusionStep.forward`` vs ``.step``).

The legacy ``ARTrajectory`` type is deleted — ``TextSegment`` (in
``unirl/types/segments/text.py``) replaces it.
"""

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
    """Rollout-level AR stage: ``C → TextSegment``.

    Schedule-equivalent (``sampling_params``) is passed at call time, not
    held on the instance. Returned ``TextSegment`` is varlen-packed.

    The conditions type ``C`` is per-bundle: each AR bundle declares its
    own typed conditions container.

    ``replay`` recomputes per-token log-probs for a stored rollout's
    response tokens via a single teacher-forced forward over
    ``prompt + response``. Returns a packed-varlen ``[total_tokens]``
    tensor aligned with ``segment.log_probs``. Used by GRPO/PPO-style
    policy-gradient training.
    """

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
    """Per-token AR transition kernel — two levels, mirroring ``DiffusionStep``.

    - ``sample(logits)`` — the math kernel: given ``[B, vocab]`` logits at
      the current position, sample the next token and return
      ``(token_id [B], log_prob [B])``.
    - ``init_state`` / ``step`` — the full transition. ``init_state`` builds
      the model-facing decode state (KV cache, initial model_kwargs) without
      running a forward; ``step`` runs one token's model forward, ``sample``,
      and the state advance, returning ``(token_id, log_prob, state)``.

    Unlike ``DiffusionStep`` (whose per-model state is default-constructed
    by the stage and lazily filled on step 0), AR gets an explicit
    ``init_state`` because the KV cache must be pre-sized with
    ``prompt_len + max_new_tokens`` — a loop-level quantity the per-step
    kernel otherwise never sees.

    The state type ``S`` is per-bundle (e.g. ``HunyuanImage3ARState``):
    growing ``input_ids``, the HF-style ``model_kwargs`` threaded across
    steps, and the step index.

    MIGRATION NOTE: ``Qwen3ARStep`` / ``QwenVLARStep`` still implement the
    legacy logits-only shape (``step(logits) -> (token, logp)``) and are
    invoked only by their own stages' loops; migrating them to this
    protocol is a follow-up.
    """

    def sample(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: ...

    def init_state(self, model: B, conditions: C, *, max_new_tokens: int) -> S: ...

    def step(self, model: B, conditions: C, state: S) -> Tuple[torch.Tensor, torch.Tensor, S]: ...


def left_pad_prompt(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-pad a right-padded prompt batch to LEFT-padding for batched decode.

    The chat-template stages right-pad prompts to the in-batch max (``[real |
    pad]``). That breaks batched autoregressive generation when prompts differ
    in length: the decode loop reads ``logits[:, -1, :]`` and appends the new
    token at the end, but for a short row the last column is a PAD position (and
    appended tokens land *after* the pad run), so it decodes from the wrong
    context. Left-padding (``[pad | real]``) right-aligns every row's last real
    token at the final column, so ``logits[:, -1]`` and end-append are correct
    for all rows; HF's ``prepare_inputs_for_generation`` derives the right
    ``position_ids`` from the (left-padded) ``attention_mask``.

    Returns ``(left_padded_ids, left_padded_mask)`` trimmed to the in-batch max
    *real* length (excess right-pad columns are dropped).

    NO-OP for an equal-length batch — the validated same-prompt-group recipe
    (``forward_batch_size == samples_per_prompt``) batches identical prompts, so
    every row is already full-length and the output is byte-identical to the
    input. Only mixed-length batches (currently mis-decoded) are rewritten.
    """
    real_lens = attention_mask.long().sum(dim=1)  # [B]
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
        # Gather row b's real tokens (mask==1, in order) and right-align them.
        real_tokens = input_ids[b][bool_mask[b]][:max_real]
        lp_ids[b, max_real - n :] = real_tokens
        lp_mask[b, max_real - n :] = 1
    return lp_ids, lp_mask


__all__ = ["ARSamplingParams", "ARStage", "ARStep", "left_pad_prompt"]
