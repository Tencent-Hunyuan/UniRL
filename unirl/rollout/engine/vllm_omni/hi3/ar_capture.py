"""Capture HI3 AR Stage 0 outputs (tokens + log-probs) from ``Omni`` outputs.

vLLM-Omni's stock multi-stage pipeline for HI3 it2i / moe ships Stage 0 as
``final_output: false`` (`hunyuan_image3_it2i.yaml`,
`hunyuan_image3_moe.yaml`) — its outputs are consumed internally via KV
transfer to Stage 1 and never reach ``Omni.generate``'s caller. We work
around that by flipping ``final_output: true`` on Stage 0 in the YAML we
synthesize at runtime (see ``backends/omni.py``). The orchestrator then
yields one ``OmniRequestOutput`` per final stage per request, and our
extractor here picks out Stage 0's payload.

Each per-prompt result from ``Omni.generate`` is a list-shaped collection
that, post-our-yaml-tweak, looks like::

    [
        OmniRequestOutput(stage_id=0, final_output_type="text", request_output=<vLLM RequestOutput>, ...),
        OmniRequestOutput(stage_id=1, final_output_type="image", trajectory_*, ...),
    ]

The vLLM ``RequestOutput`` on Stage 0 carries the standard
``outputs[k].token_ids`` / ``outputs[k].logprobs`` shapes — we
concatenate per-row into the unirl varlen ``TextSegment``.

If the orchestrator returns nothing for Stage 0 (e.g. because the YAML
patch didn't take effect or the AR worker doesn't surface its output),
the extractor returns ``None`` and the caller can stamp an empty
``TextSegment`` so the ``RolloutResp.tracks["ar"]`` slot is at least
present.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch


def _flatten_logprobs(logprobs: Any, fallback_len: int) -> Optional[torch.Tensor]:
    """Best-effort vLLM-logprob → ``[T]`` float tensor.

    vLLM 0.16 ``CompletionOutput.logprobs`` is ``list[dict[token_id, Logprob]]``
    of length T, where each dict has the sampled token's logprob plus
    optional top-K alternatives. We pick out the sampled-token entry per
    step and stack into a ``[T]`` tensor.

    Returns ``None`` when ``logprobs`` is missing or empty (matches the
    vLLM-Omni AR config's ``detokenize=False`` path which can also drop
    logprobs unless explicitly requested).
    """
    if logprobs is None:
        return None
    if not isinstance(logprobs, Sequence) or len(logprobs) == 0:
        return None
    values: List[float] = []
    for step in logprobs:
        if step is None:
            values.append(0.0)
            continue
        # vLLM Logprob objects expose ``.logprob``; dicts usually have one
        # entry whose value is the Logprob object. Try both shapes.
        if hasattr(step, "logprob"):
            values.append(float(step.logprob))
            continue
        if isinstance(step, dict) and step:
            entry = next(iter(step.values()))
            values.append(float(getattr(entry, "logprob", entry)))
            continue
        values.append(0.0)
    if not values:
        return None
    if len(values) != fallback_len and fallback_len > 0:
        # Truncate or pad-with-zeros so downstream stack stays well-shaped.
        # Pad case is rare — only happens if vLLM emits fewer logprobs than
        # tokens (e.g. due to early-stop / retokenize mismatches).
        if len(values) > fallback_len:
            values = values[:fallback_len]
        else:
            values.extend([0.0] * (fallback_len - len(values)))
    return torch.tensor(values, dtype=torch.float32)


def _extract_completion(out: Any) -> Tuple[List[int], Optional[torch.Tensor]]:
    """Pull ``(token_ids, per_token_logp)`` out of an ``OmniRequestOutput`` shaped Stage 0 result."""
    request_output = getattr(out, "request_output", None)
    if request_output is None:
        return [], None
    completions = getattr(request_output, "outputs", None) or []
    if not completions:
        return [], None
    completion = completions[0]
    tokens = list(getattr(completion, "token_ids", []) or [])
    logp = _flatten_logprobs(getattr(completion, "logprobs", None), fallback_len=len(tokens))
    return tokens, logp


def extract_ar_segment(per_request_outputs: Sequence[Sequence[Any]]) -> Optional[Any]:
    """Build a ``TextSegment`` from the AR Stage 0 outputs of one batch.

    ``per_request_outputs[i]`` is the list of ``OmniRequestOutput``s the
    orchestrator yielded for request ``i`` — one per final stage
    (post-YAML-tweak: Stage 0 + Stage 1 = two entries). We pick the
    Stage 0 entry, gather its tokens + per-token logprobs, and hand
    per-sample lists to :meth:`TextSegment.pack`, which packs along dim
    0 and derives the framework-managed ``cu_seqlens``.

    Segment rows are 1:1 with samples (one AR segment per sample).

    Returns ``None`` when no Stage 0 output is found in any row — the
    caller should then either stamp an empty ``TextSegment`` or omit
    ``tracks["ar"]`` entirely.
    """
    # Local import to keep this module importable without pulling
    # unirl.types' torch dependency until it's actually used.
    from unirl.types.segments.text import TextSegment

    rows_tokens: List[List[int]] = []
    rows_logps: List[Optional[torch.Tensor]] = []
    found_any_stage0 = False

    for outputs in per_request_outputs:
        stage0 = None
        for out in outputs:
            if getattr(out, "stage_id", None) == 0:
                stage0 = out
                break
        if stage0 is None:
            rows_tokens.append([])
            rows_logps.append(None)
            continue
        toks, logp = _extract_completion(stage0)
        if toks:
            found_any_stage0 = True
        rows_tokens.append(toks)
        rows_logps.append(logp)

    if not found_any_stage0:
        return None

    # Build per-sample tensor lists; ``TextSegment.pack`` packs them along
    # dim 0 and derives the framework-managed ``cu_seqlens``. ``log_probs``
    # is all-or-nothing across rows: if any row with tokens is missing
    # logp, we drop the whole field rather than emit a ragged-shaped one.
    tokens_list: List[torch.Tensor] = [torch.tensor(toks, dtype=torch.long) for toks in rows_tokens]
    have_logp = all(lp is not None for toks, lp in zip(rows_tokens, rows_logps) if toks)
    log_probs_list: Optional[List[torch.Tensor]] = None
    if have_logp:
        log_probs_list = [lp if lp is not None else torch.zeros(0, dtype=torch.float32) for lp in rows_logps]

    return TextSegment.pack(
        tokens=tokens_list,
        log_probs=log_probs_list,
    )


__all__ = ["extract_ar_segment"]
