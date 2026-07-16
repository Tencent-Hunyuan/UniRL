"""Sampling resolution — the single consolidation of the three param sources.

The predecessor resolved sampling inline across ``generate`` and the async
helper, re-deriving the precedence per field. This is the one place it happens
now: typed ``ARSamplingParams`` (``req.sampling_params['ar']``) > the
``req.stage_config['ar']`` bag > engine-config defaults, including the
``top_k`` translation and the ``samples_pre_expanded`` n-logic. Pure —
table-testable with config/req stand-ins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from unirl.types.rollout_req import RolloutReq


@dataclass(frozen=True)
class ResolvedSampling:
    """One ``generate`` call's resolved sampling, ready for the wire.

    ``block`` is the SRT ``sampling_params`` sub-dict (``n`` included);
    ``system_instruction`` feeds the chat template, not the wire.
    """

    n: int
    return_logprob: bool
    system_instruction: Optional[str]
    block: Dict[str, Any] = field(default_factory=dict)


def resolve_sampling(config: Any, req: RolloutReq) -> ResolvedSampling:
    """Resolve the SRT sampling block for one request.

    Reproduces the predecessor's exact precedence:

    - ``n``: 1 when ``config.samples_pre_expanded`` (the caller already
      expanded P prompts → P*N entries, one per GRPO sibling — re-applying
      ``samples_per_prompt`` would generate N completions per expanded entry);
      else ``ar.samples_per_prompt``, else ``stage_ar['n']``, else 1.
    - ``temperature`` / ``top_p`` / ``max_new_tokens``: typed AR params, else
      the config defaults.
    - ``top_k``: typed AR params, else the config default. The value must still
      be sent so SGLang does not fall back to a model-specific generation-config
      limit. The trainer/config ``top_k=0`` (HF convention) maps to SGLang's
      ``-1`` (disabled); positive values pass through.
    - ``return_logprob`` (default True), ``system_instruction``, and the
      ``stop`` / ``stop_token_ids`` / ``skip_special_tokens`` passthroughs
      come from ``stage_config['ar']``.
    """
    ar = req.sampling_params.get("ar")
    stage_ar: Dict[str, Any] = dict(req.stage_config.get("ar") or {})

    if config.samples_pre_expanded:
        n = 1
    else:
        n = int(ar.samples_per_prompt if ar is not None else stage_ar.get("n", 1))

    raw_top_k = ar.top_k if ar is not None else config.top_k
    block: Dict[str, Any] = {
        "temperature": float(ar.temperature if ar is not None else config.temperature),
        "max_new_tokens": int(ar.max_new_tokens if ar is not None else config.max_new_tokens),
        "top_p": float(ar.top_p if ar is not None else config.top_p),
        "top_k": raw_top_k if raw_top_k > 0 else -1,
        "n": n,
    }
    for key in ("stop", "stop_token_ids", "skip_special_tokens"):
        if key in stage_ar:
            block[key] = stage_ar[key]

    return ResolvedSampling(
        n=n,
        return_logprob=bool(stage_ar.get("return_logprob", True)),
        system_instruction=stage_ar.get("system_instruction") or config.system_instruction,
        block=block,
    )


__all__ = ["ResolvedSampling", "resolve_sampling"]
