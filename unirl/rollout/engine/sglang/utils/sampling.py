"""Resolve Sample-native AR sampling parameters for SGLang."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from unirl.types.sample import Sample


@dataclass(frozen=True)
class ResolvedSampling:
    """One ``generate`` call's resolved sampling, ready for the wire."""

    n: int
    return_logprob: bool
    system_instruction: Optional[str]
    block: Dict[str, Any] = field(default_factory=dict)


def resolve_sampling(config: Any, sample: Sample) -> ResolvedSampling:
    """Resolve the SRT sampling block for one request ``Sample``."""
    input_part, gen_part = sample.parts[0], sample.parts[-1]
    ar = gen_part.sampling_params
    control_ar: Dict[str, Any] = dict(input_part.control.get("ar") or {})

    parent_part = sample.parts[-2] if len(sample.parts) >= 2 else input_part
    n_parent = len(parent_part.sample_ids)
    n = (len(gen_part.sample_ids) // n_parent) if n_parent else 1

    raw_top_k = ar.top_k if ar is not None else config.top_k
    block: Dict[str, Any] = {
        "temperature": float(ar.temperature if ar is not None else config.temperature),
        "max_new_tokens": int(ar.max_new_tokens if ar is not None else config.max_new_tokens),
        "top_p": float(ar.top_p if ar is not None else config.top_p),
        "top_k": raw_top_k if raw_top_k > 0 else -1,
        "n": n,
    }
    for key in ("stop", "stop_token_ids", "skip_special_tokens"):
        if key in control_ar:
            block[key] = control_ar[key]

    return ResolvedSampling(
        n=n,
        return_logprob=bool(control_ar.get("return_logprob", True)),
        system_instruction=control_ar.get("system_instruction") or config.system_instruction,
        block=block,
    )


__all__ = ["ResolvedSampling", "resolve_sampling"]
