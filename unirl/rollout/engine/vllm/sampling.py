"""Resolve Sample-native AR sampling parameters for direct vLLM."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from unirl.types.sample import Sample

_MAX_VLLM_SAMPLING_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class ResolvedSampling:
    """Resolved per-call vLLM sampling parameters."""

    n: int
    fanout: int
    return_logprob: bool
    system_instruction: Optional[str]
    block: Dict[str, Any]
    base_seed: Optional[int] = None


def _validated_base_seed(base_seed: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise ValueError(f"base_seed must be an integer, got {base_seed!r}")
    seed = int(base_seed)
    if not 0 <= seed <= _MAX_VLLM_SAMPLING_SEED:
        raise ValueError(f"base_seed must fit vLLM's non-negative int64 sampling seed range, got {base_seed!r}")
    return seed


def resolve_sampling(config: Any, sample: Sample) -> ResolvedSampling:
    """Resolve one direct-vLLM request, including per-sample native seeds."""
    input_part, gen_part = sample.parts[0], sample.parts[-1]
    ar = gen_part.sampling_params
    control_ar: Dict[str, Any] = dict(input_part.control.get("ar") or {})

    parent_part = sample.parts[-2] if len(sample.parts) >= 2 else input_part
    n_parent = len(parent_part.sample_ids)
    fanout = (len(gen_part.sample_ids) // n_parent) if n_parent else 1
    base_seed = _validated_base_seed(ar.seed) if ar is not None and ar.seed is not None else None
    n = 1 if base_seed is not None else fanout

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
        fanout=fanout,
        return_logprob=bool(control_ar.get("return_logprob", True)),
        system_instruction=control_ar.get("system_instruction") or config.system_instruction,
        block=block,
        base_seed=base_seed,
    )


__all__ = ["ResolvedSampling", "resolve_sampling"]
