"""Resolve Sample-native AR sampling parameters for SGLang."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, Optional, cast

from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

_DEFAULT_SGLANG_SAMPLING_SEED = 42
_MAX_SGLANG_SAMPLING_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class ResolvedSampling:
    """Resolved per-call SGLang sampling; ``n`` is per payload and ``fanout`` is per parent."""

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
    if not 0 <= seed <= _MAX_SGLANG_SAMPLING_SEED:
        raise ValueError(f"base_seed must fit SGLang's non-negative int64 sampling seed range, got {base_seed!r}")
    return seed


def derive_sampling_seed(base_seed: int, sample_id: str) -> int:
    """Derive one order- and shard-invariant SGLang sampling seed."""
    seed = _validated_base_seed(base_seed)
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("sample_id must be a non-empty stable string identity")
    payload = f"{seed}::{sample_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & _MAX_SGLANG_SAMPLING_SEED


def deterministic_inference_enabled(engine_kwargs: Dict[str, Any]) -> bool:
    """Match the two SGLang server routes that enable deterministic inference."""
    return (
        engine_kwargs.get("enable_deterministic_inference") is True
        or engine_kwargs.get("rl_on_policy_target") is not None
    )


def _require_ar_frontier(sample: Sample) -> Part:
    """Require ``ARSamplingParams`` on the request ``Sample`` frontier."""
    try:
        return sample.frontier_gen_part(ARSamplingParams)
    except ValueError as exc:
        raise TypeError(
            "SGLang generation requires ARSamplingParams on the request Sample frontier. "
            f"{exc}. Direct callers must fork(..., sampling_params=ARSamplingParams(...)); "
            "in-tree trainer, PE, and agentic paths already stamp this. Engine-config "
            "temperature / top_p / top_k / max_new_tokens is no longer a fallback."
        ) from exc


def resolve_sampling(config: Any, sample: Sample) -> ResolvedSampling:
    """Resolve the SRT sampling block from the request Sample's ARSamplingParams."""
    gen_part = _require_ar_frontier(sample)
    ar = cast(ARSamplingParams, gen_part.sampling_params)
    input_part = sample.parts[0]
    control_ar: Dict[str, Any] = dict(input_part.control.get("ar") or {})

    parent_part = sample.parts[-2] if len(sample.parts) >= 2 else input_part
    n_parent = len(parent_part.sample_ids)
    fanout = (len(gen_part.sample_ids) // n_parent) if n_parent else 1

    engine_kwargs = getattr(config, "engine_kwargs", None) or {}
    deterministic = deterministic_inference_enabled(engine_kwargs)
    configured_seed = _validated_base_seed(ar.seed) if ar.seed is not None else None
    if configured_seed is not None and not deterministic:
        raise ValueError(
            "sampling.seed requires deterministic SGLang inference via "
            "engine_kwargs.enable_deterministic_inference=true or engine_kwargs.rl_on_policy_target"
        )
    base_seed = configured_seed
    if base_seed is None and deterministic:
        base_seed = _DEFAULT_SGLANG_SAMPLING_SEED  # SGLang's own deterministic default; see rollout/engine README
    n = 1 if base_seed is not None else fanout

    raw_top_k = ar.top_k
    block: Dict[str, Any] = {
        "temperature": float(ar.temperature),
        "max_new_tokens": int(ar.max_new_tokens),
        "top_p": float(ar.top_p),
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


__all__ = ["ResolvedSampling", "derive_sampling_seed", "deterministic_inference_enabled", "resolve_sampling"]
