"""Sampling resolution — the single consolidation of the three param sources.

The predecessor resolved sampling inline across ``generate`` and the async
helper, re-deriving the precedence per field. This is the one place it happens
now: typed ``ARSamplingParams`` (``req.sampling_params['ar']``) > the
``req.task_config['ar']`` bag > engine-config defaults, including the
``top_k`` translation, deterministic base seed, and the
``samples_pre_expanded`` n-logic. Pure — table-testable with config/req
stand-ins.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, Optional

from unirl.types.rollout_req import RolloutReq

_MAX_SGLANG_SAMPLING_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class ResolvedSampling:
    """One ``generate`` call's resolved sampling, ready for the wire.

    ``block`` is the SRT ``sampling_params`` sub-dict (``n`` included);
    ``system_instruction`` feeds the chat template, not the wire.
    ``base_seed`` stays separate because the adapter derives one SGLang
    ``sampling_seed`` per expanded request entry.
    """

    n: int
    return_logprob: bool
    system_instruction: Optional[str]
    block: Dict[str, Any] = field(default_factory=dict)
    base_seed: Optional[int] = None


def _validated_base_seed(base_seed: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise ValueError(f"base_seed must be an integer, got {base_seed!r}")
    seed = int(base_seed)
    if not 0 <= seed <= _MAX_SGLANG_SAMPLING_SEED:
        raise ValueError(f"base_seed must fit SGLang's non-negative int64 sampling seed range, got {base_seed!r}")
    return seed


def derive_sampling_seed(base_seed: int, sample_id: str) -> int:
    """Derive the SGLang seed for one expanded request entry.

    ``sample_id`` is authored before DP sharding, so this mapping is invariant
    to request order, rank count, and asynchronous completion order. Python's
    process-randomized ``hash`` is deliberately avoided.
    """
    seed = _validated_base_seed(base_seed)
    identity = str(sample_id)
    if not identity.strip():
        raise ValueError("sample_id must be a non-empty stable identity")
    payload = f"{seed}::{identity}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & _MAX_SGLANG_SAMPLING_SEED


def resolve_sampling(config: Any, req: RolloutReq) -> ResolvedSampling:
    """Resolve the SRT sampling block for one request.

    Reproduces the predecessor's exact precedence:

    - ``n``: 1 when ``config.samples_pre_expanded`` (the caller already
      expanded P prompts → P*N entries, one per GRPO sibling — re-applying
      ``samples_per_prompt`` would generate N completions per expanded entry);
      else ``ar.samples_per_prompt``, else ``task_ar['n']``, else 1.
    - ``temperature`` / ``top_p`` / ``max_new_tokens``: typed AR params, else
      the config defaults.
    - ``top_k``: typed AR params, else the config default. The value must still
      be sent so SGLang does not fall back to a model-specific generation-config
      limit. The trainer/config ``top_k=0`` (HF convention) maps to SGLang's
      ``-1`` (disabled); positive values pass through.
    - ``base_seed``: typed AR ``seed`` only. It remains out of ``block`` so
      ``seed=None`` leaves the legacy payload byte-for-byte unchanged. Seeded
      requests require SGLang deterministic inference and one output per
      pre-expanded entry; adapters derive the wire ``sampling_seed`` from this
      base plus the stable ``sample_id``.
    - ``return_logprob`` (default True), ``system_instruction``, and the
      ``stop`` / ``stop_token_ids`` / ``skip_special_tokens`` passthroughs
      come from ``task_config['ar']``.
    """
    ar = req.sampling_params.get("ar")
    task_ar: Dict[str, Any] = dict(req.task_config.get("ar") or {})

    if config.samples_pre_expanded:
        n = 1
    else:
        n = int(ar.samples_per_prompt if ar is not None else task_ar.get("n", 1))

    base_seed = _validated_base_seed(ar.seed) if ar is not None and ar.seed is not None else None
    if base_seed is not None:
        if n != 1:
            raise ValueError(
                "Seeded SGLang sampling requires one output per request entry (n=1); "
                "pre-expand siblings so each output has its own stable sample_id"
            )
        engine_kwargs = getattr(config, "engine_kwargs", None) or {}
        if engine_kwargs.get("enable_deterministic_inference") is not True:
            raise ValueError(
                "sampling.seed requires "
                "rollout.config.engine_kwargs.enable_deterministic_inference=true; "
                "SGLang otherwise ignores sampling_seed"
            )

    raw_top_k = ar.top_k if ar is not None else config.top_k
    block: Dict[str, Any] = {
        "temperature": float(ar.temperature if ar is not None else config.temperature),
        "max_new_tokens": int(ar.max_new_tokens if ar is not None else config.max_new_tokens),
        "top_p": float(ar.top_p if ar is not None else config.top_p),
        "top_k": raw_top_k if raw_top_k > 0 else -1,
        "n": n,
    }
    for key in ("stop", "stop_token_ids", "skip_special_tokens"):
        if key in task_ar:
            block[key] = task_ar[key]

    return ResolvedSampling(
        n=n,
        return_logprob=bool(task_ar.get("return_logprob", True)),
        system_instruction=task_ar.get("system_instruction") or config.system_instruction,
        block=block,
        base_seed=base_seed,
    )


__all__ = ["ResolvedSampling", "derive_sampling_seed", "resolve_sampling"]
