"""Per-sample AR seeds for vLLM-Omni calls; see the engine README's Gotchas."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

from unirl.rollout.engine.sglang.utils.sampling import derive_sampling_seed
from unirl.rollout.engine.vllm_omni.backends import STAGE_KIND_AR, GenerateCall


def seed_ar_calls(calls: List[GenerateCall], seed: Optional[int], sample_ids: List[str]) -> List[GenerateCall]:
    """Split explicitly seeded AR calls by row, preserving prompt order and diffusion sampling."""
    if seed is None:
        return calls

    rows = ((call, prompt) for call in calls for prompt in call.prompts)
    seeded_calls: List[GenerateCall] = []
    for (call, prompt), sample_id in zip(rows, sample_ids, strict=True):
        child_seed = derive_sampling_seed(seed, sample_id)
        sampling = [
            replace(stage, kwargs={**stage.kwargs, "seed": child_seed}) if stage.kind == STAGE_KIND_AR else stage
            for stage in call.sampling
        ]
        seeded_calls.append(replace(call, prompts=[prompt], sampling=sampling))
    return seeded_calls


__all__ = ["seed_ar_calls"]
