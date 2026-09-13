"""Driver-side reward adapter: build requests and attach scalar responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from unirl.types.primitives import PrimitiveValue, primitive_modality_key
from unirl.types.reward import RewardRequest
from unirl.types.sample import Sample, _part_with_field
from unirl.types.sampling import ARSamplingParams

if TYPE_CHECKING:
    from unirl.distributed.group.handle import Handle


def _build_reward_request(sample: Sample) -> RewardRequest:
    """Project a Sample onto the row-aligned fields consumed by reward backends."""
    frontier = sample.parts[-1]
    primitives: Dict[str, PrimitiveValue] = {}
    for primitive in sample.conditioning():
        primitives[primitive_modality_key(primitive)] = primitive

    generated = dict(frontier.primitives)
    audio_sample_rate: Optional[int] = None
    audio_metadata = frontier.primitive_metadata.get("audio", {})
    if "audio" in generated and audio_metadata.get("sample_rate") is not None:
        audio_sample_rate = int(audio_metadata["sample_rate"])

    response_lengths: Optional[list[int]] = None
    max_new_tokens: Optional[int] = None
    sampling = frontier.sampling_params
    if isinstance(sampling, ARSamplingParams) and frontier.segment is not None:
        lengths = getattr(frontier.segment, "lengths", None)
        if lengths is not None and lengths.numel() == frontier.batch_size:
            response_lengths = [int(value) for value in lengths.tolist()]
            max_new_tokens = int(sampling.max_new_tokens)

    metadata = sample.root_metadata(-1)
    return RewardRequest(
        primitives=primitives,
        generated=generated,
        metadata=metadata if any(value is not None for value in metadata) else None,
        prompt_ids=[str(sample_id) for sample_id in frontier.sample_ids],
        sample_ids=list(frontier.sample_ids),
        group_ids=list(frontier.group_ids),
        response_lengths=response_lengths,
        audio_sample_rate=audio_sample_rate,
        max_new_tokens=max_new_tokens,
    )


class RewardClient:
    """Driver-side reward API preserving the historical score-and-attach surface."""

    def __init__(self, handle: "Handle") -> None:
        self._handle = handle

    @property
    def dp_size(self) -> int:
        return self._handle.dp_size

    def score_and_attach(self, sample: Sample) -> Sample:
        """Score the frontier and return a copy sharing every non-reward field."""
        if not sample.parts:
            raise ValueError("RewardClient.score_and_attach: Sample has no Parts.")
        frontier = sample.parts[-1]
        if frontier.rewards is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on the frontier Part.")
        if not frontier.primitives:
            raise ValueError("RewardClient.score_and_attach: frontier Part has no generated primitives to score.")

        response = self._handle.score(_build_reward_request(sample))
        rewards = torch.tensor(response.rewards, dtype=torch.float32)
        component_rewards = {
            str(name): torch.tensor(list(values or []), dtype=torch.float32)
            for name, values in dict(response.component_rewards or {}).items()
        }
        scored = _part_with_field(frontier, "rewards", rewards)
        scored = _part_with_field(scored, "component_rewards", component_rewards)
        return sample.with_parts([*sample.parts[:-1], scored])

    def score_differentiable(self, *args: Any, **kwargs: Any) -> Any:
        return self._handle.score_differentiable(*args, **kwargs)

    def get_memory_stats(self, *args: Any, **kwargs: Any) -> Any:
        return self._handle.get_memory_stats(*args, **kwargs)


__all__ = ["RewardClient"]
