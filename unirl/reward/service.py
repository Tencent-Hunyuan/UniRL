"""Reward service: score a response :class:`~unirl.types.sample.Sample` in place."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.primitives import primitive_modality_key
from unirl.types.reward import RewardRequest, RewardResponse
from unirl.types.sample import Primitive, Sample, _part_with_field
from unirl.types.sampling import ARSamplingParams

from .base import DifferentiableReward, RewardBackend

logger = logging.getLogger(__name__)


def _build_reward_request(sample: Sample, preferred_input_kind: str) -> RewardRequest:
    """Assemble a :class:`RewardRequest` from a response ``Sample``."""
    frontier = sample.parts[-1]
    primitives: Dict[str, Primitive] = {}
    for prim in sample.conditioning():
        primitives[primitive_modality_key(prim)] = prim

    if preferred_input_kind not in frontier.primitives:
        raise ValueError(
            f"Reward backend consumes {preferred_input_kind!r} but the frontier Part generated "
            f"{sorted(frontier.primitives)!r}; check the recipe's reward/model pairing."
        )

    metadata = sample.root_metadata(-1)
    generated = dict(frontier.primitives)
    audio_sample_rate: Optional[int] = None
    audio_metadata = frontier.primitive_metadata.get("audio", {})
    if "audio" in generated and audio_metadata.get("sample_rate") is not None:
        audio_sample_rate = int(audio_metadata["sample_rate"])
    return RewardRequest(
        primitives=primitives,
        generated=generated,
        audio_sample_rate=audio_sample_rate,
        prompt_ids=[str(sid) for sid in frontier.sample_ids],
        sample_ids=list(frontier.sample_ids),
        group_ids=list(frontier.group_ids),
        metadata=(metadata if any(m is not None for m in metadata) else None),
    )


class RewardService(Remote):
    """Actor-side reward entry: one backend, scores a Sample's frontier Part in place."""

    def __init__(
        self,
        backend: RewardBackend,
        truncated_reward: str = "zero",
        overlong_buffer_len: int = 4096,
        overlong_penalty_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.truncated_reward = str(truncated_reward)
        self.overlong_buffer_len = int(overlong_buffer_len)
        self.overlong_penalty_factor = float(overlong_penalty_factor)
        if self.truncated_reward not in ("zero", "keep", "soft"):
            raise ValueError(f"truncated_reward must be zero|keep|soft, got {self.truncated_reward!r}")
        logger.info(
            "RewardService initialized with backend=%s, truncated_reward=%s",
            backend.get_model_name() or type(backend).__name__,
            self.truncated_reward,
        )

    @property
    def preferred_input_kind(self) -> str:
        """The decoded media kind the backend consumes (image/video/text)."""
        kind = str(getattr(self.backend, "preferred_input_kind", "") or "").strip().lower()
        if kind not in {"image", "video", "text"}:
            raise ValueError(
                f"Reward backend must expose preferred_input_kind as 'image', 'video', or 'text'. Got {kind!r}."
            )
        return kind

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        return self.backend.compute_rewards(request)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score_differentiable(
        self,
        media_tensor: torch.Tensor,
        prompts: List[str],
        records: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """ReFL scoring of grad-carrying ``media_tensor`` against ``prompts``; returns ``[B]`` with grad_fn intact."""
        if not isinstance(self.backend, DifferentiableReward):
            raise TypeError(
                f"RewardService.score_differentiable: backend "
                f"{type(self.backend).__name__} is not a DifferentiableReward — ReFL "
                f"needs a differentiable in-process reward (e.g. pickscore/clip/hpsv2)."
            )
        return self.backend.compute_rewards_differentiable(media_tensor, list(prompts), records=records)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score_and_attach(self, sample: Sample) -> Sample:
        """Score the frontier (last) Part's generated media and return the updated Sample."""
        frontier = sample.parts[-1]
        if frontier.rewards is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on the frontier Part.")
        if not frontier.primitives:
            raise ValueError("RewardService.score_and_attach: frontier Part has no generated primitives to score.")

        request = _build_reward_request(sample, self.preferred_input_kind)
        reward_response = self.compute_rewards(request)

        failed = [(i, e) for i, (ok, e) in enumerate(zip(reward_response.successes, reward_response.errors)) if not ok]
        if failed:
            raise RuntimeError(
                f"Reward computation flagged {len(failed)} of {len(reward_response.successes)} "
                f"sample(s) as failure. First few: {failed[:3]}"
            )

        rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)

        sp = frontier.sampling_params
        if self.truncated_reward != "keep" and isinstance(sp, ARSamplingParams) and frontier.segment is not None:
            seg_lengths = getattr(frontier.segment, "lengths", None)
            if seg_lengths is not None and seg_lengths.numel() == rewards.numel():
                seg_lengths = seg_lengths.to(rewards.device).float()
                max_len = float(int(sp.max_new_tokens))
                if self.truncated_reward == "zero":
                    truncated = seg_lengths >= max_len
                    rewards = torch.where(truncated, torch.zeros_like(rewards), rewards)
                else:
                    buf = float(self.overlong_buffer_len)
                    exceed = seg_lengths - (max_len - buf)
                    penalty = torch.clamp(-exceed / buf * self.overlong_penalty_factor, max=0.0)
                    rewards = rewards + penalty

        component_rewards = {
            str(name): torch.tensor(list(values or []), dtype=torch.float32)
            for name, values in dict(reward_response.component_rewards or {}).items()
        }
        scored = _part_with_field(frontier, "rewards", rewards)
        scored = _part_with_field(scored, "component_rewards", component_rewards)
        return sample.with_parts([*sample.parts[:-1], scored])

    def is_available(self) -> bool:
        return self.backend.is_available()

    def offload(self) -> None:
        self.backend.offload()

    def onload(self) -> None:
        self.backend.onload()

    def dispose(self) -> None:
        self.backend.dispose()


__all__ = [
    "RewardService",
]
