"""Reward service: score a response :class:`~unirl.types.sample.Sample` in place."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.primitives import PrimitiveValue, Texts, primitive_modality_key
from unirl.types.reward import RewardRequest
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams

from .base import DifferentiableReward, PromptVideoReward, RewardBackend

logger = logging.getLogger(__name__)


class RewardService(Remote):
    """Actor-side reward entry: one backend, scores a Sample's frontier Part in place."""

    def __init__(
        self,
        backend: RewardBackend,
        truncated_reward: str = "zero",
        overlong_buffer_len: int = 4096,
        overlong_penalty_factor: float = 1.0,
        require_prompt_video: bool = True,
    ) -> None:
        super().__init__()
        if require_prompt_video and isinstance(backend, PromptVideoReward) and not backend.covers_prompt_video():
            raise ValueError(
                f"RewardService: backend {type(backend).__name__} gives prompt-to-video alignment no positive "
                "weight, so a policy trained on it can raise the score without following the prompt "
                "(imagebind mode='audio_video' collapses both streams together). Use imagebind "
                "mode='text_video' or 'all', or a T2AVCompositeScorer with a prompt-video term; score it "
                "under eval_rewards to measure it without training on it; or set "
                "require_prompt_video: false to train on it anyway."
            )
        self.backend = backend
        kind = backend.preferred_input_kind
        if kind not in {"image", "video", "text"}:
            raise ValueError(
                f"Reward backend must expose preferred_input_kind as 'image', 'video', or 'text'. Got {kind!r}."
            )
        self.preferred_input_kind = kind
        self.truncated_reward = str(truncated_reward)
        self.overlong_buffer_len = int(overlong_buffer_len)
        self.overlong_penalty_factor = float(overlong_penalty_factor)
        if self.truncated_reward not in ("zero", "keep", "soft"):
            raise ValueError(f"truncated_reward must be zero|keep|soft, got {self.truncated_reward!r}")
        if self.truncated_reward == "soft" and self.overlong_buffer_len <= 0:
            raise ValueError(
                f"overlong_buffer_len must be > 0 when truncated_reward='soft', got {self.overlong_buffer_len}"
            )
        logger.info(
            "RewardService initialized with backend=%s, truncated_reward=%s",
            backend.model_name or type(backend).__name__,
            self.truncated_reward,
        )

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
        return self.backend.compute_rewards_differentiable(media_tensor, prompts, records=records)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score_and_attach(self, sample: Sample) -> Sample:
        """Score the frontier (last) Part's generated media and return the updated Sample."""
        frontier = sample.parts[-1]
        if frontier.rewards is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on the frontier Part.")
        if not frontier.primitives:
            raise ValueError("RewardService.score_and_attach: frontier Part has no generated primitives to score.")

        original_prompt: Optional[Texts] = None
        generation_prompt: Optional[Texts] = None
        conditioning: Dict[str, PrimitiveValue] = {}
        for turn in sample.turns():
            prim = turn.content
            if isinstance(prim, Texts):
                if original_prompt is None:
                    original_prompt = prim
                generation_prompt = prim
            else:
                conditioning[primitive_modality_key(prim)] = prim

        if self.preferred_input_kind not in frontier.primitives:
            raise ValueError(
                f"Reward backend consumes {self.preferred_input_kind!r} but the frontier Part generated "
                f"{sorted(frontier.primitives)!r}; check the recipe's reward/model pairing."
            )

        metadata = sample.root_metadata(-1)
        generated = dict(frontier.primitives)
        audio_sample_rate: Optional[int] = None
        audio_metadata = frontier.primitive_metadata.get("audio", {})
        if "audio" in generated and audio_metadata.get("sample_rate") is not None:
            audio_sample_rate = int(audio_metadata["sample_rate"])
        request = RewardRequest(
            generated=generated,
            conditioning=conditioning,
            original_prompt=original_prompt,
            generation_prompt=generation_prompt,
            audio_sample_rate=audio_sample_rate,
            sample_ids=list(frontier.sample_ids),
            group_ids=list(frontier.group_ids),
            metadata=(metadata if any(m is not None for m in metadata) else None),
        )
        reward_response = self.backend.compute_rewards(request)

        failed = [(i, e) for i, (ok, e) in enumerate(zip(reward_response.successes, reward_response.errors)) if not ok]
        if failed:
            raise RuntimeError(
                f"Reward computation flagged {len(failed)} of {len(reward_response.successes)} "
                f"sample(s) as failure. First few: {failed[:3]}"
            )

        rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)

        # Hitting max_new_tokens means truncated text, but fixed-length AR media
        # reaches that length by construction and must keep its reward.
        sampling_params = frontier.sampling_params
        if (
            self.truncated_reward != "keep"
            and isinstance(sampling_params, ARSamplingParams)
            and not sampling_params.emits_fixed_length
            and frontier.segment is not None
        ):
            seg_lengths = frontier.segment.lengths
            if seg_lengths is not None and seg_lengths.numel() == rewards.numel():
                seg_lengths = seg_lengths.to(rewards.device).float()
                max_len = sampling_params.max_new_tokens
                if self.truncated_reward == "zero":
                    truncated = seg_lengths >= max_len
                    rewards = torch.where(truncated, torch.zeros_like(rewards), rewards)
                else:
                    buf = self.overlong_buffer_len
                    exceed = seg_lengths - (max_len - buf)
                    penalty = torch.clamp(-exceed / buf * self.overlong_penalty_factor, max=0.0)
                    rewards = rewards + penalty

        component_rewards = {
            name: torch.tensor(values, dtype=torch.float32)
            for name, values in reward_response.component_rewards.items()
        }
        scored = replace(frontier, rewards=rewards, component_rewards=component_rewards)
        return sample.with_parts([*sample.parts[:-1], scored])

    # Broadcast, not scatter: every reward worker holds its own copy of the
    # scorer, so each has to move its own weights. Undecorated, these were
    # unreachable through the role's Handle, which is why nothing had ever
    # driven reward residency from the training loop.
    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def offload(self) -> None:
        self.backend.offload()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload(self) -> None:
        self.backend.onload()

    def shutdown(self) -> None:
        """Worker teardown hook: release backend sessions and managed children."""
        self.backend.dispose()


__all__ = [
    "RewardService",
]
