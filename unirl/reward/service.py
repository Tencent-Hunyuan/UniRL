"""Worker-side reward service: score a row-aligned request."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.reward import RewardRequest, RewardResponse

from .base import DifferentiableReward, RewardBackend

logger = logging.getLogger(__name__)


class RewardService(Remote):
    """Actor-side reward entry owning one backend and its shaping policy."""

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
    def score(self, request: RewardRequest) -> RewardResponse:
        """Score one DP shard and return row-aligned scalar results."""
        input_kind = self.preferred_input_kind
        if input_kind not in request.generated:
            raise ValueError(
                f"Reward backend consumes {input_kind!r} but the request generated "
                f"{sorted(request.generated)!r}; check the recipe's reward/model pairing."
            )
        reward_response = self.compute_rewards(request)

        failed = [(i, e) for i, (ok, e) in enumerate(zip(reward_response.successes, reward_response.errors)) if not ok]
        if failed:
            raise RuntimeError(
                f"Reward computation flagged {len(failed)} of {len(reward_response.successes)} "
                f"sample(s) as failure. First few: {failed[:3]}"
            )

        if len(reward_response.rewards) != request.batch_size:
            raise RuntimeError(
                f"Reward backend returned {len(reward_response.rewards)} rewards for {request.batch_size} request rows."
            )
        rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)
        if self.truncated_reward != "keep" and request.response_lengths is not None:
            if request.max_new_tokens is None or len(request.response_lengths) != rewards.numel():
                raise ValueError(
                    "RewardService.score requires response_lengths and max_new_tokens aligned with reward rows."
                )
            lengths = torch.tensor(request.response_lengths, dtype=torch.float32)
            max_len = float(request.max_new_tokens)
            if self.truncated_reward == "zero":
                rewards = torch.where(lengths >= max_len, torch.zeros_like(rewards), rewards)
            else:
                buf = float(self.overlong_buffer_len)
                exceed = lengths - (max_len - buf)
                penalty = torch.clamp(-exceed / buf * self.overlong_penalty_factor, max=0.0)
                rewards = rewards + penalty

        return replace(reward_response, rewards=[float(value) for value in rewards.tolist()])

    def is_available(self) -> bool:
        return self.backend.is_available()

    def offload(self) -> None:
        self.backend.offload()

    def onload(self) -> None:
        self.backend.onload()

    def dispose(self) -> None:
        self.backend.dispose()

    def shutdown(self) -> None:
        """Worker teardown hook: release backend sessions and managed children."""
        self.dispose()


__all__ = [
    "RewardService",
]
