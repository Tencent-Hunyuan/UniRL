"""Base abstractions for reward backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from unirl.types.reward import DEFAULT_PROMPT_SOURCE, PROMPT_SOURCES, PromptSource, RewardRequest, RewardResponse

if TYPE_CHECKING:
    import torch


class RewardBackend(ABC):
    """Turn a :class:`RewardRequest` into a :class:`RewardResponse`."""

    input_kind = "image"

    def __init__(
        self,
        model_name: str = "",
        batch_size: int = 8,
        timeout: float = 60.0,
        prompt_source: PromptSource = DEFAULT_PROMPT_SOURCE,
        **kwargs,
    ) -> None:
        if prompt_source not in PROMPT_SOURCES:
            raise ValueError(f"RewardBackend.prompt_source must be 'original' or 'generation', got {prompt_source!r}.")
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout
        self.prompt_source = prompt_source

    def get_model_name(self) -> str:
        """Name of the reward model/component this backend serves."""
        return self.model_name

    @property
    def preferred_input_kind(self) -> str:
        """The decoded media kind this backend consumes (image/video/text)."""
        return str(getattr(self, "input_kind", "image") or "image").strip().lower()

    def prompts(self, request: RewardRequest) -> List[str]:
        """Return the prompt semantics declared by this backend."""
        prompt = request.original_prompt if self.prompt_source == "original" else request.generation_prompt
        if prompt is None:
            raise ValueError(
                f"{type(self).__name__} requires prompt_source={self.prompt_source!r}, "
                "but the selected prompt is absent from RewardRequest. "
                "Ensure the scored Sample has a text ancestor."
            )
        return list(prompt.texts)

    @abstractmethod
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Score the request."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend is ready to score."""

    def offload(self) -> None:
        """Optional lifecycle hook: release device memory."""

    def onload(self) -> None:
        """Optional lifecycle hook: reacquire device memory."""

    def dispose(self) -> None:
        """Optional lifecycle hook: terminal cleanup."""


@runtime_checkable
class PromptVideoReward(Protocol):
    """Optional capability for rewards that score prompt-to-video alignment."""

    def covers_prompt_video(self) -> bool:
        """Whether the configured reward gives prompt-to-video alignment positive weight."""
        ...


@runtime_checkable
class DifferentiableReward(Protocol):
    """Optional capability: in-process ``nn.Module`` rewards returning a grad-carrying score tensor for ReFL."""

    def compute_rewards_differentiable(
        self,
        media_tensor: "torch.Tensor",
        prompts: List[str],
        records: Optional[List[dict[str, object]]] = None,
    ) -> "torch.Tensor":
        """Score grad-carrying image ``[B,C,H,W]`` or video ``[B,C,T,H,W]`` media."""
        ...


class BaseRewardComponentSpec(ABC):
    """Marker base for every reward backend spec."""


@dataclass
class PromptRewardComponentSpec(BaseRewardComponentSpec):
    """Config capability for rewards that select a request prompt."""

    prompt_source: PromptSource = DEFAULT_PROMPT_SOURCE


__all__ = [
    "BaseRewardComponentSpec",
    "DifferentiableReward",
    "PromptRewardComponentSpec",
    "PromptVideoReward",
    "RewardBackend",
]
