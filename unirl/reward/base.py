"""Base abstractions for reward backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from unirl.types.reward import RewardRequest, RewardResponse

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
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout = timeout

    def get_model_name(self) -> str:
        """Name of the reward model/component this backend serves."""
        return self.model_name

    @property
    def preferred_input_kind(self) -> str:
        """The decoded media kind this backend consumes (image/video/text)."""
        return str(getattr(self, "input_kind", "image") or "image").strip().lower()

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


__all__ = [
    "BaseRewardComponentSpec",
    "DifferentiableReward",
    "RewardBackend",
]
