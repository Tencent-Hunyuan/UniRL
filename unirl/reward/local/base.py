"""Shared helpers for built-in local reward scorers."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import List, Optional

import torch

from unirl.reward.base import RewardBackend
from unirl.types.reward import DEFAULT_PROMPT_SOURCE, PromptSource, RewardRequest, RewardResponse


class LocalRewardBackend(RewardBackend):
    """Common lifecycle and error handling for local built-in scorers."""

    canonical_model_name: Optional[str] = None

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        batch_size: int = 8,
        timeout: float = 60.0,
        prompt_source: PromptSource = DEFAULT_PROMPT_SOURCE,
        **model_kwargs,
    ) -> None:
        raw_name = "" if model_name is None else str(model_name).strip().lower()
        expected_name = "" if self.canonical_model_name is None else str(self.canonical_model_name).strip().lower()
        if expected_name:
            if raw_name and raw_name != expected_name:
                raise ValueError(f"{type(self).__name__} only supports model_name={expected_name!r}, got {raw_name!r}.")
            resolved_model_name = expected_name
        else:
            resolved_model_name = raw_name
        super().__init__(
            model_name=resolved_model_name,
            batch_size=batch_size,
            timeout=timeout,
            prompt_source=prompt_source,
        )
        self.device = device
        self.dtype = dtype
        self.model_kwargs = dict(model_kwargs)
        self.model = None
        self.processor = None

        self._load_model()

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        rewards = self._compute_model_rewards(request)
        return RewardResponse(
            rewards=rewards,
            successes=[True] * len(rewards),
            errors=[None] * len(rewards),
            compute_time=time.time() - start,
        )

    @abstractmethod
    def _load_model(self) -> None:
        """Load scorer-specific model state."""

    @abstractmethod
    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        """Compute per-sample rewards."""

    def offload(self) -> None:
        if self.model is not None and hasattr(self.model, "cpu"):
            self.model = self.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def onload(self) -> None:
        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
