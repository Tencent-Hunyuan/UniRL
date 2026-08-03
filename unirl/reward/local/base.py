"""Shared helpers for built-in local reward scorers."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import List, Optional

import torch

from unirl.reward.base import RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse


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
        **model_kwargs,
    ) -> None:
        resolved_model_name = self._resolve_model_name(model_name)
        super().__init__(
            model_name=resolved_model_name or "",
            batch_size=batch_size,
            timeout=timeout,
        )
        self.device = device
        self.dtype = dtype
        self.model_kwargs = dict(model_kwargs)
        self.model = None
        self.processor = None
        self._is_loaded = False

        self._load_model()
        self._is_loaded = True

    @classmethod
    def _resolve_model_name(cls, model_name: Optional[str]) -> str:
        raw_name = str(model_name or "").strip().lower()
        expected_name = str(cls.canonical_model_name or "").strip().lower()
        if expected_name:
            if raw_name and raw_name != expected_name:
                raise ValueError(f"{cls.__name__} only supports model_name={expected_name!r}, got {raw_name!r}.")
            return expected_name
        return raw_name

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        if not self._is_loaded:
            raise RuntimeError(
                f"{type(self).__name__}.compute_rewards called before _load_model "
                f"completed (model_name={self.model_name!r}, batch_size={request.batch_size})."
            )
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

    def is_available(self) -> bool:
        return self._is_loaded

    def offload(self) -> None:
        if self.model is not None and hasattr(self.model, "cpu"):
            self.model = self.model.cpu()
            torch.cuda.empty_cache()

    def onload(self) -> None:
        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
