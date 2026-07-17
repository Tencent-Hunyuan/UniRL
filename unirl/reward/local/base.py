"""Shared helpers for built-in local reward scorers."""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import torch

from unirl.reward.base import RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

logger = logging.getLogger(__name__)


def _model_tensor_stats(model: torch.nn.Module, *, device_type: Optional[str] = None) -> Tuple[int, int]:
    """Return logical bytes/count for unique registered parameters and buffers."""
    seen: set[int] = set()
    tensor_bytes = 0
    tensor_count = 0
    for tensor in (*model.parameters(), *model.buffers()):
        if id(tensor) in seen or tensor.is_meta:
            continue
        seen.add(id(tensor))
        if device_type is not None and tensor.device.type != device_type:
            continue
        tensor_bytes += tensor.numel() * tensor.element_size()
        tensor_count += 1
    return tensor_bytes, tensor_count


def _synchronize_cuda(device: str) -> None:
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


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

    # Differentiable scoring (ReFL) is an OPTIONAL capability, not part of this
    # base: scorers that wrap a differentiable nn.Module define
    # ``compute_rewards_differentiable`` and thereby satisfy the
    # ``unirl.reward.base.DifferentiableReward`` Protocol (e.g. PickScore).

    def is_available(self) -> bool:
        return self._is_loaded

    @property
    def supports_model_parking(self) -> bool:
        return bool(
            isinstance(self.model, torch.nn.Module)
            and hasattr(self.model, "cpu")
            and hasattr(self.model, "to")
            and torch.device(self.device).type == "cuda"
        )

    def offload(self) -> Dict[str, float]:
        """Move only registered reward-model tensors to CPU and reclaim its cache."""
        if self.model is None or not hasattr(self.model, "cpu"):
            return {
                "reward_model_tensors_parked": 0.0,
                "reward_model_bytes_parked": 0.0,
                "reward_model_park_host_time_s": 0.0,
            }

        tensor_bytes, tensor_count = (
            _model_tensor_stats(self.model, device_type="cuda") if isinstance(self.model, torch.nn.Module) else (0, 0)
        )
        started = time.perf_counter()
        if torch.device(self.device).type == "cuda":
            _synchronize_cuda(self.device)
        self.model = self.model.cpu()
        if torch.device(self.device).type == "cuda":
            _synchronize_cuda(self.device)
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - started
        report = {
            "reward_model_tensors_parked": float(tensor_count),
            "reward_model_bytes_parked": float(tensor_bytes),
            "reward_model_park_host_time_s": elapsed,
        }
        logger.info(
            "Reward model park: backend=%s tensors=%d bytes=%d host_time_s=%.6f",
            type(self).__name__,
            tensor_count,
            tensor_bytes,
            elapsed,
        )
        return report

    def onload(self) -> Dict[str, float]:
        """Restore only registered reward-model tensors to the configured device."""
        if self.model is None or not hasattr(self.model, "to"):
            return {
                "reward_model_tensors_restored": 0.0,
                "reward_model_bytes_restored": 0.0,
                "reward_model_restore_host_time_s": 0.0,
            }

        target_type = torch.device(self.device).type
        started = time.perf_counter()
        self.model = self.model.to(self.device)
        _synchronize_cuda(self.device)
        elapsed = time.perf_counter() - started
        tensor_bytes, tensor_count = (
            _model_tensor_stats(self.model, device_type=target_type)
            if isinstance(self.model, torch.nn.Module)
            else (0, 0)
        )
        report = {
            "reward_model_tensors_restored": float(tensor_count),
            "reward_model_bytes_restored": float(tensor_bytes),
            "reward_model_restore_host_time_s": elapsed,
        }
        logger.info(
            "Reward model restore: backend=%s tensors=%d bytes=%d host_time_s=%.6f",
            type(self).__name__,
            tensor_count,
            tensor_bytes,
            elapsed,
        )
        return report
