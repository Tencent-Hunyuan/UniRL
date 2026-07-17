from __future__ import annotations

from typing import List

import torch

from unirl.reward.local import base as local_base
from unirl.reward.local.base import LocalRewardBackend
from unirl.types.reward import RewardRequest


class _MovableModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.location = "cuda"
        self.moves: list[str] = []

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())

    def cpu(self):
        self.moves.append("cpu")
        self.location = "cpu"
        return self

    def to(self, device):
        self.moves.append(str(device))
        self.location = torch.device(device).type
        return self


class _LocalBackend(LocalRewardBackend):
    def _load_model(self) -> None:
        self.model = _MovableModel()

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        return [0.0] * request.batch_size


def test_model_tensor_stats_counts_unique_parameters_and_buffers_exactly() -> None:
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(5, dtype=torch.float32))
    model.weight_alias = model.weight
    model.register_buffer("scale", torch.zeros(3, dtype=torch.float64))

    tensor_bytes, tensor_count = local_base._model_tensor_stats(model, device_type="cpu")

    assert tensor_count == 2
    assert tensor_bytes == 5 * 4 + 3 * 8


def test_local_backend_reports_exact_model_tensor_transfer_metrics(monkeypatch) -> None:
    backend = _LocalBackend(device="cuda")
    empty_cache_calls: list[None] = []

    def stats(model, *, device_type=None):
        return (3 * 1024, 3) if model.location == device_type else (0, 0)

    monkeypatch.setattr(local_base, "_model_tensor_stats", stats)
    monkeypatch.setattr(local_base, "_synchronize_cuda", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_cache_calls.append(None))

    assert backend.supports_model_parking is True
    parked = backend.offload()
    restored = backend.onload()

    assert backend.model.moves == ["cpu", "cuda"]
    assert len(empty_cache_calls) == 1
    assert parked["reward_model_tensors_parked"] == 3.0
    assert parked["reward_model_bytes_parked"] == 3 * 1024
    assert parked["reward_model_park_host_time_s"] >= 0.0
    assert restored["reward_model_tensors_restored"] == 3.0
    assert restored["reward_model_bytes_restored"] == 3 * 1024
    assert restored["reward_model_restore_host_time_s"] >= 0.0
