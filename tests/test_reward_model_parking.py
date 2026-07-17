from __future__ import annotations

from typing import List

import pytest
import torch

from unirl.reward.base import RewardBackend
from unirl.reward.service import RewardService
from unirl.types.primitives import Images, Texts
from unirl.types.reward import RewardRequest, RewardResponse


class _ParkingBackend(RewardBackend):
    def __init__(self, *, supported: bool = True) -> None:
        super().__init__(model_name="test")
        self.supported = supported
        self.resident = True
        self.events: list[str] = []
        self.fail_score = False
        self.fail_park = False
        self.fail_restore = False
        self.response = RewardResponse(
            rewards=[0.25, 0.75],
            successes=[True, True],
            errors=[None, None],
            compute_time=1.0,
        )

    @property
    def supports_model_parking(self) -> bool:
        return self.supported

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        del request
        self.events.append("score")
        assert self.resident
        if self.fail_score:
            raise RuntimeError("injected score failure")
        return self.response

    def is_available(self) -> bool:
        return True

    def offload(self):
        self.events.append("park")
        if self.fail_park:
            raise RuntimeError("injected park failure")
        self.resident = False
        return {
            "reward_model_tensors_parked": 2.0,
            "reward_model_bytes_parked": 4096.0,
            "reward_model_park_host_time_s": 0.25,
        }

    def onload(self):
        self.events.append("restore")
        self.resident = True
        if self.fail_restore:
            raise RuntimeError("injected restore failure")
        return {
            "reward_model_tensors_restored": 2.0,
            "reward_model_bytes_restored": 4096.0,
            "reward_model_restore_host_time_s": 0.5,
        }

    def compute_rewards_differentiable(self, images_tensor, prompts: List[str], records=None):
        del prompts, records
        return images_tensor.mean(dim=(1, 2, 3))


def test_scalar_reward_parking_has_cpu_steady_state_and_preserves_response() -> None:
    backend = _ParkingBackend()
    service = RewardService(backend=backend, park_backend_between_calls=True)

    assert backend.events == ["park"]
    assert backend.resident is False

    response = service.compute_rewards(RewardRequest())
    assert response is backend.response
    assert response.rewards == [0.25, 0.75]
    assert backend.events == ["park", "restore", "score", "park"]
    assert backend.resident is False
    assert service._backend_parked is True
    assert service._backend_parking_metrics == {
        "reward_model_tensors_parked": 4.0,
        "reward_model_bytes_parked": 8192.0,
        "reward_model_park_host_time_s": 0.5,
        "reward_model_tensors_restored": 2.0,
        "reward_model_bytes_restored": 4096.0,
        "reward_model_restore_host_time_s": 0.5,
    }


def test_parking_is_default_off_and_preserves_existing_lifecycle() -> None:
    backend = _ParkingBackend(supported=False)
    service = RewardService(backend=backend)

    assert service.compute_rewards(RewardRequest()) is backend.response
    assert backend.events == ["score"]
    assert backend.resident is True


def test_score_failure_reparks_backend_and_allows_a_later_score() -> None:
    backend = _ParkingBackend()
    service = RewardService(backend=backend, park_backend_between_calls=True)
    backend.fail_score = True

    with pytest.raises(RuntimeError, match="injected score failure"):
        service.compute_rewards(RewardRequest())

    assert backend.resident is False
    assert service._backend_parked is True
    assert backend.events == ["park", "restore", "score", "park"]

    backend.fail_score = False
    assert service.compute_rewards(RewardRequest()) is backend.response
    assert backend.events[-3:] == ["restore", "score", "park"]
    assert backend.resident is False


def test_restore_failure_forces_repark_and_allows_a_later_score() -> None:
    backend = _ParkingBackend()
    service = RewardService(backend=backend, park_backend_between_calls=True)
    backend.fail_restore = True

    with pytest.raises(RuntimeError, match="injected restore failure"):
        service.compute_rewards(RewardRequest())

    assert backend.events == ["park", "restore", "park"]
    assert backend.resident is False
    assert service._backend_parked is True

    backend.fail_restore = False
    assert service.compute_rewards(RewardRequest()) is backend.response
    assert backend.events[-3:] == ["restore", "score", "park"]


def test_park_failure_after_successful_score_is_fatal() -> None:
    backend = _ParkingBackend()
    service = RewardService(backend=backend, park_backend_between_calls=True)
    backend.fail_park = True

    with pytest.raises(RuntimeError, match="failed to park its backend"):
        service.compute_rewards(RewardRequest())

    assert backend.events == ["park", "restore", "score", "park"]
    assert service._backend_parked is False


def test_cleanup_failure_does_not_mask_score_error_and_marks_state_unsafe() -> None:
    backend = _ParkingBackend()
    service = RewardService(backend=backend, park_backend_between_calls=True)
    backend.fail_score = True
    backend.fail_park = True

    with pytest.raises(RuntimeError, match="injected score failure"):
        service.compute_rewards(RewardRequest())

    assert service._backend_parked is False
    with pytest.raises(RuntimeError, match="lost its CPU-resident steady state"):
        service.compute_rewards(RewardRequest())


def test_automatic_parking_rejects_unsupported_backend() -> None:
    backend = _ParkingBackend(supported=False)
    with pytest.raises(ValueError, match="requires a CUDA reward backend"):
        RewardService(backend=backend, park_backend_between_calls=True)
    assert backend.events == []


def test_automatic_parking_rejects_differentiable_scoring_without_restoring() -> None:
    backend = _ParkingBackend()
    service = RewardService(backend=backend, park_backend_between_calls=True)
    images = Images(pixels=torch.zeros(1, 3, 2, 2))
    prompts = Texts(texts=["prompt"])

    with pytest.raises(RuntimeError, match="scalar-scoring only"):
        service.score_differentiable(images=images, prompts=prompts)

    assert backend.events == ["park"]
    assert backend.resident is False
