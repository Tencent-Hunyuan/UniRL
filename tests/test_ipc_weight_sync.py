from __future__ import annotations

from types import SimpleNamespace

import pytest

from unirl.distributed.weight_sync.full.ipc import IPCWeightSync


class _Backend:
    model = SimpleNamespace()
    rollout_adapter_name = "default"

    @staticmethod
    def expert_weight_export_transform():
        return None


class _Rollout:
    def __init__(self, weight_update_timeout_s: float) -> None:
        self.cfg = SimpleNamespace(timeout_for=lambda _command: weight_update_timeout_s)

    @staticmethod
    def component_name() -> str:
        return "vllm"


def test_control_timeout_must_outlive_weight_update() -> None:
    with pytest.raises(ValueError, match="control_timeout_s must exceed"):
        IPCWeightSync(
            backend=_Backend(),
            rollout=_Rollout(weight_update_timeout_s=30),
            control_timeout_s=30,
        )


def test_control_timeout_accepts_larger_budget() -> None:
    sync = IPCWeightSync(
        backend=_Backend(),
        rollout=_Rollout(weight_update_timeout_s=30),
        control_timeout_s=31,
    )

    assert sync._control_timeout.total_seconds() == 31
