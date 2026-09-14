from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType

import pytest

from unirl.rollout.engine.vllm_omni.patches.compat_moe_workspace import patch_moe_workspace_pool


class _Allocator:
    def __init__(self) -> None:
        self.current_tag = None
        self.allocator_and_pools: dict[str, tuple[object, object]] = {}
        self.entered_pool_keys: list[str] = []

    @contextmanager
    def use_memory_pool(self, tag: str):
        self.entered_pool_keys.append(tag)
        self.allocator_and_pools[tag] = (object(), object())
        yield


def _install_fake_vllm(
    monkeypatch: pytest.MonkeyPatch,
    workspace_manager: type,
    allocator: _Allocator,
    ubatch_id: ContextVar[int],
    lane: ContextVar[int],
) -> None:
    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.device_allocator": ModuleType("vllm.device_allocator"),
        "vllm.device_allocator.cumem": ModuleType("vllm.device_allocator.cumem"),
        "vllm.v1": ModuleType("vllm.v1"),
        "vllm.v1.worker": ModuleType("vllm.v1.worker"),
        "vllm.v1.worker.ubatching": ModuleType("vllm.v1.worker.ubatching"),
        "vllm.v1.worker.workspace": ModuleType("vllm.v1.worker.workspace"),
    }

    class CuMemAllocator:
        instance = allocator

    modules["vllm.device_allocator.cumem"].CuMemAllocator = CuMemAllocator
    modules["vllm.v1.worker.ubatching"].dbo_current_ubatch_id = ubatch_id.get
    modules["vllm.v1.worker.workspace"].WorkspaceManager = workspace_manager
    modules["vllm.v1.worker.workspace"]._workspace_lane = lane
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_invalid_lane_does_not_release_adjacent_ubatch(monkeypatch: pytest.MonkeyPatch) -> None:
    ubatch_id = ContextVar("test_ubatch", default=0)
    lane = ContextVar("test_lane", default=0)
    other_workspace = object()
    other_pool = (object(), object())

    class WorkspaceManager:
        def __init__(self) -> None:
            self._num_lanes = 1
            self._current_workspaces = [None, other_workspace]

        def _workspace_size_bytes(self, workspace: object | None) -> int:
            return int(workspace is not None)

        def is_locked(self) -> bool:
            return False

        def _ensure_workspace_size(self, required_bytes: int):
            current_lane = lane.get()
            if current_lane >= self._num_lanes:
                raise RuntimeError(f"Workspace lane {current_lane} is not configured")
            return self._current_workspaces[ubatch_id.get() * self._num_lanes + current_lane]

    allocator = _Allocator()
    allocator.allocator_and_pools["moe_workspace:1"] = other_pool
    _install_fake_vllm(monkeypatch, WorkspaceManager, allocator, ubatch_id, lane)
    patch_moe_workspace_pool()

    manager = WorkspaceManager()
    token = lane.set(1)
    try:
        with pytest.raises(RuntimeError, match="lane 1 is not configured"):
            manager._ensure_workspace_size(2)
    finally:
        lane.reset(token)

    assert manager._current_workspaces[1] is other_workspace
    assert allocator.allocator_and_pools["moe_workspace:1"] is other_pool


def test_two_ubatches_by_two_lanes_use_distinct_pool_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    ubatch_id = ContextVar("test_ubatch", default=0)
    lane = ContextVar("test_lane", default=0)

    class WorkspaceManager:
        def __init__(self) -> None:
            self._num_lanes = 2
            self._current_workspaces = [None] * 4

        def _workspace_size_bytes(self, workspace: object | None) -> int:
            return int(workspace is not None)

        def is_locked(self) -> bool:
            return False

        def _ensure_workspace_size(self, required_bytes: int):
            workspace_id = ubatch_id.get() * self._num_lanes + lane.get()
            self._current_workspaces[workspace_id] = object()
            return self._current_workspaces[workspace_id]

    allocator = _Allocator()
    _install_fake_vllm(monkeypatch, WorkspaceManager, allocator, ubatch_id, lane)
    patch_moe_workspace_pool()

    manager = WorkspaceManager()
    for ubatch in range(2):
        for workspace_lane in range(2):
            ubatch_token = ubatch_id.set(ubatch)
            lane_token = lane.set(workspace_lane)
            try:
                manager._ensure_workspace_size(1)
            finally:
                lane.reset(lane_token)
                ubatch_id.reset(ubatch_token)

    assert allocator.entered_pool_keys == [
        "moe_workspace:0",
        "moe_workspace:1",
        "moe_workspace:2",
        "moe_workspace:3",
    ]
    assert all(workspace is not None for workspace in manager._current_workspaces)
