from __future__ import annotations

import sys
import types
import weakref
from contextlib import contextmanager

import pytest

from unirl.rollout.engine.vllm_omni.patches.compat_moe_workspace import (
    MOE_WORKSPACE_TAG,
    _patch_workspace_manager_class,
    _workspace_pool_key,
)


class _FakePool:
    pass


class _FakeAllocator:
    def __init__(self) -> None:
        self.allocator_and_pools: dict[str, tuple[_FakePool, object]] = {}
        self.current_tag = "default"
        self.new_pool_entries = 0
        self.entry_checks: list[bool] = []
        self.on_pool_entry = lambda: True

    @contextmanager
    def use_memory_pool(self, tag: str):
        self.new_pool_entries += 1
        self.entry_checks.append(self.on_pool_entry())
        old_tag = self.current_tag
        self.current_tag = tag
        pool = _FakePool()
        self.allocator_and_pools[tag] = (pool, object())
        try:
            yield
        finally:
            self.current_tag = old_tag


def _install_fake_torch(monkeypatch, synchronizations: list[None]) -> None:
    torch_module = types.ModuleType("torch")
    torch_module.accelerator = types.SimpleNamespace(
        synchronize=lambda: synchronizations.append(None),
        empty_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)


def test_workspace_growth_replaces_only_its_ubatch_pool(monkeypatch) -> None:
    allocator = _FakeAllocator()
    synchronizations: list[None] = []
    _install_fake_torch(monkeypatch, synchronizations)
    current_ubatch = [0]
    allocation_tags: list[str] = []

    class FakeWorkspace:
        def __init__(self, size: int) -> None:
            self.size = size

    class FakeWorkspaceManager:
        def __init__(self) -> None:
            self._current_workspaces: list[FakeWorkspace | None] = [None, None]
            self._locked = False

        @staticmethod
        def _workspace_size_bytes(workspace: FakeWorkspace | None) -> int:
            return 0 if workspace is None else workspace.size

        def _ensure_workspace_size(self, required_bytes: int) -> FakeWorkspace:
            ubatch_id = current_ubatch[0]
            workspace = self._current_workspaces[ubatch_id]
            if workspace is None or workspace.size < required_bytes:
                allocation_tags.append(allocator.current_tag)
                workspace = FakeWorkspace(required_bytes)
                self._current_workspaces[ubatch_id] = workspace
            return workspace

    assert _patch_workspace_manager_class(
        FakeWorkspaceManager,
        allocator_provider=lambda: allocator,
        ubatch_provider=lambda: current_ubatch[0],
    )
    manager = FakeWorkspaceManager()

    current_ubatch[0] = 0
    manager._ensure_workspace_size(1)
    pool0 = allocator.allocator_and_pools[_workspace_pool_key(0)]
    current_ubatch[0] = 1
    manager._ensure_workspace_size(1)
    pool1 = allocator.allocator_and_pools[_workspace_pool_key(1)]

    old_workspace = weakref.ref(manager._current_workspaces[0])
    allocator.on_pool_entry = lambda: old_workspace() is None
    current_ubatch[0] = 0
    manager._ensure_workspace_size(2)

    assert old_workspace() is None
    assert allocator.entry_checks == [True, True, True]
    assert allocator.allocator_and_pools[_workspace_pool_key(0)] is not pool0
    assert allocator.allocator_and_pools[_workspace_pool_key(1)] is pool1
    assert allocation_tags == [MOE_WORKSPACE_TAG] * 3
    assert len(synchronizations) == 1
    assert allocator.current_tag == "default"
    assert not _patch_workspace_manager_class(
        FakeWorkspaceManager,
        allocator_provider=lambda: allocator,
        ubatch_provider=lambda: current_ubatch[0],
    )


def test_workspace_without_allocator_uses_original_path() -> None:
    calls: list[int] = []

    class FakeWorkspaceManager:
        def __init__(self) -> None:
            self._current_workspaces = [0]

        @staticmethod
        def _workspace_size_bytes(workspace: int) -> int:
            return workspace

        def _ensure_workspace_size(self, required_bytes: int) -> int:
            calls.append(required_bytes)
            self._current_workspaces[0] = required_bytes
            return required_bytes

    assert _patch_workspace_manager_class(
        FakeWorkspaceManager,
        allocator_provider=lambda: None,
        ubatch_provider=lambda: 0,
    )

    assert FakeWorkspaceManager()._ensure_workspace_size(8) == 8
    assert calls == [8]


def test_locked_growth_preserves_the_existing_workspace(monkeypatch) -> None:
    allocator = _FakeAllocator()
    synchronizations: list[None] = []
    _install_fake_torch(monkeypatch, synchronizations)

    class FakeWorkspace:
        def __init__(self, size: int) -> None:
            self.size = size

    class FakeWorkspaceManager:
        def __init__(self) -> None:
            self._current_workspaces = [FakeWorkspace(1)]
            self._locked = True

        @staticmethod
        def _workspace_size_bytes(workspace: FakeWorkspace) -> int:
            return workspace.size

        def _ensure_workspace_size(self, required_bytes: int) -> FakeWorkspace:
            if self._locked and self._current_workspaces[0].size < required_bytes:
                raise AssertionError("locked")
            return self._current_workspaces[0]

    assert _patch_workspace_manager_class(
        FakeWorkspaceManager,
        allocator_provider=lambda: allocator,
        ubatch_provider=lambda: 0,
    )
    manager = FakeWorkspaceManager()
    workspace = manager._current_workspaces[0]

    with pytest.raises(AssertionError, match="locked"):
        manager._ensure_workspace_size(2)

    assert manager._current_workspaces[0] is workspace
    assert allocator.new_pool_entries == 0
    assert synchronizations == []


def test_private_api_drift_falls_back_to_the_original_method() -> None:
    calls: list[int] = []

    class DriftedWorkspaceManager:
        def _ensure_workspace_size(self, required_bytes: int) -> int:
            calls.append(required_bytes)
            return required_bytes

    assert _patch_workspace_manager_class(
        DriftedWorkspaceManager,
        allocator_provider=lambda: object(),
        ubatch_provider=lambda: 0,
    )

    assert DriftedWorkspaceManager()._ensure_workspace_size(16) == 16
    assert calls == [16]
