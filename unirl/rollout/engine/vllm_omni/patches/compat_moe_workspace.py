"""Route vLLM's reusable MoE workspace through CuMem pools so sleep can release it."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any, Callable, Iterator

MOE_WORKSPACE_TAG = "moe_workspace"
_WORKSPACE_MANAGER_MARKER = "_diffrl_moe_workspace_pool_installed"


def _workspace_pool_key(workspace_id: int) -> str:
    return f"{MOE_WORKSPACE_TAG}:{workspace_id}"


@contextmanager
def _workspace_pool_context(allocator: Any, workspace_id: int) -> Iterator[None]:
    """Create one replaceable pool per workspace while keeping one sleep tag."""
    # The dictionary key is per-workspace so DBO and lane slots do not replace
    # one another, while AllocationData keeps one model-level workspace tag.
    pool_key = _workspace_pool_key(workspace_id)
    previous_tag = allocator.current_tag
    with allocator.use_memory_pool(tag=pool_key):
        allocator.current_tag = MOE_WORKSPACE_TAG
        try:
            yield
        finally:
            # vLLM 0.20's context does not restore current_tag on exceptions.
            allocator.current_tag = previous_tag


def _release_workspace_pool(allocator: Any, workspace_id: int) -> None:
    """Release a retained MemPool before its pluggable allocator."""
    pool_state = allocator.allocator_and_pools.pop(_workspace_pool_key(workspace_id), None)
    if pool_state is None:
        return
    memory_pool, pluggable_allocator = pool_state
    del pool_state
    del memory_pool
    gc.collect()
    del pluggable_allocator


def _patch_workspace_manager_class(
    workspace_manager_class: type,
    *,
    allocator_provider: Callable[[], Any],
    workspace_id_provider: Callable[[Any], int],
) -> None:
    """Route ``_ensure_workspace_size`` growth through the dedicated CuMem tag."""
    if getattr(workspace_manager_class, _WORKSPACE_MANAGER_MARKER, False):
        return

    original_ensure_workspace_size = workspace_manager_class._ensure_workspace_size

    def ensure_workspace_size(manager: Any, required_bytes: int) -> Any:
        workspace_id = workspace_id_provider(manager)
        try:
            current_workspace = manager._current_workspaces[workspace_id]
            current_size = manager._workspace_size_bytes(current_workspace)
            workspace_locked = manager.is_locked()
        except AttributeError:
            return original_ensure_workspace_size(manager, required_bytes)
        if current_size >= required_bytes:
            return original_ensure_workspace_size(manager, required_bytes)

        allocator = allocator_provider()
        if allocator is None:
            return original_ensure_workspace_size(manager, required_bytes)

        # Preserve the native locked-workspace error without releasing the
        # current allocation first.
        if workspace_locked:
            return original_ensure_workspace_size(manager, required_bytes)

        import torch

        # ``use_memory_pool`` replaces the retained MemPool for a key. Release
        # this slot's old tensor before replacing its pool; reversing that
        # order aborts in MemPool::~MemPool on torch 2.11.
        replacing_workspace = current_workspace is not None
        if replacing_workspace:
            # vLLM 0.20's CuMem free callback can unmap immediately, so a
            # device-wide sync must drain every stream first.
            torch.accelerator.synchronize()
            manager._current_workspaces[workspace_id] = None
            del current_workspace
            gc.collect()
        _release_workspace_pool(allocator, workspace_id)
        if replacing_workspace:
            torch.accelerator.empty_cache()

        with _workspace_pool_context(allocator, workspace_id):
            return original_ensure_workspace_size(manager, required_bytes)

    workspace_manager_class._ensure_workspace_size = ensure_workspace_size
    setattr(workspace_manager_class, _WORKSPACE_MANAGER_MARKER, True)


def patch_moe_workspace_pool() -> None:
    """Enroll MoE workspace allocations in sleep-managed CuMem pools."""
    try:
        from vllm.device_allocator.cumem import CuMemAllocator
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id
        from vllm.v1.worker.workspace import WorkspaceManager

        WorkspaceManager._ensure_workspace_size
    except (ImportError, AttributeError):
        return

    try:
        from vllm.v1.worker.workspace import _workspace_lane
    except ImportError:
        _workspace_lane = None

    def workspace_id_provider(manager: Any) -> int:
        # vLLM 0.28 slots workspaces by lane within ubatch; these defaults
        # reproduce the flat per-ubatch layout that earlier versions used.
        lane = 0 if _workspace_lane is None else _workspace_lane.get()
        return dbo_current_ubatch_id() * getattr(manager, "_num_lanes", 1) + lane

    _patch_workspace_manager_class(
        WorkspaceManager,
        allocator_provider=lambda: CuMemAllocator.instance,
        workspace_id_provider=workspace_id_provider,
    )


__all__ = ["patch_moe_workspace_pool"]
