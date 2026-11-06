"""Route vLLM's reusable MoE workspace through CuMem pools so sleep can release it."""

from __future__ import annotations

import gc
from typing import Any

MOE_WORKSPACE_TAG = "moe_workspace"
_WORKSPACE_MANAGER_MARKER = "_unirl_moe_workspace_pool_installed"


def patch_moe_workspace_pool() -> None:
    """Enroll MoE workspace allocations in sleep-managed CuMem pools."""
    from vllm.device_allocator.cumem import CuMemAllocator
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id
    from vllm.v1.worker.workspace import WorkspaceManager, _workspace_lane

    if getattr(WorkspaceManager, _WORKSPACE_MANAGER_MARKER, False):
        return

    original_ensure_workspace_size = WorkspaceManager._ensure_workspace_size

    def ensure_workspace_size(manager: Any, required_bytes: int) -> Any:
        try:
            lane = _workspace_lane.get()
            num_lanes = manager._num_lanes
            if lane >= num_lanes:
                return original_ensure_workspace_size(manager, required_bytes)
            workspace_id = dbo_current_ubatch_id() * num_lanes + lane
            current_workspace = manager._current_workspaces[workspace_id]
            current_size = manager._workspace_size_bytes(current_workspace)
            workspace_locked = manager.is_locked()
        except (AttributeError, IndexError):
            return original_ensure_workspace_size(manager, required_bytes)
        if current_size >= required_bytes:
            return original_ensure_workspace_size(manager, required_bytes)

        allocator = CuMemAllocator.instance
        if allocator is None or workspace_locked:
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
        pool_key = f"{MOE_WORKSPACE_TAG}:{workspace_id}"
        pool_state = allocator.allocator_and_pools.pop(pool_key, None)
        if pool_state is not None:
            memory_pool, pluggable_allocator = pool_state
            del pool_state
            del memory_pool
            gc.collect()
            del pluggable_allocator
        if replacing_workspace:
            torch.accelerator.empty_cache()

        # The dictionary key is per-workspace so DBO and lane slots do not
        # replace one another, while AllocationData keeps one model-level tag.
        previous_tag = allocator.current_tag
        with allocator.use_memory_pool(tag=pool_key):
            allocator.current_tag = MOE_WORKSPACE_TAG
            try:
                return original_ensure_workspace_size(manager, required_bytes)
            finally:
                # vLLM 0.20's context does not restore current_tag on exceptions.
                allocator.current_tag = previous_tag

    WorkspaceManager._ensure_workspace_size = ensure_workspace_size
    setattr(WorkspaceManager, _WORKSPACE_MANAGER_MARKER, True)


__all__ = ["patch_moe_workspace_pool"]
