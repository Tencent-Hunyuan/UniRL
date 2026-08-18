"""Route vLLM's reusable MoE workspace through a CuMem pool so sleep() can offload it."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any, Callable, Iterator

MOE_WORKSPACE_TAG = "moe_workspace"
_WORKSPACE_MANAGER_MARKER = "_diffrl_moe_workspace_pool_installed"
_EXECUTOR_SLEEP_MARKER = "_diffrl_moe_workspace_sleep_tag"


def _workspace_pool_key(ubatch_id: int) -> str:
    return f"{MOE_WORKSPACE_TAG}:{ubatch_id}"


@contextmanager
def _workspace_memory_pool(allocator: Any, ubatch_id: int) -> Iterator[None]:
    """Create one replaceable pool per ubatch while keeping one sleep tag."""
    # The dictionary key is ubatch-specific so DBO slots do not replace one
    # another. Allocation callbacks still receive MOE_WORKSPACE_TAG so sleep
    # and tagged wake treat every slot as one logical resource.
    pool_key = _workspace_pool_key(ubatch_id)
    with allocator.use_memory_pool(tag=pool_key):
        pool_key_tag = allocator.current_tag
        allocator.current_tag = MOE_WORKSPACE_TAG
        try:
            yield
        finally:
            allocator.current_tag = pool_key_tag


def _patch_workspace_manager_class(
    workspace_manager_class: type,
    *,
    allocator_provider: Callable[[], Any],
    ubatch_provider: Callable[[], int],
) -> bool:
    """Route ``_ensure_workspace_size`` growth through the dedicated CuMem tag."""
    if getattr(workspace_manager_class, _WORKSPACE_MANAGER_MARKER, False):
        return False

    original_ensure_workspace_size = workspace_manager_class._ensure_workspace_size

    def ensure_workspace_size(manager: Any, required_bytes: int) -> Any:
        ubatch_id = int(ubatch_provider())
        try:
            current_workspace = manager._current_workspaces[ubatch_id]
            current_size = manager._workspace_size_bytes(current_workspace)
        except AttributeError:
            return original_ensure_workspace_size(manager, required_bytes)
        if current_size < required_bytes:
            allocator = allocator_provider()
            if allocator is None:
                return original_ensure_workspace_size(manager, required_bytes)
            if not all(hasattr(allocator, attr) for attr in ("allocator_and_pools", "current_tag", "use_memory_pool")):
                return original_ensure_workspace_size(manager, required_bytes)

            # Preserve the native locked-workspace error without releasing the
            # current allocation first.
            if getattr(manager, "_locked", False):
                return original_ensure_workspace_size(manager, required_bytes)

            import torch

            # ``use_memory_pool`` replaces the retained MemPool for a key.
            # Release this ubatch's old tensor before replacing its pool; doing
            # it in the opposite order aborts in MemPool::~MemPool on torch 2.11.
            if current_workspace is not None:
                torch.accelerator.synchronize()
            manager._current_workspaces[ubatch_id] = None
            del current_workspace
            old_pool_data = allocator.allocator_and_pools.pop(_workspace_pool_key(ubatch_id), None)
            del old_pool_data
            gc.collect()
            torch.accelerator.empty_cache()

            with _workspace_memory_pool(allocator, ubatch_id):
                return original_ensure_workspace_size(manager, required_bytes)
        del current_workspace
        return original_ensure_workspace_size(manager, required_bytes)

    workspace_manager_class._ensure_workspace_size = ensure_workspace_size
    setattr(workspace_manager_class, _WORKSPACE_MANAGER_MARKER, True)
    return True


def patch_moe_workspace_pool() -> None:
    """Tag the MoE workspace pool and register it in the multiproc executor's sleep state."""
    try:
        from vllm.device_allocator.cumem import CuMemAllocator
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id
        from vllm.v1.worker.workspace import WorkspaceManager
    except (ImportError, AttributeError):
        return

    try:
        WorkspaceManager._ensure_workspace_size
        original_sleep = MultiprocExecutor.sleep
    except AttributeError:
        return

    _patch_workspace_manager_class(
        WorkspaceManager,
        allocator_provider=lambda: CuMemAllocator.instance,
        ubatch_provider=dbo_current_ubatch_id,
    )

    if getattr(original_sleep, _EXECUTOR_SLEEP_MARKER, False):
        return

    def _patched_sleep(self, level: int = 1, original=original_sleep) -> None:
        original(self, level)
        if getattr(self, "is_sleeping", False):
            sleeping_tags = getattr(self, "sleeping_tags", None)
            if sleeping_tags is not None:
                sleeping_tags.add(MOE_WORKSPACE_TAG)

    _patched_sleep._diffrl_moe_workspace_sleep_tag = True  # type: ignore[attr-defined]
    MultiprocExecutor.sleep = _patched_sleep


__all__ = ["MOE_WORKSPACE_TAG", "patch_moe_workspace_pool"]
