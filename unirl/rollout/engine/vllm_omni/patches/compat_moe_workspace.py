"""Route vLLM's reusable MoE workspace through a CuMem pool so sleep() can offload it."""

from __future__ import annotations

from typing import Any, Callable

MOE_WORKSPACE_TAG = "moe_workspace"
_WORKSPACE_MANAGER_MARKER = "_diffrl_moe_workspace_pool_installed"
_EXECUTOR_SLEEP_MARKER = "_diffrl_moe_workspace_sleep_tag"


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
        current_workspace = manager._current_workspaces[ubatch_id]
        current_size = manager._workspace_size_bytes(current_workspace)
        if current_size < required_bytes:
            allocator = allocator_provider()
            if allocator is None:
                return original_ensure_workspace_size(manager, required_bytes)
            with allocator.use_memory_pool(tag=MOE_WORKSPACE_TAG):
                return original_ensure_workspace_size(manager, required_bytes)
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

    _patch_workspace_manager_class(
        WorkspaceManager,
        allocator_provider=lambda: CuMemAllocator.instance,
        ubatch_provider=dbo_current_ubatch_id,
    )

    _orig_sleep = MultiprocExecutor.sleep
    if getattr(_orig_sleep, _EXECUTOR_SLEEP_MARKER, False):
        return

    def _patched_sleep(self, level: int = 1, _orig=_orig_sleep) -> None:
        _orig(self, level)
        if getattr(self, "is_sleeping", False):
            self.sleeping_tags.add(MOE_WORKSPACE_TAG)

    _patched_sleep._diffrl_moe_workspace_sleep_tag = True  # type: ignore[attr-defined]
    MultiprocExecutor.sleep = _patched_sleep


__all__ = ["MOE_WORKSPACE_TAG", "patch_moe_workspace_pool"]
