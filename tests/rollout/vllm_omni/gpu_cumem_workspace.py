"""Real-GPU CuMem workspace isolation and sleep/wake smoke."""

from __future__ import annotations

import gc

import torch


def main() -> None:
    assert torch.cuda.is_available()

    from vllm.device_allocator.cumem import CuMemAllocator, cumem_available
    from vllm.v1.worker import ubatching, workspace
    from vllm.v1.worker.workspace import WorkspaceManager, use_workspace_lane

    from unirl.rollout.engine.vllm_omni.patches.compat_moe_workspace import (
        MOE_WORKSPACE_TAG,
        patch_moe_workspace_pool,
    )

    assert cumem_available
    current_ubatch = 0

    def ubatch_id() -> int:
        return current_ubatch

    ubatching.dbo_current_ubatch_id = ubatch_id
    workspace.dbo_current_ubatch_id = ubatch_id

    allocator = CuMemAllocator.get_instance()
    patch_moe_workspace_pool()
    manager = WorkspaceManager(torch.device("cuda"), num_ubatches=2, num_lanes=2)

    for ubatch in range(2):
        current_ubatch = ubatch
        for lane in range(2):
            with use_workspace_lane(lane):
                tensor = manager._ensure_workspace_size(1024 * 1024 + ubatch * 4096 + lane * 1024)
                tensor.fill_(ubatch * 2 + lane)

    expected_keys = {f"{MOE_WORKSPACE_TAG}:{index}" for index in range(4)}
    assert expected_keys.issubset(allocator.allocator_and_pools)
    assert len({id(tensor) for tensor in manager._current_workspaces}) == 4

    current_ubatch = 0
    saved_workspace = manager._current_workspaces[2]
    saved_pool = allocator.allocator_and_pools[f"{MOE_WORKSPACE_TAG}:2"]
    with use_workspace_lane(2):
        try:
            manager._ensure_workspace_size(2 * 1024 * 1024)
        except RuntimeError as error:
            assert "not configured" in str(error)
        else:
            raise AssertionError("invalid workspace lane was accepted")
    assert manager._current_workspaces[2] is saved_workspace
    assert allocator.allocator_and_pools[f"{MOE_WORKSPACE_TAG}:2"] is saved_pool

    for _ in range(2):
        allocator.sleep(offload_tags=("weights",))
        allocator.wake_up()
        for tensor in manager._current_workspaces:
            tensor.zero_()
        torch.accelerator.synchronize()

    manager._current_workspaces = [None] * 4
    gc.collect()
    allocator.release_pools()
    CuMemAllocator.instance = None
    print("gpu_cumem_workspace_ok")


if __name__ == "__main__":
    main()
