"""Fix colocate device-id resolution in CudaPlatformBase.get_available_gpu_memory."""

from __future__ import annotations

from typing import Any

import torch


def patch_platform_device() -> None:
    import psutil
    from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase

    if getattr(CudaPlatformBase, "_unirl_get_avail_mem_guard", False):
        return

    @classmethod
    def get_available_gpu_memory(
        cls,
        device_id: int = 0,
        distributed: bool = False,
        empty_cache: bool = True,
        cpu_group: Any = None,
    ) -> float:
        if empty_cache:
            torch.cuda.empty_cache()

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            if rank < torch.cuda.device_count():
                device_id = rank

        device_props = torch.cuda.get_device_properties(device_id)
        if device_props.is_integrated:
            free_gpu_memory = psutil.virtual_memory().available
        else:
            free_gpu_memory, _ = torch.cuda.mem_get_info(device_id)

        if distributed:
            import torch.distributed as dist

            tensor = torch.tensor(free_gpu_memory, dtype=torch.float32, device="cuda")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
            free_gpu_memory = float(tensor.item())

        return free_gpu_memory / (1 << 30)

    CudaPlatformBase.get_available_gpu_memory = get_available_gpu_memory
    CudaPlatformBase._unirl_get_avail_mem_guard = True
