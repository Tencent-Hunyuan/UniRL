"""Fix colocate device-id resolution in CudaPlatformBase.get_available_gpu_memory.

Upstream (LIN-365 target) overrides the requested ``device_id`` with
``torch.distributed.get_rank()`` whenever a process group is initialized:

    if torch.distributed.is_initialized():
        device_id = torch.distributed.get_rank()
    device_props = torch.cuda.get_device_properties(device_id)

That holds for sglang's own multi-process TP (rank == local visible device), but
BREAKS UniRL **colocate**: the trainer's FSDP group spans N global ranks
while each Ray rollout actor sees only ONE GPU (``CUDA_VISIBLE_DEVICES``), so
``get_rank()`` returns e.g. 6 and ``get_device_properties(6)`` raises
``AssertionError: Invalid device id``. The fork never hit this because the
upstream ServerArgs auto-tuner ``maybe_adjust_auto_component_residency_after_offload``
(the first caller during ``ServerArgs.from_kwargs`` init) did not exist at the
fork point.

Fix: only adopt the rank as the device id when it is a valid LOCAL device;
otherwise honor the passed ``device_id``. Strictly safer than upstream (it also
fixes the case where upstream would have crashed). Patches the base class so all
``CudaPlatformBase`` subclasses (Nvml / NonNvml) inherit it.
"""

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
            # Colocate: the FSDP rank may exceed this actor's visible device
            # count (CUDA_VISIBLE_DEVICES restricts the actor to 1 GPU). Only
            # adopt the rank as the device id when it is a valid local device;
            # otherwise honor the caller-passed (local) device_id.
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
