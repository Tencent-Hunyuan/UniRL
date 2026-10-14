"""Worker-extension class installed on the HI3 DiT stage of vllm-omni."""

from __future__ import annotations

from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
    BucketedIPCReceiveMixin,
)
from unirl.rollout.engine.vllm_omni.worker.nccl_receive_mixin import (
    NcclBroadcastReceiveMixin,
)


class DiTWeightSyncExtension(
    BucketedIPCReceiveMixin,
    NcclBroadcastReceiveMixin,
    CustomPipelineWorkerExtension,
):
    """Receive-side extension for the HI3 DiT stage."""

    def unirl_weight_sync_device_info(self) -> dict:
        """Identify the physical GPU and TP rank for UniRL's native IPC topology check."""
        import torch
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        return {
            "tp_rank": tp_group.rank_in_group,
            "tp_world_size": tp_group.world_size,
            "device_uuid": str(torch.cuda.get_device_properties(self.device).uuid),
        }

    def sleep(self, level: int = 1) -> int:
        """Drop cached prompt embeddings first: they live outside the CuMem pools that ``allocator.sleep`` frees."""
        if self.model_runner is not None:
            self.model_runner.clear_prompt_embed_cache()
        return super().sleep(level)


__all__ = ["DiTWeightSyncExtension"]
