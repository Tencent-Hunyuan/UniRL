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

    def sleep(self, level: int = 1) -> int:
        """Drop cached prompt embeddings first: they live outside the CuMem pools that ``allocator.sleep`` frees."""
        if self.model_runner is not None:
            self.model_runner.clear_prompt_embed_cache()
        return super().sleep(level)


__all__ = ["DiTWeightSyncExtension"]
