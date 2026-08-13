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

    pass


__all__ = ["DiTWeightSyncExtension"]
