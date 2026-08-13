"""Worker-extension class installed on the HI3 AR stage of vllm-omni."""

from __future__ import annotations

from unirl.rollout.engine.vllm_omni.patches.compat_tokenizer import HI3ARWorkerExtension
from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
    BucketedIPCReceiveMixin,
)
from unirl.rollout.engine.vllm_omni.worker.nccl_receive_mixin import (
    NcclBroadcastReceiveMixin,
)


class HI3ARWeightSyncExtension(
    BucketedIPCReceiveMixin,
    NcclBroadcastReceiveMixin,
    HI3ARWorkerExtension,
):
    """Receive-side extension for the HI3 AR stage."""

    pass


__all__ = ["HI3ARWeightSyncExtension"]
