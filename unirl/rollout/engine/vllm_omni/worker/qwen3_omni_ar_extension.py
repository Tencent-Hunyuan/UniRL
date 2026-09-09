"""Weight-sync extension for the Qwen3-Omni Thinker AR worker."""

from __future__ import annotations

from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
    BucketedIPCReceiveMixin,
)
from unirl.rollout.engine.vllm_omni.worker.nccl_receive_mixin import (
    NcclBroadcastReceiveMixin,
)


class Qwen3OmniARWeightSyncExtension(
    BucketedIPCReceiveMixin,
    NcclBroadcastReceiveMixin,
):
    """Receive-side extension for the Qwen3-Omni thinker AR stage."""

    pass


__all__ = ["Qwen3OmniARWeightSyncExtension"]
