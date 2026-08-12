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


class Qwen3OmniTalkerWeightSyncExtension(
    BucketedIPCReceiveMixin,
    NcclBroadcastReceiveMixin,
):
    """Talker-only receiver that rejects MTP/Thinker/Code2Wav LoRA keys."""

    @staticmethod
    def _diffrl_validate_lora_tensors(tensors) -> None:
        from unirl.distributed.weight_sync.lora.qwen3_omni_talker import (
            validate_talker_lora_keys,
        )

        validate_talker_lora_keys(tensors, engine_envelope=True)


__all__ = [
    "Qwen3OmniARWeightSyncExtension",
    "Qwen3OmniTalkerWeightSyncExtension",
]
