"""Worker-extension class installed on the HI3 AR stage of vllm-omni.

Composes:

- ``BucketedIPCReceiveMixin`` — bucketed CUDA-IPC ``update_weights_from_ipc``
  + LoRA-bucket dispatch + ``VLLMOmniHijack`` install in ``__new__``.
- ``NcclBroadcastReceiveMixin`` — SGLang-shape NCCL primitives
  (``init_weights_update_group``, ``update_weights_from_distributed``,
  ``destroy_weights_update_group``).
- ``HI3ARWorkerExtension`` (``compat/tokenizer``) — preserves the
  module-import side effect that patches ``PreTrainedTokenizer.convert_tokens_to_ids``
  for the Base ckpt's missing ratio tokens.

The AR worker (``GPUARWorker`` → ``OmniGPUWorkerBase`` → upstream
``vllm.v1.worker.gpu_worker.Worker``) already inherits upstream's
``init_weight_transfer_engine`` / ``update_weights(update_info)`` for
the ``WeightTransferEngine`` path. We use the SGLang-shape NCCL methods
on top of (not instead of) those — both are reachable via collective_rpc.
"""

from __future__ import annotations

from unirl.rollout.engine.vllm_omni.compat.tokenizer import HI3ARWorkerExtension
from unirl.rollout.engine.vllm_omni.weight_sync.ipc_receive_mixin import (
    BucketedIPCReceiveMixin,
)
from unirl.rollout.engine.vllm_omni.weight_sync.nccl_receive_mixin import (
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
