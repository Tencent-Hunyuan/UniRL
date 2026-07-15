"""Receive-side worker extensions for vllm-omni AR stages.

Composes:

- ``BucketedIPCReceiveMixin`` — bucketed CUDA-IPC ``update_weights_from_ipc``
  + LoRA-bucket dispatch + ``VLLMOmniHijack`` install in ``__new__``.
- ``NcclBroadcastReceiveMixin`` — SGLang-shape NCCL primitives
  (``init_weights_update_group``, ``update_weights_from_distributed``,
  ``destroy_weights_update_group``).
- HI3 installs its tokenizer and MoE-LoRA compatibility patches lazily from
  ``HI3ARWeightSyncExtension.__new__``.
- BAGEL uses ``BagelARWeightSyncExtension``, which has no HI3 imports or
  patches.

The AR worker (``GPUARWorker`` → ``OmniGPUWorkerBase`` → upstream
``vllm.v1.worker.gpu_worker.Worker``) already inherits upstream's
``init_weight_transfer_engine`` / ``update_weights(update_info)`` for
the ``WeightTransferEngine`` path. We use the SGLang-shape NCCL methods
on top of (not instead of) those — both are reachable via collective_rpc.
"""

from __future__ import annotations

from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
    BucketedIPCReceiveMixin,
)
from unirl.rollout.engine.vllm_omni.worker.nccl_receive_mixin import (
    NcclBroadcastReceiveMixin,
)


class HI3ARWeightSyncExtension(
    BucketedIPCReceiveMixin,
    NcclBroadcastReceiveMixin,
):
    """Receive-side extension for the HI3 AR stage."""

    def __new__(cls, **kwargs):
        # Importing this module installs both HI3-specific compatibility
        # patches.  Keep the import here so resolving the neutral BAGEL class
        # from this module cannot mutate tokenizer or HI3 MoE-LoRA behavior.
        from unirl.rollout.engine.vllm_omni.patches.compat_tokenizer import (
            install,
        )

        install()
        return super().__new__(cls, **kwargs)


class BagelARWeightSyncExtension(
    BucketedIPCReceiveMixin,
    NcclBroadcastReceiveMixin,
):
    """Neutral full-weight receiver for BAGEL's AR stage."""

    pass


__all__ = ["HI3ARWeightSyncExtension", "BagelARWeightSyncExtension"]
