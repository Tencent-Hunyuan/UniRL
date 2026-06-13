"""Worker-extension class installed on the HI3 DiT stage of vllm-omni.

Composes three mixins on top of vllm-omni's ``CustomPipelineWorkerExtension``
(the base extension class on every diffusion stage):

- ``BucketedIPCReceiveMixin`` — bucketed CUDA-IPC ``update_weights_from_ipc``
  + LoRA-bucket dispatch + ``VLLMOmniHijack`` install in ``__new__``.
- ``NcclBroadcastReceiveMixin`` — SGLang-shape NCCL primitives
  (``init_weights_update_group``, ``update_weights_from_distributed``,
  ``destroy_weights_update_group``).

Together they cover all three weight-sync transports against the DiT
stage. ``CustomPipelineWorkerExtension`` is the only base that requires
inheritance order to come last (it's the "real" parent).
"""

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
