"""Weight-sync subsystem for the vllm-omni rollout engine.

Modules:

- ``bucketed_transfer``: ``BucketedWeightSender`` / ``BucketedWeightReceiver``
  for ZMQ + CUDA-IPC bucketed weight transport (lifted from upstream verl).
  Pure-Python — does NOT pull vllm-omni at import time.
- ``ipc_dispatch``: shared constants + ZMQ-handle naming used by both the
  driver-side IPC weight-sync handler and the rollout-side extensions.
  Pure-Python — does NOT pull vllm-omni at import time.
- ``ipc_receive_mixin`` / ``nccl_receive_mixin``: receive-side mixins for
  the worker extensions. Light, but ``ipc_receive_mixin`` imports
  ``unirl.rollout.engine.vllm_omni.vllm_patches`` (sibling module
  holding the monkey-patch bundle for vllm / vllm-omni).
- ``ar_extension`` / ``dit_extension``: per-stage worker extension classes
  installed via stage YAML ``engine_args.worker_extension_cls``. Pull
  vllm-omni base classes at import time — heaviest.

Eager-export only the pure-Python pieces so the trainer side
(the IPC weight-sync handler etc.) and the bucketed-transfer test can import
``unirl.rollout.engine.vllm_omni.weight_sync`` without a working
vllm-omni install (which requires a CUDA-13-compatible driver). The
extension classes are reachable via their fully-qualified module path —
which is also what the stage YAML uses for ``worker_extension_cls``.
"""

from unirl.rollout.engine.vllm_omni.weight_sync.bucketed_transfer import (
    BucketedWeightReceiver,
    BucketedWeightSender,
)
from unirl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
    DIFFRL_LORA_INT_ID,
    DIFFRL_LORA_NAME,
    DIFFRL_LORA_PATH,
    replica_rank_from_env,
    zmq_handle,
)

# Lazy access for the heavy modules — only imported when the symbol is
# actually requested. Keeps `import .weight_sync` cheap and CUDA-driver
# independent.
_LAZY_TARGETS = {
    "BucketedIPCReceiveMixin": ("ipc_receive_mixin", "BucketedIPCReceiveMixin"),
    "NcclBroadcastReceiveMixin": ("nccl_receive_mixin", "NcclBroadcastReceiveMixin"),
    "HI3ARWeightSyncExtension": ("ar_extension", "HI3ARWeightSyncExtension"),
    "DiTWeightSyncExtension": ("dit_extension", "DiTWeightSyncExtension"),
}


def __getattr__(name: str):
    target = _LAZY_TARGETS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod_name, attr = target
    mod = importlib.import_module(f"{__name__}.{mod_name}")
    value = getattr(mod, attr)
    globals()[name] = value
    return value


__all__ = [
    "BucketedIPCReceiveMixin",
    "BucketedWeightReceiver",
    "BucketedWeightSender",
    "DIFFRL_LORA_INT_ID",
    "DIFFRL_LORA_NAME",
    "DIFFRL_LORA_PATH",
    "DiTWeightSyncExtension",
    "HI3ARWeightSyncExtension",
    "NcclBroadcastReceiveMixin",
    "replica_rank_from_env",
    "zmq_handle",
]
