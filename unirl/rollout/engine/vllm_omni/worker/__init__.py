"""Worker-side weight-sync receive extensions (role 8 — worker subprocess).

The classes here are the ``engine_args.worker_extension_cls`` qualname targets
in the v2 stage YAMLs; vllm-omni composes them onto its worker class at
instantiation time. They are the receive side of the three transports:

- ``BucketedIPCReceiveMixin`` — bucketed ZMQ + CUDA-IPC ``update_weights_from_ipc``
  + the LoRA tensor-bag receivers (``set_lora_from_tensor_dict[_copy]``) +
  the checksum read-backs. Installs the patch bundle in ``__new__``.
- ``NcclBroadcastReceiveMixin`` — NCCL ``init_weights_update_group`` /
  ``update_weights_from_distributed`` / ``destroy_weights_update_group``.
- ``HI3ARWeightSyncExtension`` / ``DiTWeightSyncExtension`` — the per-stage
  compositions the YAMLs reference.

Host contract (the vllm-omni worker ``self`` these mixins extend): ``device``,
``local_rank``, ``load_weights(weights)``, ``add_lora(req)`` /
``remove_lora(int_id)`` — documented per-mixin.

The pure transfer mechanics (bucketed sender/receiver, ZMQ handle naming,
checksum formulas) live in the engine-neutral
``unirl.distributed.weight_sync.transfer`` package — both sides import it.

Lazy access: ``dit_extension`` pulls vllm-omni base classes at import time,
so the heavy modules resolve only when their symbol (or their qualname, via
the stage YAML) is requested — keeping this package importable without a
CUDA-capable vllm-omni install.
"""

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

    module = importlib.import_module(f"{__name__}.{target[0]}")
    return getattr(module, target[1])


__all__ = list(_LAZY_TARGETS)
