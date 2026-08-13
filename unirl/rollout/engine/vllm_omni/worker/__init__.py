"""Worker-side weight-sync receive extensions (role 8 — worker subprocess)."""

_LAZY_TARGETS = {
    "BucketedIPCReceiveMixin": ("ipc_receive_mixin", "BucketedIPCReceiveMixin"),
    "NcclBroadcastReceiveMixin": ("nccl_receive_mixin", "NcclBroadcastReceiveMixin"),
    "HI3ARWeightSyncExtension": ("ar_extension", "HI3ARWeightSyncExtension"),
    "Qwen3OmniARWeightSyncExtension": ("qwen3_omni_ar_extension", "Qwen3OmniARWeightSyncExtension"),
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
