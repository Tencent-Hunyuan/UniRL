"""Add UniRL-only distributed sync and tensor-update cache reset to GPUWorker."""

from __future__ import annotations

import torch


def patch_gpu_worker() -> None:
    """Extend the v0.5.19 worker without replacing its native post-training API."""
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

    _patch_flattened_bucket_payload()

    if not getattr(GPUWorker.__init__, "_unirl_gpu_worker", False):
        orig_init = GPUWorker.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self._weights_update_groups: dict = {}

        __init__._unirl_gpu_worker = True  # type: ignore[attr-defined]
        GPUWorker.__init__ = __init__

    if getattr(GPUWorker, "_unirl_gpu_worker_methods", False):
        return

    orig_update_from_tensor = GPUWorker.update_weights_from_tensor

    def update_weights_from_tensor(self, req) -> tuple[bool, str]:
        success, message = orig_update_from_tensor(self, req)
        if success and getattr(req, "flush_cache", True):
            _reset_teacache(self.pipeline, req.target_modules)
        return success, message

    GPUWorker.init_weights_update_group = _init_weights_update_group
    GPUWorker.destroy_weights_update_group = _destroy_weights_update_group
    GPUWorker.update_weights_from_tensor = update_weights_from_tensor
    GPUWorker.update_weights_from_distributed = _update_weights_from_distributed
    GPUWorker._unirl_gpu_worker_methods = True


def _patch_flattened_bucket_payload() -> None:
    """Accept UniRL's direct one-module flattened-bucket payload."""
    from sglang.multimodal_gen.runtime.post_training.weights_updater import (
        WeightsUpdater,
    )

    orig = WeightsUpdater._resolve_module_payloads
    if getattr(orig, "_unirl_flattened_bucket", False):
        return

    def _resolve_module_payloads(self, named_tensors, modules_to_update):
        if (
            isinstance(named_tensors, dict)
            and "flattened_tensor" in named_tensors
            and "metadata" in named_tensors
            and len(modules_to_update) == 1
        ):
            return {modules_to_update[0][0]: named_tensors}
        return orig(self, named_tensors, modules_to_update)

    _resolve_module_payloads._unirl_flattened_bucket = True  # type: ignore[attr-defined]
    WeightsUpdater._resolve_module_payloads = _resolve_module_payloads


def _init_weights_update_group(
    self,
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
    group_name: str = "weight_update_group",
    backend: str = "nccl",
) -> tuple[bool, str]:
    """Initialize a custom process group for external weight broadcasts."""
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
    from sglang.srt.utils.common import init_custom_process_group

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")
    if group_name in self._weights_update_groups:
        return True, f"Group {group_name} already initialized."

    try:
        rank = int(rank_offset) + int(self.rank)
        self._weights_update_groups[group_name] = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=int(world_size),
            rank=rank,
            group_name=group_name,
        )
        return True, "Succeeded to initialize custom process group."
    except Exception as exc:
        logger.error("Failed to initialize custom process group: %s", exc)
        return False, f"Failed to initialize custom process group: {exc}"


def _destroy_weights_update_group(
    self,
    group_name: str = "weight_update_group",
) -> tuple[bool, str]:
    """Destroy a custom external weight-update process group."""
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")
    if group_name not in self._weights_update_groups:
        return False, "The group to be destroyed does not exist."

    try:
        import torch.distributed as dist

        process_group = self._weights_update_groups.pop(group_name)
        dist.destroy_process_group(process_group)
        return True, "Succeeded to destroy custom process group."
    except Exception as exc:
        logger.error("Failed to destroy custom process group: %s", exc)
        return False, f"Failed to destroy custom process group: {exc}"


def _reset_teacache(pipeline, target_modules: list[str] | None) -> None:
    from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
    from sglang.multimodal_gen.runtime.post_training.weights_updater import (
        get_updatable_modules,
    )

    modules = get_updatable_modules(pipeline)
    for name in target_modules or ["transformer"]:
        module = modules.get(name)
        if isinstance(module, TeaCacheMixin):
            module.reset_teacache_state()


def _update_weights_from_distributed(
    self,
    names: list[str],
    dtypes: list[str],
    shapes: list[list[int]],
    group_name: str = "weight_update_group",
    target_modules: list[str] | None = None,
    flush_cache: bool = True,
) -> tuple[bool, str]:
    """Receive a broadcast and delegate application to v0.5.19's updater."""
    import torch.distributed as dist
    from sglang.multimodal_gen.runtime.post_training.weights_updater import (
        WeightsUpdater,
    )
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")
    if not self.pipeline:
        return False, "Pipeline is not initialized"
    if group_name not in self._weights_update_groups:
        return False, f"Group {group_name} is not initialized."
    if not (len(names) == len(dtypes) == len(shapes)):
        return False, "names, dtypes and shapes must have the same length"

    try:
        received: list[tuple[str, torch.Tensor]] = []
        handles = []
        process_group = self._weights_update_groups[group_name]
        device = torch.device("cuda", torch.cuda.current_device())
        updater = WeightsUpdater(self.pipeline)
        for name, dtype, shape in zip(names, dtypes, shapes):
            tensor = torch.empty(
                shape,
                dtype=updater._normalize_torch_dtype(dtype),
                device=device,
            )
            received.append((name, tensor))
            handles.append(dist.broadcast(tensor, src=0, group=process_group, async_op=True))
        for handle in handles:
            handle.wait()

        success, message = updater.update_weights_from_tensor(
            named_tensors=received,
            target_modules=target_modules,
        )
        if success and flush_cache:
            _reset_teacache(self.pipeline, target_modules)
        return success, message
    except Exception as exc:
        logger.error(
            "update_weights_from_distributed failed: %s",
            exc,
            exc_info=True,
        )
        return False, f"Failed to update weights from distributed: {exc}"
