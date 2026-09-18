"""Re-home the ``sglang-drl`` fork's ``GPUWorker`` RL additions onto stock upstream."""

from __future__ import annotations

from typing import List, Optional, Union

import torch


def patch_gpu_worker() -> None:
    """Install the fork's ``GPUWorker`` RL additions on stock upstream sglang."""
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

    if not getattr(GPUWorker.__init__, "_unirl_gpu_worker", False):
        _orig_init = GPUWorker.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)

            from sglang.srt.utils.torch_memory_saver_adapter import (
                TorchMemorySaverAdapter,
            )

            from unirl.rollout.engine.sglang_diffusion._patches.memory_saver import (
                MemorySaverHandler,
            )

            self._sleeping: bool = False
            self._sleep_restore_map: dict[str, str] = {}
            self._weights_update_groups: dict = {}

            self._memory_saver = MemorySaverHandler(
                adapter=TorchMemorySaverAdapter.create(enable=getattr(self.server_args, "enable_memory_saver", False)),
                pipeline=self.pipeline,
                local_rank=self.local_rank,
                pin_cpu_memory=getattr(self.server_args, "pin_cpu_memory", True),
            )
            self._dirty_modules = self._memory_saver.dirty_modules

        __init__._unirl_gpu_worker = True  # type: ignore[attr-defined]
        GPUWorker.__init__ = __init__

    if getattr(GPUWorker, "_unirl_gpu_worker_methods", False):
        return

    GPUWorker.is_sleeping = _is_sleeping
    GPUWorker._to_torch_dtype = _to_torch_dtype
    GPUWorker.init_weights_update_group = _init_weights_update_group
    GPUWorker.destroy_weights_update_group = _destroy_weights_update_group
    GPUWorker.update_weights_from_tensor = _update_weights_from_tensor
    GPUWorker.update_weights_from_distributed = _update_weights_from_distributed
    GPUWorker.encode_prompt = _encode_prompt
    GPUWorker.get_weights_detail = _get_weights_detail
    GPUWorker.set_lora_from_tensors = _set_lora_from_tensors
    GPUWorker._get_module_device = _get_module_device
    GPUWorker._move_unregistered_tensors = _move_unregistered_tensors
    GPUWorker._move_modules = _move_modules
    GPUWorker.release_memory_occupation = _release_memory_occupation
    GPUWorker.resume_memory_occupation = _resume_memory_occupation

    GPUWorker._unirl_gpu_worker_methods = True


def _is_sleeping(self) -> bool:
    return self._sleeping


@staticmethod
def _to_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).replace("torch.", "")
    if not hasattr(torch, normalized):
        raise ValueError(f"Unsupported dtype: {dtype}")
    return getattr(torch, normalized)


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

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")

    if group_name in self._weights_update_groups:
        return True, f"Group {group_name} already initialized."

    try:
        from sglang.srt.utils.common import init_custom_process_group

        rank = int(rank_offset) + int(self.rank)
        self._weights_update_groups[group_name] = init_custom_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=int(world_size),
            rank=rank,
            group_name=group_name,
        )
        return True, "Succeeded to initialize custom process group."
    except Exception as e:
        logger.error("Failed to initialize custom process group: %s", e)
        return False, f"Failed to initialize custom process group: {e}"


def _destroy_weights_update_group(
    self,
    group_name: str = "weight_update_group",
) -> tuple[bool, str]:
    """Destroy a custom process group for external weight broadcasts."""
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")

    if group_name not in self._weights_update_groups:
        return False, "The group to be destroyed does not exist."
    try:
        import torch.distributed as dist

        pg = self._weights_update_groups.pop(group_name)
        dist.destroy_process_group(pg)
        return True, "Succeeded to destroy custom process group."
    except Exception as e:
        logger.error("Failed to destroy custom process group: %s", e)
        return False, f"Failed to destroy custom process group: {e}"


def _update_weights_from_tensor(
    self,
    serialized_named_tensors: list[str | bytes],
    target_modules: list[str] | None = None,
    load_format: str | None = None,
    flush_cache: bool = True,
) -> tuple[bool, str]:
    """Update model weights from serialized tensors."""
    from sglang.multimodal_gen.runtime.distributed import get_tp_rank
    from sglang.multimodal_gen.runtime.loader.weights_updater import WeightsUpdater
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")

    if not self.pipeline:
        return False, "Pipeline is not initialized"
    if not serialized_named_tensors:
        return False, "serialized_named_tensors is required"

    try:
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
    except Exception as e:
        return False, f"Failed to import tensor serializer utilities: {e}"

    try:
        monkey_patch_torch_reductions()
        payload_idx = min(int(get_tp_rank()), len(serialized_named_tensors) - 1)
        named_tensors = MultiprocessingSerializer.deserialize(serialized_named_tensors[payload_idx])
        updater = WeightsUpdater(self.pipeline)
        return updater.update_weights_from_named_tensors(
            named_tensors=named_tensors,
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
        )
    except Exception as e:
        logger.error("update_weights_from_tensor failed: %s", e, exc_info=True)
        return False, f"Failed to update weights from tensor: {e}"


def _update_weights_from_distributed(
    self,
    names: list[str],
    dtypes: list[str],
    shapes: list[list[int]],
    group_name: str = "weight_update_group",
    target_modules: list[str] | None = None,
    flush_cache: bool = True,
) -> tuple[bool, str]:
    """Update model weights from a custom distributed broadcast group."""
    import torch.distributed as dist
    from sglang.multimodal_gen.runtime.loader.weights_updater import WeightsUpdater
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")

    if not self.pipeline:
        return False, "Pipeline is not initialized"
    if group_name not in self._weights_update_groups:
        return False, f"Group {group_name} is not initialized."

    if not (len(names) == len(dtypes) == len(shapes)):
        return False, "names, dtypes and shapes must have the same length"

    try:
        recv_tensors: list[tuple[str, torch.Tensor]] = []
        handles = []
        pg = self._weights_update_groups[group_name]
        device = torch.device("cuda", torch.cuda.current_device())
        for name, dtype, shape in zip(names, dtypes, shapes):
            tensor = torch.empty(
                shape,
                dtype=self._to_torch_dtype(dtype),
                device=device,
            )
            recv_tensors.append((name, tensor))
            handles.append(dist.broadcast(tensor, src=0, group=pg, async_op=True))
        for handle in handles:
            handle.wait()

        updater = WeightsUpdater(self.pipeline)
        return updater.update_weights_from_named_tensors(
            named_tensors=recv_tensors,
            target_modules=target_modules,
            load_format=None,
            flush_cache=flush_cache,
        )
    except Exception as e:
        logger.error("update_weights_from_distributed failed: %s", e, exc_info=True)
        return False, f"Failed to update weights from distributed: {e}"


def _encode_prompt(self, prompts: list[str]) -> dict:
    """Encode prompts: ``prompt_embeds [B, seq, hidden]``, ``pooled [B, hidden]``, ``mask [B, seq]``."""
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")

    if self.pipeline is None:
        return {"error": "Pipeline is not initialized"}

    from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
        TextEncodingStage,
    )

    text_stage = self.pipeline.get_stage("text_encoding_stage")
    if text_stage is None or not isinstance(text_stage, TextEncodingStage):
        return {"error": "Pipeline does not have a text encoding stage"}

    try:
        embeds_list, masks_list, pooled_list = text_stage.encode_text(
            prompts,
            self.server_args,
            encoder_index=list(range(len(text_stage.text_encoders))),
            return_attention_mask=True,
        )

        result: dict = {}

        seq_embeds = [e for e in embeds_list if e.ndim >= 3]
        pooled_embeds = [e for e in embeds_list if e.ndim == 2]

        if seq_embeds:
            result["prompt_embeds"] = torch.cat(seq_embeds, dim=1) if len(seq_embeds) > 1 else seq_embeds[0]

        if not pooled_embeds:
            pooled_embeds = list(pooled_list)
        if pooled_embeds:
            result["pooled_prompt_embeds"] = (
                torch.cat(pooled_embeds, dim=-1) if len(pooled_embeds) > 1 else pooled_embeds[0]
            )

        seq_masks = [m for m in masks_list if m.ndim == 2]
        if seq_masks:
            result["encoder_attention_mask"] = torch.cat(seq_masks, dim=1) if len(seq_masks) > 1 else seq_masks[0]

        return result
    except Exception as e:
        logger.error("encode_prompt failed: %s", e, exc_info=True)
        return {"error": f"Encoding failed: {e}"}


def _get_weights_detail(self, module_names: list[str] | None = None) -> dict:
    """Get per-parameter details: names, shapes, dtypes, count, checksums."""
    from sglang.multimodal_gen.runtime.loader.weight_utils import (
        compute_weights_checksum,
    )
    from sglang.multimodal_gen.runtime.loader.weights_updater import (
        get_updatable_modules,
    )

    try:
        from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
            iter_materialized_weights,
        )
    except ImportError:  # pre-reorg flat layout (<= v0.5.12.post1)
        from sglang.multimodal_gen.runtime.managers.layerwise_offload import (
            iter_materialized_weights,
        )

    if not self.pipeline:
        return {"error": "Pipeline is not initialized"}

    all_modules = get_updatable_modules(self.pipeline)
    names = module_names if module_names is not None else list(all_modules.keys())

    result: dict = {}
    for module_name in names:
        module = all_modules.get(module_name)
        if module is None:
            result[module_name] = {"error": "not_found"}
            continue

        param_names = []
        param_shapes = {}
        param_dtypes = {}
        param_checksums = {}
        total_numel = 0
        for pname, ptensor in iter_materialized_weights(module):
            param_names.append(pname)
            param_shapes[pname] = list(ptensor.shape)
            param_dtypes[pname] = str(ptensor.dtype)
            total_numel += ptensor.numel()
            param_checksums[pname] = compute_weights_checksum([(pname, ptensor)])

        result[module_name] = {
            "param_count": len(param_names),
            "total_numel": total_numel,
            "param_names": sorted(param_names),
            "param_shapes": param_shapes,
            "param_dtypes": param_dtypes,
            "param_checksums": param_checksums,
        }
    return result


def _set_lora_from_tensors(
    self,
    lora_nickname: str,
    lora_tensors: dict,
    target: Union[str, List[str]] = "all",
    strength: Union[float, List[float]] = 1.0,
    lora_alpha: Optional[float] = None,
):
    """Set LoRA adapter from in-memory tensors."""
    from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

    if not isinstance(self.pipeline, LoRAPipeline):
        return OutputBatch(error="Lora is not enabled")
    self.pipeline.set_lora(
        lora_nickname,
        lora_path=None,
        target=target,
        strength=strength,
        lora_tensors=lora_tensors,
        lora_alpha=lora_alpha,
    )
    return OutputBatch()


def _get_module_device(self, module: torch.nn.Module) -> str:
    """Return best-effort device string for a module."""
    param = next(module.parameters(), None)
    if param is not None:
        return str(param.device)
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return str(buffer.device)

    for key, val in vars(module).items():
        if key.startswith("_"):
            continue
        if isinstance(val, torch.Tensor):
            return str(val.device)

    return "cpu"


def _move_unregistered_tensors(self, module: torch.nn.Module, device: str) -> None:
    """Move tensor attributes that are not covered by `module.to(device)`."""
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")

    def move_tensors(obj):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: move_tensors(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [move_tensors(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(move_tensors(v) for v in obj)
        return obj

    attrs = module.__dict__
    for attr_name, attr_value in list(attrs.items()):
        if attr_name in {"_parameters", "_buffers", "_modules"}:
            continue

        try:
            moved_value = move_tensors(attr_value)
        except Exception as e:
            logger.warning(
                f"[move_unregistered_tensors] attr move failed: module={module.__class__.__name__} attr={attr_name} type={type(attr_value)} target={device} error={e}",
            )
            raise e

        if moved_value is not attr_value:
            attrs[attr_name] = moved_value


def _move_modules(self, names: list[str], device: str) -> bool:
    """Move selected modules to device."""
    from sglang.multimodal_gen.runtime.loader.weights_updater import (
        get_updatable_modules,
    )
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")

    moved: list[str] = []

    if self.pipeline is None:
        raise RuntimeError(f"_move_modules called but pipeline is None, target={device}")

    modules = get_updatable_modules(self.pipeline)
    src_device_map: dict[str, str] = {}
    try:
        for name in names:
            module = modules.get(name)
            if module is None:
                raise RuntimeError(f"module not found during move: name={name}, target={device}")

            src_device_map[name] = self._get_module_device(module)
            module.to(device)
            moved.append(name)
            self._move_unregistered_tensors(module, device)
    except Exception as e:
        logger.warning(
            f"[_move_modules] move failed, rollback started: target={device} moved={moved} error={e}",
        )
        for name in moved:
            module = modules.get(name)
            src_dev = src_device_map.get(name)
            module.to(src_dev)
            self._move_unregistered_tensors(module, src_dev)
        raise RuntimeError(f"failed to move modules to {device}; rollback finished: error={e}") from e

    return True


def _release_memory_occupation(self, tags: list[str] | None = None, cpu_backup_tags: list[str] | None = None) -> dict:
    import gc

    from sglang.multimodal_gen.runtime.loader.weights_updater import (
        get_updatable_modules,
    )
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")

    logger.info(f"[SLEEP] GPUWorker.release_memory_occupation rank={self.rank}")
    if self._sleeping:
        return {"success": True, "sleeping": True, "message": "already sleeping"}
    if self.pipeline is None:
        return {
            "success": False,
            "sleeping": False,
            "message": "pipeline not initialized",
        }

    if self._memory_saver.enabled:
        result = self._memory_saver.release(tags, cpu_backup_tags)
        self._sleeping = result.get("sleeping", False)
        return result

    try:
        modules = get_updatable_modules(self.pipeline)
        restore_map: dict[str, str] = {}
        for name, m in modules.items():
            try:
                dev_str = self._get_module_device(m)
            except RuntimeError as e:
                logger.debug(
                    f"[SLEEP] module device query failed; skip module. rank={self.rank} module={name} error={e}",
                )
                continue
            if not dev_str.startswith("cpu"):
                restore_map[name] = dev_str

        self._move_modules(list(restore_map.keys()), "cpu")
        device = torch.get_device_module()
        device.synchronize()
        gc.collect()
        device.empty_cache()

        self._sleep_restore_map = restore_map
        self._sleeping = True
        return {
            "success": True,
            "sleeping": True,
            "message": "released GPU memory (moved active modules to CPU)",
        }
    except Exception as e:
        logger.warning(
            f"[SLEEP] release_memory_occupation failed. rank={self.rank} error={e}",
        )
        return {
            "success": False,
            "sleeping": self._sleeping,
            "message": f"offload failed; rolled back to keep state consistent: {e}",
        }


def _resume_memory_occupation(self, tags: list[str] | None = None) -> dict:
    """Resume previously released GPU memory occupation."""
    from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

    logger = init_logger("sglang.multimodal_gen.runtime.managers.gpu_worker")

    logger.info(f"[WAKE] GPUWorker.resume_memory_occupation rank={self.rank}")
    if not self._sleeping:
        return {"success": True, "sleeping": False, "message": "already awake"}
    if self.pipeline is None:
        return {
            "success": False,
            "sleeping": True,
            "message": "pipeline not initialized",
        }

    if self._memory_saver.enabled:
        result = self._memory_saver.resume(tags)
        self._sleeping = result.get("sleeping", False)
        return result

    try:
        if not self._sleep_restore_map:
            self._sleeping = False
            return {
                "success": True,
                "sleeping": False,
                "message": "no restore map; marked awake",
            }

        for dev_str in sorted(set(self._sleep_restore_map.values())):
            names = [n for n, d in self._sleep_restore_map.items() if d == dev_str]
            self._move_modules(names, dev_str)

        self._sleep_restore_map = {}
        self._sleeping = False
        return {
            "success": True,
            "sleeping": False,
            "message": "resumed GPU memory (restored modules to original devices)",
        }
    except Exception as e:
        logger.warning(
            f"[WAKE] resume_memory_occupation failed. rank={self.rank} error={e}",
        )
        return {
            "success": False,
            "sleeping": self._sleeping,
            "message": f"resume failed; rolled back to keep state consistent: {e}",
        }
