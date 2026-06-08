"""Re-home the ``sglang-drl`` fork's ``GPUWorker`` RL additions onto stock upstream.

The fork added, to ``runtime/managers/gpu_worker.py:GPUWorker``:
  * ``__init__`` instance state for sleep/wake + distributed weight updates, and a
    ``MemorySaverHandler`` (zero-copy GPU sleep/wake).
  * ~14 net-new methods: ``is_sleeping``, ``_to_torch_dtype``,
    ``init_weights_update_group`` / ``destroy_weights_update_group``,
    ``update_weights_from_tensor`` / ``update_weights_from_distributed``,
    ``encode_prompt``, ``get_weights_detail``, ``set_lora_from_tensors``,
    ``_get_module_device`` / ``_move_unregistered_tensors`` / ``_move_modules``,
    ``release_memory_occupation`` / ``resume_memory_occupation``.

All method bodies are copied verbatim from the fork diff
(``e9b570654..HEAD`` for ``gpu_worker.py``); they are only re-homed as
``setattr`` (and an AROUND-wrapped ``__init__``) so UniRL can track
upstream instead of carrying a hard fork. NO sglang source is edited.

The fork called several names as gpu_worker module globals (``get_tp_rank``,
``WeightsUpdater``, ``get_updatable_modules``, ``iter_materialized_weights``,
``compute_weights_checksum``, ``LoRAPipeline``); since these patched functions
live in this module, they import those names locally (import-safe, idempotent).

Cross-patch dependencies (added by sibling patches, called here exactly as the
fork does):
  * ``WeightsUpdater.update_weights_from_named_tensors`` -- fork-only, added by
    ``patch_weights_updater``. Used by ``update_weights_from_tensor`` /
    ``update_weights_from_distributed``.
  * ``LoRAPipeline.set_lora(..., lora_tensors=...)`` + the tensor-load path --
    fork-only, added by ``patch_lora_pipeline``. Used by ``set_lora_from_tensors``.

See the module-level RISKS docstring at the bottom for upstream gaps.
"""

from __future__ import annotations

from typing import List, Union

import torch


def patch_gpu_worker() -> None:
    """Install the fork's ``GPUWorker`` RL additions on stock upstream sglang."""
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

    # -- AROUND-wrap __init__: add fork instance state + MemorySaverHandler -----
    if not getattr(GPUWorker.__init__, "_unirl_gpu_worker", False):
        _orig_init = GPUWorker.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)

            # Lazy import: import-safe + avoids a hard dep at patch-install time.
            from sglang.srt.utils.torch_memory_saver_adapter import (
                TorchMemorySaverAdapter,
            )

            from unirl.rollout.engine.sglang._patches.memory_saver import (
                MemorySaverHandler,
            )

            self._sleeping: bool = False
            self._sleep_restore_map: dict[str, str] = {}
            self._weights_update_groups: dict = {}

            # Memory saver handler (zero-copy sleep/wake).
            # NOTE: stock-upstream multimodal_gen ServerArgs lacks
            # ``enable_memory_saver`` (only ``pin_cpu_memory`` exists), so read
            # both defensively via getattr -- see RISKS. The fork read them as
            # plain attributes (server_args.enable_memory_saver / .pin_cpu_memory).
            self._memory_saver = MemorySaverHandler(
                adapter=TorchMemorySaverAdapter.create(enable=getattr(self.server_args, "enable_memory_saver", False)),
                pipeline=self.pipeline,
                local_rank=self.local_rank,
                pin_cpu_memory=getattr(self.server_args, "pin_cpu_memory", True),
            )
            self._dirty_modules = self._memory_saver.dirty_modules

        __init__._unirl_gpu_worker = True  # type: ignore[attr-defined]
        GPUWorker.__init__ = __init__

    # -- setattr the net-new methods (verbatim fork bodies) --------------------
    # Idempotency guard: all methods share one sentinel attr on the class.
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
    # NOTE: get_weights_checksum is NOT set -- it exists in stock upstream.

    GPUWorker._unirl_gpu_worker_methods = True


# ===========================================================================
# Module-level patched method bodies (copied verbatim from the fork diff).
# ``self`` is the GPUWorker instance; module globals the fork relied on are
# imported locally inside each body (import-safe).
# ===========================================================================


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
    """Encode text prompts into embeddings using the pipeline's text encoding stage.

    Returns a dict mapping tensor names to torch.Tensor values:
      - prompt_embeds: [B, seq, hidden] sequence embeddings (concatenated along
        seq dim when multiple encoders produce 3-D output)
      - pooled_prompt_embeds: [B, hidden] pooled embeddings (concatenated along
        hidden dim when multiple 2-D outputs exist)
      - encoder_attention_mask: [B, seq] attention mask for sequence encoders
    """
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

        # Separate 3D sequence embeds from 2D pooled embeds
        seq_embeds = [e for e in embeds_list if e.ndim >= 3]
        pooled_embeds = [e for e in embeds_list if e.ndim == 2]

        # prompt_embeds: concat sequence embeds along seq dim
        if seq_embeds:
            result["prompt_embeds"] = torch.cat(seq_embeds, dim=1) if len(seq_embeds) > 1 else seq_embeds[0]

        # pooled_prompt_embeds: from 2D embeds first, fallback to pooled_list
        # (don't merge both — Flux has duplicates across the two sources)
        if not pooled_embeds:
            pooled_embeds = list(pooled_list)
        if pooled_embeds:
            result["pooled_prompt_embeds"] = (
                torch.cat(pooled_embeds, dim=-1) if len(pooled_embeds) > 1 else pooled_embeds[0]
            )

        # Attention masks for sequence encoders
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
    """
    Move tensor attributes that are not covered by `module.to(device)`.

    `module.to` handles parameters/buffers/submodules, but some models keep tensor
    caches in plain Python attributes. We traverse `module.__dict__` and move tensor
    leaves inside tensors / dict / list / tuple while keeping non-tensor objects.
    """
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
    """
    Move selected modules to device.

    This function has all-or-nothing semantics:
    - Stop on first failure (missing module / device query / move / sanitize).
    - Roll back modules already moved in this call.
    - Raise RuntimeError to caller after rollback.
    """
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
        # TODO (mengyang, chenyang): If exception is raised
        # during rollback, the original exception detail is lost.
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

    # --- memory_saver path: per-component region pause ---
    if self._memory_saver.enabled:
        result = self._memory_saver.release(tags, cpu_backup_tags)
        self._sleeping = result.get("sleeping", False)
        return result

    # --- legacy path: .to("cpu") offload ---
    # Accept any tags (or None) — legacy path moves all modules regardless.

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

    # --- memory_saver path: per-component region resume ---
    if self._memory_saver.enabled:
        result = self._memory_saver.resume(tags)
        self._sleeping = result.get("sleeping", False)
        return result

    # --- legacy path: .to(device) restore ---

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


# ===========================================================================
# RISKS (upstream gaps vs. the fork) -- surfaced per task requirements.
# ===========================================================================
#
# 1. ServerArgs.enable_memory_saver MISSING upstream.
#    Stock-upstream ``multimodal_gen/runtime/server_args.py`` defines
#    ``pin_cpu_memory: bool = True`` (line 208) but NOT ``enable_memory_saver``
#    (that field lives only in srt ServerArgs and in the fork's multimodal_gen
#    ServerArgs). __init__ above therefore reads BOTH via getattr with the
#    fork's defaults (enable_memory_saver=False, pin_cpu_memory=True). Net effect
#    on the SD3/dance pilots: ``self._memory_saver.enabled`` is False, so
#    release/resume take the legacy ``.to("cpu")`` path -- functionally fine.
#    To enable the zero-copy memory_saver path, upstream (or a ServerArgs patch)
#    must add ``enable_memory_saver``. Worker reads server args as
#    ``self.server_args`` (confirmed upstream gpu_worker:118).
#
# 2. encode_prompt: encode_text return-arity DRIFT (RISK).
#    The fork unpacks a 3-tuple
#    ``embeds_list, masks_list, pooled_list = text_stage.encode_text(...)``,
#    but stock upstream ``TextEncodingStage.encode_text(return_type="list",
#    return_attention_mask=True)`` now returns a 5-tuple
#    ``(embeds_list, attn_masks_list, pooled_embeds_list, embeds_masks_list,
#    seq_lens_list)``. The verbatim fork body will raise "too many values to
#    unpack" against upstream. Left verbatim (battle-tested) and flagged: this is
#    the conditions/text-embed path, NOT on the SD3/dance pilots
#    (populate_conditions=False), so it is exercised only if encode_prompt is
#    actually called. Fix when adopting the conditions path (re-sync the unpack
#    or pass return_type="dict").
#
# 3. set_lora_from_tensors: depends on a sibling LoRA patch (RISK).
#    Stock upstream ``LoRAPipeline.set_lora`` does NOT accept ``lora_tensors=``
#    (and lacks the ``load_lora_adapter_from_tensors`` / ``normalize_lora_state_dict``
#    helpers the fork added). The body here calls set_lora exactly as the fork
#    does; it only works once ``patch_lora_pipeline`` re-homes those fork
#    additions onto upstream. Not on the SD3/dance pilot path.
#
# 4. WeightsUpdater.update_weights_from_named_tensors is fork-only.
#    The class exists upstream (weights_updater.py:154) but this method does NOT.
#    update_weights_from_tensor / update_weights_from_distributed call it as the
#    fork does; it must be provided by sibling ``patch_weights_updater``.
#
# 5. Forward path (_req_to_output_batch) NOT wrapped -- conditions path DEFERRED.
#    The fork wrapped an inline ``OutputBatch(...)`` in the forward loop to add
#    prompt_embeds / pooled_prompt_embeds / encoder_attention_mask /
#    negative_* and trajectory_log_probs / trajectory_noise_preds.
#    Upstream has since refactored this into a @staticmethod
#    ``GPUWorker._req_to_output_batch(result)`` whose OutputBatch ALREADY carries
#    the native-logprob payload via ``rollout_trajectory_data`` (schedule_batch
#    OutputBatch:416) and ``trajectory_latents`` -- so the trajectory/log-prob
#    needs of the SD3/dance pilots are met without any wrap. Upstream OutputBatch
#    has NO trajectory_log_probs / trajectory_noise_preds / pooled_prompt_embeds /
#    encoder_attention_mask / neg_pooled_prompt_embeds / negative_attention_mask
#    fields, and the conditions return_prompt_embeds/return_negative_prompt_embeds
#    flags are not on the pilot path (populate_conditions=False). We therefore
#    SKIP wrapping the forward / _req_to_output_batch. Revisit when adopting the
#    conditions path: it needs both new OutputBatch fields and an AROUND-wrap of
#    the static ``_req_to_output_batch`` (or its merge helpers).
