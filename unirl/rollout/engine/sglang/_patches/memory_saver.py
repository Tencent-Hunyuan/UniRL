"""Memory saver utilities for zero-copy GPU sleep/wake via CUDA virtual memory.

Encapsulates per-component region management, CPU backup stashing, and
dirty-module tracking. Used by GPUWorker to delegate memory_saver operations.

Copied verbatim from the sglang-drl fork
(``sglang/multimodal_gen/runtime/utils/memory_saver.py``) for the LIN-365
migration; its imports all resolve against stock upstream sglang.
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Iterable

import torch
from sglang.multimodal_gen.runtime.loader.weights_updater import get_updatable_modules
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (private)
# ---------------------------------------------------------------------------

# Map pipeline component names to memory_saver region tags.
_COMPONENT_TO_REGION = {
    "transformer": "transformer",
    "transformer_2": "transformer",
    "video_dit": "transformer",
    "video_dit_2": "transformer",
    "audio_dit": "transformer",
    "vae": "vae",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder",
    "text_encoder_3": "text_encoder",
    "image_encoder": "image_encoder",
}
_ALL_REGION_TAGS = list(dict.fromkeys(_COMPONENT_TO_REGION.values()))

# Request-scoped caches that become invalid after weight updates.
# Keyed by (class_name, attr_name).
_REQUEST_SCOPED_ATTRS = {
    ("GLMSelfAttention", "k_cache"),
    ("GLMSelfAttention", "v_cache"),
}


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


def get_region_tag(component_name: str) -> str | None:
    """Return the memory_saver region tag for a pipeline component name, or None."""
    return _COMPONENT_TO_REGION.get(component_name)


# ---------------------------------------------------------------------------
# Handler class
# ---------------------------------------------------------------------------


class MemorySaverHandler:
    """Manages per-component memory_saver regions: pause/resume, CPU backup, dirty tracking."""

    def __init__(
        self,
        adapter: TorchMemorySaverAdapter,
        pipeline,
        local_rank: int,
        pin_cpu_memory: bool = True,
    ) -> None:
        self.adapter = adapter
        self.pipeline = pipeline
        self.local_rank = local_rank
        self._pin_cpu_memory = pin_cpu_memory

        self._paused_tags: set[str] = set()
        self._cpu_backup_tags: set[str] = set()
        self._stashed_states: dict[str, dict] = {}
        self.dirty_modules: set[str] = set()
        # Reusable pinned CPU buffers keyed by (tag, module_name, param_key)
        self._pinned_buffers: dict[tuple[str, str, str], torch.Tensor] = {}

    @property
    def enabled(self) -> bool:
        return self.adapter.enabled

    # -- module lookup -------------------------------------------------------

    def modules_for_tag(self, tag: str) -> dict[str, torch.nn.Module]:
        """Return pipeline modules belonging to a given region tag."""
        modules = get_updatable_modules(self.pipeline)
        return {name: m for name, m in modules.items() if _COMPONENT_TO_REGION.get(name) == tag}

    # -- ephemeral cache clearing --------------------------------------------

    def clear_ephemeral_caches(self, tags: Iterable[str]) -> None:
        """Null out request-scoped caches that are invalid after weight updates."""
        for tag in tags:
            for _name, m in self.modules_for_tag(tag).items():
                for submod in m.modules():
                    cls_name = type(submod).__name__
                    for attr in list(submod.__dict__):
                        if (cls_name, attr) in _REQUEST_SCOPED_ATTRS:
                            submod.__dict__[attr] = None

    # -- pinned buffer management ---------------------------------------------

    def _get_or_alloc_pinned(self, tag: str, module_name: str, key: str, src: torch.Tensor) -> torch.Tensor:
        """Return a reusable pinned CPU buffer matching src's shape/dtype."""
        cache_key = (tag, module_name, key)
        buf = self._pinned_buffers.get(cache_key)
        if buf is not None and buf.shape == src.shape and buf.dtype == src.dtype:
            return buf
        buf = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
        self._pinned_buffers[cache_key] = buf
        return buf

    # -- CPU stash / restore -------------------------------------------------

    def _clone_gpu_tensor_to_cpu(self, src: torch.Tensor, cache_key: tuple[str, str, str]) -> torch.Tensor:
        """Copy a GPU tensor to a (pinned) CPU buffer with non_blocking."""
        if self._pin_cpu_memory:
            buf = self._pinned_buffers.get(cache_key)
            if buf is None or buf.shape != src.shape or buf.dtype != src.dtype:
                buf = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
                self._pinned_buffers[cache_key] = buf
            buf.copy_(src.detach(), non_blocking=True)
            return buf
        return src.detach().clone().cpu()

    def _clone_gpu_tensors_to_cpu_recursive(self, obj, tag: str, prefix: str):
        """Recursively clone GPU tensors to CPU; return obj unchanged if no GPU tensors."""
        if torch.is_tensor(obj) and obj.is_cuda:
            return self._clone_gpu_tensor_to_cpu(obj, (tag, prefix, "_unreg_tensor"))
        if isinstance(obj, dict):
            result = {k: self._clone_gpu_tensors_to_cpu_recursive(v, tag, f"{prefix}.{k}") for k, v in obj.items()}
            if any(result[k] is not obj[k] for k in obj):
                return result
            return obj
        if isinstance(obj, list):
            result = [self._clone_gpu_tensors_to_cpu_recursive(v, tag, f"{prefix}.{i}") for i, v in enumerate(obj)]
            if any(r is not o for r, o in zip(result, obj)):
                return result
            return obj
        if isinstance(obj, tuple):
            result = tuple(self._clone_gpu_tensors_to_cpu_recursive(v, tag, f"{prefix}.{i}") for i, v in enumerate(obj))
            if any(r is not o for r, o in zip(result, obj)):
                return result
            return obj
        return obj

    @staticmethod
    def _move_saved_to_device(obj, device, non_blocking: bool = False):
        """Recursively move saved CPU tensors back to device."""
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=non_blocking)
        if isinstance(obj, dict):
            return {k: MemorySaverHandler._move_saved_to_device(v, device, non_blocking) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MemorySaverHandler._move_saved_to_device(v, device, non_blocking) for v in obj]
        if isinstance(obj, tuple):
            return tuple(MemorySaverHandler._move_saved_to_device(v, device, non_blocking) for v in obj)
        return obj

    def stash_tag(self, tag: str) -> None:
        """Clone entire module state (params + buffers + unregistered tensors) to CPU.

        Uses pinned memory and non_blocking copies when ``_pin_cpu_memory`` is True.
        Caller must call ``torch.cuda.synchronize()`` after all stash_tag calls.
        """
        state: dict = {"params_and_buffers": {}, "unregistered": {}}
        for name, m in self.modules_for_tag(tag).items():
            # Use named_parameters() + named_buffers() instead of state_dict()
            # because state_dict() excludes persistent=False buffers (e.g. CLIP position_ids)
            saved: dict[str, torch.Tensor] = {}
            for k, v in m.named_parameters():
                saved[k] = self._clone_gpu_tensor_to_cpu(v, (tag, name, k))
            for k, v in m.named_buffers():
                saved[k] = self._clone_gpu_tensor_to_cpu(v, (tag, name, f"buf.{k}"))
            state["params_and_buffers"][name] = saved

            # Stash unregistered GPU tensor attrs (Qwen RoPE etc.)
            for submod_name, submod in m.named_modules():
                prefix = f"{name}.{submod_name}" if submod_name else name
                for attr_name, attr_value in list(submod.__dict__.items()):
                    if attr_name in ("_parameters", "_buffers", "_modules"):
                        continue
                    cloned = self._clone_gpu_tensors_to_cpu_recursive(attr_value, tag, f"{prefix}.{attr_name}")
                    if cloned is not attr_value:
                        state["unregistered"][(prefix, attr_name)] = cloned
        self._stashed_states[tag] = state

    def restore_tag(self, tag: str) -> None:
        """Restore stashed state for a tag after resume.

        Uses non_blocking copies when stashed tensors are in pinned memory.
        Caller must call ``torch.cuda.synchronize()`` after all restore_tag calls.
        """
        state = self._stashed_states.pop(tag, None)
        if state is None:
            return
        non_blocking = self._pin_cpu_memory
        device = f"cuda:{self.local_rank}"
        modules = get_updatable_modules(self.pipeline)
        # Restore params + buffers (including non-persistent)
        for name, saved_tensors in state["params_and_buffers"].items():
            m = modules.get(name)
            if m is None:
                continue
            current: dict[str, torch.Tensor] = {}
            for k, v in m.named_parameters():
                current[k] = v
            for k, v in m.named_buffers():
                current[k] = v
            for k, saved_v in saved_tensors.items():
                target = current.get(k)
                if target is not None:
                    target.data.copy_(saved_v, non_blocking=non_blocking)
        # Restore unregistered tensor attrs
        all_submodules: dict[str, torch.nn.Module] = {}
        for name, m in modules.items():
            for sn, sm in m.named_modules():
                prefix = f"{name}.{sn}" if sn else name
                all_submodules[prefix] = sm
        for (prefix, attr_name), saved_value in state["unregistered"].items():
            submod = all_submodules.get(prefix)
            if submod is not None:
                submod.__dict__[attr_name] = self._move_saved_to_device(saved_value, device, non_blocking=non_blocking)

    # -- release / resume orchestration --------------------------------------

    def release(
        self,
        tags: list[str] | None = None,
        cpu_backup_tags: list[str] | None = None,
    ) -> dict:
        """Pause memory_saver regions and optionally stash CPU backups.

        Returns a result dict with ``success``, ``sleeping``, ``message`` keys.
        """
        try:
            t_start = time.monotonic()
            all_tags = tags if tags is not None else list(_ALL_REGION_TAGS)
            backup_set = set(cpu_backup_tags or [])

            # 1. Clear request-scoped caches (e.g. GLM KV caches)
            self.clear_ephemeral_caches(all_tags)
            t_clear = time.monotonic()

            # 2. Stash state for CPU-backup tags (frozen modules)
            #    Uses non_blocking copies when pinned memory is enabled.
            for tag in all_tags:
                if tag in backup_set:
                    self.stash_tag(tag)
            # Ensure all async GPU→CPU copies complete before vunmap
            if self._pin_cpu_memory:
                torch.cuda.synchronize()
            t_stash = time.monotonic()

            # 3. Pause each tag (zero-copy since enable_cpu_backup=False)
            for tag in all_tags:
                self.adapter.pause(tag)
            t_pause = time.monotonic()

            torch.cuda.synchronize()
            t_sync = time.monotonic()
            gc.collect()
            t_gc = time.monotonic()
            torch.cuda.empty_cache()
            t_empty = time.monotonic()

            logger.info(
                "[SLEEP] memory_saver.release timing: "
                "clear_caches=%.3fs  stash_cpu=%.3fs  pause=%.3fs  "
                "cuda_sync=%.3fs  gc_collect=%.3fs  empty_cache=%.3fs  total=%.3fs",
                t_clear - t_start,
                t_stash - t_clear,
                t_pause - t_stash,
                t_sync - t_pause,
                t_gc - t_sync,
                t_empty - t_gc,
                t_empty - t_start,
            )

            self._paused_tags = set(all_tags)
            self._cpu_backup_tags = backup_set

            # Modules in non-backup tags are dirty (garbage until weight sync)
            non_backup_tags = set(all_tags) - backup_set
            self.dirty_modules = set()
            for tag in non_backup_tags:
                self.dirty_modules.update(self.modules_for_tag(tag).keys())

            return {
                "success": True,
                "sleeping": True,
                "message": (
                    f"released GPU memory via memory_saver (paused tags={all_tags}, backed up={list(backup_set)})"
                ),
            }
        except Exception as e:
            logger.warning(
                f"[SLEEP] memory_saver release failed. error={e}",
                exc_info=True,
            )
            return {
                "success": False,
                "sleeping": False,
                "message": f"memory_saver release failed: {e}",
            }

    def resume(self, tags: list[str] | None = None) -> dict:
        """Resume memory_saver regions and restore CPU backups.

        Returns a result dict with ``success``, ``sleeping``, ``message`` keys.
        """
        try:
            t_start = time.monotonic()
            tags_to_resume = set(tags) if tags is not None else set(self._paused_tags)

            # 1. Resume all tags (zero-copy remap)
            for tag in tags_to_resume:
                self.adapter.resume(tag)
            t_resume = time.monotonic()

            # 2. Restore stashed state for CPU-backed tags
            #    Uses non_blocking copies when pinned memory is enabled.
            for tag in tags_to_resume:
                if tag in self._cpu_backup_tags:
                    self.restore_tag(tag)
            # Ensure all async CPU→GPU copies complete before inference
            if self._pin_cpu_memory:
                torch.cuda.synchronize()
            t_restore = time.monotonic()

            logger.info(
                "[WAKE] memory_saver.resume timing: resume=%.3fs  restore_cpu=%.3fs  total=%.3fs",
                t_resume - t_start,
                t_restore - t_resume,
                t_restore - t_start,
            )

            self._paused_tags -= tags_to_resume
            still_sleeping = len(self._paused_tags) > 0
            # Non-backed tags remain dirty until weight sync
            return {
                "success": True,
                "sleeping": still_sleeping,
                "message": (f"resumed via memory_saver (tags={list(tags_to_resume)}, dirty={self.dirty_modules})"),
            }
        except Exception as e:
            logger.warning(
                f"[WAKE] memory_saver resume failed. error={e}",
                exc_info=True,
            )
            return {
                "success": False,
                "sleeping": True,
                "message": f"memory_saver resume failed: {e}",
            }
