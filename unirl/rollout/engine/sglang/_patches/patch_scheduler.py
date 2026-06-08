"""Re-host the ``sglang-drl`` fork's ``Scheduler`` RL additions on stock upstream.

Stock upstream ``Scheduler`` (``runtime/managers/scheduler.py``) dispatches ZMQ
requests via ``self.request_handlers`` (keyed by ``type(req)``) and ships only
the disk-weight + checksum handlers. The fork added nine RL request handlers
(distributed weight sync, tensor weight sync, LoRA-from-tensors, encode-prompt,
sleep/wake memory occupation, per-param weight detail) plus two private helpers,
and gated ``_handle_generation`` / ``_handle_update_weights_from_disk`` behind
sleep/dirty-module guards.

This patch ports all of that WITHOUT editing sglang source:

  * ``setattr`` the 9 handlers + 2 helpers onto ``Scheduler`` (bodies verbatim
    from the fork diff). They call ``self.worker.<method>`` -- those worker
    methods (``is_sleeping``, ``_dirty_modules``, ``set_lora_from_tensors``,
    ``update_weights_from_tensor``, ...) are installed by ``patch_gpu_worker``.
  * AROUND-wrap ``Scheduler.__init__`` so that AFTER the upstream ``__init__``
    builds ``self.request_handlers`` we ``.update(...)`` it with the 9 fork
    entries. The dict is keyed by the SAME request classes the UniRL
    adapter sends: the 8 ``*ReqInput`` from ``_patches.io_struct`` and
    ``SetLoraFromTensorsReq`` from ``_patches.lora_req`` (the single definition
    sites), so ``type(req)`` dispatch matches.
  * AROUND-wrap ``_handle_generation`` and ``_handle_update_weights_from_disk``
    to PREPEND the fork's sleep/dirty guards, then delegate to the upstream
    body with the original args/kwargs forwarded untouched (upstream
    ``_handle_generation`` takes a keyword-only ``allow_dynamic_batching``).

Idempotent via sentinel guards. Import-safe (sglang imported inside the fn).
"""

from __future__ import annotations

from typing import Any, Callable, List

_INIT_SENTINEL = "_unirl_request_handlers"
_GEN_SENTINEL = "_unirl_sleep_dirty_guard"
_DISK_SENTINEL = "_unirl_sleep_dirty_guard"
_HANDLERS_SENTINEL = "_unirl_rl_handlers"


def patch_scheduler() -> None:
    """Install the fork's RL request handlers + guards on upstream ``Scheduler``."""
    from sglang.multimodal_gen.runtime.managers.scheduler import (
        Scheduler,
        logger,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

    # NOTE: ``logger`` (the scheduler module's logger) is bound here so the
    # verbatim handler bodies below — nested fns whose free ``logger`` resolves
    # to THIS enclosing scope (LEGB) — log through the exact upstream object,
    # keeping log routing identical to the fork.
    # The request classes used to key the dispatch dict. Imported from the
    # UniRL single-definition sites so they are the SAME objects the
    # adapter sends (dispatch is keyed by identity of `type(req)`).
    from unirl.rollout.engine.sglang._patches.io_struct import (
        DestroyWeightsUpdateGroupReqInput,
        EncodePromptReqInput,
        GetWeightsDetailReqInput,
        InitWeightsUpdateGroupReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightsFromDistributedReqInput,
        UpdateWeightsFromTensorReqInput,
    )
    from unirl.rollout.engine.sglang._patches.lora_req import (
        SetLoraFromTensorsReq,
    )

    # --- (1) setattr the 9 handlers + 2 helpers onto Scheduler (verbatim) ---
    if not getattr(Scheduler, _HANDLERS_SENTINEL, False):

        def _clear_dirty_modules(self, target_modules: "list[str] | None") -> None:
            """Clear dirty module tracking after a successful weight update."""
            if not self.worker._dirty_modules:
                return
            if target_modules:
                self.worker._dirty_modules -= set(target_modules)
            else:
                # target_modules=None means all modules were updated
                self.worker._dirty_modules.clear()

        def _handle_set_lora_from_tensors(self, reqs: List[Any]) -> OutputBatch:
            req = reqs[0]
            return self.worker.set_lora_from_tensors(req.lora_nickname, req.lora_tensors, req.target, req.strength)

        def _handle_get_weights_detail(self, reqs: List[Any]) -> OutputBatch:
            """Handle get_weights_detail request — per-param names, shapes, checksums."""
            req = reqs[0]
            details = self.worker.get_weights_detail(module_names=req.module_names)
            return OutputBatch(output=details)

        def _handle_init_weights_update_group(self, reqs: List[Any]) -> OutputBatch:
            req = reqs[0]
            success, message = self.worker.init_weights_update_group(
                master_address=req.master_address,
                master_port=req.master_port,
                rank_offset=req.rank_offset,
                world_size=req.world_size,
                group_name=req.group_name,
                backend=req.backend,
            )
            return OutputBatch(
                output={"success": success, "message": message},
                error=None if success else message,
            )

        def _handle_destroy_weights_update_group(self, reqs: List[Any]) -> OutputBatch:
            req = reqs[0]
            success, message = self.worker.destroy_weights_update_group(
                group_name=req.group_name,
            )
            return OutputBatch(
                output={"success": success, "message": message},
                error=None if success else message,
            )

        def _handle_update_weights_from_tensor(self, reqs: List[Any]) -> OutputBatch:
            req = reqs[0]
            success, message = self.worker.update_weights_from_tensor(
                serialized_named_tensors=req.serialized_named_tensors,
                target_modules=req.target_modules,
                load_format=req.load_format,
                flush_cache=req.flush_cache,
            )
            if success:
                self._clear_dirty_modules(req.target_modules)
            return OutputBatch(
                output={"success": success, "message": message},
                error=None if success else message,
            )

        def _handle_update_weights_from_distributed(self, reqs: List[Any]) -> OutputBatch:
            req = reqs[0]
            success, message = self.worker.update_weights_from_distributed(
                names=req.names,
                dtypes=req.dtypes,
                shapes=req.shapes,
                group_name=req.group_name,
                target_modules=req.target_modules,
                flush_cache=req.flush_cache,
            )
            if success:
                self._clear_dirty_modules(req.target_modules)
            return OutputBatch(
                output={"success": success, "message": message},
                error=None if success else message,
            )

        def _handle_encode_prompt(self, reqs: List[Any]) -> OutputBatch:
            """Handle encode_prompt request for RL workflows."""
            req = reqs[0]
            if self.worker.is_sleeping():
                return OutputBatch(error="Server is sleeping. Call resume_memory_occupation first.")
            if self.worker._dirty_modules:
                return OutputBatch(
                    error=f"Modules {self.worker._dirty_modules} have garbage weights after resume. Update weights first."
                )
            result = self.worker.encode_prompt(prompts=req.prompts)
            if isinstance(result, dict) and "error" in result:
                return OutputBatch(error=result["error"])
            return OutputBatch(output=result)

        def _handle_memory_occupation(
            self,
            tag: str,
            operation_name: str,
            worker_call: "Callable[[], dict[str, Any]]",
        ) -> OutputBatch:
            logger.info(f"[{tag}] {operation_name} on rank={self.gpu_id}")

            try:
                detail = worker_call()
            except Exception as e:
                logger.exception(f"[{tag}] {operation_name} failed on rank={self.gpu_id}")
                detail = {"success": False, "message": str(e)}

            if not isinstance(detail, dict):
                detail = {
                    "success": False,
                    "message": f"invalid worker response: {detail}",
                }

            normalized_detail = dict(detail)
            normalized_detail["success"] = bool(normalized_detail.get("success", False))
            normalized_detail["sleeping"] = self.worker.is_sleeping()
            normalized_detail.setdefault("message", "memory occupation operation finished")
            return OutputBatch(output=normalized_detail)

        def _handle_release_memory_occupation(self, reqs: List[Any]) -> OutputBatch:
            req = reqs[0]
            tags = getattr(req, "tags", None)
            cpu_backup_tags = getattr(req, "cpu_backup_tags", None)
            return self._handle_memory_occupation(
                tag="SLEEP",
                operation_name="handle_release_memory_occupation",
                worker_call=lambda: self.worker.release_memory_occupation(tags=tags, cpu_backup_tags=cpu_backup_tags),
            )

        def _handle_resume_memory_occupation(self, reqs: List[Any]) -> OutputBatch:
            req = reqs[0]
            tags = getattr(req, "tags", None)
            return self._handle_memory_occupation(
                tag="WAKE",
                operation_name="handle_resume_memory_occupation",
                worker_call=lambda: self.worker.resume_memory_occupation(tags=tags),
            )

        Scheduler._clear_dirty_modules = _clear_dirty_modules
        Scheduler._handle_set_lora_from_tensors = _handle_set_lora_from_tensors
        Scheduler._handle_get_weights_detail = _handle_get_weights_detail
        Scheduler._handle_init_weights_update_group = _handle_init_weights_update_group
        Scheduler._handle_destroy_weights_update_group = _handle_destroy_weights_update_group
        Scheduler._handle_update_weights_from_tensor = _handle_update_weights_from_tensor
        Scheduler._handle_update_weights_from_distributed = _handle_update_weights_from_distributed
        Scheduler._handle_encode_prompt = _handle_encode_prompt
        Scheduler._handle_memory_occupation = _handle_memory_occupation
        Scheduler._handle_release_memory_occupation = _handle_release_memory_occupation
        Scheduler._handle_resume_memory_occupation = _handle_resume_memory_occupation
        setattr(Scheduler, _HANDLERS_SENTINEL, True)

    # --- (2) AROUND-wrap __init__: extend request_handlers after upstream ----
    if not getattr(Scheduler.__init__, _INIT_SENTINEL, False):
        _orig_init = Scheduler.__init__

        def __init__(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            # Bound-method references resolve through the setattr'd attrs above.
            self.request_handlers.update(
                {
                    SetLoraFromTensorsReq: self._handle_set_lora_from_tensors,
                    GetWeightsDetailReqInput: self._handle_get_weights_detail,
                    InitWeightsUpdateGroupReqInput: self._handle_init_weights_update_group,
                    DestroyWeightsUpdateGroupReqInput: self._handle_destroy_weights_update_group,
                    UpdateWeightsFromTensorReqInput: self._handle_update_weights_from_tensor,
                    UpdateWeightsFromDistributedReqInput: self._handle_update_weights_from_distributed,
                    ReleaseMemoryOccupationReqInput: self._handle_release_memory_occupation,
                    ResumeMemoryOccupationReqInput: self._handle_resume_memory_occupation,
                    EncodePromptReqInput: self._handle_encode_prompt,
                }
            )

        __init__._unirl_request_handlers = True  # type: ignore[attr-defined]
        Scheduler.__init__ = __init__

    # --- (3) AROUND-wrap _handle_generation: prepend sleep/dirty guards -------
    if not getattr(Scheduler._handle_generation, _GEN_SENTINEL, False):
        _orig_handle_generation = Scheduler._handle_generation

        def _handle_generation(self, *args, **kwargs):
            if self.worker.is_sleeping():
                return OutputBatch(error="Server is sleeping. Call resume_memory_occupation first.")
            if self.worker._dirty_modules:
                return OutputBatch(
                    error=f"Modules {self.worker._dirty_modules} have garbage weights after resume. Update weights first."
                )
            return _orig_handle_generation(self, *args, **kwargs)

        _handle_generation._unirl_sleep_dirty_guard = True  # type: ignore[attr-defined]
        Scheduler._handle_generation = _handle_generation

    # --- (4) AROUND-wrap _handle_update_weights_from_disk: guards + clear -----
    # The fork's disk handler is identical to upstream's body EXCEPT it adds the
    # `_clear_dirty_modules` call on success. Upstream does not clear, so wrap to
    # add it after a successful disk update (parse the success flag off the
    # returned OutputBatch.error, which upstream sets to None on success).
    if not getattr(Scheduler._handle_update_weights_from_disk, _DISK_SENTINEL, False):
        _orig_handle_disk = Scheduler._handle_update_weights_from_disk

        def _handle_update_weights_from_disk(self, reqs: List[Any]) -> OutputBatch:
            out = _orig_handle_disk(self, reqs)
            if out.error is None:
                self._clear_dirty_modules(reqs[0].target_modules)
            return out

        _handle_update_weights_from_disk._unirl_sleep_dirty_guard = True  # type: ignore[attr-defined]
        Scheduler._handle_update_weights_from_disk = _handle_update_weights_from_disk
