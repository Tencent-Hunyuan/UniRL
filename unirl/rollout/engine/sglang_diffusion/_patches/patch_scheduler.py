"""Register UniRL-only scheduler verbs around SGLang's native RL handlers."""

from __future__ import annotations

from typing import Any, Callable, List

_INIT_SENTINEL = "_unirl_request_handlers"
_GEN_SENTINEL = "_unirl_sleep_dirty_guard"
_DISK_SENTINEL = "_unirl_sleep_dirty_guard"
_TENSOR_SENTINEL = "_unirl_dirty_guard"
_HANDLERS_SENTINEL = "_unirl_rl_handlers"


def patch_scheduler() -> None:
    """Add distributed sync and tagged sleep while preserving v0.5.19 handlers."""
    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler, logger
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

    from unirl.rollout.engine.sglang_diffusion._patches.io_struct import (
        DestroyWeightsUpdateGroupReqInput,
        InitWeightsUpdateGroupReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightsFromDistributedReqInput,
    )

    if not getattr(Scheduler, _HANDLERS_SENTINEL, False):

        def _clear_dirty_modules(
            self,
            target_modules: list[str] | None,
        ) -> None:
            dirty_modules = self.worker._dirty_modules
            if target_modules:
                dirty_modules.difference_update(target_modules)
            else:
                dirty_modules.clear()

        def _handle_init_weights_update_group(
            self,
            reqs: List[Any],
        ) -> OutputBatch:
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

        def _handle_destroy_weights_update_group(
            self,
            reqs: List[Any],
        ) -> OutputBatch:
            req = reqs[0]
            success, message = self.worker.destroy_weights_update_group(
                group_name=req.group_name,
            )
            return OutputBatch(
                output={"success": success, "message": message},
                error=None if success else message,
            )

        def _handle_update_weights_from_distributed(
            self,
            reqs: List[Any],
        ) -> OutputBatch:
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

        def _handle_memory_occupation(
            self,
            tag: str,
            operation_name: str,
            worker_call: Callable[[], dict[str, Any]],
        ) -> OutputBatch:
            logger.info("[%s] %s on rank=%s", tag, operation_name, self.gpu_id)
            try:
                detail = worker_call()
            except Exception as exc:
                logger.exception(
                    "[%s] %s failed on rank=%s",
                    tag,
                    operation_name,
                    self.gpu_id,
                )
                detail = {"success": False, "message": str(exc)}

            normalized_detail = dict(detail)
            normalized_detail["success"] = bool(normalized_detail.get("success", False))
            normalized_detail["sleeping"] = self.worker.is_sleeping()
            normalized_detail.setdefault(
                "message",
                "memory occupation operation finished",
            )
            return OutputBatch(output=normalized_detail)

        def _handle_release_memory_occupation(
            self,
            reqs: List[Any],
        ) -> OutputBatch:
            req = reqs[0]
            return self._handle_memory_occupation(
                tag="SLEEP",
                operation_name="handle_release_memory_occupation",
                worker_call=lambda: self.worker.release_memory_occupation(
                    tags=getattr(req, "tags", None),
                    cpu_backup_tags=getattr(req, "cpu_backup_tags", None),
                ),
            )

        def _handle_resume_memory_occupation(
            self,
            reqs: List[Any],
        ) -> OutputBatch:
            req = reqs[0]
            return self._handle_memory_occupation(
                tag="WAKE",
                operation_name="handle_resume_memory_occupation",
                worker_call=lambda: self.worker.resume_memory_occupation(
                    tags=getattr(req, "tags", None),
                ),
            )

        Scheduler._clear_dirty_modules = _clear_dirty_modules
        Scheduler._handle_init_weights_update_group = _handle_init_weights_update_group
        Scheduler._handle_destroy_weights_update_group = _handle_destroy_weights_update_group
        Scheduler._handle_update_weights_from_distributed = _handle_update_weights_from_distributed
        Scheduler._handle_memory_occupation = _handle_memory_occupation
        Scheduler._handle_release_memory_occupation = _handle_release_memory_occupation
        Scheduler._handle_resume_memory_occupation = _handle_resume_memory_occupation
        setattr(Scheduler, _HANDLERS_SENTINEL, True)

    if not getattr(Scheduler.__init__, _INIT_SENTINEL, False):
        orig_init = Scheduler.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.request_handlers.update(
                {
                    InitWeightsUpdateGroupReqInput: (self._handle_init_weights_update_group),
                    DestroyWeightsUpdateGroupReqInput: (self._handle_destroy_weights_update_group),
                    UpdateWeightsFromDistributedReqInput: (self._handle_update_weights_from_distributed),
                    ReleaseMemoryOccupationReqInput: (self._handle_release_memory_occupation),
                    ResumeMemoryOccupationReqInput: (self._handle_resume_memory_occupation),
                }
            )

        __init__._unirl_request_handlers = True  # type: ignore[attr-defined]
        Scheduler.__init__ = __init__

    if not getattr(Scheduler._handle_generation, _GEN_SENTINEL, False):
        orig_handle_generation = Scheduler._handle_generation

        def _handle_generation(self, *args, **kwargs):
            if self.worker.is_sleeping():
                return OutputBatch(error="Server is sleeping. Call resume_memory_occupation first.")
            if self.worker._dirty_modules:
                return OutputBatch(
                    error=(
                        f"Modules {self.worker._dirty_modules} have garbage weights after resume. Update weights first."
                    )
                )
            return orig_handle_generation(self, *args, **kwargs)

        _handle_generation._unirl_sleep_dirty_guard = True  # type: ignore[attr-defined]
        Scheduler._handle_generation = _handle_generation

    if not getattr(Scheduler._handle_update_weights_from_disk, _DISK_SENTINEL, False):
        orig_handle_disk = Scheduler._handle_update_weights_from_disk

        def _handle_update_weights_from_disk(
            self,
            reqs: List[Any],
        ) -> OutputBatch:
            output = orig_handle_disk(self, reqs)
            if output.error is None:
                self._clear_dirty_modules(reqs[0].target_modules)
            return output

        _handle_update_weights_from_disk._unirl_sleep_dirty_guard = True  # type: ignore[attr-defined]
        Scheduler._handle_update_weights_from_disk = _handle_update_weights_from_disk

    if not getattr(Scheduler._handle_update_weights_from_tensor, _TENSOR_SENTINEL, False):
        orig_handle_tensor = Scheduler._handle_update_weights_from_tensor

        def _handle_update_weights_from_tensor(
            self,
            reqs: List[Any],
        ) -> OutputBatch:
            output = orig_handle_tensor(self, reqs)
            if output.error is None:
                self._clear_dirty_modules(reqs[0].target_modules)
            return output

        _handle_update_weights_from_tensor._unirl_dirty_guard = True  # type: ignore[attr-defined]
        Scheduler._handle_update_weights_from_tensor = _handle_update_weights_from_tensor
