"""Register UniRL-only scheduler verbs around SGLang's native RL handlers."""

from __future__ import annotations

from typing import Any, List

_INIT_SENTINEL = "_unirl_request_handlers"
_HANDLERS_SENTINEL = "_unirl_rl_handlers"


def patch_scheduler() -> None:
    """Add distributed sync while preserving v0.5.19 handlers."""
    _patch_scheduler_client_fanout()

    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

    from unirl.rollout.engine.sglang_diffusion._patches.io_struct import (
        DestroyWeightsUpdateGroupReqInput,
        InitWeightsUpdateGroupReqInput,
        UpdateWeightsFromDistributedReqInput,
    )

    if not getattr(Scheduler, _HANDLERS_SENTINEL, False):

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
            return OutputBatch(
                output={"success": success, "message": message},
                error=None if success else message,
            )

        Scheduler._handle_init_weights_update_group = _handle_init_weights_update_group
        Scheduler._handle_destroy_weights_update_group = _handle_destroy_weights_update_group
        Scheduler._handle_update_weights_from_distributed = _handle_update_weights_from_distributed
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
                }
            )

        __init__._unirl_request_handlers = True  # type: ignore[attr-defined]
        Scheduler.__init__ = __init__


def _patch_scheduler_client_fanout() -> None:
    """Treat any DP replica's explicit failure payload as a fanout failure."""
    import sglang.multimodal_gen.runtime.scheduler_client as scheduler_client
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

    orig = scheduler_client._merge_fanout_results
    if getattr(orig, "_unirl_failure_payload", False):
        return

    def _merge_fanout_results(results):
        for result in results:
            if not isinstance(result, OutputBatch):
                continue
            if result.error:
                return result
            output = result.output
            if isinstance(output, dict) and not bool(output.get("success", True)):
                message = str(output.get("message", "replica operation failed"))
                return OutputBatch(output=output, error=message)
        return orig(results)

    _merge_fanout_results._unirl_failure_payload = True  # type: ignore[attr-defined]
    scheduler_client._merge_fanout_results = _merge_fanout_results
