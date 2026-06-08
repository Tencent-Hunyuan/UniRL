"""Pipeline-level fixes for stock-upstream sglang colocate rollout (LIN-365).

**Grouped-path component_residency_manager.** Upstream
``ComposedPipelineBase.forward`` (the single-request path) sets
``self.executor.component_residency_manager = get_global_component_residency_manager(...)``
before executing (composed_pipeline_base.py:922-926). But the grouped path
``forward_batch`` (used when a request expands to ``num_outputs_per_prompt>1`` --
UniRL GRPO collapses K identical samples per prompt into one nopp=K request)
goes straight to ``executor.execute_group_with_profiling`` WITHOUT that setup, so
``PipelineExecutor.begin_component_residency_request`` dereferences a ``None``
manager -> ``AttributeError: 'NoneType' object has no attribute 'begin_request'``.

The fork never hit this (``component_residency_manager`` is a post-fork upstream
addition). Fix: AROUND-wrap ``forward_batch`` to mirror ``forward``'s manager
setup for the grouped (len>1) case. (len==1 delegates to ``forward``, which sets
it itself, so we only need the len>1 branch.) Surfaced by the SD3 GRPO e2e.

Idempotent; AROUND-wrap only -- no sglang source edits.
"""

from __future__ import annotations


def patch_pipeline() -> None:
    try:
        from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
            get_global_component_residency_manager,
        )
    except ImportError:  # pre-reorg flat layout (<= v0.5.12.post1)
        from sglang.multimodal_gen.runtime.managers.component_manager import (
            get_global_component_residency_manager,
        )
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )

    orig = ComposedPipelineBase.__dict__.get("forward_batch")
    if orig is None:
        raise AttributeError("ComposedPipelineBase.forward_batch missing upstream")
    if getattr(orig, "_unirl_grouped_residency", False):
        return

    def forward_batch(self, batches, server_args):
        # Grouped path (len>1) skips the residency-manager setup that `forward`
        # does; mirror it so executor.execute_group_with_profiling has a manager.
        if len(batches) > 1:
            self.component_residency_manager = get_global_component_residency_manager(self, server_args)
            self.executor.component_residency_manager = self.component_residency_manager
        return orig(self, batches, server_args)

    forward_batch._unirl_grouped_residency = True  # type: ignore[attr-defined]
    ComposedPipelineBase.forward_batch = forward_batch

    # The grouped executor passes a LIST of batches to the residency hook, but
    # ComponentResidencyManager.begin_request reads ``batch.is_warmup`` (a single
    # Req) -> ``'list' object has no attribute 'is_warmup'``. All grouped batches
    # share is_warmup (same request group), and before_stage/after_stage don't
    # touch batch, so unwrap to a representative batch[0]. AROUND-wrap the base
    # PipelineExecutor hook so all executor subclasses inherit it.
    from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
        PipelineExecutor,
    )

    ex_orig = PipelineExecutor.__dict__.get("begin_component_residency_request")
    if ex_orig is not None and not getattr(ex_orig, "_unirl_residency_list", False):

        def begin_component_residency_request(self, stages, batch, server_args):
            rep = batch[0] if isinstance(batch, (list, tuple)) else batch
            return ex_orig(self, stages, rep, server_args)

        begin_component_residency_request._unirl_residency_list = True  # type: ignore[attr-defined]
        PipelineExecutor.begin_component_residency_request = begin_component_residency_request
