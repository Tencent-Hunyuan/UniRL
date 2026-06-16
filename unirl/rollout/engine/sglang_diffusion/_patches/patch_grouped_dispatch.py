"""Grouped-stage dispatch bridge for sglang v0.5.12.post1 (LIN-365).

post1 shipped an UNFINISHED grouped-pipeline refactor. ``ParallelExecutor.
_execute_stages`` (parallel_executor.py) accepts a ``run_stage`` callback -- for
the grouped path ``execute_group`` passes
``lambda stage, batch: stage.run_grouped_requests(batch, server_args)`` -- but
the body never calls it; every paradigm branch runs ``batch = stage(batch,
server_args)`` (i.e. ``PipelineStage.__call__``) directly. So a GRPO grouped
request (a *list* of K Reqs, from ``forward_batch``) reaches ``__call__`` ->
``verify_input`` -> ``batch.seed`` -> ``AttributeError: 'list' object has no
attribute 'seed'`` on the first rollout.

Upstream finished the refactor in 3142278c5 (2026-05-26, "layerwise NVTX
markers"), which post1 (2026-05-23) predates by 3 days; the verified base
2fc548f25 has it, where ``_execute_stages`` routes every stage through the
``run_grouped_requests`` callback. The default ``run_grouped_requests`` lives on
``StageDedupMixin`` (the base of every ``PipelineStage`` -- confirmed present in
post1) and is just ``[self(b, server_args) for b in batches]`` -- per-request
fan-out -- with dedup / latent-preparation stages overriding it for stage-local
reuse.

Rather than backport the ~1000-line NVTX commit, AROUND-wrap
``PipelineStage.__call__`` so a *list* batch dispatches to
``run_grouped_requests`` -- exactly what 2fc's ``_execute_stages`` does. On post1
this bridges ``stage(list)``; on any sglang >= 3142278c5 (incl. 2fc and the
forthcoming 0.5.13 tag) ``_execute_stages`` calls ``run_grouped_requests``
directly and never hands a list to ``__call__``, so the wrap is a no-op --
forward-safe and self-removing once the pin moves to a release with the fix.

Idempotent; AROUND-wrap only -- no sglang source edits.
"""

from __future__ import annotations


def patch_grouped_dispatch() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
        PipelineStage,
    )

    orig_call = PipelineStage.__call__
    if getattr(orig_call, "_unirl_grouped_call", False):
        return

    def __call__(self, batch, server_args):
        # post1's ParallelExecutor._execute_stages calls stage(batch) instead of
        # honoring its run_grouped_requests dispatch; when batch is a grouped
        # list of Reqs, route it through run_grouped_requests (StageDedupMixin)
        # the way upstream >= 3142278c5 / 2fc does. Single-Req calls are
        # untouched, and newer sglang never passes a list here, so this is a
        # no-op there.
        if isinstance(batch, list):
            return self.run_grouped_requests(batch, server_args)
        return orig_call(self, batch, server_args)

    __call__._unirl_grouped_call = True  # type: ignore[attr-defined]
    PipelineStage.__call__ = __call__
