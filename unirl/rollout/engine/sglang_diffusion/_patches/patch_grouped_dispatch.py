"""Grouped-stage dispatch bridge for sglang v0.5.12.post1."""

from __future__ import annotations


def patch_grouped_dispatch() -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
        PipelineStage,
    )

    orig_call = PipelineStage.__call__
    if getattr(orig_call, "_unirl_grouped_call", False):
        return

    def __call__(self, batch, server_args):
        if isinstance(batch, list):
            return self.run_grouped_requests(batch, server_args)
        return orig_call(self, batch, server_args)

    __call__._unirl_grouped_call = True  # type: ignore[attr-defined]
    PipelineStage.__call__ = __call__
