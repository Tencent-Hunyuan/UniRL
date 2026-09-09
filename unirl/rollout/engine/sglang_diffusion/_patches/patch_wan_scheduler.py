"""Use the flow-match (single-step SDE) scheduler for WAN rollout, not UniPC."""

from __future__ import annotations


def patch_wan_scheduler() -> None:
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from sglang.multimodal_gen.runtime.pipelines.wan_pipeline import WanPipeline

    orig = WanPipeline.initialize_pipeline
    if getattr(orig, "_unirl_flowmatch_scheduler", False):
        return

    def initialize_pipeline(self, server_args) -> None:
        orig(self, server_args)
        flow_shift = server_args.pipeline_config.flow_shift
        if flow_shift is None:
            flow_shift = 1.0
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

    initialize_pipeline._unirl_flowmatch_scheduler = True  # type: ignore[attr-defined]
    WanPipeline.initialize_pipeline = initialize_pipeline

    step_orig = FlowMatchEulerDiscreteScheduler.step
    if not getattr(step_orig, "_unirl_drop_eta", False):

        def step(self, *args, **kwargs):
            kwargs.pop("eta", None)
            return step_orig(self, *args, **kwargs)

        step._unirl_drop_eta = True  # type: ignore[attr-defined]
        FlowMatchEulerDiscreteScheduler.step = step
