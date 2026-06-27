"""Use the RL-capable flow-match scheduler for LTX-2 rollout.

sglang's LTX-2 pipeline (``_BaseLTX2Pipeline.initialize_pipeline``) builds
``LTX2FlowMatchScheduler``, which subclasses the **diffusers** stock
``FlowMatchEulerDiscreteScheduler`` (``from diffusers import ...`` at the top of
``ltx_2_pipeline.py``). That class does NOT inherit ``SchedulerRLMixin``, so the
GRPO rollout (``rollout=True``, ``rollout_sde_type="sde"``) raises
``Scheduler <LTX2FlowMatchScheduler> does not support rollout`` the moment the
denoising mixin checks ``isinstance(self.scheduler, SchedulerRLMixin)``.

Swap it for sglang's OWN ``FlowMatchEulerDiscreteScheduler``
(``runtime.models.schedulers.scheduling_flow_match_euler_discrete``), which DOES
carry the RL mixin (the SDE-step log-prob / variance-noise path) and whose
``set_timesteps`` accepts the driver's externally pinned σ schedule (shift
neutralized by ``patch_set_timesteps``). This is the exact shape WAN uses
(``patch_wan_scheduler``); single-step flow-match transitions also match what the
trainer replays trainside (``FlowSDEStrategy``), keeping GRPO rollout↔replay
consistency.

AROUND-wraps ``_BaseLTX2Pipeline.initialize_pipeline`` (T2V ``LTX2Pipeline``
inherits it; the two-stage HQ pipeline overrides it and is out of scope here),
rebuilding the swapped scheduler ``from_config`` so the LTX-2 shift/config
survive. Idempotent via a sentinel; import-safe (sglang imported inside, guarded
by ``_safe_apply`` in ``hijack``).
"""

from __future__ import annotations


def patch_ltx2_scheduler() -> None:
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import _BaseLTX2Pipeline

    orig = _BaseLTX2Pipeline.initialize_pipeline
    if getattr(orig, "_unirl_flowmatch_scheduler", False):
        return

    def initialize_pipeline(self, server_args) -> None:
        # Stock init builds the (diffusers-based) LTX2FlowMatchScheduler; run it so
        # any future additions survive, then overwrite the scheduler module with the
        # RL-mixin-carrying sglang scheduler, preserving the LTX-2 config (shift, …).
        orig(self, server_args)
        sched = self.get_module("scheduler")
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(sched.config)

    initialize_pipeline._unirl_flowmatch_scheduler = True  # type: ignore[attr-defined]
    _BaseLTX2Pipeline.initialize_pipeline = initialize_pipeline
