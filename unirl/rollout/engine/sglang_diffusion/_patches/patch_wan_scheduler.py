"""Use the flow-match (single-step SDE) scheduler for WAN rollout, not UniPC.

``sglang``'s ``WanPipeline.initialize_pipeline`` hardcodes
``FlowUniPCMultistepScheduler`` (the Wan2.1 official sampler, a multistep ODE
solver). That class does NOT inherit ``SchedulerRLMixin`` — it has no
``_rollout_variance_noise`` / SDE-step log-prob path — so the GRPO rollout
(``rollout=True``, ``rollout_sde_type="sde"``) cannot run on it; the first thing
that breaks is ``set_timesteps`` rejecting the driver's externally pinned σ list
with ``assert isinstance(sigmas, np.ndarray)``.

The other diffusion families (SD3, FLUX, Qwen-Image) all roll out on
``FlowMatchEulerDiscreteScheduler``, which DOES carry the RL mixin and whose
``set_timesteps`` accepts an external σ schedule (coerced to ndarray, with the
shift neutralized by ``patch_set_timesteps`` so the driver-pinned schedule
survives the σ-consistency check). WAN's own DMD pipeline
(``wan_dmd_pipeline.py``) already builds exactly this scheduler the same way, so
swapping is the sanctioned shape for WAN, not a divergence.

This AROUND-wraps ``WanPipeline.initialize_pipeline`` to replace the scheduler
with ``FlowMatchEulerDiscreteScheduler(shift=flow_shift)`` after the stock init
(which only sets the scheduler module). Single-step flow-match transitions also
match what the trainer replays trainside (FlowSDE), keeping GRPO's rollout↔replay
consistency. Idempotent via a sentinel; import-safe (sglang imported inside).

Depends on ``patch_set_timesteps`` also being applied: it neutralizes the shift
so the driver-pinned σ schedule survives ``set_timesteps``' consistency check.
Both are installed together by the patch suite's ``hijack()``.
"""

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
        # Stock init builds the (UniPC) scheduler + nothing else; run it so any
        # future additions survive, then overwrite the scheduler module.
        orig(self, server_args)
        # ``flow_shift`` is a pipeline_config field typed ``float | None``: WAN
        # variants set a concrete value (3/5/8/…), but the base default is None,
        # which is a LEGAL config meaning "use the scheduler's own default shift".
        # We cannot forward None (the scheduler computes ``shift * sigmas / …`` and
        # ``None * sigmas`` would raise), so map None to the scheduler default (1.0).
        flow_shift = server_args.pipeline_config.flow_shift
        if flow_shift is None:
            flow_shift = 1.0  # FlowMatchEulerDiscreteScheduler's own default shift
        self.modules["scheduler"] = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

    initialize_pipeline._unirl_flowmatch_scheduler = True  # type: ignore[attr-defined]
    WanPipeline.initialize_pipeline = initialize_pipeline

    # WAN 2.2-A14B dual-expert: sglang's dual-transformer denoising path wraps the
    # scheduler/cache so that ``DenoisingStage.prepare_extra_func_kwargs`` inspects a
    # step with a generic ``**kwargs`` signature. That helper only FILTERS its
    # {generator, eta, batch} dict when the inspected step has NO var-kwargs, so with
    # the wrapper it stops filtering and leaks the diffusers-style ``eta`` (= batch.eta)
    # into the call. The RL-mixin ``step`` has no ``eta`` param — SDE strength comes from
    # ``batch`` + the driver-pinned σ schedule, not ``eta`` — and raises
    # "unexpected keyword argument 'eta'". (wan21 single-transformer keeps the explicit
    # step signature, so eta is filtered and never reaches here.) Drop the stray ``eta``
    # so the dual-expert path matches wan21 semantics. The wrapper itself takes
    # ``**kwargs`` and pops only ``eta`` → identical {generator, batch} reaches the real
    # step for every family; a strict no-op wherever eta was already filtered.
    step_orig = FlowMatchEulerDiscreteScheduler.step
    if not getattr(step_orig, "_unirl_drop_eta", False):

        def step(self, *args, **kwargs):
            kwargs.pop("eta", None)
            return step_orig(self, *args, **kwargs)

        step._unirl_drop_eta = True  # type: ignore[attr-defined]
        FlowMatchEulerDiscreteScheduler.step = step
