"""Use driver-provided σ as-is in FlowMatch ``set_timesteps`` (gap, LIN-365).

UniRL pins the SDE σ schedule (GRPO σ-consistency: the *same* schedule must
drive rollout and replay) and ships it via ``SamplingParams.sigmas`` ->
``batch.sigmas`` -> ``TimestepPreparationStage`` ->
``scheduler.set_timesteps(sigmas=...)``. But upstream's
``FlowMatchEulerDiscreteScheduler.set_timesteps`` ALWAYS applies a shift to
provided sigmas, double-shifting the driver's already-final schedule
(``sigma_verify`` then fails -- "Worker did NOT use the σ we sent"). There are
three schedule-mutation code paths inside set_timesteps and we neutralize all:

1. **Static resolution shift** (used by SD3 etc., ``shift!=1``):
   ``sigmas = shift*σ/(1+(shift-1)*σ)``. This is the identity at ``shift==1``
   (``1*σ/(1+0*σ) == σ``), so we run the stock method with ``_shift``
   temporarily set to ``1.0``.

2. **Dynamic mu shift** (used by FLUX-family / Klein,
   ``use_dynamic_shifting=True``): ``sigmas = time_shift(mu, 1.0, sigmas)``.
   Neutralized by binding ``time_shift`` to identity for the delegated call
   (instance attribute shadows the class method; ``del`` restores), exactly like
   the terminal-stretch path below. This is correct for BOTH
   ``time_shift_type="exponential"`` and ``"linear"``; the older ``mu = 0.0``
   trick was the identity only for the exponential form and would zero the whole
   schedule under ``linear``. The driver already baked μ into the sigmas.

3. **Terminal stretch** (used by Qwen-Image, ``shift_terminal: 0.02``;
   SD3/Flux ship ``null``): ``stretch_shift_to_terminal`` rescales the whole
   schedule to terminate at ``shift_terminal`` -- gated only on the config
   value, NOT on external sigmas, so it would mutate the driver's final σ just
   like the two shift paths. Neutralized by temporarily binding the instance
   method to identity for the delegated call (instance attribute shadows the
   class method; ``del`` restores).

The driver applied its own model-specific schedule transforms (Klein
empirical-mu shift, Qwen-Image terminal stretch -- ``FlowMatchSchedulePolicy``
``shift_terminal``) to the sigmas before shipping them, so the worker MUST NOT
re-apply any of them; this neutralization is how we keep GRPO's σ-consistency
contract.

Robust to upstream changes elsewhere in ``set_timesteps`` (karras/beta)
because we still delegate to ``orig``.
"""

from __future__ import annotations


def patch_set_timesteps() -> None:
    from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )

    orig = FlowMatchEulerDiscreteScheduler.set_timesteps
    if getattr(orig, "_unirl_external_sigmas", False):
        return

    def set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
    ):
        # External (driver-pinned) sigmas are already the final schedule.
        # Neutralize whichever schedule-mutation paths the scheduler is
        # configured for so the driver's sigmas pass through unchanged.
        if sigmas is None:
            return orig(
                self,
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                timesteps=timesteps,
            )

        cfg = getattr(self, "config", None)
        dynamic = bool(getattr(cfg, "use_dynamic_shifting", False))
        stretches = bool(getattr(cfg, "shift_terminal", None))
        saved_shift = self.shift
        try:
            self.set_shift(1.0)
            if stretches:
                # Identity instance-binding shadows the class method for this
                # call only; the driver already baked the terminal stretch in.
                self.stretch_shift_to_terminal = lambda t: t
            if dynamic:
                # Same identity-shadow trick for the mu-shift: correct for BOTH
                # exponential and linear time_shift_type (mu=0.0 was the identity
                # only for exponential and would zero the schedule under linear).
                # The driver already baked μ into the sigmas.
                self.time_shift = lambda mu, sigma, t: t
            return orig(
                self,
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                timesteps=timesteps,
            )
        finally:
            self.set_shift(saved_shift)
            if stretches:
                del self.stretch_shift_to_terminal
            if dynamic:
                del self.time_shift

    set_timesteps._unirl_external_sigmas = True  # type: ignore[attr-defined]
    FlowMatchEulerDiscreteScheduler.set_timesteps = set_timesteps
