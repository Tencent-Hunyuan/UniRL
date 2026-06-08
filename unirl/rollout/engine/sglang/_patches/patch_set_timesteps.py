"""Use driver-provided σ as-is in FlowMatch ``set_timesteps`` (gap, LIN-365).

UniRL pins the SDE σ schedule (GRPO σ-consistency: the *same* schedule must
drive rollout and replay) and ships it via ``SamplingParams.sigmas`` ->
``batch.sigmas`` -> ``TimestepPreparationStage`` ->
``scheduler.set_timesteps(sigmas=...)``. But upstream's
``FlowMatchEulerDiscreteScheduler.set_timesteps`` ALWAYS applies a shift to
provided sigmas, double-shifting the driver's already-final schedule
(``sigma_verify`` then fails -- "Worker did NOT use the σ we sent"). There are
two shift code paths inside set_timesteps and we neutralize both:

1. **Static resolution shift** (used by SD3 etc., ``shift!=1``):
   ``sigmas = shift*σ/(1+(shift-1)*σ)``. This is the identity at ``shift==1``
   (``1*σ/(1+0*σ) == σ``), so we run the stock method with ``_shift``
   temporarily set to ``1.0``.

2. **Dynamic mu shift** (used by FLUX-family / Klein,
   ``use_dynamic_shifting=True``): ``sigmas = time_shift(mu, 1.0, sigmas)``,
   where ``time_shift(mu, t, x) = exp(mu) / (exp(mu) + (1/x - 1)**t)``. With
   ``mu == 0`` this collapses to ``1/(1/x) == x`` -- also the identity. So we
   force ``mu = 0.0`` for the call (the scheduler still requires a non-None mu
   here, which is why we pass 0.0 instead of None).

The driver applied its own Klein-specific empirical-mu shift to the sigmas
before shipping them, so the worker MUST NOT re-shift; this neutralization is
how we keep GRPO's σ-consistency contract.

Robust to upstream changes elsewhere in ``set_timesteps``
(karras/beta/stretch_shift_to_terminal) because we still delegate to ``orig``.
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
        # Neutralize whichever of the two shift paths the scheduler is
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
        saved_shift = self.shift
        try:
            self.set_shift(1.0)
            # time_shift(mu=0, sigma=1.0, t=x) == x; neutralizes dynamic path.
            effective_mu = 0.0 if dynamic else mu
            return orig(
                self,
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=effective_mu,
                timesteps=timesteps,
            )
        finally:
            self.set_shift(saved_shift)

    set_timesteps._unirl_external_sigmas = True  # type: ignore[attr-defined]
    FlowMatchEulerDiscreteScheduler.set_timesteps = set_timesteps
