"""Use driver-provided σ as-is in FlowMatch ``set_timesteps``."""

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
                self.stretch_shift_to_terminal = lambda t: t
            if dynamic:
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
