"""FLUX.2-klein empirical-mu schedule policy."""

from __future__ import annotations

from dataclasses import dataclass

from unirl.sde.runtime import FlowMatchSchedulePolicy

from .flux2_klein_utils import compute_empirical_mu


@dataclass
class Flux2KleinSchedulePolicy(FlowMatchSchedulePolicy):
    """Policy subclass with FLUX.2-klein empirical-mu shifting."""

    def compute_mu(self, image_seq_len: int, num_inference_steps: int) -> float:
        return compute_empirical_mu(image_seq_len, num_inference_steps)


def build_flux2_klein_schedule_policy(shift: float = 1.0) -> Flux2KleinSchedulePolicy:
    """Build the shared FLUX.2-klein empirical-mu schedule policy."""

    return Flux2KleinSchedulePolicy(
        shift=float(shift),
        use_dynamic_shifting=True,
        vae_scale_factor=8,
        patch_size=2,
        time_shift_type="exponential",
    )


__all__ = ["Flux2KleinSchedulePolicy", "build_flux2_klein_schedule_policy"]
