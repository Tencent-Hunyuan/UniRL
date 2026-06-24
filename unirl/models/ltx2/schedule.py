"""LTX-2 FlowMatch schedule policy — constant-μ dynamic exponential shift.

LTX-2 uses ``use_dynamic_shifting`` but, unlike Flux/Qwen-Image, its μ is
NOT resolution-dependent: the diffusers reference pins the ``image_seq_len``
argument of ``calculate_shift`` to ``max_image_seq_len`` (see
``diffusers/pipelines/ltx2/pipeline_ltx2.py``::

    mu = calculate_shift(
        self.scheduler.config.get("max_image_seq_len", 4096),   # <- pinned to MAX
        self.scheduler.config.get("base_image_seq_len", 1024),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.95),
        self.scheduler.config.get("max_shift", 2.05),
    )

Since ``calculate_shift`` is a linear interpolation between ``base_shift`` and
``max_shift`` over ``[base_image_seq_len, max_image_seq_len]``, evaluating it
at ``image_seq_len == max_image_seq_len`` always yields ``max_shift``. So
μ ≡ ``max_shift`` (= 2.05 by default), a CONSTANT, applied via the exponential
time-shift ``σ = e^μ / (e^μ + (1/t - 1))``.

The previous LTX-2 path had no ``build_schedule_policy``, so the engine fell
back to a static ``shift=1.0`` (identity) schedule. That under-resolves the
trajectory (too little time spent at high noise) → blurry/garbled frames, even
though GRPO reward still rises (rollout and replay share the same σ, so the
log-prob ratio stays valid). This policy restores the diffusers schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

from unirl.sde.runtime import FlowMatchSchedulePolicy

# LTX-2 scheduler_config.json defaults (diffusers ``.get`` fallbacks).
_LTX2_BASE_SHIFT = 0.95
_LTX2_MAX_SHIFT = 2.05
_LTX2_BASE_IMAGE_SEQ_LEN = 1024
_LTX2_MAX_IMAGE_SEQ_LEN = 4096
# LTX-2 stretches the σ grid so its terminal value is 0.1 (scheduler_config
# ``shift_terminal``); the diffusers scheduler applies this in set_timesteps.
_LTX2_SHIFT_TERMINAL = 0.1


@dataclass
class LTX2SchedulePolicy(FlowMatchSchedulePolicy):
    """Constant-μ (== ``max_shift``) exponential-shift policy for LTX-2.

    Overrides only :meth:`compute_mu` to return ``max_shift`` regardless of
    the request resolution, matching diffusers' ``image_seq_len``-pinned
    ``calculate_shift`` call. The base-class :meth:`compute_sigma` then builds
    the σ grid with the diffusers exponential time-shift.
    """

    def compute_mu(self, image_seq_len: int, num_inference_steps: int) -> float:
        # image_seq_len is intentionally ignored: diffusers pins it to
        # max_image_seq_len, so the linear interpolation degenerates to max_shift.
        return float(self.max_shift)


def build_ltx2_schedule_policy(shift: float = 1.0) -> LTX2SchedulePolicy:
    """Build the LTX-2 constant-μ exponential-shift schedule policy."""
    return LTX2SchedulePolicy(
        shift=float(shift),
        use_dynamic_shifting=True,
        base_shift=_LTX2_BASE_SHIFT,
        max_shift=_LTX2_MAX_SHIFT,
        base_image_seq_len=_LTX2_BASE_IMAGE_SEQ_LEN,
        max_image_seq_len=_LTX2_MAX_IMAGE_SEQ_LEN,
        time_shift_type="exponential",
        shift_terminal=_LTX2_SHIFT_TERMINAL,
    )


__all__ = ["LTX2SchedulePolicy", "build_ltx2_schedule_policy"]
