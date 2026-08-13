"""LTX-2 FlowMatch schedule policy — constant-μ dynamic exponential shift."""

from __future__ import annotations

from dataclasses import dataclass

from unirl.sde.runtime import FlowMatchSchedulePolicy

_LTX2_BASE_SHIFT = 0.95
_LTX2_MAX_SHIFT = 2.05
_LTX2_BASE_IMAGE_SEQ_LEN = 1024
_LTX2_MAX_IMAGE_SEQ_LEN = 4096
_LTX2_SHIFT_TERMINAL = 0.1


@dataclass
class LTX2SchedulePolicy(FlowMatchSchedulePolicy):
    """Constant-μ (== ``max_shift``) exponential-shift policy for LTX-2."""

    def compute_mu(self, image_seq_len: int, num_inference_steps: int) -> float:
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
