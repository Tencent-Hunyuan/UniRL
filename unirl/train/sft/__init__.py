"""SFT domain package — worker-side supervised track builders.

The losses live in ``unirl/algorithms/sft.py`` (peers of GRPO/FlowGRPO); the
driver is ``unirl/trainer/sft.py``; this package holds only the piece that is
genuinely new to supervision: turning dataset records into stage-ready tracks.
"""

from unirl.train.sft.track_builder import (
    ARSupervisedTrackBuilder,
    DiffusionSupervisedTrackBuilder,
    SupervisedTrackBuilder,
)

__all__ = [
    "ARSupervisedTrackBuilder",
    "DiffusionSupervisedTrackBuilder",
    "SupervisedTrackBuilder",
]
