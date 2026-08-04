"""Rollout managers (LIN-693). See ``unirl/rollout/README.md``.

The layer between the trainer and the rollout engines: admission, acceptance and
disposal over time. Holds no model. Named *manager* because this tree already calls
the LR scheduler, the diffusion noise scheduler and SGLang's subprocesses schedulers.
"""

from unirl.rollout.manager.agentic import AgenticManager
from unirl.rollout.manager.batch import BatchManager, InflightPool, launch_ceiling
from unirl.rollout.manager.buffers import PendingGroups, VersionedBuffer, root_of

__all__ = [
    "AgenticManager",
    "BatchManager",
    "InflightPool",
    "PendingGroups",
    "VersionedBuffer",
    "launch_ceiling",
    "root_of",
]
