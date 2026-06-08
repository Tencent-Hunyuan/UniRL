"""Full base-weight sync handlers for the v2 trainer.

- ``NCCLWeightSync``   — separate slabs (cross-node capable).
- ``TensorWeightSync`` — colocate, serialized-tensor handoff.
- ``IPCWeightSync``    — colocate, bucketed CUDA-IPC over ZMQ (same-node).

All subclass ``FullWeightSync`` and are referenced from configs via ``_target_``.
"""

from unirl.distributed.weight_sync.full.base import FullWeightSync
from unirl.distributed.weight_sync.full.ipc import IPCWeightSync
from unirl.distributed.weight_sync.full.nccl import NCCLWeightSync
from unirl.distributed.weight_sync.full.tensor import TensorWeightSync

__all__ = ["FullWeightSync", "NCCLWeightSync", "TensorWeightSync", "IPCWeightSync"]
