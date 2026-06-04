"""Engine-neutral weight-transfer helpers shared by trainer and rollout sides.

Pure-Python modules (torch/zmq/stdlib only — no vllm-omni, no engine
imports), so trainer-side senders (``distributed/weight_sync``,
``train/backend/fsdp.py``) and engine-side receivers (rollout worker
extensions) can both import them without a cycle:

- ``bucketed_transfer``: ``BucketedWeightSender`` / ``BucketedWeightReceiver``
  for ZMQ + CUDA-IPC bucketed weight transport.
- ``ipc_dispatch``: ZMQ socket layout + the fixed LoRA-slot identifiers both
  sides must agree on.
- ``checksum``: the single tensor-fingerprint formula for trainer ↔ worker
  value-correctness checks.

Formerly ``unirl.rollout.engine.vllm_omni.weight_sync.{bucketed_transfer,
ipc_dispatch,checksum}`` — hoisted here so no engine package owns
cross-engine transfer code; the old paths re-export for back-compat until
the v1 engine retires.
"""

from unirl.distributed.weight_sync.transfer.bucketed_transfer import (
    BucketedWeightReceiver,
    BucketedWeightSender,
)
from unirl.distributed.weight_sync.transfer.checksum import (
    compute_lora_checksums_post_optimize,
    compute_param_checksums,
    fingerprint_tensor,
)
from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
    DIFFRL_LORA_INT_ID,
    DIFFRL_LORA_NAME,
    DIFFRL_LORA_PATH,
    replica_rank_from_env,
    zmq_handle,
)

__all__ = [
    "BucketedWeightReceiver",
    "BucketedWeightSender",
    "DIFFRL_LORA_INT_ID",
    "DIFFRL_LORA_NAME",
    "DIFFRL_LORA_PATH",
    "compute_lora_checksums_post_optimize",
    "compute_param_checksums",
    "fingerprint_tensor",
    "replica_rank_from_env",
    "zmq_handle",
]
