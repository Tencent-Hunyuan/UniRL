"""Shared bookkeeping for the bucketed-CUDA-IPC weight-sync path.

Both the engine-side dispatch (``VLLMOmniRolloutEngine.update_weights_from_ipc``
fan-out to per-stage workers via ``collective_rpc``) and the trainer-side
IPC weight-sync handler need to agree on:

- The ZMQ socket layout (one socket per ``(replica, stage, rank)`` tuple).
- The fixed LoRA identifier used for the single adapter we hot-swap.

Centralizing here keeps both sides in sync. No vllm-omni / vllm imports
at module level — this is pure-Python so it's importable from any side.
"""

from __future__ import annotations

import os

# Single LoRA "slot" we hot-swap on rollout. The integer id has to match
# between the trainer's add/remove calls and the worker's lookup. Mirrors
# verl-omni's VLLM_LORA_INT_ID / NAME / PATH triple.
DIFFRL_LORA_INT_ID: int = 1
DIFFRL_LORA_NAME: str = "diffrl_lora"
# Used as a placeholder ``lora_path`` on ``OmniTensorLoRARequest``; the
# hijacked ``_load_adapter`` ignores it for tensor-bag requests.
DIFFRL_LORA_PATH: str = "diffrl_lora_in_memory"

# Default ZMQ-IPC root. Override per call by passing zmq_handle directly
# or by setting DIFFRL_IPC_DIR in the env before construction.
_DEFAULT_IPC_DIR: str = os.environ.get("DIFFRL_IPC_DIR", "/tmp")


def zmq_handle(replica_rank: int, stage_id: int, local_rank: int, *, ipc_dir: str | None = None) -> str:
    """Return the IPC socket path for one ``(replica, stage, rank)`` peer pair.

    The receiver-side worker is identified by its ``local_rank`` within
    the stage and the global ``replica_rank`` of the rollout actor. The
    sender (trainer) computes the same path so they meet on the same
    socket.
    """
    root = ipc_dir if ipc_dir is not None else _DEFAULT_IPC_DIR
    return f"ipc://{root}/diffrl-zmq-replica-{int(replica_rank)}-stage-{int(stage_id)}-rank-{int(local_rank)}.sock"


def replica_rank_from_env() -> int:
    """Read the rollout-actor replica rank from env (default 0).

    Trainer-side and worker-side both consult this so the socket path
    matches without an explicit kwarg every call. Trainers and rollout
    actors should set ``DIFFRL_REPLICA_RANK`` consistently when there is
    more than one rollout replica.
    """
    return int(os.environ.get("DIFFRL_REPLICA_RANK", "0"))


__all__ = [
    "DIFFRL_LORA_INT_ID",
    "DIFFRL_LORA_NAME",
    "DIFFRL_LORA_PATH",
    "zmq_handle",
    "replica_rank_from_env",
]
