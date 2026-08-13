"""Shared bookkeeping for the bucketed-CUDA-IPC weight-sync path."""

from __future__ import annotations

import os

DIFFRL_LORA_INT_ID: int = 1
DIFFRL_LORA_NAME: str = "diffrl_lora"
DIFFRL_LORA_PATH: str = "diffrl_lora_in_memory"

_DEFAULT_IPC_DIR: str = os.environ.get("DIFFRL_IPC_DIR", "/tmp")


def zmq_handle(replica_rank: int, stage_id: int, local_rank: int, *, ipc_dir: str | None = None) -> str:
    """Return the IPC socket path for one ``(replica, stage, rank)`` peer pair."""
    root = ipc_dir if ipc_dir is not None else _DEFAULT_IPC_DIR
    return f"ipc://{root}/diffrl-zmq-replica-{int(replica_rank)}-stage-{int(stage_id)}-rank-{int(local_rank)}.sock"


def replica_rank_from_env() -> int:
    """Read the rollout-actor replica rank from env (default 0)."""
    return int(os.environ.get("DIFFRL_REPLICA_RANK", "0"))


__all__ = [
    "DIFFRL_LORA_INT_ID",
    "DIFFRL_LORA_NAME",
    "DIFFRL_LORA_PATH",
    "zmq_handle",
    "replica_rank_from_env",
]
