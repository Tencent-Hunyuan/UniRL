"""UniRL-only request structs absent from SGLang's post-training API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InitWeightsUpdateGroupReqInput:
    """Initialize a temporary process group for distributed weight updates."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str = "weight_update_group"
    backend: str = "nccl"


@dataclass
class DestroyWeightsUpdateGroupReqInput:
    """Destroy a temporary distributed weight-update process group."""

    group_name: str = "weight_update_group"


@dataclass
class UpdateWeightsFromDistributedReqInput:
    """Receive weight tensors from an external source via distributed broadcast."""

    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str = "weight_update_group"
    target_modules: list[str] | None = None
    flush_cache: bool = True


@dataclass
class ReleaseMemoryOccupationReqInput:
    """Request to release (sleep) GPU memory occupation for the diffusion engine."""

    tags: list[str] | None = None
    cpu_backup_tags: list[str] | None = None


@dataclass
class ResumeMemoryOccupationReqInput:
    """Request to resume (wake) GPU memory occupation for the diffusion engine."""

    tags: list[str] | None = None
