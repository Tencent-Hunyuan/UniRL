"""RL request structs the fork added to sglang's post_training ``io_struct``.

Stock upstream sglang ships only ``UpdateWeightFromDiskReqInput`` and
``GetWeightsChecksumReqInput``; these 8 are fork-new. Defining them here is the
**single definition site** -- both the UniRL adapter
(``rollout/engine/sglang/engine.py``) and ``patch_scheduler`` import them from
here, so the scheduler's ``request_handlers`` dict (keyed by ``type(req)``) and
the objects the adapter sends are the **same classes** and dispatch matches.

Copied verbatim from
``sglang-drl/.../runtime/entrypoints/post_training/io_struct.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GetWeightsDetailReqInput:
    """Get per-parameter details: names, shapes, dtypes, count, checksums."""

    module_names: list[str] | None = None


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
class UpdateWeightsFromTensorReqInput:
    """Update weights from serialized named tensors."""

    serialized_named_tensors: list[str | bytes]
    target_modules: list[str] | None = None
    load_format: str | None = None
    flush_cache: bool = True


@dataclass
class EncodePromptReqInput:
    """Request to encode text prompts into embeddings without running diffusion."""

    prompts: list[str]


@dataclass
class ReleaseMemoryOccupationReqInput:
    """Request to release (sleep) GPU memory occupation for the diffusion engine."""

    tags: list[str] | None = None
    cpu_backup_tags: list[str] | None = None


@dataclass
class ResumeMemoryOccupationReqInput:
    """Request to resume (wake) GPU memory occupation for the diffusion engine."""

    tags: list[str] | None = None
