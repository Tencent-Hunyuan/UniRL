"""Shared NCCL-broadcast receive mixin for HI3 worker-extension classes."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_DTYPE_FROM_STR: Dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.bfloat16": torch.bfloat16,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name in _DTYPE_FROM_STR:
        return _DTYPE_FROM_STR[name]
    short = name.replace("torch.", "")
    full = f"torch.{short}"
    if full in _DTYPE_FROM_STR:
        return _DTYPE_FROM_STR[full]
    raise KeyError(f"_resolve_dtype: unsupported dtype name {name!r}")


class NcclBroadcastReceiveMixin:
    """Adds ``init_weights_update_group`` + ``update_weights_from_distributed`` to a worker by inheritance."""

    _diffrl_weight_groups: Dict[str, "dist.ProcessGroup"] = {}

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        """Join the trainer-coordinated process group as the receiver."""
        from unirl.utils.distributed_utils import (
            init_process_group as _diffrl_init_pg,
        )

        local_rank = int(getattr(self, "local_rank", 0))
        global_rank = int(rank_offset) + local_rank
        new_group = _diffrl_init_pg(
            backend=backend,
            init_method=f"tcp://{master_address}:{int(master_port)}",
            world_size=int(world_size),
            rank=global_rank,
            group_name=group_name,
        )
        type(self)._diffrl_weight_groups[group_name] = new_group
        logger.info(
            "%s.init_weights_update_group: joined %r at global_rank=%d/%d (master=%s:%d, backend=%s)",
            type(self).__name__,
            group_name,
            global_rank,
            int(world_size),
            master_address,
            int(master_port),
            backend,
        )

    def destroy_weights_update_group(self, *, group_name: str) -> None:
        """Drop the named group's handle. Counterpart to ``init_weights_update_group``."""
        type(self)._diffrl_weight_groups.pop(group_name, None)

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
    ) -> None:
        """Receive a bucket of named tensors by ``dist.broadcast`` from rank 0, then call ``self.load_weights``."""
        del target_modules, flush_cache  # accepted for SGLang-shape parity
        group = type(self)._diffrl_weight_groups.get(group_name)
        if group is None:
            raise RuntimeError(
                f"{type(self).__name__}.update_weights_from_distributed: no "
                f"process group {group_name!r}; call init_weights_update_group first."
            )
        device = getattr(self, "device", None)
        if device is None:
            raise RuntimeError(
                f"{type(self).__name__}.update_weights_from_distributed: worker has no `device` attribute."
            )
        if not (len(names) == len(dtypes) == len(shapes)):
            raise ValueError(f"names/dtypes/shapes length mismatch: {len(names)} / {len(dtypes)} / {len(shapes)}")

        bucket: list[tuple[str, torch.Tensor]] = []
        for name, dtype_name, shape in zip(names, dtypes, shapes):
            dt = _resolve_dtype(dtype_name)
            tensor = torch.empty(tuple(shape), dtype=dt, device=device)
            dist.broadcast(tensor, src=0, group=group)
            bucket.append((str(name), tensor))

        loader = getattr(self, "_diffrl_load_weights", None)
        if loader is None:
            self.load_weights(bucket)
        else:
            loader(bucket)
        logger.debug(
            "%s.update_weights_from_distributed: received %d tensors via group %r",
            type(self).__name__,
            len(bucket),
            group_name,
        )


__all__ = ["NcclBroadcastReceiveMixin"]
