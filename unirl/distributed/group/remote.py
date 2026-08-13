"""RankInfo and Remote — logical worker base classes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed

if TYPE_CHECKING:
    from unirl.distributed.tensor import TensorTransport


@dataclass
class RankInfo:
    """Parallelism rank information for a logical worker."""

    rank: int = 0
    world_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    sp_rank: int = 0
    sp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1

    @property
    def is_pipeline_last_stage(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @property
    def is_dp_rank_zero(self) -> bool:
        return self.dp_rank == 0

    def __repr__(self) -> str:
        parts = [f"rank={self.rank}", f"world_size={self.world_size}"]
        if self.dp_size > 1:
            parts.append(f"dp={self.dp_rank}/{self.dp_size}")
        if self.tp_size > 1:
            parts.append(f"tp={self.tp_rank}/{self.tp_size}")
        if self.pp_size > 1:
            parts.append(f"pp={self.pp_rank}/{self.pp_size}")
        if self.sp_size > 1:
            parts.append(f"sp={self.sp_rank}/{self.sp_size}")
        if self.ep_size > 1:
            parts.append(f"ep={self.ep_rank}/{self.ep_size}")
        return f"RankInfo({', '.join(parts)})"


class Remote:
    """Base class for logical workers. Users inherit this."""

    def __init__(self) -> None:
        self.transport: Optional[TensorTransport] = None
        self.device: Optional[str] = None
        self.rank_info: Optional[RankInfo] = None
        self.dist_env: Dict[str, str] = {}
        self._get_sibling = None
        self._grad_inputs: Dict[str, List[torch.Tensor]] = {}
        self._grad_outputs: Dict[str, List[torch.Tensor]] = {}

    def setup(
        self,
        transport: "TensorTransport",
        device: str,
        rank_info: RankInfo,
        dist_env: Optional[Dict[str, str]] = None,
        get_sibling=None,
    ) -> None:
        """Inject dependencies. Called by Worker.add_remote()."""
        self.transport = transport
        self.device = device
        self.rank_info = rank_info
        self.dist_env = dist_env or {}
        self._get_sibling = get_sibling
        if self.dist_env:
            os.environ.update(self.dist_env)

    def get_sibling(self, name: str) -> "Remote":
        """Look up a colocated Remote by name on the same Worker."""
        if self._get_sibling is None:
            raise RuntimeError("get_sibling not available (Worker did not provide lookup)")
        return self._get_sibling(name)

    def initialize(self, *args, **kwargs) -> None:
        """User-facing init hook. Override to load models, create sub-PG, etc."""
        pass

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def _auto_backward(
        self,
        call_id: str,
        out_grads: tuple,
        in_grads: tuple,
    ) -> tuple:
        """Framework backward RPC, dispatched with DP_SCATTER by _run_auto_backward."""
        saved_out: List[torch.Tensor] = self._grad_outputs.pop(call_id, [])
        saved_in: List[torch.Tensor] = self._grad_inputs.pop(call_id, [])

        for t, g in zip(saved_in, in_grads):
            if g is not None:
                t.grad = g

        pairs = [
            (t, g) for t, g in zip(saved_out, out_grads) if g is not None and (t.requires_grad or t.grad_fn is not None)
        ]
        if pairs:
            tensors, grads = zip(*pairs)
            torch.autograd.backward(list(tensors), list(grads))

        result = tuple(t.grad for t in saved_in)
        torch.cuda.empty_cache()
        return result

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def _cleanup_all_grads(self) -> None:
        """Discard ALL saved grad tensors on this worker."""
        self._grad_inputs.clear()
        self._grad_outputs.clear()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def get_memory_stats(
        self,
        reset_peak: bool = False,
        log_stage: Optional[str] = None,
        empty_cache: bool = False,
        dump_snapshot_tag: Optional[str] = None,
    ) -> Dict[str, float]:
        """Worker-side memory probe reached by the CUDA-less driver via BROADCAST."""
        if not torch.cuda.is_available():
            return {}
        from unirl.utils.memory_utils import (
            aggressive_empty_cache,
            get_memory_info,
            get_process_snapshot_sampler,
            log_memory_usage,
        )

        info = log_memory_usage(log_stage) if log_stage else get_memory_info()
        if empty_cache:
            aggressive_empty_cache()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        out = {**info, "rank": float(self.rank_info.rank)}
        if dump_snapshot_tag:
            sampler = get_process_snapshot_sampler()
            if sampler is not None:
                report = sampler.dump(dump_snapshot_tag)
                if report:
                    out["snapshot_report"] = report
        return out
