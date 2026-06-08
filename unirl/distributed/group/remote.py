"""RankInfo and Remote — logical worker base classes.

RankInfo holds parallelism rank information (DP/TP/PP/SP/EP).
Remote is the base class users inherit to define worker logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed

if TYPE_CHECKING:
    from unirl.distributed.tensor.transport import TensorTransport


@dataclass
class RankInfo:
    """Parallelism rank information for a logical worker.

    Set by RoleHandle when registering the remote on a Worker.
    """

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
    """Base class for logical workers. Users inherit this.

    A Remote runs inside a Worker (physical Ray actor).
    Multiple Remotes can share the same Worker (colocated).

    Attributes set by Worker.add_remote():
        transport: TensorTransport (owned by the Worker)
        device:    GPU device string (e.g. "cuda:0")
        rank_info: RankInfo for this worker group
        dist_env:  Group-level dist env vars (MASTER_ADDR, MASTER_PORT, etc.)
    """

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
        """Inject dependencies. Called by Worker.add_remote().

        Writes dist_env to os.environ once for convenience (env:// init_method).
        """
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
        """User-facing init hook. Override to load models, create sub-PG, etc.

        Called explicitly by user via wg.initialize(*args, **kwargs).
        At this point self.transport, self.device, self.rank_info, self.dist_env
        are all available, and dist_env is already in os.environ.
        """
        pass

    # ── Auto-backward (framework-injected, not user-facing) ───────────────────

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def _auto_backward(
        self,
        call_id: str,
        out_grads: tuple,
        in_grads: tuple,
    ) -> tuple:
        """Framework backward RPC, dispatched with DP_SCATTER by _run_auto_backward.

        dispatch_mode = DP_SCATTER is intentional and covers all currently supported
        forward dispatch modes:

          DP_SCATTER      forward → DP_SCATTER backward  (grad shards align with output shards)
          DP_SCATTER_HEAD forward → DP_SCATTER backward  (all ranks must participate in backward,
                                                          not just the DP head ranks)

        !! IMPORTANT — adding a new forward dispatch_mode !!
        If you add a new Dispatch variant, check resolve_backward_dispatch_mode() in
        dispatch.py to decide whether DP_SCATTER backward is still correct, or
        whether _auto_backward needs a new dispatch variant / a hard error.

        Args:
            call_id:   Matches the key in _grad_inputs/_grad_outputs.
            out_grads: tuple[Optional[Tensor], ...] — external grad for each
                       saved output tensor.  None means no gradient.
                       Elements are split element-wise by pytree_chunk (tuple
                       semantics), so each worker receives its own shard.
            in_grads:  tuple[Optional[Tensor], ...] — existing accumulated grad
                       for each saved input tensor (fan-out accumulation support).
                       Assigned to tensor.grad before backward so PyTorch +=.

        Returns:
            tuple[Optional[Tensor], ...] — new .grad for each saved input tensor.
            None at position i means input i has no gradient.
            Elements are merged element-wise by pytree_cat across workers.
        """
        saved_out: List[torch.Tensor] = self._grad_outputs.pop(call_id, [])
        saved_in: List[torch.Tensor] = self._grad_inputs.pop(call_id, [])

        # Pre-assign existing in_grads so PyTorch accumulates via grad +=
        for t, g in zip(saved_in, in_grads):
            if g is not None:
                t.grad = g

        # Run backward for outputs that have an external gradient
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
        """Discard ALL saved grad tensors on this worker.

        Called on every remote in the context when the GradContext exits (whether
        normally or via exception), ensuring no stale _grad_inputs/_grad_outputs
        linger in worker memory across training steps.
        """
        self._grad_inputs.clear()
        self._grad_outputs.clear()
