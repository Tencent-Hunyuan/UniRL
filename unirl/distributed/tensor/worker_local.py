"""WorkerLocalTransport — worker-resident transports + the ``localize`` routing.

The ``TensorTransport`` subclass that worker-resident backends (colocate, gpu)
extend: ref-count lifecycle, cross-worker NCCL transfer, on-worker compute, and
the ``localize`` find/move/replace routing. ``isinstance(t, WorkerLocalTransport)``
is the controller's locality discriminator.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import ray
import torch

from unirl.distributed.tensor.ref import TensorRef, TensorSpan, map_tree
from unirl.distributed.tensor.transport import TensorTransport
from unirl.distributed.utils import collect_leaves


def _apply_tensor_op(t: torch.Tensor, op: str, *args) -> torch.Tensor:
    if op == "getitem":
        return t[args[0]]
    if op == "reshape":
        return t.reshape(args[0])
    if op == "permute":
        return t.permute(args[0])
    raise ValueError(f"Unknown tensor op: {op!r}")


class WorkerLocalTransport(TensorTransport):
    """Worker-resident transport: ref-count lifecycle, cross-worker NCCL, and ``localize``.

    GLOBAL backends (transfer queue) are plain ``TensorTransport`` and implement none
    of this; ``isinstance(t, WorkerLocalTransport)`` is the locality discriminator.
    """

    # Capability methods the controller may invoke via the Worker's transport_op relay.
    REMOTE_OPS: ClassVar[frozenset] = frozenset({"incref", "decref", "tensor_op", "get_cpu", "nccl_send", "nccl_recv"})

    # ---- lifecycle (ref-counting) ----

    def incref(self, key: Any) -> None:
        """Increment the ref count. No-op by default."""

    def decref(self, key: Any) -> None:
        """Decrement the ref count; free at zero. No-op by default."""

    # ---- locality + cross-worker transfer ----

    def setup_transfer(self, global_rank: int, world_size: int) -> None:
        """Initialize the cross-worker transfer group."""

    def nccl_send(self, dst_rank: int, spans: List[TensorSpan]) -> None:
        raise NotImplementedError("transport does not support cross-worker send")

    def nccl_recv(self, src_rank: int, shapes: List[tuple], dtypes: List[torch.dtype]) -> List[Any]:
        raise NotImplementedError("transport does not support cross-worker recv")

    @classmethod
    def _is_local(cls, ref: Any, dst_worker_id: str, dst_device_id: int, pool: Any) -> bool:
        """Per-backend locality predicate. Base: local iff produced by the dst worker."""
        return ref.source_id == dst_worker_id

    @classmethod
    def _move_key(cls, span: Any, dst: Tuple[str, int], pool: Any) -> Optional[tuple]:
        """A span's transfer identity wrt ``dst``, or ``None`` if already resolvable there.

        Short-circuit: object_ref (CPU/plasma) resolves anywhere; else ``_is_local``;
        else the by-VALUE key ``(src_device, dst_device, store_key, start, stop)`` so
        identical foreign slices to one device dedup to a single transfer.
        """
        dst_worker_id, dst_device_id = dst
        h = span.handle
        if getattr(h, "object_ref", None) is not None:
            return None
        if cls._is_local(h, dst_worker_id, dst_device_id, pool):
            return None
        return (pool.device_id_of(h.source_id), dst_device_id, h.store_key, span.start, span.stop)

    @classmethod
    def _replace_leaf(cls, moved: Dict[tuple, Any], dst: Tuple[str, int], pool: Any) -> Callable[[Any], Any]:
        """``map_tree`` leaf for REPLACE: swap foreign spans for their moved result.

        An all-local ref is returned UNCHANGED (same object), preserving grad /
        retain_grad_flag / _packed_cu_seqlens that ``with_spans`` would drop.
        """

        def leaf(o: Any) -> Any:
            if isinstance(o, TensorRef):
                new_spans = [moved.get(cls._move_key(s, dst, pool), s) for s in o.spans]
                if all(ns is s for ns, s in zip(new_spans, o.spans)):
                    return o
                return o.with_spans(new_spans)
            return o

        return leaf

    @classmethod
    def _move(cls, pool: Any, to_move: Dict[tuple, Any]) -> Dict[tuple, Any]:
        """One batched NCCL hop per ``(src, dst)`` device group; return key → received span.

        Ordering invariant: each group's ``keys`` list is reused in the SAME order for the
        send, the recv shapes/dtypes, and ``zip(keys, recv_handles)`` — do not reorder one
        without the others. All sends + recvs post before any ``ray.get``. Recv shapes are
        the SLICED span shapes (exactly the rows shipped), not the full handle block.
        """
        groups: Dict[Tuple[int, int], List[tuple]] = {}
        for key in to_move:
            groups.setdefault((key[0], key[1]), []).append(key)

        send_refs, recv_refs = [], []
        for (src_device_id, dst_device_id), keys in groups.items():
            spans = [to_move[k] for k in keys]
            send_refs.append(pool.slot0_worker(src_device_id).transport_op.remote("nccl_send", dst_device_id, spans))
            recv_refs.append(
                pool.slot0_worker(dst_device_id).transport_op.remote(
                    "nccl_recv", src_device_id, [s.shape for s in spans], [s.dtype for s in spans]
                )
            )
        ray.get(send_refs)
        recv_results = ray.get(recv_refs)

        moved: Dict[tuple, Any] = {}
        for ((src_device_id, dst_device_id), keys), new_handles in zip(groups.items(), recv_results):
            dst_worker = pool.slot0_worker(dst_device_id)
            for key, new_h in zip(keys, new_handles):
                new_h.rebind(dst_worker)
                moved[key] = TensorSpan(new_h, 0, int(new_h.shape[0]))
        return moved

    @classmethod
    def localize(cls, shards: list, pool: Any, device_ids: List[int], worker_ids: List[str]) -> list:
        """Make every ref resolvable on its dst worker — FIND (pure) / MOVE (NCCL) / REPLACE (pure).

        The shared skeleton for all worker-local backends; only ``_is_local`` varies. Shards
        are returned untouched when nothing is foreign.
        """
        dsts = list(zip(worker_ids, device_ids))

        to_move: Dict[tuple, Any] = {}
        for (s_args, s_kwargs), dst in zip(shards, dsts):
            for ref in collect_leaves(s_args, TensorRef) + collect_leaves(s_kwargs, TensorRef):
                for s in ref.spans:
                    key = cls._move_key(s, dst, pool)
                    if key is not None:
                        to_move.setdefault(key, s)
        if not to_move:
            return shards

        moved = cls._move(pool, to_move)
        return [
            (
                map_tree(s_args, cls._replace_leaf(moved, dst, pool)),
                map_tree(s_kwargs, cls._replace_leaf(moved, dst, pool)),
            )
            for (s_args, s_kwargs), dst in zip(shards, dsts)
        ]

    # ---- remote compute (controller-triggered) ----

    def tensor_op(self, handle: Any, op: str, *op_args) -> Any:
        """Round-trip resolve → op → put. Backends with on-worker compute override."""
        result = _apply_tensor_op(self._resolve_handles([handle])[0], op, *op_args).contiguous()
        return self.put(result)

    def get_cpu(self, handle: Any) -> torch.Tensor:
        """Return the stored tensor as a CPU tensor."""
        return self._resolve_handles([handle])[0].cpu()


__all__ = ["WorkerLocalTransport"]
