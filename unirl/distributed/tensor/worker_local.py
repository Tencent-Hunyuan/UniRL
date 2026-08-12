"""WorkerLocalTransport — worker-resident transports + the ``localize`` routing.

The ``TensorTransport`` subclass that worker-resident backends (colocate, gpu)
extend: ref-count lifecycle, cross-worker NCCL transfer, on-worker compute, and
the ``localize`` find/move/replace routing. ``isinstance(t, WorkerLocalTransport)``
is the controller's locality discriminator.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, List, Optional, Set, Tuple

import ray
import torch

from unirl.distributed.tensor.ref import TensorRef, TensorSpan, map_tree, ref_is_required
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

    REMOTE_OPS: ClassVar[frozenset] = frozenset({"incref", "decref", "tensor_op", "get_cpu", "nccl_send", "nccl_recv"})

    def incref(self, key: Any) -> None:
        """Increment the ref count. No-op by default."""

    def decref(self, key: Any) -> None:
        """Decrement the ref count; free at zero. No-op by default."""

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
        without the others. Device pairs run in sorted order, one at a time; recv posts
        before send, and both complete before the next pair starts. Recv shapes are the
        SLICED span shapes (exactly the rows shipped), not the full handle block.
        """
        groups: Dict[Tuple[int, int], List[tuple]] = {}
        for key in to_move:
            groups.setdefault((key[0], key[1]), []).append(key)

        moved: Dict[tuple, Any] = {}
        for src_device_id, dst_device_id in sorted(groups):
            keys = groups[(src_device_id, dst_device_id)]
            spans = [to_move[k] for k in keys]
            recv_ref = pool.slot0_worker(dst_device_id).transport_op.remote(
                "nccl_recv", src_device_id, [s.shape for s in spans], [s.dtype for s in spans]
            )
            send_ref = pool.slot0_worker(src_device_id).transport_op.remote("nccl_send", dst_device_id, spans)
            new_handles, _ = ray.get([recv_ref, send_ref])

            dst_worker = pool.slot0_worker(dst_device_id)
            for key, new_h in zip(keys, new_handles):
                new_h.rebind(dst_worker)
                moved[key] = TensorSpan(new_h, 0, int(new_h.shape[0]))
        return moved

    @classmethod
    def localize(
        cls,
        shards: list,
        pool: Any,
        device_ids: List[int],
        worker_ids: List[str],
        required: Optional[List[Optional[Set[str]]]] = None,
    ) -> list:
        """Make every ref resolvable on its dst worker — FIND (pure) / MOVE (NCCL) / REPLACE (pure).

        The shared skeleton for all worker-local backends; only ``_is_local`` varies. Shards
        are returned untouched when nothing is foreign.

        ``required`` is the optional per-shard partial-localization mask (see
        ``@distributed(reads=...)``): a ref outside its shard's mask is left
        foreign, so the callee receives it dehydrated and it never crosses the
        wire. REPLACE needs no mask of its own — a skipped span's move key was
        never entered into ``moved``, so it substitutes to itself.
        """
        dsts = list(zip(worker_ids, device_ids))
        masks = list(required) if required is not None else [None] * len(shards)

        to_move: Dict[tuple, Any] = {}
        for (s_args, s_kwargs), dst, mask in zip(shards, dsts, masks):
            for ref in collect_leaves(s_args, TensorRef) + collect_leaves(s_kwargs, TensorRef):
                if not ref_is_required(ref, mask):
                    continue
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

    def tensor_op(self, handle: Any, op: str, *op_args) -> Any:
        """Round-trip resolve → op → put. Backends with on-worker compute override."""
        result = _apply_tensor_op(self._resolve_handles([handle])[0], op, *op_args).contiguous()
        return self.put(result)

    def get_cpu(self, handle: Any) -> torch.Tensor:
        """Return the stored tensor as a CPU tensor."""
        return self._resolve_handles([handle])[0].cpu()


__all__ = ["WorkerLocalTransport"]
