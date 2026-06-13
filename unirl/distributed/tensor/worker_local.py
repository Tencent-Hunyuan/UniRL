"""WorkerLocalTransport — worker-resident transports + the ``localize`` routing.

The ``TensorTransport`` subclass that worker-resident backends (colocate, gpu)
extend: ref-count lifecycle (``incref`` / ``decref``), controller-orchestrated
cross-worker transfer (``setup_transfer`` / ``nccl_send`` / ``nccl_recv``), and
on-worker compute (``tensor_op`` / ``get_cpu``). It owns ``localize`` — the
find / move / replace skeleton that makes every ref resolvable on its target
worker via one batched NCCL hop per ``(src, dst)`` device pair.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

import ray
import torch

from unirl.distributed.tensor.ref import TensorRef, TensorSpan, map_tree
from unirl.distributed.tensor.transport import TensorTransport
from unirl.distributed.utils import collect_leaves

def _apply_tensor_op(t: torch.Tensor, op: str, *args) -> torch.Tensor:
    """Apply a named tensor op. Shared by the default ``tensor_op`` round-trip."""
    if op == "getitem":
        return t[args[0]]
    if op == "reshape":
        return t.reshape(args[0])
    if op == "permute":
        return t.permute(args[0])
    raise ValueError(f"Unknown tensor op: {op!r}")




class WorkerLocalTransport(TensorTransport):
    """The V2 Worker/Handle storage contract — worker-resident backends only.

    Adds the storage-engine machinery only a worker-resident store needs:
    ref-count lifecycle (``incref``/``decref``), controller-orchestrated
    cross-worker transfer (``setup_transfer``/``nccl_send``/``nccl_recv``), and
    on-worker remote compute (``tensor_op``/``cat``/``get_cpu``). The universal
    materialization surface (``get_batch``/``put_batch``) lives on
    the base. GLOBAL backends (e.g. the transfer queue) are plain
    :class:`TensorTransport` and implement none of this capability.

    ``isinstance(t, WorkerLocalTransport)`` is the locality discriminator the
    controller uses to decide whether cross-worker routing is required.
    """

    # Methods the controller may invoke on this transport via the Worker actor's
    # ``transport_op`` relay (TensorHandle GC/compute + Handle NCCL routing).
    # Adding a capability method below means adding its name here. Excludes
    # setup_transfer, which the Worker injects identity into via setup_global_pg.
    REMOTE_OPS: ClassVar[frozenset] = frozenset({"incref", "decref", "tensor_op", "get_cpu", "nccl_send", "nccl_recv"})

    # ---- lifecycle (ref-counting) ------------------------------------------

    def incref(self, key: Any) -> None:
        """Increment the ref count for a stored tensor. No-op by default."""

    def decref(self, key: Any) -> None:
        """Decrement the ref count; free at zero. No-op by default."""

    # ---- locality + cross-worker transfer (localize) -------------------

    def setup_transfer(self, global_rank: int, world_size: int) -> None:
        """Initialize the cross-worker transfer group."""

    def nccl_send(self, dst_rank: int, handles: List[Any]) -> None:
        raise NotImplementedError("transport does not support cross-worker send")

    def nccl_recv(self, src_rank: int, shapes: List[tuple], dtypes: List[torch.dtype]) -> List[Any]:
        raise NotImplementedError("transport does not support cross-worker recv")

    @classmethod
    def _is_local(cls, ref: Any, dst_worker_id: str, dst_device_id: int, pool: Any) -> bool:
        """True if ``ref`` is already resolvable on the dst worker (no transfer needed).

        The one per-backend locality decision. Base: a ref is local only if produced by
        the dst worker (per-process store). gpu overrides to also accept same physical
        device, since its per-GPU TensorWorker is shared across that GPU's slots.
        """
        return ref.source_id == dst_worker_id

    @classmethod
    def _move_key(cls, span: Any, dst: Tuple[str, int], pool: Any) -> Optional[tuple]:
        """Transfer identity of a span wrt a destination — or ``None`` if already resolvable there.

        The single span classifier shared by ``localize``'s FIND and REPLACE passes. Pure
        (no Ray): reads only the handle's identity fields, ``pool.device_id_of``, and the
        per-backend ``_is_local``. Checked in short-circuit order so ``device_id_of`` is
        never called on a resolvable ref:

          1. ``object_ref`` present (CPU/plasma) → resolvable anywhere → ``None``
          2. ``_is_local`` on the dst worker → ``None``
          3. else → the value key ``(src_device, dst_device, store_key, start, stop)``

        The key is by VALUE, not ``id()``: identical foreign slices headed to the same dst
        device collapse to one transfer, and the received result is shared across every
        tree position. ``store_key`` is unique per source device, and ``object_ref`` handles
        (``store_key is None``) never reach branch 3, so the key cannot alias.
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
        """Build the ``map_tree`` leaf for REPLACE: swap each foreign span for its moved result.

        A factory (not an inline loop closure) so the per-shard ``dst`` is a real parameter —
        no loop late-binding. A ref whose spans all stay put is returned UNCHANGED (same
        object): this skips a needless rebuild and preserves the fields ``with_spans`` drops
        (``grad`` / ``retain_grad_flag`` / ``_packed_cu_seqlens``). Keys are recomputed on the
        original spans, so they match exactly what FIND recorded.
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
        """Run one batched NCCL hop per ``(src_device, dst_device)`` group; return key → received span.

        Ordering invariant: each group's ``keys`` list is built once and reused — in the
        SAME order — for (a) the ``nccl_send`` items, (b) the ``nccl_recv`` shapes/dtypes,
        and (c) the ``zip(keys, recv_handles)`` that fills ``moved``. ``dict`` preserves
        insertion order, which is what makes this sound; do not reorder one without the
        others. All sends AND recvs are posted before any ``ray.get`` so a send cannot block
        on an unposted recv.

        Recv shapes/dtypes come off the representative span's ``.shape`` / ``.dtype`` — the
        SLICED shape (exactly the rows the send ships), not the full handle block.
        """
        groups: Dict[Tuple[int, int], List[tuple]] = {}  # (src_dev, dst_dev) → [key, ...] (ordered)
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
                # The recv handle holds exactly the sliced rows → full-range span.
                moved[key] = TensorSpan(new_h, 0, int(new_h.shape[0]))
        return moved

    @classmethod
    def localize(cls, shards: list, pool: Any, device_ids: List[int], worker_ids: List[str]) -> list:
        """Make every ref in each shard resolvable on its dst worker — find / move / replace.

        Shared skeleton for all worker-local backends; the only thing that varies is
        ``_is_local`` (the locality predicate, reached via ``_move_key``). Three stages:

          FIND     read-only walk → the unique foreign slices to move, keyed by transfer
                   identity; identical slices to one dst device dedup (``_move_key``).
          MOVE     one batched NCCL hop per ``(src_device, dst_device)`` group between the
                   two devices' slot0 workers (``_move``).
          REPLACE  rebuild each shard, swapping foreign spans for their received result and
                   leaving local spans (and untouched refs) as-is (``_replace_leaf``).

        Names no backend type — works through ``span.handle`` / ``pool`` / ``map_tree`` /
        the ``transport_op`` relay. FIND and REPLACE are pure (no Ray); only MOVE does I/O,
        so when nothing is foreign the shards are returned untouched.
        """
        dsts = list(zip(worker_ids, device_ids))

        # FIND — the unique foreign slices, keyed by transfer identity (setdefault dedups).
        to_move: Dict[tuple, Any] = {}  # key → representative span
        for (s_args, s_kwargs), dst in zip(shards, dsts):
            for ref in collect_leaves(s_args, TensorRef) + collect_leaves(s_kwargs, TensorRef):
                for s in ref.spans:
                    key = cls._move_key(s, dst, pool)
                    if key is not None:
                        to_move.setdefault(key, s)
        if not to_move:
            return shards

        # MOVE — one batched NCCL hop per (src_device, dst_device) group.
        moved = cls._move(pool, to_move)

        # REPLACE — rebuild each shard, swapping foreign spans for their received result.
        return [
            (map_tree(s_args, cls._replace_leaf(moved, dst, pool)), map_tree(s_kwargs, cls._replace_leaf(moved, dst, pool)))
            for (s_args, s_kwargs), dst in zip(shards, dsts)
        ]

    # ---- remote compute (controller-triggered) ----------------------------

    def tensor_op(self, handle: Any, op: str, *op_args) -> Any:
        """Apply a named op (getitem/reshape/permute) to a stored tensor.

        Default: round-trip resolve -> op -> put. Backends with on-worker compute
        override to avoid moving data.
        """
        result = _apply_tensor_op(self._resolve_handles([handle])[0], op, *op_args).contiguous()
        return self.put(result)

    def get_cpu(self, handle: Any) -> torch.Tensor:
        """Return the stored tensor as a CPU tensor."""
        return self._resolve_handles([handle])[0].cpu()




__all__ = ["WorkerLocalTransport"]
