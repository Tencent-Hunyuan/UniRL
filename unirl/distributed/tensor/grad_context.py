"""GradContext — controller-side autograd for RPC call chains.

enable_grad() tracks forward RPC calls and automatically issues backward
RPCs in reverse order when the context exits, propagating gradients through
the chain of worker computations.

Usage::

    with enable_grad():
        gen    = actor.generate_samples(batch)
        reward = reward_role.score(gen)
        _      = actor.forward_backward_loss(reward)

    actor.step()

The framework issues _auto_backward RPCs automatically on __exit__.
Gradients are accumulated worker-side using PyTorch's native .grad += mechanism,
supporting fan-out (same TensorRef used as input to multiple RPCs).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from unirl.distributed.tensor.ref import TensorRef

if TYPE_CHECKING:
    from unirl.distributed.group.dispatch import Dispatch
    from unirl.distributed.group.handle import Handle

logger = logging.getLogger(__name__)


_tls = threading.local()


def current_grad_context() -> Optional["GradContext"]:
    """Return the active GradContext for the current thread, or None."""
    return getattr(_tls, "ctx", None)


@dataclass
class RPCBackwardNode:
    """Records a single forward RPC call for later backward dispatch.

    input_metas and output_metas are ordered lists — index i corresponds to
    _grad_inputs[call_id][i] and _grad_outputs[call_id][i] on the worker side.
    Both sides use the same depth-first sorted-key traversal via
    collect_leaves(x, TensorRef) / collect_leaves(x, Tensor) to guarantee alignment.
    """

    role_proxy: "Handle"
    call_id: str  # key prefix for worker _grad_inputs/_grad_outputs
    dispatch_mode: "Dispatch"  # backward dispatch mode (always DP_SCATTER currently)
    input_metas: List["TensorRef"]  # forward input TensorMetas, in traversal order
    output_metas: List["TensorRef"]  # forward output TensorMetas, in traversal order


class GradContext:
    """Context manager that tracks forward RPCs and runs backward on exit."""

    def __init__(self) -> None:
        self.nodes: List[RPCBackwardNode] = []

    def __enter__(self) -> "GradContext":
        if getattr(_tls, "ctx", None) is not None:
            raise RuntimeError(
                "Nested enable_grad() is not supported. Exit the outer context before entering a new one."
            )
        _tls.ctx = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _tls.ctx = None

        if exc_type is not None:
            _cleanup_all(self)
            return False  # don't suppress exception

        _run_backward(self)

        _cleanup_all(self)

        seen: set = set()
        for node in self.nodes:
            for tm in node.input_metas + node.output_metas:
                if id(tm) not in seen:
                    seen.add(id(tm))
                    if not tm.retain_grad_flag:
                        tm.grad = None


def enable_grad() -> GradContext:
    """Return a GradContext to track RPC calls and run auto-backward on exit."""
    return GradContext()


def _run_backward(ctx: GradContext) -> None:
    """Traverse nodes in reverse, issue _auto_backward RPCs."""
    errors = []

    for node in reversed(ctx.nodes):
        if node.output_metas and all(tm.grad is None for tm in node.output_metas):
            continue

        try:
            _run_auto_backward(node)
        except Exception as e:
            errors.append((node.call_id, e))

    if errors:
        raise RuntimeError(
            f"backward failed on {len(errors)} node(s): " + ", ".join(cid for cid, _ in errors)
        ) from errors[0][1]


def _run_auto_backward(node: RPCBackwardNode) -> None:
    """Call _auto_backward proxy on workers using node's dispatch_mode.

    out_grads and in_grads are tuples of Optional[TensorRef].  pytree_chunk
    recurses into tuple elements, so each TensorRef is chunked by dp_size
    giving worker_i its own grad shard.  pytree_cat does the inverse on
    return values.  No manual per-worker dispatch needed.
    """
    out_grads = tuple(tm.grad for tm in node.output_metas)
    in_grads = tuple(tm.grad for tm in node.input_metas)

    new_in_grads = node.role_proxy._auto_backward(node.call_id, out_grads, in_grads)

    if new_in_grads is not None:
        for tm, grad in zip(node.input_metas, new_in_grads):
            if grad is not None:
                tm.grad = grad


def _cleanup_all(ctx: GradContext) -> None:
    """Tell every role proxy involved in this context to clear all saved grad tensors.

    Calls _cleanup_all_grads() on each unique role proxy once (BROADCAST RPC),
    so every worker clears its _grad_inputs and _grad_outputs dicts entirely.
    Called both after successful backward and on forward exception.
    """
    seen_proxies: set = set()
    for node in ctx.nodes:
        proxy = node.role_proxy
        if id(proxy) not in seen_proxies:
            seen_proxies.add(id(proxy))
            try:
                proxy._cleanup_all_grads()
            except Exception:
                logger.debug("Failed to clean up remote gradient tensors.", exc_info=True)
