"""MoE Router Replay (R2/R3) controller and context managers for UniRL."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Dict, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

ReplayMode = Literal["record", "replay"]


def _is_in_checkpoint_recompute() -> bool:
    """Detect whether execution is inside ``torch.utils.checkpoint`` backward recomputation."""
    frame = sys._getframe(1)
    while frame is not None:
        code = frame.f_code
        if "checkpoint" in code.co_filename and code.co_name in (
            "recompute_fn",
            "backward",
            "unpack",
            "_recompute",
        ):
            return True
        frame = frame.f_back
    return False


def _upstream():
    """Return VeOmni's router-replay module, which owns the active controller."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    import importlib

    return importlib.import_module("veomni.utils.moe_router_replay")


def get_active_replay() -> Optional["MoERouterReplayController"]:
    """Return the currently active MoE router-replay controller, or ``None``."""
    return _upstream().get_active_replay()


def set_active_replay(controller: Optional["MoERouterReplayController"]) -> None:
    """Install or clear the active MoE router-replay controller."""
    _upstream().set_active_replay(controller)


def maybe_replay_indices(
    module: nn.Module,
    routing_scores: torch.Tensor,
    top_indices: torch.Tensor,
) -> torch.Tensor:
    """Forward-time router hook delegating to ``veomni.utils.moe_router_replay``."""
    return _upstream().maybe_replay_indices(module, routing_scores, top_indices)


class MoERouterReplayController:
    """Record and replay MoE top-k expert routing indices across RL passes (see ``../../README.md``)."""

    def __init__(
        self,
        mode: ReplayMode = "record",
        *,
        recorded: Optional[Dict[int, List[torch.Tensor]]] = None,
        storage_dtype: torch.dtype = torch.int32,
        offload_to_cpu: bool = False,
    ) -> None:
        if mode not in ("record", "replay"):
            raise ValueError(f"MoERouterReplayController: mode must be 'record' or 'replay', got {mode!r}")
        self.mode: ReplayMode = mode
        self.storage_dtype = storage_dtype
        self.offload_to_cpu = bool(offload_to_cpu)
        self._records: Dict[int, List[torch.Tensor]] = (
            {int(k): list(v) for k, v in recorded.items()} if recorded is not None else {}
        )
        self._module_order: List[int] = list(self._records.keys())
        self._cursors: Dict[int, int] = {int(k): 0 for k in self._records}
        self._last_fwd_idx: Dict[int, int] = {int(k): 0 for k in self._records}
        self._recompute_stack: Dict[int, List[int]] = {int(k): [] for k in self._records}
        self._replay_hits: int = 0
        self._record_hits: int = 0

    @property
    def num_tracked_routers(self) -> int:
        return len(self._records)

    @property
    def record_hits(self) -> int:
        return self._record_hits

    @property
    def replay_hits(self) -> int:
        return self._replay_hits

    def reset_cursors(self) -> None:
        """Reset per-router replay cursors before a new forward pass."""
        for k in self._cursors:
            self._cursors[k] = 0
            self._last_fwd_idx[k] = 0
            self._recompute_stack[k] = []

    def switch_to_replay(self) -> "MoERouterReplayController":
        """Switch this controller from ``record`` mode to ``replay`` mode and reset cursors."""
        self.mode = "replay"
        self.reset_cursors()
        return self

    def on_router_forward(
        self,
        module: nn.Module,
        routing_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Duck-typed callback invoked by ``maybe_replay_indices`` in MoE layers."""
        mod_id = id(module)
        in_recompute = _is_in_checkpoint_recompute()
        if self.mode == "record":
            if in_recompute:
                return top_indices
            if mod_id not in self._records:
                self._records[mod_id] = []
                self._module_order.append(mod_id)
                self._cursors[mod_id] = 0
                self._last_fwd_idx[mod_id] = 0
                self._recompute_stack[mod_id] = []
            saved = top_indices.detach().to(dtype=self.storage_dtype)
            if self.offload_to_cpu:
                saved = saved.cpu()
            self._records[mod_id].append(saved)
            self._record_hits += 1
            return top_indices

        # mode == "replay"
        bucket = self._records.get(mod_id)
        if not bucket:
            return top_indices
        if in_recompute:
            stack = self._recompute_stack.get(mod_id)
            idx = stack.pop() if stack else self._last_fwd_idx.get(mod_id, 0)
        else:
            cursor = self._cursors.get(mod_id, 0)
            idx = cursor % len(bucket)
            self._cursors[mod_id] = cursor + 1
            self._last_fwd_idx[mod_id] = idx
            self._recompute_stack.setdefault(mod_id, []).append(idx)
        target = bucket[idx]
        if target.shape != top_indices.shape:
            raise RuntimeError(
                f"MoERouterReplayController: shape mismatch on router id={mod_id}: "
                f"recorded {tuple(target.shape)} vs current {tuple(top_indices.shape)}."
            )
        self._replay_hits += 1
        return target.to(device=top_indices.device, dtype=top_indices.dtype)

    def export_state(self) -> Dict[int, List[torch.Tensor]]:
        """Export recorded per-router index lists."""
        return {k: list(v) for k, v in self._records.items()}

    def export_ordered_layers(self) -> List[List[torch.Tensor]]:
        """Export recorded indices in router execution order (for R3 rollout-to-train transfer)."""
        return [list(self._records[k]) for k in self._module_order if k in self._records]

    def bind_ordered_layers(
        self,
        routers: List[nn.Module],
        ordered_layer_indices: List[Union[torch.Tensor, List[torch.Tensor]]],
    ) -> None:
        """Bind R3 rollout-captured per-layer indices to ``routers`` in layer order."""
        if len(routers) != len(ordered_layer_indices):
            raise ValueError(
                f"MoERouterReplayController.bind_ordered_layers: router count {len(routers)} "
                f"!= layer index count {len(ordered_layer_indices)}."
            )
        self._records.clear()
        self._module_order.clear()
        self._cursors.clear()
        self._last_fwd_idx.clear()
        self._recompute_stack.clear()
        for router, entry in zip(routers, ordered_layer_indices):
            mod_id = id(router)
            tensors = [entry] if isinstance(entry, torch.Tensor) else list(entry)
            self._records[mod_id] = [t.detach().to(dtype=self.storage_dtype) for t in tensors]
            self._module_order.append(mod_id)
            self._cursors[mod_id] = 0
            self._last_fwd_idx[mod_id] = 0
            self._recompute_stack[mod_id] = []


def patch_moe_router_replay(model: nn.Module) -> int:
    """Ensure every MoE router in ``model`` participates in ``maybe_replay_indices`` (see ``../../README.md``)."""
    patched = 0
    for _fqn, module in model.named_modules():
        if getattr(module, "_supports_native_router_replay", False):
            continue
        gate = getattr(module, "gate", None)
        if not isinstance(gate, nn.Module) or not hasattr(module, "experts"):
            continue
        if getattr(gate, "_unirl_router_replay_patched", False):
            continue
        try:
            src = inspect.getsource(type(module))
            if "maybe_replay_indices" in src:
                continue
        except (OSError, TypeError):
            pass

        orig_gate_forward = gate.forward

        def _make_wrapped_gate_forward(g: nn.Module, orig_fn):
            def _wrapped(*args, **kwargs):
                out = orig_fn(*args, **kwargs)
                if get_active_replay() is None:
                    return out
                if isinstance(out, tuple) and len(out) == 3:
                    router_logits, routing_weights, selected_experts = out
                    scores = F.softmax(router_logits, dtype=torch.float32, dim=-1)
                    replayed = maybe_replay_indices(g, scores, selected_experts)
                    if not torch.equal(replayed, selected_experts):
                        routing_weights = scores.gather(1, replayed)
                        if getattr(g, "norm_topk_prob", False):
                            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(
                                1e-20
                            )
                        routing_weights = routing_weights.to(router_logits.dtype)
                    return router_logits, routing_weights, replayed
                return out

            return _wrapped

        gate.forward = _make_wrapped_gate_forward(gate, orig_gate_forward)  # type: ignore[method-assign]
        gate._unirl_router_replay_patched = True  # type: ignore[attr-defined]
        patched += 1
    return patched


@contextmanager
def record_moe_routing(
    controller: Optional[MoERouterReplayController] = None,
    *,
    offload_to_cpu: bool = False,
) -> Iterator[MoERouterReplayController]:
    """Context manager that records MoE top-k expert indices during forward passes."""
    ctrl = (
        controller
        if controller is not None
        else MoERouterReplayController(mode="record", offload_to_cpu=offload_to_cpu)
    )
    ctrl.mode = "record"
    prev = get_active_replay()
    set_active_replay(ctrl)
    try:
        yield ctrl
    finally:
        set_active_replay(prev)


@contextmanager
def replay_moe_routing(
    controller_or_state: Union[MoERouterReplayController, Dict[int, List[torch.Tensor]]],
) -> Iterator[MoERouterReplayController]:
    """Context manager that replays recorded MoE top-k expert indices during forward/backward."""
    if isinstance(controller_or_state, MoERouterReplayController):
        ctrl = controller_or_state
        ctrl.switch_to_replay()
    else:
        ctrl = MoERouterReplayController(mode="replay", recorded=controller_or_state)
    prev = get_active_replay()
    set_active_replay(ctrl)
    try:
        yield ctrl
    finally:
        set_active_replay(prev)


__all__ = [
    "MoERouterReplayController",
    "ReplayMode",
    "get_active_replay",
    "maybe_replay_indices",
    "patch_moe_router_replay",
    "record_moe_routing",
    "replay_moe_routing",
    "set_active_replay",
]
