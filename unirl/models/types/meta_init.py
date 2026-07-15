"""Meta-init support for bundles feeding :class:`VeOmniBackend`.

Materializing a meta-built transformer with ``to_empty()`` clobbers every
init-computed tensor the checkpoint doesn't carry — non-persistent buffers
(diffusers ``PatchEmbed.pos_embed``, rope ``freqs``) and plain ``__dict__``
tensors (Qwen-Image rope). :func:`build_meta_init_transformer` builds under
``init_empty_weights(include_buffers=False)`` (parameters on meta, those tensors
real on CPU) and captures them; callers stash the capture on
``bundle._meta_init_state`` for ``load_trainable_weights`` to restore after the
weight load.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


def capture_init_state(model: nn.Module) -> dict:
    """Capture ``model``'s init-computed non-persistent state as a picklable dict.

    Returns ``{"buffers": {fqn: cpu_tensor}, "attrs": {(mod, attr): cpu_tensor}}``
    — non-persistent buffers plus plain ``__dict__`` tensors, cloned to CPU so the
    capture survives transport (Ray pickling, a rebuilt module). Raises
    ``ValueError`` if any tensor is still on meta (model built under
    ``torch.device("meta")`` instead of ``init_empty_weights(include_buffers=False)``).
    """
    persistent = set(model.state_dict().keys())
    buffers = {name: buf.detach().cpu().clone() for name, buf in model.named_buffers() if name not in persistent}
    attrs = {}
    for mod_name, module in model.named_modules():
        for attr, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                attrs[(mod_name, attr)] = value.detach().cpu().clone()

    on_meta = [name for name, value in buffers.items() if value.is_meta]
    on_meta += [f"{mod_name}.{attr}" for (mod_name, attr), value in attrs.items() if value.is_meta]
    if on_meta:
        raise ValueError(
            "capture_init_state: captured init-state is on the meta device "
            "— nothing real to capture. Build the model under "
            "accelerate.init_empty_weights(include_buffers=False) (parameters on "
            "meta, buffers/attrs real on CPU), not torch.device('meta'). "
            f"Offending tensor(s): {on_meta[:8]}"
        )
    return {"buffers": buffers, "attrs": attrs}


def restore_init_state(model: nn.Module, captured: Optional[dict]) -> int:
    """Copy a :func:`capture_init_state` snapshot back onto a materialized module.

    Buffers are ``copy_``-ed into the live buffers (dtype/device cast); plain attrs
    are re-attached as CPU tensors (forwards ``.to(device)`` them on use). Idempotent;
    ``captured=None`` -> no-op. Returns the number of tensors restored.
    """
    if not captured:
        return 0
    buffers = captured.get("buffers", {})
    attrs = captured.get("attrs", {})
    modules = dict(model.named_modules())
    for fqn, value in buffers.items():
        mod_name, _, buf_name = fqn.rpartition(".")
        owner = modules.get(mod_name) if mod_name else model
        if owner is None or not hasattr(owner, buf_name):
            continue
        live = getattr(owner, buf_name)
        live.copy_(value.to(device=live.device, dtype=live.dtype))
    for (mod_name, attr), value in attrs.items():
        owner = modules.get(mod_name)
        if owner is not None:
            owner.__dict__[attr] = value
    n = len(buffers) + len(attrs)
    if n:
        logger.info("restore_init_state: recovered %d non-persistent buffer(s) + plain attr(s)", n)
    return n


def finalize_meta_init(transformer: nn.Module, *, dtype: torch.dtype) -> nn.Module:
    """Cast a meta-built transformer to ``dtype`` and no-op its ``init_weights``.

    On meta the cast is metadata-only (``to_empty`` later materializes in
    ``dtype``); the no-op stops VeOmni's ``parallelize`` from re-initializing
    after ``to_empty``. Warns if non-persistent buffers are still on meta (built
    under ``torch.device("meta")`` instead of :func:`build_meta_init_transformer`)
    — ``to_empty`` leaves those garbage and they cannot be recovered.

    ``nn.Module.to`` is in place; callers rebind by convention.
    """
    transformer = transformer.to(dtype)
    transformer.init_weights = lambda: None
    non_persistent = set(n for n, _ in transformer.named_buffers()) - set(transformer.state_dict())
    on_meta = sorted(n for n, b in transformer.named_buffers() if n in non_persistent and b.is_meta)
    if on_meta:
        logger.warning(
            "finalize_meta_init: %d non-persistent buffer(s) on META, unrecoverable "
            "after to_empty: %s. Build via build_meta_init_transformer and stash its "
            "capture on bundle._meta_init_state.",
            len(on_meta),
            on_meta[:8],
        )
    return transformer


def build_meta_init_transformer(
    factory: Callable[[], nn.Module],
    *,
    dtype: torch.dtype,
) -> Tuple[nn.Module, dict]:
    """Build ``factory()`` on meta, capturing init-computed non-persistent state.

    Builds under ``init_empty_weights(include_buffers=False)`` (parameters on
    meta, buffers / ``__dict__`` tensors real on CPU), captures that state before
    the dtype cast, then finalizes.

    Returns ``(transformer, captured)``. **Stash** ``captured`` on the bundle as
    ``bundle._meta_init_state``; ``load_trainable_weights`` restores it after the
    sharded weight load. Model-specific quirks stay in the bundle.
    """
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        transformer = factory()
    captured = capture_init_state(transformer)
    transformer = finalize_meta_init(transformer, dtype=dtype)
    return transformer, captured


__all__ = [
    "capture_init_state",
    "restore_init_state",
    "finalize_meta_init",
    "build_meta_init_transformer",
]
