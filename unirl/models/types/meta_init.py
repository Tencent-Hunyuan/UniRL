"""Meta-init support for bundles feeding :class:`VeOmniBackend`.

A transformer built under ``torch.device("meta")`` and later materialized by
``to_empty()`` loses every tensor that checkpoints don't carry:

* **non-persistent registered buffers** (in ``named_buffers()`` but not in
  ``state_dict()``) — e.g. diffusers ``PatchEmbed.pos_embed`` sincos tables;
* **plain tensor attributes** in a module's ``__dict__`` — e.g. Qwen-Image's
  complex rope tables (deliberately not buffers upstream).

:func:`stamp_init_state_restore` captures that init-computed state directly
from the freshly-built model — which must be built under
``accelerate.init_empty_weights(include_buffers=False)`` so its parameters
land on meta while its buffers / ``__dict__`` tensors stay real on CPU — and
stamps a deferred op (the ``unirl.train.deferred`` contract) that restores it
onto the materialized module. The backend drains it via ``apply_deferred_ops``
*after* the post-parallelize weight load, so persistent weights and
init-computed state never clobber each other.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from unirl.train.deferred import _stamp

logger = logging.getLogger(__name__)


def capture_init_state(model: nn.Module) -> dict:
    """Capture ``model``'s init-computed non-persistent state as a picklable dict.

    Returns ``{"buffers": {fqn: cpu_tensor}, "attrs": {(mod, attr): cpu_tensor}}``
    — every registered buffer NOT in ``state_dict()`` (non-persistent) plus every
    plain ``__dict__`` tensor attribute, cloned to CPU so the capture survives any
    transport (Ray pickling, a rebuilt module).

    ``model`` must be built so these tensors are **real on CPU** (parameters may
    live on meta) — i.e. under ``accelerate.init_empty_weights(include_buffers=False)``,
    *not* ``with torch.device("meta")`` (which forces buffers to meta too). Raises
    ``ValueError`` if any captured tensor is still on meta — the tell-tale of the
    wrong build context, which would otherwise restore garbage.
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

    Buffers are ``copy_``-ed into the live (device) buffers with dtype/device cast;
    plain attrs are re-attached as CPU tensors (forwards ``.to(device)`` them on
    use, matching upstream). Idempotent and safe to call on a non-meta-init model
    (``captured=None`` -> no-op). Returns the number of tensors restored.
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


def stamp_init_state_restore(model: nn.Module) -> int:
    """Capture ``model``'s own init-computed tensors; stamp the deferred restore.

    Thin wrapper over :func:`capture_init_state` + :func:`restore_init_state`: the
    capture is closed over and replayed by ``apply_deferred_ops`` after the backend
    materializes the module. This is the IN-PROCESS path; for the live trainer the
    capture is *also* carried on the bundle and replayed by ``load_trainable_weights``
    (the closure on ``model._deferred_ops`` can be lost when the bundle crosses Ray
    actors). Both restores are idempotent. Returns the number of captured tensors.
    """
    captured = capture_init_state(model)
    n = len(captured["buffers"]) + len(captured["attrs"])
    if n == 0:
        return 0
    _stamp(model, lambda materialized: restore_init_state(materialized, captured))
    return n


def finalize_meta_init(transformer: nn.Module, *, dtype: torch.dtype) -> nn.Module:
    """Finalize a transformer just built on the meta device for the backends'
    ``load_sharded`` path (shared by every meta-init bundle):

    * dtype-cast — on meta this only sets the dtype (no storage, no data move),
      so the backend's ``to_empty`` later materializes directly in ``dtype``;
    * stamp ``init_weights`` to a no-op — VeOmni's ``parallelize`` calls it
      unconditionally after ``to_empty``; the real weights load afterwards;
    * warn about non-persistent buffers absent from the checkpoint — if the
      model relies on their init-time values they must be restored via
      :func:`stamp_init_state_restore` (see SD3's sincos ``pos_embed`` and
      Qwen-Image's rope tables).

    ``nn.Module.to`` is in place and returns ``self``; callers rebind by
    convention. Quirk fixes that must run *before* the cast (e.g. rebuilding
    rope modules whose tables stay on meta) should be applied by the caller
    before invoking this.
    """
    transformer = transformer.to(dtype)
    transformer.init_weights = lambda: None
    non_persistent = sorted(set(n for n, _ in transformer.named_buffers()) - set(transformer.state_dict()))
    if non_persistent:
        logger.warning(
            "finalize_meta_init: %d non-persistent buffer(s) absent from the "
            "checkpoint and NOT restored by the weight load: %s%s. If the model "
            "relies on their init-time values, stamp stamp_init_state_restore.",
            len(non_persistent),
            non_persistent[:8],
            " ..." if len(non_persistent) > 8 else "",
        )
    return transformer


__all__ = [
    "stamp_init_state_restore",
    "capture_init_state",
    "restore_init_state",
    "finalize_meta_init",
]
