"""Meta-init support for bundles feeding :class:`VeOmniBackend`."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn

logger = logging.getLogger(__name__)


def capture_init_state(model: nn.Module) -> dict:
    """Capture ``model``'s init-computed non-persistent state as a picklable dict."""
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
    """Copy a :func:`capture_init_state` snapshot back onto a materialized module."""
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
        tgt = live.to_local() if hasattr(live, "to_local") else live
        src = value.to(device=tgt.device, dtype=tgt.dtype)
        if tuple(tgt.shape) != tuple(src.shape):
            raise RuntimeError(
                f"restore_init_state: captured buffer {fqn!r} shape {tuple(src.shape)} "
                f"does not match live local shape {tuple(tgt.shape)}."
            )
        tgt.copy_(src)
    for (mod_name, attr), value in attrs.items():
        owner = modules.get(mod_name)
        if owner is not None:
            owner.__dict__[attr] = value
    n = len(buffers) + len(attrs)
    if n:
        logger.info("restore_init_state: recovered %d non-persistent buffer(s) + plain attr(s)", n)
    return n


def recover_rope_inv_freq(model: nn.Module) -> int:
    """Guaranteed post-materialize RoPE ``inv_freq`` recovery (idempotent)."""
    device = None
    for p in model.parameters():
        loc = p.to_local() if hasattr(p, "to_local") else p
        device = loc.device
        break
    n = 0
    for module_name, m in model.named_modules():
        if getattr(m, "inv_freq", None) is None:
            continue
        cfg = getattr(m, "config", None) or getattr(model, "config", None)
        if cfg is None:
            raise RuntimeError(f"recover_rope_inv_freq: rotary module {module_name!r} has no config.")
        rope_config = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None)
        rope_type = getattr(m, "rope_type", None)
        if rope_type is None and isinstance(rope_config, dict):
            rope_type = rope_config.get("rope_type") or rope_config.get("type")
        rope_type = rope_type or "default"

        if rope_type == "default":
            theta = getattr(cfg, "rope_theta", None)
            if theta is None and isinstance(rope_config, dict):
                theta = rope_config.get("rope_theta")
            theta = theta or 10000.0
            hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
            inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=device) / hd))
        else:
            try:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

                rope_init = ROPE_INIT_FUNCTIONS[rope_type]
                inv_freq, attention_scaling = rope_init(cfg, device)
            except Exception as exc:
                raise RuntimeError(
                    f"recover_rope_inv_freq: failed to initialize rope_type={rope_type!r} for module {module_name!r}."
                ) from exc
            m.attention_scaling = attention_scaling
        with torch.no_grad():
            for bn in ("inv_freq", "original_inv_freq"):
                b = getattr(m, bn, None)
                if b is None:
                    continue
                tgt = b.to_local() if hasattr(b, "to_local") else b
                src = inv_freq.to(device=tgt.device, dtype=tgt.dtype)
                if tuple(tgt.shape) != tuple(src.shape):
                    raise RuntimeError(
                        f"recover_rope_inv_freq: {module_name}.{bn} shape "
                        f"{tuple(tgt.shape)} != recomputed {tuple(src.shape)}."
                    )
                tgt.copy_(src)
        n += 1
    if n:
        logger.info("recover_rope_inv_freq: recomputed inv_freq on %d rotary module(s)", n)
    return n


def _pin_fp32(transformer: nn.Module, keep_in_fp32: Sequence[str]) -> int:
    """Re-cast params/buffers whose name matches ``keep_in_fp32`` back to fp32."""
    patterns = tuple(keep_in_fp32)
    matched = 0
    for name, tensor in list(transformer.named_parameters()) + list(transformer.named_buffers()):
        if not tensor.dtype.is_floating_point or not any(pattern in name for pattern in patterns):
            continue
        matched += 1
        if tensor.dtype != torch.float32:
            tensor.data = tensor.data.to(torch.float32)
    return matched


def finalize_meta_init(
    transformer: nn.Module,
    *,
    dtype: torch.dtype,
    keep_in_fp32: Optional[Sequence[str]] = None,
) -> nn.Module:
    """Apply the shared post-build contract for a meta transformer."""
    if not any(param.is_meta for param in transformer.parameters()):
        raise ValueError("finalize_meta_init requires a transformer with meta parameters.")
    transformer = transformer.to(dtype)
    if keep_in_fp32:
        pinned = _pin_fp32(transformer, keep_in_fp32)
        if pinned == 0:
            raise ValueError(
                f"finalize_meta_init: keep_in_fp32={tuple(keep_in_fp32)!r} matched no floating parameters or buffers"
            )
    transformer.init_weights = lambda: None
    return transformer


def build_meta_init_transformer(
    factory: Callable[[], nn.Module],
    *,
    dtype: torch.dtype,
    keep_in_fp32: Optional[Sequence[str]] = None,
) -> Tuple[nn.Module, dict]:
    """Build ``factory()`` on meta, capturing init-computed non-persistent state."""
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        transformer = factory()
    captured = capture_init_state(transformer)
    transformer = finalize_meta_init(transformer, dtype=dtype, keep_in_fp32=keep_in_fp32)
    return transformer, captured


__all__ = [
    "capture_init_state",
    "restore_init_state",
    "recover_rope_inv_freq",
    "finalize_meta_init",
    "build_meta_init_transformer",
]
