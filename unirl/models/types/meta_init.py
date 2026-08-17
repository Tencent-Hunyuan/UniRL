"""Meta-init support for bundles feeding :class:`VeOmniBackend`."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

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


def _canonical_named_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Index modules by names from before activation-checkpoint wrapping."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    def unwrap(module: nn.Module) -> nn.Module:
        seen = set()
        while id(module) not in seen:
            seen.add(id(module))
            if isinstance(module, CheckpointWrapper):
                module = module._checkpoint_wrapped_module
                continue
            get_base_layer = getattr(module, "get_base_layer", None)
            if callable(get_base_layer):
                base_layer = get_base_layer()
                if isinstance(base_layer, nn.Module) and base_layer is not module:
                    module = base_layer
                    continue
            break
        return module

    modules: dict[str, nn.Module] = {}
    visited = set()

    def visit(module: nn.Module, name: str) -> None:
        module = unwrap(module)
        if id(module) in visited:
            return
        visited.add(id(module))
        modules[name] = module
        for child_name, child in module.named_children():
            child_fqn = f"{name}.{child_name}" if name else child_name
            visit(child, child_fqn)

    visit(model, "")
    return modules


def restore_init_state(model: nn.Module, captured: Optional[dict]) -> int:
    """Copy a :func:`capture_init_state` snapshot back onto a materialized module."""
    if not captured:
        return 0
    buffers = captured.get("buffers", {})
    attrs = captured.get("attrs", {})
    modules = _canonical_named_modules(model)
    buffer_targets = []
    attr_targets = []
    missing = []
    for fqn, value in buffers.items():
        mod_name, _, buf_name = fqn.rpartition(".")
        owner = modules.get(mod_name)
        if owner is None or not hasattr(owner, buf_name):
            missing.append(fqn)
            continue
        live = getattr(owner, buf_name)
        buffer_targets.append((fqn, live, value))
    for (mod_name, attr), value in attrs.items():
        owner = modules.get(mod_name)
        if owner is None or attr not in vars(owner):
            missing.append(f"{mod_name}.{attr}" if mod_name else attr)
            continue
        attr_targets.append((owner, attr, value))
    if missing:
        raise RuntimeError(
            "restore_init_state: could not resolve captured tensor owner(s) after model wrapping; "
            f"missing {len(missing)} entry(s): {missing[:8]}"
        )

    for fqn, live, value in buffer_targets:
        tgt = live.to_local() if hasattr(live, "to_local") else live
        src = value.to(device=tgt.device, dtype=tgt.dtype)
        if tuple(tgt.shape) != tuple(src.shape):
            raise RuntimeError(
                f"restore_init_state: captured buffer {fqn!r} shape {tuple(src.shape)} "
                f"does not match live local shape {tuple(tgt.shape)}."
            )
        tgt.copy_(src)
    for owner, attr, value in attr_targets:
        owner.__dict__[attr] = value
    n = len(buffer_targets) + len(attr_targets)
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


def finalize_meta_init(transformer: nn.Module, *, dtype: torch.dtype) -> nn.Module:
    """Apply the shared post-build contract for a meta transformer."""
    if not any(param.is_meta for param in transformer.parameters()):
        raise ValueError("finalize_meta_init requires a transformer with meta parameters.")
    transformer = transformer.to(dtype)
    transformer.init_weights = lambda: None
    return transformer


def build_meta_init_transformer(
    factory: Callable[[], nn.Module],
    *,
    dtype: torch.dtype,
) -> Tuple[nn.Module, dict]:
    """Build ``factory()`` on meta, capturing init-computed non-persistent state."""
    from accelerate import init_empty_weights

    with init_empty_weights(include_buffers=False):
        transformer = factory()
    captured = capture_init_state(transformer)
    transformer = finalize_meta_init(transformer, dtype=dtype)
    return transformer, captured


__all__ = [
    "capture_init_state",
    "restore_init_state",
    "recover_rope_inv_freq",
    "finalize_meta_init",
    "build_meta_init_transformer",
]
