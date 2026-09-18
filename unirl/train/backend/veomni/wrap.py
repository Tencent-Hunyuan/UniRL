"""VeOmni FSDP2 model wrapping for the VeOmni backend."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from unirl.utils.dtypes import canonical_torch_dtype_name, parse_torch_dtype

logger = logging.getLogger(__name__)


def veomni_parallelize(
    model: nn.Module,
    *,
    block_class_names: Tuple[str, ...],
    param_dtype: str = "bf16",
    master_dtype: Optional[str] = None,
    reshard_after_forward: bool = True,
    activation_checkpointing: bool = False,
    use_torch_compile: bool = False,
) -> None:
    """Parallelize ``model`` (on meta) in place via VeOmni FSDP2."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2

    compute_dtype = parse_torch_dtype(param_dtype, field_name="training.fsdp.param_dtype")
    dtype_name = canonical_torch_dtype_name(compute_dtype, field_name="training.fsdp.param_dtype")

    master_t = (
        parse_torch_dtype(master_dtype, field_name="training.fsdp.master_dtype") if master_dtype else compute_dtype
    )
    model.to(master_t)

    mixed_precision = MixedPrecisionConfig(
        enable=True,
        param_dtype=dtype_name,
        reduce_dtype="float32",
    )
    parallelize_model_fsdp2(
        model,
        weights_path=None,
        enable_reshard_after_forward=bool(reshard_after_forward),
        mixed_precision=mixed_precision,
        basic_modules=list(block_class_names),
        init_device="meta",
        enable_fsdp_offload=False,
    )

    block_instances = _enumerate_block_instances(model, block_class_names)

    if activation_checkpointing and not block_instances:
        raise RuntimeError(
            "veomni_parallelize: activation_checkpointing=True but no blocks of class "
            f"{tuple(block_class_names)!r} matched — AC would silently be a no-op and "
            "training would OOM. Check block_class_names against the model."
        )

    if activation_checkpointing:
        from torch.utils import checkpoint as _ckpt

        def _make_ckpt_forward(orig_fwd: object) -> object:
            def wrapped(*args: object, **kwargs: object) -> object:
                def fn(*a: object) -> object:
                    return orig_fwd(*a, **kwargs)

                return _ckpt.checkpoint(fn, *args, use_reentrant=False)

            return wrapped

        for layer in block_instances:
            layer.forward = _make_ckpt_forward(layer.forward)

    if use_torch_compile:
        for layer in block_instances:
            layer.forward = torch.compile(layer.forward)

    # Populate VeOmni EP parameter groups or EP-aware gradient clipping crashes.
    _attach_extra_parallel_param_groups(model)

    if _current_rank() == 0:
        logger.info(
            "veomni_parallelize: wrapped %d block(s) of class %r + root (dtype=%s, reshard=%s, ac=%s, compile=%s)",
            len(block_instances),
            tuple(block_class_names),
            dtype_name,
            reshard_after_forward,
            activation_checkpointing,
            use_torch_compile,
        )


def _attach_extra_parallel_param_groups(model: nn.Module) -> None:
    """Classify params into EP vs non-EP groups and cache them as ``_extra_parallel_param_groups`` on the model."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    if not getattr(ps, "any_extra_parallel_enabled", False):
        return

    try:
        from torch.distributed.tensor import DTensor
    except Exception:  # pragma: no cover - older torch
        from torch.distributed._tensor import DTensor

    para_names = list(ps.extra_parallel_names)
    groups: dict = {para: [] for para in para_names}
    non_ep: list = []
    for _name, p in model.named_parameters():
        matched = False
        if isinstance(p, DTensor):
            mesh = getattr(p, "device_mesh", None)
            dim_names = getattr(mesh, "mesh_dim_names", ()) if mesh is not None else ()
            for para in para_names:
                if f"{para}_fsdp" in dim_names:
                    groups[para].append(p)
                    matched = True
                    break
        if not matched:
            non_ep.append(p)
    groups["non_extra_parallel"] = non_ep
    model._extra_parallel_param_groups = groups

    if _current_rank() == 0:
        counts = {k: len(v) for k, v in groups.items()}
        logger.info("veomni_parallelize: attached _extra_parallel_param_groups %s", counts)


def _enumerate_block_instances(
    model: nn.Module,
    class_names: Tuple[str, ...],
) -> Tuple[nn.Module, ...]:
    if not class_names:
        return ()
    names = set(class_names)
    return tuple(m for _, m in model.named_modules() if type(m).__name__.removeprefix("FSDP") in names)


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["veomni_parallelize"]
