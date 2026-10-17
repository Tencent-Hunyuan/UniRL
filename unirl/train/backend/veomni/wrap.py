"""VeOmni FSDP2 model wrapping for the VeOmni backend."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

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
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2

    compute_dtype = parse_torch_dtype(param_dtype, field_name="training.fsdp.param_dtype")
    dtype_name = canonical_torch_dtype_name(compute_dtype, field_name="training.fsdp.param_dtype")

    master_t = (
        parse_torch_dtype(master_dtype, field_name="training.fsdp.master_dtype") if master_dtype else compute_dtype
    )
    for p in model.parameters():
        if p.dtype.is_floating_point and p.dtype != master_t:
            p.data = p.data.to(master_t)

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

    names = set(block_class_names)
    block_instances = (
        tuple(m for _, m in model.named_modules() if type(m).__name__.removeprefix("FSDP") in names) if names else ()
    )

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
                return _ckpt.checkpoint(lambda *a: orig_fwd(*a, **kwargs), *args, use_reentrant=False)

            return wrapped

        for layer in block_instances:
            layer.forward = _make_ckpt_forward(layer.forward)

    if use_torch_compile:
        for layer in block_instances:
            layer.forward = torch.compile(layer.forward)

    rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    # Populate VeOmni EP parameter groups or EP-aware gradient clipping crashes.
    ps = get_parallel_state()
    if ps.any_extra_parallel_enabled:
        para_names = list(ps.extra_parallel_names)
        groups: dict = {para: [] for para in para_names}
        non_ep: list = []
        for _name, p in model.named_parameters():
            matched = False
            if isinstance(p, DTensor):
                mesh = p.device_mesh
                dim_names = () if mesh.mesh_dim_names is None else mesh.mesh_dim_names
                for para in para_names:
                    if f"{para}_fsdp" in dim_names:
                        groups[para].append(p)
                        matched = True
                        break
            if not matched:
                non_ep.append(p)
        for para in para_names:
            if not groups[para]:
                raise RuntimeError(
                    f"veomni_parallelize: {para} is enabled, but 0 parameters matched the '{para}_fsdp' mesh."
                )
        groups["non_extra_parallel"] = non_ep
        model._extra_parallel_param_groups = groups
        if rank0:
            logger.info(
                "veomni_parallelize: attached _extra_parallel_param_groups %s",
                {k: len(v) for k, v in groups.items()},
            )

    if rank0:
        logger.info(
            "veomni_parallelize: wrapped %d block(s) of class %r + root (dtype=%s, reshard=%s, ac=%s, compile=%s)",
            len(block_instances),
            tuple(block_class_names),
            dtype_name,
            reshard_after_forward,
            activation_checkpointing,
            use_torch_compile,
        )


__all__ = ["veomni_parallelize"]
