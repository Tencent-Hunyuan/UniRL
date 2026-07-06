"""VeOmni FSDP2 model wrapping for the VeOmni backend.

Calls VeOmni's *inner* ``parallelize_model_fsdp2`` directly — the outer
``build_parallelize_model`` would first upcast the model to fp32 master
weights (``model.float()``), apply HF-API gradient checkpointing, and a
vestigial TP path; bypassing it keeps bf16 master weights and the same
memory/numerics regime as ``unirl.train.backend.fsdp.wrap.fsdp_wrap``
(which force-casts to bf16), so the two backends are A/B-comparable.

Differences vs the fsdp wrap (by VeOmni design, accepted for v1):
* the model root IS ``fully_shard``-ed (root auto-no-reshard) — fine for
  single-module trainables; composites (WAN22/HI3) are out of scope.
* requires the model on the meta device (`init_device="meta"` is asserted
  by VeOmni); materialization happens inside the call (``to_empty`` + the
  model's ``init_weights``, which the bundle stamps to a no-op — real
  weights load afterwards in ``backend.py``).

Runs in the backend constructor after structural injection
(``unirl.train.lora`` / ``unirl.train.ema``) and before the weight load.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
}


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
    """Parallelize ``model`` (on meta) in place via VeOmni FSDP2.

    ``block_class_names`` feeds VeOmni's ``basic_modules`` (its per-module
    ``fully_shard`` targets, unioned with the model's ``_no_split_modules``).

    ``master_dtype`` (e.g. ``"fp32"``) keeps the sharded master weights + optimizer
    states at that dtype while ``MixedPrecisionPolicy(param_dtype)`` still casts the
    all-gathered compute copy to ``param_dtype`` (bf16) — the standard "fp32 master +
    bf16 compute" recipe. Essential for full-finetune RL with tiny gradients (e.g.
    DRPO/GRPO grad-norm ~1e-2): a bf16 master rounds those updates to zero. ``None``
    (default) follows ``param_dtype`` for the master (the prior all-bf16 behavior;
    fine for LoRA, where the trainable adapter update scale is large).
    """
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2

    compute_dtype = parse_torch_dtype(param_dtype, field_name="training.fsdp.param_dtype")
    dtype_name = _DTYPE_NAMES.get(compute_dtype)
    if dtype_name is None:
        raise ValueError(f"veomni_parallelize: unsupported param_dtype {param_dtype!r}")

    # Master-weight dtype: cast on meta (dtype-only, no data) so to_empty
    # materializes storage in this dtype; MixedPrecisionPolicy(param_dtype) then
    # casts the compute copy to bf16. master_dtype=None -> master follows
    # param_dtype (all-bf16). Mirrors fsdp_wrap's master/compute split.
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


def _enumerate_block_instances(
    model: nn.Module,
    class_names: Tuple[str, ...],
) -> Tuple[nn.Module, ...]:
    if not class_names:
        return ()
    names = set(class_names)
    # ``parallelize_model_fsdp2`` (run before this) ``fully_shard``s each block,
    # which renames its class to ``FSDP<OriginalName>``; strip that prefix before
    # matching (a no-op when absent) so AC actually finds the post-shard blocks.
    return tuple(m for _, m in model.named_modules() if type(m).__name__.removeprefix("FSDP") in names)


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = ["veomni_parallelize"]
