"""Wrap functions that mutate the nn.Module tree at build time.

Each function injects structural state (adapters, shadow parameters,
FSDP DTensors) onto the model and stamps ``model._deferred_ops`` with
post-materialize work.  A single call to :func:`apply_deferred_ops`
drains them all — feature-agnostic.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn.parameter import Parameter

from unirl.config.require import require
from unirl.train.shadow import Shadow
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Deferred ops bookkeeping
# ------------------------------------------------------------------


def _stamp(model: nn.Module, op: Callable[[nn.Module], None]) -> None:
    if not hasattr(model, "_deferred_ops"):
        model._deferred_ops: List[Callable[[nn.Module], None]] = []
    model._deferred_ops.append(op)


def apply_deferred_ops(model: nn.Module) -> None:
    """Drain ``_deferred_ops`` after materialize.  Feature-agnostic."""
    for op in getattr(model, "_deferred_ops", []):
        op(model)
    model._deferred_ops = []


# ------------------------------------------------------------------
# inject_lora — no handle
# ------------------------------------------------------------------


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
    adapter_name: str = "default",
) -> None:
    """Inject a single LoRA adapter.  No Shadow, no EMA."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias=str(bias),
        task_type=str(task_type),
    )
    inject_adapter_in_model(peft_cfg, model, adapter_name=adapter_name)

    if _current_rank() == 0:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(
            "inject_lora: adapter %r (rank=%d, alpha=%d, target_modules=%s) — %d trainable params",
            adapter_name,
            rank,
            alpha,
            tuple(target_modules),
            n_trainable,
        )

    _stamp(model, partial(_reset_adapter, name=adapter_name))


# ------------------------------------------------------------------
# inject_nft — returns Shadow handle
# ------------------------------------------------------------------


def inject_nft(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    default: str = "default",
    shadow: str = "old",
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
) -> Shadow:
    """Inject dual LoRA adapters for DiffusionNFT-style EMA.  Returns Shadow."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias=str(bias),
        task_type=str(task_type),
    )
    inject_adapter_in_model(peft_cfg, model, adapter_name=default)
    inject_adapter_in_model(peft_cfg, model, adapter_name=shadow)

    # peft's inject_adapter_in_model installs the LoRA layers but does not flip
    # diffusers' PeftAdapterMixin `_hf_peft_config_loaded` flag, so the model-level
    # `set_adapter` raises "No adapter loaded". Activate `default` the same
    # per-LoraLayer way swap_out does (works for diffusers + plain modules), and
    # mark the flag so downstream diffusers adapter ops stay consistent.
    if hasattr(model, "_hf_peft_config_loaded"):
        model._hf_peft_config_loaded = True
    _activate(model, default)
    _freeze_adapter(model, shadow)

    if _current_rank() == 0:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logger.info(
            "inject_nft: adapters %r + %r (rank=%d, alpha=%d) — %d trainable params",
            default,
            shadow,
            rank,
            alpha,
            n_trainable,
        )

    _stamp(model, partial(_reset_adapter, name=default))
    _stamp(model, partial(_reset_adapter, name=shadow))
    _stamp(model, partial(_copy_adapter, src=default, dst=shadow))

    return Shadow(
        iter_pairs=lambda: _adapter_pairs(model, default, shadow),
        swap_in=lambda: _activate(model, shadow),
        swap_out=lambda: _activate(model, default),
    )


# ------------------------------------------------------------------
# inject_mirror — returns Shadow handle
# ------------------------------------------------------------------


def inject_mirror(
    model: nn.Module,
    *,
    prefix: str = "shadow_",
) -> Shadow:
    """Register shadow_* parameters for full-model EMA.  Returns Shadow."""
    pairs: List[Tuple[nn.Module, str, str]] = []

    for fqn, p in list(model.named_parameters()):
        if not p.requires_grad:
            continue
        parent, attr = _parent_and_attr(model, fqn)
        shadow_attr = prefix + attr
        shadow_param = Parameter(torch.empty_like(p), requires_grad=False)
        parent.register_parameter(shadow_attr, shadow_param)
        pairs.append((parent, attr, shadow_attr))

    if _current_rank() == 0:
        logger.info("inject_mirror: registered %d shadow parameters (prefix=%r)", len(pairs), prefix)

    _stamp(model, partial(_copy_mirror, pairs=pairs))

    return Shadow(
        iter_pairs=lambda: ((getattr(m, a), getattr(m, s)) for m, a, s in pairs),
        swap_in=lambda: _swap_mirror(pairs),
        swap_out=lambda: _swap_mirror(pairs),
    )


# ------------------------------------------------------------------
# fsdp_wrap — no handle
# ------------------------------------------------------------------


def fsdp_wrap(
    model: nn.Module,
    stage: Optional[object] = None,
    *,
    block_class_names: Optional[Tuple[str, ...]] = None,
    param_dtype: str = "bf16",
    cpu_offload: bool = False,
    mixed_precision: bool = True,
    fsdp_mode: str = "full",
    reshard_after_forward: bool = True,
    forward_prefetch: bool = False,
    activation_checkpointing: bool = False,
    use_torch_compile: bool = False,
    master_dtype: Optional[str] = None,
    root_wrap: bool = True,
) -> None:
    """Apply FSDP2 wrapping to the model.  No handle returned — DTensors
    ARE the handle.  Ported from FSDPPolicy._wrap_model.

    If ``block_class_names`` is supplied, it takes precedence and
    ``stage`` is ignored for discovery.  Otherwise we fall back to
    ``_discover_block_classes(model, stage)`` (model __mro__ then stage
    source chain).

    ``root_wrap`` (default ON) adds a root ``fully_shard(model)`` after the
    per-block wrap so the leftover params (embed / final norm / lm_head)
    are sharded + mp_policy'd instead of staying plain replicated tensors.
    The root group deliberately does NOT inherit ``reshard_after_forward``:
    FSDP2's auto policy keeps the root's params materialized after forward,
    which stages rely on for direct post-forward submodule calls (e.g. the
    chunked ``lm_head`` in Qwen3 replay). See ``FSDPConfig.root_wrap`` for
    when to disable it.
    """
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
    from torch.distributed.tensor import DTensor

    target_dtype = parse_torch_dtype(param_dtype, field_name="training.fsdp.param_dtype")
    # Optional high-precision optimizer master for the TRAINABLE (LoRA) params. When set
    # (e.g. fp32) the trainable params are upcast to this dtype in the cast loop below — even
    # under mixed precision — while the frozen base and the all-gathered COMPUTE copy stay
    # param_dtype (bf16) via MixedPrecisionPolicy, so the forward math (and the on-policy
    # GRPO ratio) is unchanged and only the optimizer accumulation gains precision. This lets
    # a bf16-loaded 7B base carry an fp32 LoRA master; without it bf16 master weights lose the
    # ~1e-4 GRPO updates to rounding and the policy drifts into a degenerate (all-white)
    # reward-hack. None (default) leaves the master dtype to the load/mixed-precision policy
    # in the cast loop below (an fp32-LOADED model already keeps an fp32 master for free).
    trainable_dtype = (
        parse_torch_dtype(master_dtype, field_name="training.fsdp.master_dtype") if master_dtype is not None else None
    )

    fsdp_kwargs: Dict[str, object] = {
        "reshard_after_forward": bool(reshard_after_forward),
    }
    if mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=target_dtype,
            reduce_dtype=torch.float32,
        )
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    mesh = _create_device_mesh(fsdp_mode)
    if mesh is not None:
        fsdp_kwargs["mesh"] = mesh

    if block_class_names is None:
        block_class_names = _discover_block_classes(model, stage)
    block_instances = _enumerate_block_instances(model, block_class_names)

    casts = 0
    # Three pre-cast regimes (see trainable_dtype above + MixedPrecisionPolicy),
    # applied uniformly to EVERY param — blocks and leftovers alike; the wrap
    # topology below is orthogonal to the dtype policy:
    #   * explicit master_dtype  → upcast the TRAINABLE (LoRA) params to it even under mixed
    #     precision; the mp_policy still all-gathers them as param_dtype for compute, so only
    #     the optimizer master gains precision (the bf16-base + fp32-LoRA-master case).
    #   * no mp_policy            → storage dtype IS the compute dtype, so pre-cast every
    #     param to param_dtype.
    #   * mixed precision, no master_dtype → do NOT pre-cast: fully_shard keeps shards in the
    #     loaded dtype and casts to mp_policy.param_dtype per forward, so an fp32-loaded model
    #     gets Megatron-style fp32 master weights for free. Pre-casting to bf16 here would
    #     round away the ~1e-6 AdamW steps. (Historically a no-op: models were loaded in bf16.)
    for p in model.parameters():
        if isinstance(p, DTensor) or not p.dtype.is_floating_point:
            continue  # already-wrapped params and ints never cast
        if trainable_dtype is not None and p.requires_grad:
            dst = trainable_dtype
        elif not mixed_precision:
            dst = target_dtype
        else:
            continue
        if p.dtype != dst:
            p.data = p.data.to(dst)
            casts += 1

    for layer in block_instances:
        fully_shard(layer, **fsdp_kwargs)

    if root_wrap and not isinstance(model, FSDPModule):
        # Root wrap: claim the leftover params (everything the block wraps
        # above did not own — embed / final norm / lm_head / time+patch
        # embeds) into a root fully_shard group. The ``isinstance`` guard
        # makes the wrap idempotent and skips the degenerate case where
        # ``model`` itself is a wrapped block instance.
        #
        # The root group must NOT inherit reshard_after_forward: FSDP2's auto
        # policy never reshards the root after forward, keeping its params
        # materialized for post-forward direct submodule calls (Qwen3's chunked
        # lm_head) and activation-checkpoint recomputes. Everything else
        # (mesh / mp_policy / offload_policy) is shared with the block groups.
        root_kwargs = dict(fsdp_kwargs)
        root_kwargs.pop("reshard_after_forward", None)
        fully_shard(model, **root_kwargs)
    else:
        # No root wrap: a TRAINABLE param outside every fully_shard group
        # would receive grads no collective ever DP-syncs (the manual
        # sync_unsharded_grads net was removed with the default root wrap),
        # so its replicas would silently drift apart across ranks. Frozen
        # leftovers (the bagel / hunyuan_image3 LoRA recipes) are fine —
        # they carry no grads — and a single rank has no replicas to drift.
        # Fail fast rather than corrupt the run.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            stray = [n for n, p in model.named_parameters() if p.requires_grad and not isinstance(p, DTensor)]
            require(
                not stray,
                f"fsdp_wrap(root_wrap=false): {len(stray)} trainable param(s) sit outside "
                f"every fully_shard group (e.g. {stray[:3]}); their grads would never be "
                "DP-synced and replicas drift. Enable training.fsdp.root_wrap or freeze them.",
            )

    if forward_prefetch:
        # Cross-block forward prefetch: chain each FSDP group to prefetch the
        # NEXT group's all-gather during its own forward, in forward
        # (named_modules) order — root → group 0 → … → group N — so the
        # per-group all-gather overlaps compute instead of stalling the critical
        # path (a multi-node win; ~no-op over NVLink). Iterates the ACTUAL FSDP
        # groups (root + blocks + any separately-wrapped leftover group), not
        # just block_instances, matching set_grad_sync's walk — so no wrapped
        # group is left unchained. Needs the root wrapped (the default root wrap
        # above) so FSDP2 has initialized the shared all-gather comm context.
        if not isinstance(model, FSDPModule):
            raise ValueError(
                "fsdp_wrap: forward_prefetch=True needs the model root-wrapped so FSDP2 "
                "initializes the shared all-gather comm context, but root_wrap did not run "
                "(training.fsdp.root_wrap=False). Set root_wrap=True, or forward_prefetch=False."
            )
        fsdp_groups = [m for m in model.modules() if isinstance(m, FSDPModule)]
        for cur, nxt in zip(fsdp_groups, fsdp_groups[1:]):
            cur.set_modules_to_forward_prefetch([nxt])

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
            "fsdp_wrap: wrapped %d block(s) of class %r "
            "(%s, cpu_offload=%s, mixed_precision=%s, reshard=%s, prefetch=%s, "
            "ac=%s, compile=%s, dtype_casts=%d, master_dtype=%s, root_wrap=%s)",
            len(block_instances),
            tuple(block_class_names),
            "HSDP" if mesh is not None else "FSDP2",
            cpu_offload,
            mixed_precision,
            reshard_after_forward,
            forward_prefetch,
            activation_checkpointing,
            use_torch_compile,
            casts,
            master_dtype,
            root_wrap,
        )


# ------------------------------------------------------------------
# Block-class discovery (ported from FSDPPolicy)
# ------------------------------------------------------------------


def _discover_block_classes(model: nn.Module, stage: object) -> Tuple[str, ...]:
    for cls in type(model).__mro__:
        attr = getattr(cls, "_no_split_modules", None)
        if attr:
            return tuple(str(n) for n in attr)
    leaf_source = stage
    while hasattr(leaf_source, "source"):
        leaf_source = leaf_source.source
    attr = getattr(type(leaf_source), "_no_split_modules", None)
    if attr:
        return tuple(str(n) for n in attr)
    if _current_rank() == 0:
        logger.warning(
            "fsdp_wrap: no block classes discovered for %r (stage %r). Falling back to root-only wrap.",
            type(model).__name__,
            type(leaf_source).__name__,
        )
    return ()


def _enumerate_block_instances(
    model: nn.Module,
    class_names: Tuple[str, ...],
) -> Tuple[nn.Module, ...]:
    if not class_names:
        return ()
    names = set(class_names)
    return tuple(m for _, m in model.named_modules() if type(m).__name__ in names)


# ------------------------------------------------------------------
# HSDP mesh (ported from FSDPPolicy)
# ------------------------------------------------------------------


def _create_device_mesh(fsdp_mode: str) -> Optional[object]:
    if str(fsdp_mode).strip().lower() != "hybrid":
        return None

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None

    world_size = dist.get_world_size()
    shard_size = 8
    if world_size <= shard_size or world_size % shard_size != 0:
        return None

    from torch.distributed.device_mesh import init_device_mesh

    replicate_size = world_size // shard_size
    mesh = init_device_mesh(
        "cuda",
        (replicate_size, shard_size),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    logger.info("fsdp_wrap: HSDP mesh dp_replicate=%d x dp_shard=%d", replicate_size, shard_size)
    return mesh


# ------------------------------------------------------------------
# Peft helpers
# ------------------------------------------------------------------


def _freeze_adapter(model: nn.Module, name: str) -> None:
    from peft.tuners.lora import LoraLayer

    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if name in bank:
                bank[name].weight.requires_grad = False


def _reset_adapter(model: nn.Module, *, name: str) -> None:
    from peft.tuners.lora import LoraLayer

    n_reset = 0
    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.reset_lora_parameters(name, init_lora_weights=True)
            n_reset += 1
    if _current_rank() == 0:
        logger.info("_reset_adapter(%r): %d LoraLayer(s)", name, n_reset)


def _copy_adapter(model: nn.Module, *, src: str, dst: str) -> None:
    from peft.tuners.lora import LoraLayer

    n_copied = 0
    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if src in bank and dst in bank:
                for sp, dp in zip(bank[src].parameters(), bank[dst].parameters()):
                    dp.data.copy_(sp.data)
                n_copied += 1
    if n_copied == 0:
        raise RuntimeError(f"_copy_adapter: no adapter pairs found for {src!r} -> {dst!r}")


def _adapter_pairs(
    model: nn.Module,
    default: str,
    shadow: str,
) -> list[Tuple[torch.Tensor, torch.Tensor]]:
    from peft.tuners.lora import LoraLayer

    pairs: list[Tuple[torch.Tensor, torch.Tensor]] = []
    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if default in bank and shadow in bank:
                for sp, dp in zip(bank[default].parameters(), bank[shadow].parameters()):
                    pairs.append((sp, dp))
    return pairs


def _activate(model: nn.Module, adapter_name: str) -> None:
    from peft.tuners.lora import LoraLayer

    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.set_adapter(adapter_name)


# ------------------------------------------------------------------
# Mirror helpers
# ------------------------------------------------------------------


def _copy_mirror(model: nn.Module, *, pairs: List[Tuple[nn.Module, str, str]]) -> None:
    for mod, live_attr, shadow_attr in pairs:
        getattr(mod, shadow_attr).data.copy_(getattr(mod, live_attr).data)


def _swap_mirror(pairs: List[Tuple[nn.Module, str, str]]) -> None:
    for mod, live_attr, shadow_attr in pairs:
        live = getattr(mod, live_attr)
        shd = getattr(mod, shadow_attr)
        live.data, shd.data = shd.data, live.data


# ------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------


def _parent_and_attr(model: nn.Module, fqn: str) -> Tuple[nn.Module, str]:
    parts = fqn.rsplit(".", 1)
    if len(parts) == 1:
        return model, parts[0]
    parent = model
    for part in parts[0].split("."):
        parent = getattr(parent, part)
    return parent, parts[1]


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = [
    "inject_lora",
    "inject_nft",
    "inject_mirror",
    "fsdp_wrap",
    "apply_deferred_ops",
]
