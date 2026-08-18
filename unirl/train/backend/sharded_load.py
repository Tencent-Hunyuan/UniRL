"""Shared sharded-weight loader for the FSDP and VeOmni backends."""

from __future__ import annotations

import glob
import logging
import os
import re
from collections.abc import Callable, Collection, Sequence
from fnmatch import fnmatchcase
from typing import Dict

import torch
from torch import nn

from unirl.models.types.post_materialize import canonical_param_name
from unirl.train.backend.sharded_state import _build_state_dict_options, _current_rank
from unirl.train.backend.veomni.ep.models.qwen3_moe import build_local_fused_block
from unirl.train.backend.veomni.ep.placement import assign_local_block, ep_named_parameters

logger = logging.getLogger(__name__)

StateDict = Dict[str, object]
_PACKED_QWEN_MOE_MODEL_TYPES = frozenset({"qwen2_moe", "qwen3_moe", "qwen3_5_moe"})
_QWEN_MOE_EXPERT_RE = re.compile(r"(^|.*\.)mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")


def load_trainable_weights(
    model: nn.Module,
    bundle: object,
    *,
    device: torch.device,
    rank: int = 0,
    with_aux: tuple[str, ...] = (),
    eager_ok: bool,
) -> None:
    """Resolve a bundle's trainable-weight source and load it post-wrap."""
    weights_path = getattr(bundle, "_transformer_weights_path", None)
    if weights_path is not None:
        load_sharded(model, weights_path, device=device, strict=False)
        # Restore init-only state lost during meta materialization.
        from unirl.models.types.meta_init import restore_init_state

        n_recovered = restore_init_state(model, getattr(bundle, "_meta_init_state", None))
        # Re-tie shared weights after meta materialization.
        retied = False
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False) and hasattr(model, "tie_weights"):
            model.tie_weights()
            retied = True
        logger.info(
            "Rank %s: loaded trainable weights from %s (recovered %d non-persistent tensor(s), retied=%s)",
            rank,
            weights_path,
            n_recovered,
            retied,
        )
        return

    materialize = getattr(bundle, "materialize", None)
    if callable(materialize):
        materialize(device=device, with_aux=tuple(with_aux))
        return

    if not eager_ok:
        raise ValueError(
            "sharded_load: trainable module has no weight source — a meta-init "
            "bundle must stash `_transformer_weights_path` or provide "
            "materialize(). Eagerly-loaded bundles are FSDP-only: this backend's "
            "parallelize already materialized (to_empty) the module, so eager "
            "weights would be clobbered."
        )
    if with_aux:
        logger.info(
            "Rank %s: bundle %s loads eagerly; ignoring with_aux=%s",
            rank,
            type(bundle).__name__,
            tuple(with_aux),
        )


def load_sharded(
    module: nn.Module,
    weights_dir: str,
    *,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Materialize ``module`` from a (diffusers-layout) safetensors directory."""
    if hasattr(module, "_extra_parallel_param_groups"):
        if _module_has_meta_param(module):
            module.to_empty(device=device)
        _load_state_dict_ep_sliced(module, weights_dir, device=device, strict=strict)
        return

    state_dict = _read_safetensors_dir(weights_dir) if _current_rank() == 0 else {}
    _load_state_dict_sharded(module, state_dict, device=device, strict=strict)


def _ep_shard_dims_from_plan(module: nn.Module) -> Dict[str, int]:
    """Resolve each EP parameter's pre-sliced axis from VeOmni's parallel plan."""
    named_ep = dict(ep_named_parameters(module))
    if not named_ep:
        return {}

    resolved: Dict[str, int] = {}
    spec_by_fqn = getattr(module, "_fqn2spec_info", None) or {}
    for name, param in named_ep.items():
        spec = spec_by_fqn.get(name)
        if spec is None or getattr(spec, "para_name", None) != "ep":
            continue
        dim = getattr(getattr(spec, "placement", None), "dim", None)
        if dim is not None:
            resolved[name] = _validate_ep_shard_dim(name, int(dim), param)

    missing = [name for name in named_ep if name not in resolved]
    if missing:
        get_plan = getattr(module, "get_parallel_plan", None)
        if callable(get_plan):
            plan = get_plan()
            ep_plan = getattr(plan, "extra_parallel_plan", {}).get("ep", {})
            for name in missing:
                matches = [
                    int(placement.dim)
                    for pattern, placement in ep_plan.items()
                    if hasattr(placement, "dim") and fnmatchcase(name, pattern)
                ]
                if len(set(matches)) > 1:
                    raise RuntimeError(f"EP load: conflicting shard dims for {name!r}: {matches}.")
                if matches:
                    resolved[name] = _validate_ep_shard_dim(name, matches[0], named_ep[name])

    unresolved = [name for name in named_ep if name not in resolved]
    if unresolved:
        raise RuntimeError(
            "EP load: parallel plan does not declare a Shard placement for "
            f"{len(unresolved)} EP parameter(s): {unresolved[:8]}."
        )
    return resolved


def _read_ep_checkpoint_block(
    tensor_slice,
    *,
    name: str,
    ckpt_shape: tuple,
    global_shape: tuple,
    ep_size: int,
    ep_rank: int,
    ep_dim: int | None,
) -> torch.Tensor:
    """Read one parameter, slicing only the axis declared by the EP plan."""
    if ep_dim is not None and ep_size > 1:
        expected = list(global_shape)
        expected[ep_dim] *= ep_size
        expected_shape = tuple(expected)
        if ckpt_shape == global_shape:
            raise RuntimeError(
                f"EP load: {name!r} contains only one pre-sliced EP block "
                f"{ckpt_shape}; expected full checkpoint shape {expected_shape}."
            )
        if ckpt_shape != expected_shape:
            raise RuntimeError(
                f"EP load: EP param {name!r} shape mismatch: checkpoint "
                f"{ckpt_shape} vs plan-declared full shape {expected_shape} "
                f"(Shard({ep_dim}), local global shape {global_shape})."
            )
        width = global_shape[ep_dim]
        index = [slice(None)] * len(global_shape)
        index[ep_dim] = slice(ep_rank * width, (ep_rank + 1) * width)
        block = tensor_slice[tuple(index)]
        if tuple(block.shape) != global_shape:
            raise RuntimeError(f"EP load: sliced {name!r} to {tuple(block.shape)} != {global_shape}.")
        return block

    if ckpt_shape != global_shape:
        raise RuntimeError(
            f"EP load: shape mismatch for non-EP param {name!r}: checkpoint {ckpt_shape} vs global {global_shape}."
        )
    return tensor_slice[:]


def _validate_ep_shard_dim(name: str, dim: int, param: nn.Parameter) -> int:
    if dim < 0:
        dim += param.ndim
    if dim < 0 or dim >= param.ndim:
        raise RuntimeError(
            f"EP load: parallel plan declares invalid Shard({dim}) for {name!r} shape {tuple(param.shape)}."
        )
    return dim


def _load_state_dict_ep_sliced(
    module: nn.Module,
    weights_dir: str,
    *,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Memory-optimal EP weight load: each rank reads ONLY its expert block."""
    import glob

    from safetensors import safe_open

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    ep_size = int(ps.ep_size) if getattr(ps, "ep_enabled", False) else 1
    ep_rank = int(ps.ep_rank) if ep_size > 1 else 0

    shards = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"EP load: no *.safetensors under {weights_dir!r}")
    handles = {s: safe_open(s, framework="pt", device="cpu") for s in shards}
    key_to_handle = {}
    for s, h in handles.items():
        for k in h.keys():
            key_to_handle[k] = h
    ckpt_keys = set(key_to_handle)

    def get_checkpoint_tensor(key: str) -> torch.Tensor:
        return key_to_handle[key].get_tensor(key)

    ep_shard_dims = _ep_shard_dims_from_plan(module)
    named = {canonical_param_name(n): p for n, p in module.named_parameters()}
    named.update({canonical_param_name(n): b for n, b in module.named_buffers()})

    missing, loaded, local_elems = [], 0, 0
    for name, dst in named.items():
        ckpt_key = name
        if ckpt_key not in ckpt_keys:
            stem, _, leaf = name.rpartition(".")
            cand = f"{stem.removesuffix('.base_layer')}.{leaf}" if stem.endswith(".base_layer") else name
            ckpt_key = cand if cand in ckpt_keys else name
        if ckpt_key not in ckpt_keys:
            block = build_local_fused_block(
                fused_param_name=name,
                expected_shape=tuple(dst.shape),
                ep_rank=ep_rank,
                available_keys=ckpt_keys,
                get_tensor=get_checkpoint_tensor,
            )
            if block is not None:
                block = block.to(device=device, dtype=dst.dtype)
                local_elems += block.numel()
                assign_local_block(dst, block)
                loaded += 1
                del block
                continue
            if name in ep_shard_dims:
                raise RuntimeError(
                    f"EP load: checkpoint contains neither fused tensor {name!r} "
                    "nor the complete HF per-expert keys needed to rebuild it."
                )
            missing.append(name)
            continue

        sl = key_to_handle[ckpt_key].get_slice(ckpt_key)
        ckpt_shape = tuple(sl.get_shape())
        global_shape = tuple(dst.shape)

        block = _read_ep_checkpoint_block(
            sl,
            name=name,
            ckpt_shape=ckpt_shape,
            global_shape=global_shape,
            ep_size=ep_size,
            ep_rank=ep_rank,
            ep_dim=ep_shard_dims.get(name),
        )

        block = block.to(device=device, dtype=dst.dtype)
        local_elems += block.numel()
        assign_local_block(dst, block)
        loaded += 1
        del block

    if strict and missing:
        raise RuntimeError(f"EP load: missing {len(missing)} tensor(s) in checkpoint: {missing[:8]}")
    if missing and _current_rank() == 0:
        logger.info(
            "EP load: %d tensor(s) absent from checkpoint (e.g. non-persistent buffers): %s%s",
            len(missing),
            missing[:6],
            " ..." if len(missing) > 6 else "",
        )
    if _current_rank() == 0:
        logger.info(
            "EP load: loaded %d tensor(s) via get_slice+distribute_tensor (rank0 read %.0fM elems)",
            loaded,
            local_elems / 1e6,
        )


def _load_state_dict_sharded(
    module: nn.Module,
    state_dict: StateDict,
    *,
    device: torch.device,
    strict: bool = False,
) -> None:
    """Allocate storage for any meta params, then broadcast-load ``state_dict``."""
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    if _module_has_meta_param(module):
        module.to_empty(device=device)

    if _current_rank() == 0:
        state_dict = _remap_hf_checkpoint_keys(state_dict, module)
        state_dict = _pack_qwen_moe_expert_keys(state_dict, module)
        state_dict = _remap_lora_base_keys(state_dict, module)
        _assert_state_dict_covers_model(state_dict, module)

    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
        strict=strict,
    )
    try:
        set_model_state_dict(module, state_dict, options=options)
    except TypeError:
        set_model_state_dict(module, state_dict)


def _module_has_meta_param(module: nn.Module) -> bool:
    """True if any parameter of ``module`` (recursing into children) is on the"""
    return any(p.is_meta for p in module.parameters(recurse=True))


def _read_safetensors_dir(weights_dir: str) -> StateDict:
    """Merge all ``*.safetensors`` shards in a directory."""
    from safetensors.torch import load_file

    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"sharded_load: transformer weights dir not found: {weights_dir!r}. "
            "HF repo IDs are not supported here — point the recipe's checkpoint "
            "path at a local download."
        )
    shards = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"sharded_load: no *.safetensors files under {weights_dir!r}")
    state_dict: StateDict = {}
    for shard in shards:
        state_dict.update(load_file(shard, device="cpu"))
    return state_dict


def _qwen_moe_source_base_candidates(target: str, model: nn.Module) -> list[str]:
    """Return checkpoint prefix candidates for one packed model parameter."""
    base = target.removesuffix(".gate_up_proj").removesuffix(".down_proj")
    candidates = [base]
    prefix = str(getattr(model, "base_model_prefix", "") or "")
    if prefix:
        dotted = f"{prefix}."
        candidates.append(base.removeprefix(dotted) if base.startswith(dotted) else f"{prefix}.{base}")
    return list(dict.fromkeys(candidates))


def _qwen_moe_source_groups(
    target: str,
    checkpoint_keys: Collection[str],
    model: nn.Module,
    expert_ids: Sequence[int] | range,
) -> tuple[tuple[str, ...], ...] | None:
    """Resolve complete split-checkpoint key groups for one packed target."""
    if target.endswith(".experts.gate_up_proj"):
        leaves = ("gate_proj", "up_proj")
    elif target.endswith(".experts.down_proj"):
        leaves = ("down_proj",)
    else:
        return None

    checkpoint_keys = set(checkpoint_keys)
    expert_ids = tuple(expert_ids)
    for source_base in _qwen_moe_source_base_candidates(target, model):
        groups = tuple(tuple(f"{source_base}.{expert_id}.{leaf}.weight" for expert_id in expert_ids) for leaf in leaves)
        expected = tuple(key for group in groups for key in group)
        present = tuple(key for key in expected if key in checkpoint_keys)
        if not present:
            continue
        if len(present) != len(expected):
            missing = next(key for key in expected if key not in checkpoint_keys)
            raise RuntimeError(
                f"Qwen MoE load: incomplete per-expert checkpoint for {target!r}: "
                f"{len(present)}/{len(expected)} keys present (e.g. missing {missing!r})."
            )
        return groups
    return None


def _build_qwen_moe_packed_tensor(
    target: str,
    target_shape: tuple[int, ...],
    checkpoint_keys: Collection[str],
    get_tensor: Callable[[str], torch.Tensor],
    *,
    model: nn.Module,
    expert_ids: Sequence[int] | range,
) -> tuple[torch.Tensor, tuple[str, ...]] | None:
    """Build one packed Qwen expert tensor from split checkpoint tensors."""
    groups = _qwen_moe_source_groups(target, checkpoint_keys, model, expert_ids)
    if groups is None:
        return None

    blocks = [torch.stack([get_tensor(key) for key in group], dim=0) for group in groups]
    packed = torch.cat(blocks, dim=1) if len(blocks) == 2 else blocks[0]
    if tuple(packed.shape) != target_shape:
        raise RuntimeError(
            f"Qwen MoE load: rebuilt expert block for {target!r} has shape {tuple(packed.shape)} "
            f"!= target shape {target_shape}."
        )
    return packed.contiguous(), tuple(key for group in groups for key in group)


def _qwen_moe_num_experts(model: nn.Module) -> int:
    """Resolve expert count from text-only or multimodal Qwen configs."""
    config = getattr(model, "config", None)
    for owner in (config, getattr(config, "text_config", None)):
        value = getattr(owner, "num_experts", None)
        if value is not None and int(value) > 0:
            return int(value)
    model_type = getattr(config, "model_type", None)
    raise ValueError(f"Qwen MoE load: invalid num_experts for model_type={model_type!r}.")


def _pack_qwen_moe_expert_keys(state_dict: StateDict, model: nn.Module) -> StateDict:
    """Convert a complete split Qwen-MoE checkpoint to packed model keys."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type not in _PACKED_QWEN_MOE_MODEL_TYPES:
        return state_dict
    if not any(_QWEN_MOE_EXPERT_RE.match(key) for key in state_dict):
        return state_dict

    model_tensors = {canonical_param_name(n): t for n, t in model.named_parameters()}
    model_tensors.update({canonical_param_name(n): b for n, b in model.named_buffers()})
    targets = sorted(
        name for name in model_tensors if name.endswith((".mlp.experts.gate_up_proj", ".mlp.experts.down_proj"))
    )
    if not targets:
        raise ValueError(
            f"Qwen MoE load: checkpoint has split experts for model_type={model_type!r}, "
            "but the target model exposes no packed expert parameters."
        )

    expert_ids = range(_qwen_moe_num_experts(model))
    checkpoint_keys = set(state_dict)
    plans: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    for target in targets:
        groups = _qwen_moe_source_groups(target, checkpoint_keys, model, expert_ids)
        if target in state_dict:
            if groups is not None:
                raise ValueError(f"Qwen MoE load: both packed and split keys are present for {target!r}.")
            continue
        if groups is None:
            raise ValueError(f"Qwen MoE load: missing packed or complete split expert weights for {target!r}.")
        plans.append((target, groups))

    for target, groups in plans:
        result = _build_qwen_moe_packed_tensor(
            target,
            tuple(model_tensors[target].shape),
            checkpoint_keys,
            lambda key: state_dict[key],
            model=model,
            expert_ids=expert_ids,
        )
        assert result is not None
        packed, source_keys = result
        state_dict[target] = packed
        for key in source_keys:
            state_dict.pop(key)

    has_target_mtp = any("mtp" in name.split(".") for name in model_tensors)
    if not has_target_mtp:
        mtp_keys = [key for key in state_dict if "mtp" in key.split(".")]
        for key in mtp_keys:
            state_dict.pop(key)
        if mtp_keys:
            logger.info(
                "Qwen MoE load: ignored %d auxiliary MTP checkpoint tensor(s) absent from the target model.",
                len(mtp_keys),
            )

    leftovers = [key for key in state_dict if _QWEN_MOE_EXPERT_RE.match(key)]
    if leftovers:
        raise ValueError(
            "Qwen MoE load: split expert keys remain after packing and would be ignored with strict=False; "
            f"examples: {leftovers[:8]}."
        )

    logger.info("Packed Qwen MoE checkpoint for model_type=%s: %d tensor(s)", model_type, len(plans))
    return state_dict


def _remap_hf_checkpoint_keys(state_dict: StateDict, model: nn.Module) -> StateDict:
    """Rewrite stale HF checkpoint keys to the constructed model's naming."""
    try:
        from accelerate import init_empty_weights
        from transformers import PreTrainedModel
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightRenaming
    except Exception as exc:
        logger.warning("sharded_load: HF key-renaming unavailable (%s); skipping", exc)
        return state_dict

    unwrapped = getattr(model, "module", model)
    config = getattr(unwrapped, "config", None)
    if config is None:
        return state_dict

    hf_cls = type(unwrapped)
    try:
        from torch.distributed.fsdp import FSDPModule

        if isinstance(unwrapped, FSDPModule):
            hf_cls = type(unwrapped).__mro__[FSDPModule._orig_cls_mro_index]
    except (ImportError, IndexError):
        pass
    if not issubclass(hf_cls, PreTrainedModel):
        return state_dict

    try:
        with init_empty_weights(include_buffers=False):
            ref = hf_cls(config)
        rules = [
            (re.compile(t.source_patterns[0]), t.target_patterns[0])
            for t in get_model_conversion_mapping(ref, add_legacy=True)
            if isinstance(t, WeightRenaming)
        ]
    except Exception as exc:
        logger.warning("sharded_load: could not build HF rename rules (%r); skipping", exc)
        return state_dict

    def rename(key: str) -> str:
        for pattern, target in rules:
            key = pattern.sub(target, key)
        return key

    renamed = {rename(k): v for k, v in state_dict.items()}
    ref_keys = {canonical_param_name(n) for n, _ in ref.named_parameters()} | {
        canonical_param_name(n) for n, _ in ref.named_buffers()
    }
    matched_old = sum(k in ref_keys for k in state_dict)
    matched_new = sum(k in ref_keys for k in renamed)
    if matched_new <= matched_old:  # keys already matched — keep the original
        return state_dict
    logger.info(
        "sharded_load: applied HF checkpoint key-renaming (%d -> %d / %d keys matched)",
        matched_old,
        matched_new,
        len(ref_keys),
    )
    return renamed


def _assert_state_dict_covers_model(state_dict: StateDict, model: nn.Module) -> None:
    """Raise if the checkpoint matches almost none of the model's parameters."""
    params = {canonical_param_name(n) for n, _ in getattr(model, "module", model).named_parameters()}
    matched = sum(k in params for k in state_dict)
    if params and matched < 0.25 * len(params):
        raise ValueError(
            f"sharded_load: checkpoint matches only {matched}/{len(params)} model "
            "parameters — the checkpoint keys do not fit the constructed model "
            "(likely a transformers module-tree rename that _remap_hf_checkpoint_keys "
            "did not cover); strict=False would silently drop them all."
        )


def _remap_lora_base_keys(state_dict: StateDict, model: nn.Module) -> StateDict:
    """Translate base-checkpoint keys for LoRA-injected modules."""
    model_keys = {canonical_param_name(n) for n, _ in model.named_parameters()}
    model_keys.update(canonical_param_name(n) for n, _ in model.named_buffers())
    remapped: StateDict = {}
    for key, value in state_dict.items():
        if key not in model_keys:
            stem, _, leaf = key.rpartition(".")
            candidate = f"{stem}.base_layer.{leaf}" if stem else key
            if candidate in model_keys:
                remapped[candidate] = value
                continue
        remapped[key] = value
    return remapped


__all__ = ["load_trainable_weights", "load_sharded", "_load_state_dict_sharded", "StateDict"]
