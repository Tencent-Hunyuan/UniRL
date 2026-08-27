"""Base class for v2 full-(base-)weight sync handlers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from unirl.config.require import require
from unirl.distributed.group.remote import Remote

logger = logging.getLogger(__name__)


def _one_star(s: object) -> bool:
    """True if ``s`` is a string containing exactly one ``"*"``."""
    return isinstance(s, str) and s.count("*") == 1


def _validate_name_remap(
    name_remap: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Optional[str]]:
    """Validate + return the ordered ``name_remap`` rewrite rules (fail-closed)."""
    rules = dict(name_remap or {})
    keys = list(rules.keys())
    for i, (key, value) in enumerate(rules.items()):
        require(_one_star(key), f"name_remap key {key!r} needs one '*'.")
        require(value is None or _one_star(value), f"name_remap[{key!r}] value: null or one '*'; got {value!r}.")
        require(key != "*" or i == len(keys) - 1, f"name_remap '*' must be last; shadows {keys[i + 1 :]!r}.")
    return rules


def _apply_name_remap(name: str, name_remap: Dict[str, Optional[str]]) -> Optional[str]:
    """Apply the ordered single-``*`` rewrite; return the new name or None to drop."""
    for key, value in name_remap.items():
        pre, _, post = key.partition("*")
        # length guard: pre and post must not overlap inside name
        if name.startswith(pre) and name.endswith(post) and len(name) >= len(pre) + len(post):
            if value is None:
                return None
            vpre, _, vpost = value.partition("*")
            return vpre + name[len(pre) : len(name) - len(post)] + vpost
    return name


class FullWeightSync(Remote):
    """Base for full-weight sync Remotes."""

    def __init__(
        self,
        *,
        backend,
        bucket_size_mb: int = 512,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
    ) -> None:
        super().__init__()
        from unirl.utils.dtypes import parse_torch_dtype

        self._backend = backend
        self._bucket_bytes = int(bucket_size_mb) * 1024 * 1024
        self._flush_cache = bool(flush_cache)
        self._lora_merged = bool(lora_merged)
        ep_enabled = bool(
            getattr(self._backend.model, "_extra_parallel_param_groups", None)
            and self._backend.model._extra_parallel_param_groups.get("ep")
        )
        if ep_enabled:
            from unirl.train.backend.veomni.ep.models.qwen3_moe import is_fused_expert_param
            from unirl.train.backend.veomni.ep.placement import ep_named_parameters

            unsupported = [
                name for name, _ in ep_named_parameters(self._backend.model) if not is_fused_expert_param(name)
            ]
            if unsupported:
                raise ValueError(
                    "FullWeightSync: EP full-weight sync currently supports the "
                    "Qwen3-MoE fused expert layout only; use a LoRA sync handler "
                    f"for this model. Unsupported EP params: {unsupported[:4]}."
                )
        if ep_enabled and hasattr(self._backend.model, "peft_config"):
            from unirl.utils.peft_merge import lora_targets_ep_experts

            if lora_targets_ep_experts(self._backend.model):
                raise ValueError(
                    "FullWeightSync: LoRA targeting EP-sharded fused experts is unsupported; "
                    "target attention/shared non-EP modules instead."
                )
            if not self._lora_merged:
                raise ValueError(
                    "FullWeightSync: EP + LoRA requires lora_merged=True. "
                    "Use LocalLoraWeightSync/RemoteLoraWeightSync for adapter-only sync."
                )
        self._adapter_name = str(adapter_name) if adapter_name is not None else str(backend.rollout_adapter_name)
        if self._adapter_name != "default" and not self._lora_merged:
            raise ValueError(
                f"FullWeightSync: adapter_name={self._adapter_name!r} requires "
                f"lora_merged=True (a non-default adapter must be folded into the "
                f"pushed base weights; got lora_merged=False)."
            )
        if self._adapter_name != "default":
            logger.info(
                "%s: rollout engine will be synced from adapter %r (%s).",
                type(self).__name__,
                self._adapter_name,
                "explicit adapter_name" if adapter_name is not None else "backend.rollout_adapter_name",
            )
        self._name_remap = _validate_name_remap(name_remap)
        self._track_prefix = str(track_prefix or "")
        self._wire_dtype = parse_torch_dtype(wire_dtype, field_name="wire_dtype", allow_none=True)

    def _iter_full_tensors(self) -> Iterator[Tuple[str, "object"]]:
        """Yield ``(name, full_tensor)`` one at a time (lazy → bounded memory)."""
        from unirl.utils.peft_merge import merged_state_dict, raw_state_dict

        remap = self._name_remap
        if getattr(self._backend.model, "_extra_parallel_param_groups", None) is not None:
            yield from self._iter_full_tensors_ep()
            return

        if self._lora_merged:
            for name, tensor in merged_state_dict(
                self._backend.model, adapter_name=self._adapter_name, dtype=self._wire_dtype
            ):
                out = _apply_name_remap(name, remap)
                if out is not None:
                    yield out, tensor
            return

        for name, tensor in raw_state_dict(
            self._backend.model, adapter_name=self._adapter_name, dtype=self._wire_dtype
        ):
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                continue
            out = _apply_name_remap(name, remap)
            if out is not None:
                yield out, tensor

    def _iter_full_tensors_ep(self) -> Iterator[Tuple[str, "object"]]:
        """EP-aware weight walk: emit HF per-expert tensors from the EP-sharded model."""
        from unirl.train.backend.veomni import _compat
        from unirl.train.backend.veomni.ep.models.qwen3_moe import (
            fused_expert_kind,
            iter_hf_expert_tensors,
        )
        from unirl.train.backend.veomni.ep.placement import gather_stacked_expert_block
        from unirl.utils.peft_merge import merged_state_dict, raw_state_dict

        _compat.ensure_installed()
        from veomni.distributed.parallel_state import get_parallel_state

        ps = get_parallel_state()
        ep_size = int(ps.ep_size) if getattr(ps, "ep_enabled", False) else 1
        ep_group = ps.ep_group if ep_size > 1 else None
        remap = self._name_remap

        if self._lora_merged:
            state_walk = merged_state_dict(
                self._backend.model,
                adapter_name=self._adapter_name,
                dtype=self._wire_dtype,
            )
        else:
            state_walk = raw_state_dict(
                self._backend.model,
                adapter_name=self._adapter_name,
                dtype=self._wire_dtype,
            )

        for name, full in state_walk:
            expert_kind = fused_expert_kind(name)
            if expert_kind is not None and ep_size > 1:
                stacked = gather_stacked_expert_block(
                    full,
                    ep_size=ep_size,
                    ep_group=ep_group,
                )
                del full
            elif expert_kind is not None:
                stacked = full  # ep_size==1: already the full [E,…]
            else:
                out = _apply_name_remap(name, remap)
                if out is not None:
                    yield out, full
                continue

            for hf_name, tensor in iter_hf_expert_tensors(name, stacked):
                out = _apply_name_remap(hf_name, remap)
                if out is not None:
                    yield out, tensor
            del stacked

    def _iter_buckets(self) -> Iterator[Tuple[List[Tuple[str, "object"]], bool]]:
        """Yield ``(bucket, is_last)`` where ``bucket`` is a list of"""
        bucket: List[Tuple[str, object]] = []
        nbytes = 0
        for name, tensor in self._iter_full_tensors():
            size = tensor.numel() * tensor.element_size()
            if bucket and nbytes + size >= self._bucket_bytes:
                yield bucket, False
                bucket, nbytes = [], 0
            bucket.append((name, tensor))
            nbytes += size
        if bucket:
            yield bucket, True

    @property
    def _my_rank(self) -> int:
        ri = self.rank_info
        return ri.rank if ri is not None else 0


__all__ = ["FullWeightSync"]
