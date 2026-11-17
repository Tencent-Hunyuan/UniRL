"""Base class for v2 full-(base-)weight sync handlers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from unirl.config.require import require
from unirl.distributed.group.remote import Remote

logger = logging.getLogger(__name__)


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
        self._expert_weight_transform = backend.expert_weight_export_transform()
        if self._expert_weight_transform is not None and hasattr(self._backend.model, "peft_config"):
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
        self._adapter_name = str(adapter_name) if adapter_name is not None else backend.rollout_adapter_name
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
        rules = {} if name_remap is None else dict(name_remap)
        keys = list(rules.keys())
        for i, (key, value) in enumerate(rules.items()):
            require(isinstance(key, str) and key.count("*") == 1, f"name_remap key {key!r} needs one '*'.")
            require(
                value is None or (isinstance(value, str) and value.count("*") == 1),
                f"name_remap[{key!r}] value: null or one '*'; got {value!r}.",
            )
            require(key != "*" or i == len(keys) - 1, f"name_remap '*' must be last; shadows {keys[i + 1 :]!r}.")
        self._name_remap = rules
        self._track_prefix = "" if track_prefix is None else str(track_prefix)
        self._wire_dtype = parse_torch_dtype(wire_dtype, field_name="wire_dtype", allow_none=True)

    @staticmethod
    def _dist_ready() -> bool:
        import torch.distributed as dist

        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _iter_full_tensors(self) -> Iterator[Tuple[str, "object"]]:
        """Yield ``(name, full_tensor)`` one at a time (lazy → bounded memory)."""
        from unirl.utils.peft_merge import merged_state_dict, raw_state_dict

        if self._lora_merged:
            stream = merged_state_dict(self._backend.model, adapter_name=self._adapter_name, dtype=self._wire_dtype)
        else:
            stream = (
                (name, tensor)
                for name, tensor in raw_state_dict(
                    self._backend.model, adapter_name=self._adapter_name, dtype=self._wire_dtype
                )
                if not name.endswith((".lora_A", ".lora_B"))
            )
        if self._expert_weight_transform is not None:
            stream = self._expert_weight_transform(stream)

        remap = self._name_remap
        if not remap:
            yield from stream
            return
        for name, tensor in stream:
            out: Optional[str] = name
            for key, value in remap.items():
                pre, _, post = key.partition("*")
                if name.startswith(pre) and name.endswith(post) and len(name) >= len(pre) + len(post):
                    if value is None:
                        out = None
                    else:
                        vpre, _, vpost = value.partition("*")
                        out = vpre + name[len(pre) : len(name) - len(post)] + vpost
                    break
            if out is not None:
                yield out, tensor

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


__all__ = ["FullWeightSync"]
