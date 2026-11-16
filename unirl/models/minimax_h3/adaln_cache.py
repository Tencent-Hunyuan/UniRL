"""Frozen full-timestep AdaLN tables; see README.md Gotchas for the FSDP and invalidation contract."""

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
from torch import nn

logger = logging.getLogger(__name__)


class _Tables:
    """One model call's six modulation tensors per block, each shaped [U * 3, H]."""

    def __init__(self, owner, key):
        self.owner = owner
        self.key = key
        self.values = {}

    def get(self, module: nn.Module, temb: torch.Tensor):
        """Bypass the entire child FSDP call on a hit, including during AC recomputation."""
        if module not in self.values:
            values = module(temb)
            if any(value.requires_grad for value in values):
                raise ValueError("H3 AdaLN caching requires frozen modulation and timestep embeddings.")
            self.values[module] = tuple(value.detach() for value in values)
            self.owner.misses += 1
        else:
            self.owner.hits += 1
        return tuple(value.clone() for value in self.values[module])

    def run(self, model, block, *args):
        """Capture this call's tables in either checkpoint implementation."""
        if torch.is_grad_enabled() and model.gradient_checkpointing:
            return model._gradient_checkpointing_func(block, *args, adaln_cache_entry=self)
        return block(*args, adaln_cache_entry=self)

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for values in self.values.values() for value in values)


class FrozenAdaLNCache:
    """Bound persistent tables to 32 keys and 256 MiB per rank, excluding live autograd references."""

    max_keys = 32
    max_bytes = 256 * 2**20

    def __init__(self, model: nn.Module):
        self.model = model
        self.projections = tuple(block.adaln_proj for block in model.transformer_blocks)
        self.entries = OrderedDict()
        self.hits = self.misses = 0
        self.key_hook = None

    def clear(self):
        """Invalidate before replacing frozen weights, precision or device."""
        self.entries.clear()

    def apply(self, fn, recurse=True):
        """Invalidate on module moves, including FSDP offload/onload."""
        self.clear()
        return type(self.model)._apply(self.model, fn, recurse=recurse)

    def load(self, module, *args):
        """Invalidate before load_state_dict writes frozen parameters."""
        self.clear()

    def validate(self):
        """Reject trainable AdaLN or timestep upstream parameters after LoRA injection."""
        for module in (self.model.time_proj, self.model.time_embedder, *self.projections):
            if any(parameter.requires_grad for parameter in module.parameters()):
                raise ValueError("H3 AdaLN caching requires all adaln_proj/time_proj/time_embedder parameters frozen.")

    def shard_children(self):
        """Supply frozen children before the parent blocks are sharded."""
        self.validate()
        return self.projections, self.install_key_hook

    def install_key_hook(self):
        """Keep key capture before FSDP's prepended device/cast hook, including on the root."""
        if self.key_hook is not None:
            self.key_hook.remove()
        self.key_hook = self.model.register_forward_pre_hook(self.begin, with_kwargs=True, prepend=True)

    def begin(self, module, args, kwargs):
        """Snapshot the complete original timestep and synchronize the cache decision across ranks."""
        from torch.distributed.fsdp import FSDPModule

        self.validate()
        if not all(isinstance(projection, FSDPModule) for projection in self.projections):
            raise ValueError("H3 AdaLN caching requires UniRL FSDP2 child groups.")
        timestep = kwargs.get("timestep", args[3] if len(args) > 3 else None)
        if timestep.requires_grad:
            raise ValueError("H3 AdaLN caching requires a timestep without gradients.")
        key = (
            timestep.dtype,
            timestep.device,
            tuple(timestep.shape),
            timestep.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes(),
            torch.is_autocast_enabled(timestep.device.type),
            torch.get_autocast_dtype(timestep.device.type),
        )
        entry = self.entries.get(key)
        if torch.distributed.is_initialized():
            hit = torch.full((), int(entry is not None), device=timestep.device, dtype=torch.int32)
            torch.distributed.all_reduce(hit, op=torch.distributed.ReduceOp.MIN)
            if not hit.item():
                entry = None
        if entry is None:
            entry = _Tables(self, key)
        kwargs["adaln_cache_entry"] = entry
        return args, kwargs

    def finish(self, module, args, kwargs, output):
        """Publish only complete model calls; failed forwards cannot leave partial entries."""
        entry = kwargs.get("adaln_cache_entry")
        if output is None or entry is None or len(entry.values) != len(self.projections):
            return
        if entry.nbytes > self.max_bytes:
            return
        self.entries[entry.key] = entry
        self.entries.move_to_end(entry.key)
        while (
            len(self.entries) > self.max_keys or sum(table.nbytes for table in self.entries.values()) > self.max_bytes
        ):
            self.entries.popitem(last=False)

    def log(self):
        """Summarize cumulative layer hits/misses, including replay and AC, on rank zero."""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(
                "H3 AdaLN cache: hits=%d misses=%d keys=%d bytes=%d (cumulative, per rank)",
                self.hits,
                self.misses,
                len(self.entries),
                sum(table.nbytes for table in self.entries.values()),
            )


def install_adaln_cache(model: nn.Module) -> None:
    """Install opt-in hooks before FSDP and validate freezing after structural injection."""
    cache = FrozenAdaLNCache(model)
    model._h3_adaln_cache = cache
    model._fsdp_shard_frozen_children = cache.shard_children
    model._apply = cache.apply
    for projection in cache.projections:
        projection._h3_adaln_group = True
    cache.install_key_hook()
    model.register_forward_hook(cache.finish, with_kwargs=True, always_call=True)
    model.register_load_state_dict_pre_hook(cache.load)
