"""Frozen FSDP2 model holder for teacher/target networks.

Unlike :class:`FSDPBackend`, this role owns no optimizer, scheduler, EMA, or
checkpoint state. It only shards and materializes one frozen module so a
Self-Forcing DMD real-score teacher does not remain fully replicated per GPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.models.types.bundle import Bundle
from unirl.train.backend.base import resolve_trainable_module
from unirl.train.backend.fsdp.state import fsdp_offload, fsdp_onload
from unirl.train.backend.fsdp.wrap import fsdp_wrap
from unirl.train.backend.sharded_load import load_trainable_weights
from unirl.train.configs import FSDPConfig


class FrozenFSDPModel(Remote):
    """Shard a frozen bundle module for inference-only teacher forwards."""

    def __init__(
        self,
        *,
        bundle: Bundle,
        block_class_names: Tuple[str, ...],
        fsdp_cfg: FSDPConfig,
        device: Optional[torch.device] = None,
        rank: int = 0,
        trainable_attr: str = "transformer",
        with_aux: Tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._bundle = bundle
        self._rank = int(rank)
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = resolve_trainable_module(bundle, trainable_attr)
        model.requires_grad_(False)
        fsdp_wrap(
            model,
            block_class_names=tuple(block_class_names),
            param_dtype=fsdp_cfg.param_dtype,
            cpu_offload=fsdp_cfg.cpu_offload,
            mixed_precision=fsdp_cfg.mixed_precision,
            fsdp_mode=fsdp_cfg.fsdp_mode,
            reshard_after_forward=fsdp_cfg.reshard_after_forward,
            forward_prefetch=fsdp_cfg.forward_prefetch,
            activation_checkpointing=False,
            use_torch_compile=fsdp_cfg.use_torch_compile,
            master_dtype=None,
            root_wrap=getattr(fsdp_cfg, "root_wrap", True),
        )
        load_trainable_weights(
            model,
            bundle,
            device=self._device,
            rank=self._rank,
            with_aux=with_aux,
            eager_ok=True,
        )
        model.requires_grad_(False)
        model.eval()
        self.model = model

    def trainable_module(self) -> torch.nn.Module:
        """Compatibility name: return the held module (which is frozen)."""
        return self.model

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload(self) -> None:
        fsdp_onload(self.model, self._device)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def offload(self) -> None:
        fsdp_offload(self.model)


__all__ = ["FrozenFSDPModel"]
