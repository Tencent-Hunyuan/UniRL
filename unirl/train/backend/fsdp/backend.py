"""FSDPBackend — single-track training-state Remote on torch-native FSDP2.

Owns structural injection (LoRA / NFT / mirror EMA) and the torch-native
``fully_shard`` wrap of the trainable module exposed by a :class:`Bundle`. All
the ongoing training state (optimizer, scheduler, EMA, eval-EMA swap, checkpoint,
onload/offload) lives in :class:`~unirl.train.backend.base_backend.BaseFSDP2Backend`;
this leaf supplies only the constructor lifecycle and the five engine hooks
(grad clip, optimizer-state gather/load, model on/offload).

Does NOT hold a ``Stage`` or an algorithm — the algorithm sibling Remote owns
the stage and runs forward/backward against the same shared bundle.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from unirl.models.types.bundle import Bundle
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig, resolve_trainable_module
from unirl.train.backend.base_backend import BaseFSDP2Backend
from unirl.train.backend.fsdp.state import clip_grad_norm, fsdp_offload, fsdp_onload
from unirl.train.backend.fsdp.wrap import fsdp_wrap
from unirl.train.backend.sharded_load import load_trainable_weights
from unirl.train.backend.sharded_state import (
    StateDict,
    gather_optimizer_state_dict,
    load_optimizer_state_dict,
    trainable_params,
)
from unirl.train.configs import (
    EmaFullConfig,
    EmaLoraConfig,
    FSDPConfig,
    LoraConfig,
)
from unirl.train.deferred import apply_deferred_ops


class FSDPBackend(BaseFSDP2Backend):
    """Single-track FSDP training backend.

    One-shot construction: after ``__init__`` returns the backend is fully usable
    (model wrapped, optimizer/scheduler/EMA built). Caller is responsible for
    passing ``device`` and ``rank`` — matches :class:`BaseRolloutEngine`
    subclasses' contract.
    """

    def __init__(
        self,
        *,
        bundle: Bundle,
        block_class_names: Tuple[str, ...],
        fsdp_cfg: FSDPConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: LrSchedulerConfig,
        device: Optional[torch.device] = None,
        rank: int = 0,
        trainable_attr: str = "transformer",
        lora_cfg: Optional[LoraConfig] = None,
        ema_lora_cfg: Optional[EmaLoraConfig] = None,
        ema_cfg: Optional[EmaFullConfig] = None,
        with_aux: Tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self._check_lora_exclusivity(lora_cfg, ema_lora_cfg)

        self._bundle = bundle
        self._rank = int(rank)
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = resolve_trainable_module(bundle, trainable_attr)
        shadow = self._inject_structural(model, lora_cfg, ema_lora_cfg, ema_cfg)

        fsdp_wrap(
            model,
            block_class_names=tuple(block_class_names),
            param_dtype=fsdp_cfg.param_dtype,
            cpu_offload=fsdp_cfg.cpu_offload,
            mixed_precision=fsdp_cfg.mixed_precision,
            fsdp_mode=fsdp_cfg.fsdp_mode,
            reshard_after_forward=fsdp_cfg.reshard_after_forward,
            forward_prefetch=fsdp_cfg.forward_prefetch,
            activation_checkpointing=fsdp_cfg.activation_checkpointing,
            use_torch_compile=fsdp_cfg.use_torch_compile,
            master_dtype=getattr(fsdp_cfg, "master_dtype", None),
            root_wrap=getattr(fsdp_cfg, "root_wrap", True),
        )

        # Real weights: meta-init bundles stash a safetensors dir (load_sharded
        # to_empty-materializes the still-meta module then broadcasts); Pattern-A
        # bundles (hi3) materialize themselves; eager bundles already hold real
        # weights (fsdp_wrap sharded them in place), so eager_ok=True is a no-op.
        load_trainable_weights(
            model,
            bundle,
            device=self._device,
            rank=self._rank,
            with_aux=with_aux,
            eager_ok=True,
        )

        apply_deferred_ops(model)

        self._finalize_construction(
            model,
            shadow,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            lora_cfg=lora_cfg,
            ema_lora_cfg=ema_lora_cfg,
            ema_cfg=ema_cfg,
            fsdp_cfg=fsdp_cfg,
        )

    # ------------------------------------------------------------------
    # Engine hooks (torch-native FSDP2)
    # ------------------------------------------------------------------

    def _clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        # Every trainable grad is a sharded DTensor that FSDP reduce-scatters:
        # the root wrap claims the leftover params, and fsdp_wrap fails fast on
        # trainable params outside every group when root_wrap is disabled.
        return clip_grad_norm(list(trainable_params(self.model)), max_grad_norm)

    def _gather_optimizer_state(self) -> StateDict:
        return gather_optimizer_state_dict(self.model, self.optimizer)

    def _load_optimizer_state(self, optimizer_state: StateDict) -> None:
        load_optimizer_state_dict(self.model, self.optimizer, optimizer_state)

    def _onload_model(self) -> None:
        fsdp_onload(self.model, self._device)

    def _offload_model(self) -> None:
        fsdp_offload(self.model)


__all__ = ["FSDPBackend"]
