"""FSDPBackend — single-track training-state Remote.

Owns structural injection (LoRA / DiffusionNFT / mirror EMA / FSDP wrap) on the
trainable module exposed by a :class:`Bundle`, plus the ongoing training
state (optimizer, scheduler, EMA, eval-EMA swap, checkpoint, onload/
offload).  Does NOT hold a ``Stage`` or an algorithm — the algorithm
sibling Remote owns the stage and runs forward/backward against the
same shared bundle.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.models.types.bundle import Bundle
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.configs import (
    EmaFullConfig,
    EmaLoraConfig,
    FSDPConfig,
    LoraConfig,
)
from unirl.train.ema import EMA, make_decay_fn
from unirl.train.factories import build_lr_scheduler, build_optimizer
from unirl.train.fsdp_utils import (
    clip_grad_norm,
    fsdp_offload,
    fsdp_onload,
    gather_state_dict,
    load_model_state_dict,
    trainable_params,
)
from unirl.train.inject import (
    apply_deferred_ops,
    fsdp_wrap,
    inject_lora,
    inject_mirror,
    inject_nft,
)
from unirl.train.shadow import Shadow

logger = logging.getLogger(__name__)


class FSDPBackend(Remote):
    """Single-track FSDP training backend.

    One-shot construction: after ``__init__`` returns the backend is
    fully usable (model wrapped, optimizer/scheduler/EMA built).
    Caller is responsible for passing ``device`` and ``rank`` —
    matches :class:`BaseRolloutEngine` subclasses' contract.
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
        if lora_cfg is not None and ema_lora_cfg is not None:
            raise ValueError(
                "FSDPBackend: lora_cfg and ema_lora_cfg are mutually exclusive "
                "(both inject LoRA adapters). Use ema_lora_cfg for DiffusionNFT-style "
                "adapter EMA, or lora_cfg for plain LoRA."
            )

        self._bundle = bundle
        self._rank = int(rank)
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = getattr(bundle, trainable_attr)

        shadow: Optional[Shadow] = None

        if ema_lora_cfg is not None:
            shadow = inject_nft(
                model,
                rank=ema_lora_cfg.rank,
                alpha=ema_lora_cfg.alpha,
                target_modules=tuple(ema_lora_cfg.target_modules),
                default=ema_lora_cfg.default_adapter,
                shadow=ema_lora_cfg.shadow_adapter,
                dropout=ema_lora_cfg.dropout,
                bias=ema_lora_cfg.bias,
                task_type=ema_lora_cfg.task_type,
            )
        elif lora_cfg is not None:
            inject_lora(
                model,
                rank=lora_cfg.rank,
                alpha=lora_cfg.alpha,
                target_modules=tuple(lora_cfg.target_modules),
                dropout=lora_cfg.dropout,
                bias=lora_cfg.bias,
                task_type=lora_cfg.task_type,
            )

        if ema_cfg is not None:
            shadow = inject_mirror(model, prefix=ema_cfg.shadow_prefix)

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

        bundle_materialize = getattr(bundle, "materialize", None)
        if callable(bundle_materialize):
            bundle_materialize(device=self._device, with_aux=tuple(with_aux))
        elif with_aux:
            logger.info(
                "Rank %s: bundle %s loads eagerly; ignoring with_aux=%s",
                self._rank,
                type(bundle).__name__,
                tuple(with_aux),
            )

        apply_deferred_ops(model)

        self.ema: Optional[EMA] = None
        if shadow is not None:
            active_cfg = ema_lora_cfg or ema_cfg
            self.ema = EMA(
                shadow=shadow,
                decay_fn=make_decay_fn(active_cfg),
                timing=active_cfg.timing,
            )

        self.optimizer: torch.optim.Optimizer = build_optimizer(
            optimizer_cfg,
            params=list(trainable_params(model)),
        )
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = build_lr_scheduler(
            scheduler_cfg,
            optimizer=self.optimizer,
        )

        self.model: nn.Module = model
        self._optimizer_step_count: int = 0
        self._eval_ema_active: bool = False
        # No-sync gradient accumulation (see set_grad_sync). Only active under
        # ZeRO-2 (reshard_after_forward=False); a no-op under ZeRO-3, where the
        # per-micro reshard/re-gather interacts badly with deferred sync.
        self._defer_grad_sync: bool = bool(fsdp_cfg.defer_grad_sync) and not bool(fsdp_cfg.reshard_after_forward)
        self._grad_sync_enabled: bool = True

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def set_grad_sync(self, enable: bool) -> None:
        """Toggle the FSDP2 gradient reduce-scatter for no-sync accumulation.

        With ``defer_grad_sync`` on, the train loop disables sync on every
        micro-batch except the last, so every FSDP group accumulates gradients
        in its unsharded buffers and a single reduce-scatter runs per optimizer
        step instead of one per micro-batch. Under the default root wrap the
        whole model is sharded — the per-block groups AND the leftover root
        group (embed / final norm / lm_head) — and the loop below toggles all of
        them, so this defers one reduce-scatter per step for the *entire* model
        on a multi-node fabric.

        No-op when deferral is off (the common case) or the flag is already in
        the wanted state. ``set_is_last_backward`` does not recurse, so every
        FSDP module is toggled; ``set_requires_gradient_sync`` is idempotent
        across nesting.
        """
        if not self._defer_grad_sync or enable == self._grad_sync_enabled:
            return
        from torch.distributed.fsdp import FSDPModule

        for m in self.model.modules():
            if isinstance(m, FSDPModule):
                m.set_requires_gradient_sync(enable)
                m.set_is_last_backward(enable)
        self._grad_sync_enabled = enable

    @property
    def grad_sync_deferred(self) -> bool:
        """True when no-sync accumulation is active (``defer_grad_sync`` under ZeRO-2)."""
        return self._defer_grad_sync

    def optimizer_step(self, *, max_grad_norm: float) -> float:
        """Clip, optimizer step, scheduler step, EMA step.

        The algorithm sibling Remote is responsible for populating grads
        on this backend's model (they share the bundle).  Caller must
        only invoke this when ``has_backward`` was True for the
        accumulated micro-batches.

        Skips the whole step on a non-finite (NaN/Inf) clipped grad norm:
        stepping would scale every parameter by the bad norm and poison the
        weights, crashing the next rollout's sampling. The clipped norm is an
        all-rank scalar so the skip is identical on every rank. This is the one
        optimizer-step chokepoint every v2 trainer (PE / VLM / diffusion via
        ``TrainStack``) routes through, so the guard covers all of them.
        """
        params = list(trainable_params(self.model))
        # Every trainable grad is a sharded DTensor that FSDP reduce-scatters:
        # the root wrap claims the leftover params, and fsdp_wrap fails fast on
        # trainable params outside every group when root_wrap is disabled.
        clipped = clip_grad_norm(params, float(max_grad_norm))
        grad_norm = float(clipped.item()) if isinstance(clipped, torch.Tensor) else float(clipped or 0.0)

        if not math.isfinite(grad_norm):
            logger.warning(
                "FSDPBackend.optimizer_step: non-finite grad norm (%s) at step %d; skipping step.",
                grad_norm,
                self._optimizer_step_count,
            )
            self.optimizer.zero_grad(set_to_none=True)
            return grad_norm

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.step(self._optimizer_step_count)
        self._optimizer_step_count += 1
        return grad_norm

    def on_rollout_end(self) -> None:
        if self.ema is not None:
            self.ema.on_rollout_end(self._optimizer_step_count)

    # ------------------------------------------------------------------
    # Eval-EMA swap
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def apply_eval_ema(self) -> None:
        """Swap the EMA shadow ("old") adapter into live position for rollout.

        Driver-callable (each worker swaps its own model); the DiffusionNFT trainer
        wraps ``rollout.generate`` with this + :meth:`restore_from_eval`.
        No-op when ``ema is None`` (GRPO) or already swapped in.
        """
        if self.ema is None or self._eval_ema_active:
            return
        self.ema.apply_shadow()
        self._eval_ema_active = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def restore_from_eval(self) -> None:
        if self.ema is None or not self._eval_ema_active:
            return
        self.ema.restore_shadow()
        self._eval_ema_active = False

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Gather state on all ranks; write to ``path/checkpoint.pt`` on rank 0."""
        state: Dict[str, object] = {
            "policy_state_dict": gather_state_dict(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        if self._rank != 0:
            return
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, "checkpoint.pt"))

    def load(self, path: str) -> None:
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"FSDPBackend.load: checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self._device)

        load_model_state_dict(self.model, checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload(self) -> None:
        """Move the FSDP train state (params + grads + optimizer) back to GPU.

        Driver-callable across all DP workers (each onloads its own FSDP shard).
        Inverse of :meth:`offload`; the colocate trainers call this before the
        train backward (gated by ``enable_fsdp_offload``)."""
        fsdp_onload(self.model, self._device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self._device)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def offload(self) -> None:
        """Move the FSDP train state (params + grads + optimizer) to CPU.

        Frees GPU memory during the rollout phase so a colocate vLLM/SGLang
        engine fits. Driver-callable across all DP workers (each offloads its
        own FSDP shard). Gated by the trainer's ``enable_fsdp_offload``."""
        fsdp_offload(self.model)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def trainable_module(self) -> nn.Module:
        return self.model

    # ------------------------------------------------------------------
    # Smoke helpers
    # ------------------------------------------------------------------

    def compute_local_param_checksums(
        self,
        *,
        names: List[str],
        prefix: str = "",
    ) -> Dict[str, str]:
        from unirl.rollout.engine.vllm_omni.weight_sync.checksum import (
            fingerprint_tensor,
        )
        from unirl.utils.peft_merge import raw_state_dict

        target = set(names)
        out: Dict[str, str] = {}
        for raw_name, param in raw_state_dict(self.model):
            prefixed = prefix + raw_name
            if prefixed in target:
                out[prefixed] = fingerprint_tensor(param)
        return out

    def randomize_weights_for_smoke(self, seed: int = 0) -> None:
        from torch.distributed.tensor import DTensor

        gen = torch.Generator(device=self._device)
        gen.manual_seed(int(seed) + int(self._rank))
        with torch.no_grad():
            for p in trainable_params(self.model):
                local = p.data
                if isinstance(local, DTensor):
                    shard = local.to_local()
                    shard.copy_(
                        torch.randn(
                            shard.shape,
                            dtype=shard.dtype,
                            device=shard.device,
                            generator=gen,
                        )
                    )
                else:
                    local.copy_(
                        torch.randn(
                            local.shape,
                            dtype=local.dtype,
                            device=local.device,
                            generator=gen,
                        )
                    )
        logger.info(
            "Rank %s: randomize_weights_for_smoke complete (seed=%d)",
            self._rank,
            seed,
        )


__all__ = ["FSDPBackend"]
