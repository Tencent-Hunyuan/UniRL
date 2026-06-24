"""BaseFSDP2Backend — the shared training-state Remote for FSDP2 backends.

Holds everything identical across the torch-native and VeOmni FSDP2 backends:
the training step, the EMA eval-swap, the hardened checkpoint envelope, the
memory lifecycle, and the construction scaffolding (structural injection +
EMA/optimizer/scheduler build). The two concrete backends
(:class:`~unirl.train.backend.fsdp.FSDPBackend` and
:class:`~unirl.train.backend.veomni.VeOmniBackend`) subclass this and supply only
their engine-specific behavior:

* the constructor *lifecycle* (wrap strategy, distributed bring-up, sequence
  parallelism, eager-vs-meta weight load) stays in each leaf, written as a linear
  named sequence that calls the shared construction helpers here; and
* five small *hooks* — grad clip, optimizer-state gather/load, model on/offload —
  that the methods below dispatch through.

This module imports torch (and ema / lora / optim / deferred) at module level and
MUST NOT be imported from ``veomni/__init__`` — only from inside the two
``backend.py`` files. It imports neither ``veomni`` nor either leaf backend.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Literal, Optional

import torch
import torch.distributed as dist
from torch import nn

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.sharded_state import (
    StateDict,
    _current_rank,
    drop_meta_entries,
    gather_lora_state_dict,
    gather_state_dict,
    load_model_state_dict,
    load_sharded_model_state_dict,
    load_sharded_optimizer_state_dict,
    move_optimizer_state,
    sharded_model_state_dict,
    sharded_optimizer_state_dict,
    trainable_params,
)
from unirl.train.configs import EmaFullConfig, EmaLoraConfig, FSDPConfig, LoraConfig
from unirl.train.ema import EMA, Shadow, inject_mirror, inject_nft, make_decay_fn
from unirl.train.lora import inject_lora
from unirl.train.optim import build_lr_scheduler, build_optimizer

logger = logging.getLogger(__name__)


class BaseFSDP2Backend(Remote):
    """Shared base for the single-track FSDP2 training backends.

    Subclasses own ``__init__`` (the engine-specific lifecycle) and the five
    hooks at the bottom of this class; everything else is shared. After a leaf
    ``__init__`` returns the backend is fully usable (model wrapped, weights
    loaded, optimizer/scheduler/EMA built).
    """

    # --- attribute contract (set by the leaf ctor + _finalize_construction) ---
    _bundle: object
    _rank: int
    _device: torch.device
    model: nn.Module
    ema: Optional[EMA]
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    _optimizer_step_count: int
    _eval_ema_active: bool
    _lora_meta: Optional[Dict[str, object]]
    _rollout_adapter_name: str
    _defer_grad_sync: bool
    _grad_sync_enabled: bool

    # ------------------------------------------------------------------
    # Construction helpers (called in sequence from each leaf __init__)
    # ------------------------------------------------------------------

    def _check_lora_exclusivity(
        self,
        lora_cfg: Optional[LoraConfig],
        ema_lora_cfg: Optional[EmaLoraConfig],
    ) -> None:
        if lora_cfg is not None and ema_lora_cfg is not None:
            raise ValueError(
                f"{type(self).__name__}: lora_cfg and ema_lora_cfg are mutually "
                "exclusive (both inject LoRA adapters). Use ema_lora_cfg for "
                "NFT-style adapter EMA, or lora_cfg for plain LoRA."
            )

    def _inject_structural(
        self,
        model: nn.Module,
        lora_cfg: Optional[LoraConfig],
        ema_lora_cfg: Optional[EmaLoraConfig],
        ema_cfg: Optional[EmaFullConfig],
    ) -> Optional[Shadow]:
        """Structural injection on the (possibly meta) trainable module.

        LoRA / NFT-adapter-EMA / mirror-EMA injection — exactly the
        ``unirl.train.deferred`` contract (mutate now; the post-materialize
        resets are drained by ``apply_deferred_ops`` after the weight load).
        Returns the EMA :class:`Shadow` (or ``None``).
        """
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
        return shadow

    def _finalize_construction(
        self,
        model: nn.Module,
        shadow: Optional[Shadow],
        *,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: LrSchedulerConfig,
        lora_cfg: Optional[LoraConfig],
        ema_lora_cfg: Optional[EmaLoraConfig],
        ema_cfg: Optional[EmaFullConfig],
        fsdp_cfg: FSDPConfig,
    ) -> None:
        """Build EMA / optimizer / scheduler and set the shared train state.

        Called at the end of each leaf constructor — after the model is wrapped,
        weight-loaded, and its deferred ops drained — so the backend is fully
        usable once the leaf ``__init__`` returns.
        """
        self.model = model

        self.ema = None
        if shadow is not None:
            active_cfg = ema_lora_cfg or ema_cfg
            self.ema = EMA(
                shadow=shadow,
                decay_fn=make_decay_fn(active_cfg),
                timing=active_cfg.timing,
            )

        self.optimizer = build_optimizer(
            optimizer_cfg,
            params=list(trainable_params(model)),
        )
        self.scheduler = build_lr_scheduler(
            scheduler_cfg,
            optimizer=self.optimizer,
        )

        self._optimizer_step_count: int = 0
        self._eval_ema_active: bool = False
        # Checkpoint storage backend ("torch" legacy single-file vs "dcp"
        # sharded). save honors this; load auto-detects the on-disk format.
        checkpoint_format = str(getattr(fsdp_cfg, "checkpoint_format", "torch"))
        if checkpoint_format not in ("torch", "dcp"):
            raise ValueError(
                f"{type(self).__name__}: fsdp_cfg.checkpoint_format must be 'torch' or 'dcp', got {checkpoint_format!r}"
            )
        self._checkpoint_format: str = checkpoint_format
        self._checkpoint_async: bool = bool(getattr(fsdp_cfg, "checkpoint_async", False))
        # Checkpointed for export tooling: the LoRA fold needs scaling =
        # alpha / rank, and alpha is not derivable from the weights.
        active_lora = lora_cfg or ema_lora_cfg
        self._lora_meta = (
            {
                "rank": active_lora.rank,
                "alpha": active_lora.alpha,
                "target_modules": active_lora.target_modules,
                "dropout": active_lora.dropout,
                "bias": active_lora.bias,
                "task_type": active_lora.task_type,
            }
            if active_lora is not None
            else None
        )
        # Single source of truth for "which adapter the rollout samples under":
        # the EMA shadow ("old") for DiffusionNFT adapter-EMA, else the trainable
        # "default". The in-process eval-EMA swap and the weight sync to a
        # SEPARATE engine both derive from this, so they cannot disagree.
        self._rollout_adapter_name = str(ema_lora_cfg.shadow_adapter) if ema_lora_cfg is not None else "default"
        # No-sync gradient accumulation (see set_grad_sync). Only active under
        # ZeRO-2 (reshard_after_forward=False); a no-op under ZeRO-3, where the
        # per-micro reshard/re-gather interacts badly with deferred sync.
        self._defer_grad_sync = bool(fsdp_cfg.defer_grad_sync) and not bool(fsdp_cfg.reshard_after_forward)
        self._grad_sync_enabled = True

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def set_grad_sync(self, enable: bool) -> None:
        """Toggle the FSDP2 gradient reduce-scatter for no-sync accumulation.

        With ``defer_grad_sync`` on, the train loop disables sync on every
        micro-batch except the last, so every FSDP group accumulates gradients in
        its unsharded buffers and a single reduce-scatter runs per optimizer step
        instead of one per micro-batch (a multi-node win; ~no-op over NVLink).
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
        """Clip (via the engine hook), optimizer step, scheduler step, EMA step.

        The algorithm sibling Remote populates grads on this backend's model
        (they share the bundle); caller invokes this only when ``has_backward``
        was True for the accumulated micro-batches.

        Skips the whole step on a non-finite (NaN/Inf) clipped grad norm:
        stepping would scale every parameter by the bad norm and poison the
        weights, crashing the next rollout's sampling. The clipped norm is an
        all-rank scalar so the skip is identical on every rank. This is the one
        optimizer-step chokepoint every v2 trainer routes through.
        """
        clipped = self._clip_grad_norm(float(max_grad_norm))
        grad_norm = float(clipped.item()) if isinstance(clipped, torch.Tensor) else float(clipped or 0.0)

        if not math.isfinite(grad_norm):
            # On a skipped step (already discarded), pinpoint which params carry the
            # non-finite grad (sharded DTensor -> check the local shard) so the
            # offending module is identifiable from the log. Runs only on this skip
            # path, so it adds nothing to healthy steps.
            bad_params = []
            total_with_grad = 0
            for _name, _p in self.model.named_parameters():
                if _p.grad is None:
                    continue
                total_with_grad += 1
                _gl = _p.grad.to_local() if hasattr(_p.grad, "to_local") else _p.grad
                if _gl.numel() and not bool(torch.isfinite(_gl).all()):
                    bad_params.append(_name)
            logger.warning(
                "%s.optimizer_step: non-finite grad norm (%s) at step %d; skipping step. "
                "%d/%d grad-params non-finite; first: %s",
                type(self).__name__,
                grad_norm,
                self._optimizer_step_count,
                len(bad_params),
                total_with_grad,
                bad_params[:12],
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

    @property
    def rollout_adapter_name(self) -> str:
        """Adapter the rollout must sample under (single source of truth).

        The EMA shadow (``"old"``) for DiffusionNFT-style adapter EMA, else the
        trainable ``"default"``. The weight-sync handlers read this to decide
        which adapter to push to a SEPARATE engine, mirroring the in-process
        :meth:`apply_eval_ema` swap — so an off-policy algorithm rolls out under
        the same weights whether the engine is colocated or separate.
        """
        return self._rollout_adapter_name

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def apply_eval_ema(self) -> None:
        """Swap the EMA shadow ("old") adapter into live position for rollout.

        Driver-callable (each worker swaps its own model); the NFT trainer wraps
        ``rollout.generate`` with this + :meth:`restore_from_eval`. No-op when
        ``ema is None`` (GRPO) or already swapped in.
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
    # Checkpoint (one hardened envelope; optimizer mechanism is per-backend)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def save(self, path: str, step: Optional[int] = None, mode: str = "full") -> None:
        """Save training state; dispatch on ``checkpoint_format`` ("torch" | "dcp").

        ``step`` is the trainer's rollout step — :meth:`load` returns it so
        the loop resumes where it stopped. ``mode="adapter"`` gathers only the
        LoRA keys in the model state (MBs instead of GBs; the frozen base reloads
        from the pretrained snapshot on resume). ``mode="auto"`` selects
        adapter mode when LoRA is present, otherwise full. The optimizer state is
        identical under all modes — it only ever covers trainable params.

        "torch" gathers a full state dict to dist rank 0 and writes a single
        ``checkpoint.pt``. "dcp" writes per-rank shards directly under
        ``path`` via DCP's ``checkpoint_id`` (including its ``.metadata``) plus
        a light app-level ``metadata.pt`` on rank 0 —
        this is the path that supports 80B meta-init bundles and reshard.
        """
        mode = self._resolve_save_mode(mode)
        if mode == "adapter" and not any("lora_" in name for name, _ in self.model.named_parameters()):
            raise RuntimeError(f"{type(self).__name__}.save: mode='adapter' but the model has no LoRA params")
        if self._checkpoint_format == "dcp":
            self._save_dcp(path, step, mode)
        else:
            self._save_torch(path, step, mode)

    def _save_torch(self, path: str, step: Optional[int], mode: str) -> None:
        """Legacy single-file save: gather full state to rank 0, torch.save.

        The optimizer gather is per-backend (DCP get-state for torch-native
        FSDP, plain ``state_dict()`` for VeOmni) via :meth:`_gather_optimizer_state`.
        """
        self._reject_meta(operation="save", checkpoint_format="torch", mode=mode)
        if mode == "adapter":
            policy_state = gather_lora_state_dict(self.model)
        else:
            policy_state = gather_state_dict(self.model)
        optimizer_state = self._gather_optimizer_state()
        state: Dict[str, object] = {
            "policy_state_dict": policy_state,
            "optimizer_state_dict": optimizer_state,
            "optimizer_step_count": self._optimizer_step_count,
            "step": step,
            "save_mode": mode,
            "lora_config": self._lora_meta,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        # The gathers above populate dist rank 0 only — that rank writes. (NOT
        # self._rank: that is a constructor kwarg, identical on every worker.)
        if _current_rank() != 0:
            return
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, "checkpoint.pt"))

    def _save_dcp(self, path: str, step: Optional[int], mode: str) -> None:
        """Sharded save: every rank writes its own shard under ``path``.

        Never gathers a full tensor on any single rank, so meta-init bundles
        (whose frozen aux stays on meta) are supported — those keys carry no
        data and are dropped here. Non-tensor metadata (step / save_mode /
        lora_config / scheduler / optimizer_step_count) is light and rides in a
        rank-0 ``metadata.pt`` beside DCP's own ``.metadata``.
        """
        import torch.distributed.checkpoint as dcp

        self._reject_meta(operation="save", checkpoint_format="dcp", mode=mode)
        os.makedirs(path, exist_ok=True)
        model_sd = drop_meta_entries(sharded_model_state_dict(self.model))
        if mode == "adapter":
            model_sd = {k: v for k, v in model_sd.items() if "lora_A" in k or "lora_B" in k}
        sharded_state: Dict[str, object] = {
            "model": model_sd,
            "optim": sharded_optimizer_state_dict(self.model, self.optimizer),
        }
        dcp.save(sharded_state, checkpoint_id=path)

        if _current_rank() != 0:
            return
        meta: Dict[str, object] = {
            "optimizer_step_count": self._optimizer_step_count,
            "step": step,
            "save_mode": mode,
            "lora_config": self._lora_meta,
        }
        if self.scheduler is not None:
            meta["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(meta, os.path.join(path, "metadata.pt"))

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def load(self, path: str) -> int:
        """Restore state written by :meth:`save`; return the saved rollout step (0 if absent).

        Auto-detects the on-disk format: DCP's root ``path/.metadata`` loads
        via DCP (each rank reads only its own shard, reshard-aware); otherwise
        the legacy ``path/checkpoint.pt`` loads via torch. So checkpoints
        written before this change still resume regardless of
        ``checkpoint_format``. The optimizer restore is per-backend via the
        :meth:`_load_optimizer_state` hook.
        Adapter-mode checkpoints load non-strict — only the LoRA keys are
        present; the frozen base keeps the weights the bundle loaded.
        """
        dcp_metadata_path = os.path.join(path, ".metadata")
        metadata_path = os.path.join(path, "metadata.pt")
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        # Agree on visibility BEFORE the collectives: on multi-node, a rank
        # whose node does not mount the checkpoint path would raise alone and
        # strand the others in the load collective until the NCCL timeout.
        local_visible = {
            "dcp": os.path.exists(dcp_metadata_path),
            "metadata": os.path.exists(metadata_path),
            "torch": os.path.exists(checkpoint_path),
        }
        if dist.is_available() and dist.is_initialized():
            verdicts: List[Optional[Dict[str, bool]]] = [None] * dist.get_world_size()
            dist.all_gather_object(verdicts, local_visible)
        else:
            verdicts = [local_visible]

        saw_dcp = [rank for rank, ok in enumerate(verdicts) if ok and ok["dcp"]]
        if saw_dcp:
            missing_dcp = [rank for rank, ok in enumerate(verdicts) if not (ok and ok["dcp"])]
            if missing_dcp:
                raise FileNotFoundError(
                    f"FSDPBackend.load: DCP metadata not visible on rank(s) {missing_dcp}: {dcp_metadata_path} "
                    "(save_dir/load_dir must live on storage mounted on every node)"
                )
            missing_meta = [rank for rank, ok in enumerate(verdicts) if not (ok and ok["metadata"])]
            if missing_meta:
                raise FileNotFoundError(
                    f"FSDPBackend.load: app metadata not visible on rank(s) {missing_meta}: {metadata_path}"
                )
            return self._load_dcp(path)

        missing_on = [rank for rank, ok in enumerate(verdicts) if not (ok and ok["torch"])]
        if missing_on:
            raise FileNotFoundError(
                f"{type(self).__name__}.load: checkpoint not visible on rank(s) {missing_on}: {checkpoint_path} "
                "(save_dir/load_dir must live on storage mounted on every node)"
            )
        return self._load_torch(checkpoint_path)

    def _load_torch(self, checkpoint_path: str) -> int:
        """Legacy single-file load: read full state, broadcast from rank 0, reshard.

        Every rank loads the file: the model broadcast tolerates {} on non-zero
        ranks, but a plain-``state_dict`` optimizer restore (VeOmni) needs the
        real dict locally on every rank — restored via :meth:`_load_optimizer_state`.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        mode = checkpoint.get("save_mode", "full")
        strict = mode == "full"
        self._reject_meta(operation="load", checkpoint_format="torch", mode=mode)
        load_model_state_dict(self.model, checkpoint["policy_state_dict"], strict=strict)
        self._load_optimizer_state(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("optimizer_step_count") is not None:
            self._optimizer_step_count = int(checkpoint["optimizer_step_count"])
        return int(checkpoint.get("step") or 0)

    def _resolve_save_mode(self, mode: str) -> str:
        if mode == "auto":
            return "adapter" if self._lora_meta is not None else "full"
        if mode not in ("full", "adapter"):
            raise ValueError(f"FSDPBackend.save: unknown mode {mode!r} (use 'auto', 'full' or 'adapter')")
        return mode

    def _load_dcp(self, path: str) -> int:
        """Sharded load: each rank reads its own shard from DCP ``path``.

        Reshard-aware — the same shard dir loads under a different world size.
        The current model/optimizer sharded state dicts seed the layout, DCP
        fills them in place, then we set them back (the canonical DCP recipe).
        """
        import torch.distributed.checkpoint as dcp

        meta_path = os.path.join(path, "metadata.pt")
        meta: Dict[str, object] = torch.load(meta_path, map_location="cpu")
        mode = str(meta.get("save_mode", "full"))
        self._reject_meta(operation="load", checkpoint_format="dcp", mode=mode)
        has_meta_params = any(p.is_meta for p in self.model.parameters())
        strict = mode == "full" and not has_meta_params

        model_sd = drop_meta_entries(sharded_model_state_dict(self.model))
        if mode == "adapter":
            model_sd = {k: v for k, v in model_sd.items() if "lora_A" in k or "lora_B" in k}
        sharded_state: Dict[str, object] = {
            "model": model_sd,
            "optim": sharded_optimizer_state_dict(self.model, self.optimizer),
        }
        dcp.load(sharded_state, checkpoint_id=path)

        load_sharded_model_state_dict(self.model, sharded_state["model"], strict=strict)
        load_sharded_optimizer_state_dict(self.model, self.optimizer, sharded_state["optim"])
        if self.scheduler is not None and "scheduler_state_dict" in meta:
            self.scheduler.load_state_dict(meta["scheduler_state_dict"])
        if meta.get("optimizer_step_count") is not None:
            self._optimizer_step_count = int(meta["optimizer_step_count"])
        return int(meta.get("step") or 0)

    def _reject_meta(
        self,
        *,
        operation: Literal["save", "load"],
        checkpoint_format: Literal["torch", "dcp"],
        mode: str,
    ) -> None:
        """Single materialization guard for every checkpoint save/load path.

        Args (keyword-only):
            operation: the calling operation, for the error message ("save" / "load").
            checkpoint_format: the storage format ("torch" full-gather / "dcp" sharded).
            mode: the save mode read from the request / checkpoint ("full" / "adapter").

        Raises if a param this ``(checkpoint_format, mode)`` path is responsible
        for is still on meta. The dispatch below is the single source of truth, so
        the guard cannot drift from the matching ``_save_*`` / ``_load_*`` (the bug
        class behind the earlier duplicate / wrong-mode / missing checks). The
        verdict is a pure function of model structure, identical on every rank, so
        raising is collective-safe.

        Responsibility by ``(checkpoint_format, mode)``:

        - ``adapter`` (either format): every LoRA param — the trainable
          ``default`` AND the frozen EMA ``old`` / shadow adapter, both of which
          ride in the adapter checkpoint.
        - ``full`` + ``"dcp"``: trainable params only. This is "what must not be
          silently dropped", NOT "everything persisted": ``_save_dcp`` writes
          every non-meta entry but ``drop_meta_entries`` drops the frozen aux.
          Frozen-but-owned state (full-EMA mirror params, a frozen shadow LoRA
          under full mode) is intentionally not guarded here — see
          ``docs/dcp_checkpoint_impl.md``.
        - ``full`` + ``"torch"``: every param — the rank-0 gather encodes the
          whole model (frozen aux included) and cannot represent meta.
        """
        named = list(self.model.named_parameters())
        if mode == "adapter":
            meta = [n for n, p in named if p.is_meta and ("lora_A" in n or "lora_B" in n)]
            why = "adapter checkpointing requires materialized LoRA params"
        elif checkpoint_format == "dcp":
            meta = [n for n, p in named if p.is_meta and p.requires_grad]
            why = "trainable params on meta would be silently dropped from the DCP checkpoint (materialize missed them)"
        else:  # full + torch
            meta = [n for n, p in named if p.is_meta]
            why = (
                "full-state-dict checkpointing of meta-init bundles is not supported "
                "under checkpoint_format='torch' (use 'dcp')"
            )
        if meta:
            raise RuntimeError(
                f"{type(self).__name__}.{operation}: {len(meta)} params on meta (e.g. {meta[:3]}); {why}."
            )

    # ------------------------------------------------------------------
    # Memory lifecycle
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload(self) -> None:
        """Move the train state (params + grads + optimizer) back to GPU.

        Driver-callable across all DP workers (each onloads its own FSDP shard).
        Inverse of :meth:`offload`; the colocate trainers call this before the
        train backward (gated by ``enable_fsdp_offload``)."""
        self._onload_model()
        move_optimizer_state(self.optimizer, self._device)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def offload(self) -> None:
        """Move the train state (params + grads + optimizer) to CPU.

        Frees GPU memory during the rollout phase so a colocate vLLM/SGLang
        engine fits. Driver-callable across all DP workers (each offloads its own
        FSDP shard). Gated by the trainer's ``enable_fsdp_offload``."""
        self._offload_model()
        move_optimizer_state(self.optimizer, "cpu")
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
        from unirl.distributed.weight_sync.transfer.checksum import (
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

    # ------------------------------------------------------------------
    # Engine-specific hooks (overridden by each leaf backend)
    # ------------------------------------------------------------------

    def _clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        """Clip gradients and return the (pre-clip) global grad norm."""
        raise NotImplementedError

    def _gather_optimizer_state(self) -> StateDict:
        """Rank-0 optimizer state for the checkpoint (collective for DCP backends)."""
        raise NotImplementedError

    def _load_optimizer_state(self, optimizer_state: StateDict) -> None:
        """Restore optimizer state from a checkpoint dict (real dict on every rank)."""
        raise NotImplementedError

    def _onload_model(self) -> None:
        """Move the wrapped model's params + grads back to the live device."""
        raise NotImplementedError

    def _offload_model(self) -> None:
        """Move the wrapped model's params + grads to CPU."""
        raise NotImplementedError


__all__ = ["BaseFSDP2Backend"]
