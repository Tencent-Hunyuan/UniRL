"""BaseFSDP2Backend — the shared training-state Remote for FSDP2 backends."""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import Future
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
import torch.distributed as dist
from torch import nn

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.base import ExpertWeightExportTransform, LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.sharded_state import (
    StateDict,
    _current_rank,
    drop_meta_entries,
    fresh_adamw_state,
    gather_lora_state_dict,
    gather_state_dict,
    load_model_state_dict,
    move_optimizer_state,
    trainable_params,
)
from unirl.train.configs import EmaFullConfig, EmaLoraConfig, FSDPConfig, LoraConfig
from unirl.train.ema import EMA, Shadow, inject_mirror, inject_nft, make_decay_fn
from unirl.train.lora import FrozenAdapters, inject_lora, resolve_target_modules_pattern
from unirl.train.optim import build_lr_scheduler, build_optimizer
from unirl.utils.distributed_utils import find_dtensor_mesh, init_gloo_group

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)

_PENDING_DCP_SAVE_FUTURE: Optional[Future] = None


def _drain_pending_dcp_save() -> None:
    """Wait for the process-wide async DCP upload, if any."""
    global _PENDING_DCP_SAVE_FUTURE
    future = _PENDING_DCP_SAVE_FUTURE
    if future is None:
        return
    try:
        future.result()
    finally:
        if _PENDING_DCP_SAVE_FUTURE is future:
            _PENDING_DCP_SAVE_FUTURE = None


class BaseFSDP2Backend(Remote):
    """Shared base for the single-track FSDP2 training backends."""

    _bundle: object
    _rank: int
    _device: torch.device
    model: nn.Module
    ema: Optional[EMA]
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    fsdp_cfg: FSDPConfig
    lora_meta: Optional[Dict[str, object]]
    rollout_adapter_name: str
    grad_sync_deferred: bool
    _optimizer_step_count: int
    _eval_ema_active: bool
    _frozen_adapters: FrozenAdapters
    _grad_sync_enabled: bool
    _loss_reduction_mesh: Optional["DeviceMesh"]

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
        """Structural injection on the (possibly meta) trainable module."""
        self._frozen_adapters = FrozenAdapters()
        shadow: Optional[Shadow] = None
        if ema_lora_cfg is not None:
            shadow = inject_nft(
                model,
                rank=ema_lora_cfg.rank,
                alpha=ema_lora_cfg.alpha,
                target_modules=ema_lora_cfg.target_modules,
                exclude_modules=ema_lora_cfg.exclude_modules,
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
                target_modules=lora_cfg.target_modules,
                module_prefix=lora_cfg.module_prefix,
                exclude_modules=lora_cfg.exclude_modules,
                dropout=lora_cfg.dropout,
                bias=lora_cfg.bias,
                task_type=lora_cfg.task_type,
            )
            # Frozen sibling adapters (e.g. OPD teachers) — see ../readme.md Gotchas.
            self._frozen_adapters = FrozenAdapters.inject(model, lora_cfg.frozen_adapters)
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
        """Build EMA / optimizer / scheduler and set the shared train state."""
        if fsdp_cfg.checkpoint_format not in ("torch", "dcp"):
            raise ValueError(
                f"{type(self).__name__}: fsdp_cfg.checkpoint_format must be 'torch' or 'dcp', "
                f"got {fsdp_cfg.checkpoint_format!r}"
            )
        self.fsdp_cfg = fsdp_cfg
        self.model = model
        self._loss_reduction_mesh = find_dtensor_mesh(model)

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
            named_params=list(model.named_parameters()),
        )
        self.scheduler = build_lr_scheduler(
            scheduler_cfg,
            optimizer=self.optimizer,
        )

        self._optimizer_step_count = 0
        self._eval_ema_active = False
        active_lora = lora_cfg or ema_lora_cfg
        recorded_target_modules = None
        if active_lora is not None:
            recorded_target_modules = active_lora.target_modules
            if lora_cfg is not None:
                recorded_target_modules, _ = resolve_target_modules_pattern(
                    target_modules=lora_cfg.target_modules,
                    module_prefix=lora_cfg.module_prefix,
                )
        self.lora_meta = (
            {
                "rank": active_lora.rank,
                "alpha": active_lora.alpha,
                "target_modules": recorded_target_modules,
                "exclude_modules": active_lora.exclude_modules,
                "dropout": active_lora.dropout,
                "bias": active_lora.bias,
                "task_type": active_lora.task_type,
                "frozen_adapters": dict(self._frozen_adapters.shas),
            }
            if active_lora is not None
            else None
        )
        self.rollout_adapter_name = ema_lora_cfg.shadow_adapter if ema_lora_cfg is not None else "default"
        # Deferred gradient sync is supported only under ZeRO-2.
        self.grad_sync_deferred = fsdp_cfg.defer_grad_sync and not fsdp_cfg.reshard_after_forward
        self._grad_sync_enabled = True

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def set_grad_sync(self, enable: bool) -> None:
        """Toggle the FSDP2 gradient reduce-scatter for no-sync accumulation."""
        if not self.grad_sync_deferred or enable == self._grad_sync_enabled:
            return
        from torch.distributed.fsdp import FSDPModule

        for m in self.model.modules():
            if isinstance(m, FSDPModule):
                m.set_requires_gradient_sync(enable)
                m.set_is_last_backward(enable)
        self._grad_sync_enabled = enable

    def optimizer_step(self, *, max_grad_norm: float) -> float:
        """Clip (via the engine hook), optimizer step, scheduler step, EMA step."""
        grad_norm = float(self._clip_grad_norm(max_grad_norm).item())

        if not math.isfinite(grad_norm):
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
        # Release consumed grads before train-state offload.
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def on_rollout_end(self) -> None:
        if self.ema is not None:
            self.ema.on_rollout_end(self._optimizer_step_count)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def get_optimizer_step_count(self) -> int:
        """Return the authoritative number of committed optimizer updates."""
        return self._optimizer_step_count

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def apply_eval_ema(self) -> None:
        """Swap the EMA shadow ("old") adapter into live position for rollout."""
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

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def save(self, path: str, step: int = 0, mode: str = "full") -> None:
        """Save training state; dispatch on ``checkpoint_format`` ("torch" | "dcp")."""
        global _PENDING_DCP_SAVE_FUTURE
        if mode == "auto":
            mode = "adapter" if self.lora_meta is not None else "full"
        elif mode not in ("full", "adapter"):
            raise ValueError(f"FSDPBackend.save: unknown mode {mode!r} (use 'auto', 'full' or 'adapter')")
        if mode == "adapter" and not any("lora_" in name for name, _ in self.model.named_parameters()):
            raise RuntimeError(f"{type(self).__name__}.save: mode='adapter' but the model has no LoRA params")

        if self.fsdp_cfg.checkpoint_format == "dcp":
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                get_optimizer_state_dict,
            )

            # Finish an async save before starting another DCP collective.
            _drain_pending_dcp_save()
            self._reject_meta(operation="save", checkpoint_format="dcp", mode=mode)
            sharded_options = StateDictOptions(full_state_dict=False)
            model_sd = drop_meta_entries(dict(get_model_state_dict(self.model, options=sharded_options)))
            if mode == "adapter":
                model_sd = {k: v for k, v in model_sd.items() if self._frozen_adapters.is_trainable_lora_key(k)}
            with fresh_adamw_state(self.optimizer):
                optim_sd = get_optimizer_state_dict(self.model, self.optimizer, options=sharded_options)
            sharded_state: Dict[str, object] = {
                "model": model_sd,
                "optim": optim_sd,
            }
            meta: Dict[str, object] = {
                "optimizer_step_count": self._optimizer_step_count,
                "step": step,
                "save_mode": mode,
                "lora_config": self.lora_meta,
            }
            if self.scheduler is not None:
                meta["scheduler_state_dict"] = self.scheduler.state_dict()

            pg = None
            if self.fsdp_cfg.checkpoint_async and dist.is_available() and dist.is_initialized():
                pg = init_gloo_group()

            local_error: Optional[str] = None
            local_exc: Optional[Exception] = None
            metadata_path = os.path.join(path, "metadata.pt")
            metadata_tmp = f"{metadata_path}.tmp"
            if _current_rank() == 0:
                try:
                    os.makedirs(path, exist_ok=True)
                    torch.save(meta, metadata_tmp)
                    for stale_file in (".metadata", "checkpoint.pt"):
                        stale_path = os.path.join(path, stale_file)
                        if os.path.exists(stale_path):
                            os.remove(stale_path)
                    os.replace(metadata_tmp, metadata_path)
                except Exception as exc:
                    local_exc = exc
                    local_error = f"{type(exc).__name__}: {exc}"
                    try:
                        if os.path.exists(metadata_tmp):
                            os.remove(metadata_tmp)
                    except OSError:
                        pass

            if dist.is_available() and dist.is_initialized():
                error_box: List[Optional[str]] = [local_error]
                dist.broadcast_object_list(error_box, src=0, group=pg)
                error = error_box[0]
            else:
                error = local_error

            if error is not None:
                wrapped = RuntimeError(f"failed to prepare DCP checkpoint directory {path!r}: {error}")
                if local_exc is not None:
                    raise wrapped from local_exc
                raise wrapped

            if self.fsdp_cfg.checkpoint_async:
                if _PENDING_DCP_SAVE_FUTURE is not None:
                    raise RuntimeError("an async DCP save is already pending in this worker process")
                _PENDING_DCP_SAVE_FUTURE = dcp.async_save(sharded_state, checkpoint_id=path, process_group=pg)
            else:
                dcp.save(sharded_state, checkpoint_id=path)
            return

        self._reject_meta(operation="save", checkpoint_format="torch", mode=mode)
        policy_state = self._gather_model_state(mode)
        optimizer_state = self._gather_optimizer_state()
        state: Dict[str, object] = {
            "policy_state_dict": policy_state,
            "optimizer_state_dict": optimizer_state,
            "optimizer_step_count": self._optimizer_step_count,
            "step": step,
            "save_mode": mode,
            "lora_config": self.lora_meta,
        }
        state.update(self._torch_checkpoint_metadata())
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        if _current_rank() != 0:
            return
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, "checkpoint.pt")
        checkpoint_tmp = f"{checkpoint_path}.tmp"
        try:
            torch.save(state, checkpoint_tmp)
            for stale_dcp_file in (".metadata", "metadata.pt"):
                stale_path = os.path.join(path, stale_dcp_file)
                if os.path.exists(stale_path):
                    os.remove(stale_path)
            os.replace(checkpoint_tmp, checkpoint_path)
        except Exception:
            try:
                if os.path.exists(checkpoint_tmp):
                    os.remove(checkpoint_tmp)
            except OSError:
                pass
            raise

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wait_for_checkpoint(self) -> None:
        """Flush the last async DCP save (driver-callable across all workers)."""
        _drain_pending_dcp_save()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def load(self, path: str) -> int:
        """Restore state written by :meth:`save`; return the saved rollout step (0 if absent)."""
        # Finish an async save before reading its checkpoint.
        _drain_pending_dcp_save()
        dcp_metadata_path = os.path.join(path, ".metadata")
        metadata_path = os.path.join(path, "metadata.pt")
        checkpoint_path = os.path.join(path, "checkpoint.pt")
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
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
                get_optimizer_state_dict,
                set_model_state_dict,
                set_optimizer_state_dict,
            )

            meta: Dict[str, object] = torch.load(metadata_path, map_location="cpu")
            mode = meta["save_mode"]
            self._reject_meta(operation="load", checkpoint_format="dcp", mode=mode)
            has_meta_params = any(p.is_meta for p in self.model.parameters())
            strict = mode == "full" and not has_meta_params

            self._frozen_adapters.check_resume(meta.get("lora_config"))
            sharded_options = StateDictOptions(full_state_dict=False)
            model_sd = drop_meta_entries(dict(get_model_state_dict(self.model, options=sharded_options)))
            if mode == "adapter":
                model_sd = {k: v for k, v in model_sd.items() if self._frozen_adapters.is_trainable_lora_key(k)}
            with fresh_adamw_state(self.optimizer):
                optim_sd = get_optimizer_state_dict(self.model, self.optimizer, options=sharded_options)
            sharded_state: Dict[str, object] = {
                "model": model_sd,
                "optim": optim_sd,
            }
            dcp.load(sharded_state, checkpoint_id=path)

            set_model_state_dict(
                self.model,
                sharded_state["model"],
                options=StateDictOptions(full_state_dict=False, strict=strict),
            )
            set_optimizer_state_dict(
                self.model,
                self.optimizer,
                optim_state_dict=sharded_state["optim"],
                options=sharded_options,
            )
            if isinstance(self.optimizer, torch.optim.AdamW):
                for param in [p for p, entry in self.optimizer.state.items() if float(entry["step"]) == 0]:
                    del self.optimizer.state[param]
            if self.scheduler is not None and "scheduler_state_dict" in meta:
                self.scheduler.load_state_dict(meta["scheduler_state_dict"])
            self._optimizer_step_count = meta["optimizer_step_count"]
            return meta["step"]

        missing_on = [rank for rank, ok in enumerate(verdicts) if not (ok and ok["torch"])]
        if missing_on:
            raise FileNotFoundError(
                f"{type(self).__name__}.load: checkpoint not visible on rank(s) {missing_on}: {checkpoint_path} "
                "(save_dir/load_dir must live on storage mounted on every node)"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        mode = checkpoint["save_mode"]
        strict = mode == "full"
        self._reject_meta(operation="load", checkpoint_format="torch", mode=mode)
        self._frozen_adapters.check_resume(checkpoint.get("lora_config"))
        policy_state = checkpoint["policy_state_dict"]
        if mode == "adapter":
            policy_state = {k: v for k, v in policy_state.items() if self._frozen_adapters.is_trainable_lora_key(k)}
        self._load_model_state(policy_state, strict=strict)
        self._load_optimizer_state(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self._optimizer_step_count = checkpoint["optimizer_step_count"]
        return checkpoint["step"]

    def _reject_meta(
        self,
        *,
        operation: Literal["save", "load"],
        checkpoint_format: Literal["torch", "dcp"],
        mode: str,
    ) -> None:
        """Single materialization guard for every checkpoint save/load path."""
        named = list(self.model.named_parameters())
        if mode == "adapter":
            meta = [n for n, p in named if p.is_meta and ("lora_A" in n or "lora_B" in n)]
            why = "adapter checkpointing requires materialized LoRA params"
        elif checkpoint_format == "dcp":
            meta = [n for n, p in named if p.is_meta and p.requires_grad]
            why = "trainable params on meta would be silently dropped from the DCP checkpoint (materialize missed them)"
        else:
            meta = [n for n, p in named if p.is_meta]
            why = (
                "full-state-dict checkpointing of meta-init bundles is not supported "
                "under checkpoint_format='torch' (use 'dcp')"
            )
        if meta:
            raise RuntimeError(
                f"{type(self).__name__}.{operation}: {len(meta)} params on meta (e.g. {meta[:3]}); {why}."
            )

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload(self) -> None:
        """Move the train state (params + grads + optimizer) back to GPU."""
        self._onload_model()
        move_optimizer_state(self.optimizer, self._device)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def offload(
        self,
        *,
        model: bool = True,
        optimizer: bool = True,
        clear_gradients: bool = False,
    ) -> None:
        """Move selected train state to CPU while preserving legacy defaults."""
        if clear_gradients:
            self.optimizer.zero_grad(set_to_none=True)
        if model:
            self._offload_model()
        if optimizer:
            move_optimizer_state(self.optimizer, "cpu")
        torch.cuda.empty_cache()

    def gradient_average_world_size(self) -> int:
        """Ranks in the FSDP mesh whose gradient averaging loss scaling cancels."""
        mesh = self._loss_reduction_mesh
        if mesh is not None:
            return mesh.size()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            raise RuntimeError(
                f"{type(self).__name__}: distributed training has no FSDP DeviceMesh; "
                "cannot determine the gradient-averaging world size."
            )
        return 1

    def all_reduce_loss_sums(self, values: List[float]) -> List[float]:
        """SUM loss statistics over the same FSDP mesh used for gradients."""
        if not values:
            return []
        mesh = self._loss_reduction_mesh
        world_size = self.gradient_average_world_size()
        if world_size <= 1:
            return list(values)
        if mesh is None or not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(f"{type(self).__name__}: FSDP loss reduction requires initialized distributed state.")
        tensor = torch.tensor(values, dtype=torch.float64, device=self._device)
        for group in mesh.get_all_groups():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor.tolist()

    def trainable_module(self) -> nn.Module:
        return self.model

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

    def _clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        """Clip gradients and return the (pre-clip) global grad norm."""
        raise NotImplementedError

    def _gather_model_state(self, mode: str) -> StateDict:
        """Rank-0 model state for the single-file checkpoint."""
        if mode == "adapter":
            return gather_lora_state_dict(self.model, keep=self._frozen_adapters.is_trainable_lora_key)
        return gather_state_dict(self.model)

    def _load_model_state(self, model_state: StateDict, *, strict: bool) -> None:
        """Restore model state from a single-file checkpoint."""
        load_model_state_dict(self.model, model_state, strict=strict)

    def _torch_checkpoint_metadata(self) -> Dict[str, object]:
        """Backend-specific metadata added to a single-file checkpoint."""
        return {}

    def expert_weight_export_transform(self) -> Optional[ExpertWeightExportTransform]:
        """Return this backend's expert-weight export transform, if one is required."""
        return None

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
