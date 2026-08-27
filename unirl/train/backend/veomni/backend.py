"""VeOmniBackend — single-track training-state Remote on VeOmni FSDP2."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from unirl.models.types.bundle import Bundle
from unirl.models.types.post_materialize import apply_deferred_ops
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig, resolve_trainable_module
from unirl.train.backend.base_backend import BaseFSDP2Backend
from unirl.train.backend.sharded_load import load_trainable_weights
from unirl.train.backend.sharded_state import (
    StateDict,
    gather_optimizer_state_dict,
    load_optimizer_state_dict,
)
from unirl.train.backend.veomni.ep.checkpoint import (
    EP_CHECKPOINT_VERSION,
    gather_ep_model_state_dict,
    gather_ep_optimizer_state_dict,
    load_ep_model_state_dict,
    load_ep_optimizer_state_dict,
)
from unirl.train.backend.veomni.ep.placement import has_ep_params
from unirl.train.backend.veomni.state import clip_grad_norm, veomni_offload, veomni_onload
from unirl.train.backend.veomni.wrap import veomni_parallelize
from unirl.train.configs import (
    EmaFullConfig,
    EmaLoraConfig,
    FSDPConfig,
    LoraConfig,
)
from unirl.utils.dtypes import parse_torch_dtype


class VeOmniBackend(BaseFSDP2Backend):
    """Single-track VeOmni-FSDP2 training backend."""

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
        _validate_fsdp_cfg(fsdp_cfg)

        from unirl.train.backend.veomni import _compat

        _, _, local_rank = _compat.rank_world_local()
        _compat.ensure_dist_initialized(local_rank)
        import torch.distributed as dist

        self._rank = dist.get_rank() if dist.is_initialized() else int(rank)
        world = dist.get_world_size() if dist.is_initialized() else 1
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._weight_sync_dtype: torch.dtype = parse_torch_dtype(
            fsdp_cfg.param_dtype,
            field_name="training.fsdp.param_dtype",
        )

        _compat.ensure_installed()
        from veomni.distributed.parallel_state import init_parallel_state

        # Fold Ulysses into the FSDP shard mesh so gradients reduce-scatter across SP ranks.
        self._sp_size = int(getattr(fsdp_cfg, "sp_size", 1) or 1)
        if world % self._sp_size != 0:
            raise ValueError(f"VeOmniBackend: world_size {world} not divisible by sp_size {self._sp_size}")
        # ep_size > 1 requires get_parallel_plan() for fused expert tensors.
        self._ep_size = _validate_ep_size(getattr(fsdp_cfg, "ep_size", 1), world_size=world)
        extra_parallel_kwargs = (
            {"extra_parallel_sizes": (self._ep_size,), "extra_parallel_names": ("ep",)} if self._ep_size > 1 else {}
        )
        init_parallel_state(
            dp_size=world // self._sp_size,
            ulysses_size=self._sp_size,
            dp_mode="fsdp2",
            device_type=self._device.type,
            **extra_parallel_kwargs,
        )

        self._bundle = bundle
        model = resolve_trainable_module(bundle, trainable_attr)

        if self._ep_size > 1:
            prepare_ep = getattr(bundle, "prepare_for_expert_parallel", None)
            if not callable(prepare_ep):
                raise ValueError(
                    f"VeOmniBackend: ep_size={self._ep_size} requires the bundle to support "
                    f"expert parallelism via prepare_for_expert_parallel(); "
                    f"{type(bundle).__name__} does not implement it."
                )
            prepare_ep()

        shadow = self._inject_structural(model, lora_cfg, ema_lora_cfg, ema_cfg)

        veomni_parallelize(
            model,
            block_class_names=tuple(block_class_names),
            param_dtype=fsdp_cfg.param_dtype,
            master_dtype=getattr(fsdp_cfg, "master_dtype", None),
            reshard_after_forward=fsdp_cfg.reshard_after_forward,
            activation_checkpointing=fsdp_cfg.activation_checkpointing,
            use_torch_compile=fsdp_cfg.use_torch_compile,
        )

        # Install Ulysses hooks after VeOmni parallelization.
        from unirl.train.backend.veomni.sp import apply_sequence_parallelism

        apply_sequence_parallelism(model, self._sp_size)

        load_trainable_weights(
            model,
            bundle,
            device=self._device,
            rank=self._rank,
            with_aux=with_aux,
            eager_ok=False,
        )

        apply_deferred_ops(model)

        # Restore RoPE inv_freq after meta materialization.
        from unirl.models.types.meta_init import recover_rope_inv_freq

        recover_rope_inv_freq(model)

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

    @property
    def weight_sync_dtype(self) -> torch.dtype:
        """FSDP compute dtype used on the rollout wire."""
        return self._weight_sync_dtype

    def _clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        return clip_grad_norm(self.model, max_grad_norm)

    def _gather_model_state(self, mode: str) -> StateDict:
        if mode == "full" and has_ep_params(self.model):
            return gather_ep_model_state_dict(self.model)
        return super()._gather_model_state(mode)

    def _load_model_state(self, model_state: StateDict, *, strict: bool) -> None:
        if has_ep_params(self.model):
            load_ep_model_state_dict(self.model, model_state, strict=strict)
            return
        super()._load_model_state(model_state, strict=strict)

    def _torch_checkpoint_metadata(self) -> StateDict:
        if not has_ep_params(self.model):
            return {}
        return {
            "ep_checkpoint_version": EP_CHECKPOINT_VERSION,
            "ep_size": self._ep_size,
        }

    def _gather_optimizer_state(self) -> StateDict:
        if has_ep_params(self.model):
            return gather_ep_optimizer_state_dict(self.model, self.optimizer)
        return gather_optimizer_state_dict(self.model, self.optimizer)

    def _load_optimizer_state(self, optimizer_state: StateDict) -> None:
        if has_ep_params(self.model):
            load_ep_optimizer_state_dict(self.model, self.optimizer, optimizer_state)
            return
        load_optimizer_state_dict(self.model, self.optimizer, optimizer_state)

    def _onload_model(self) -> None:
        veomni_onload(self.model, self._device)

    def _offload_model(self) -> None:
        veomni_offload(self.model)

    def _reject_meta(self, *, operation, checkpoint_format, mode) -> None:
        super()._reject_meta(
            operation=operation,
            checkpoint_format=checkpoint_format,
            mode=mode,
        )
        if self._ep_size > 1 and checkpoint_format == "dcp" and mode == "full":
            raise RuntimeError(
                f"VeOmniBackend.{operation}: full DCP checkpoints are unsupported "
                "with ep_size>1 because the outer expert split is not encoded in "
                "the DTensor placements. Use checkpoint_format='torch' (EP-aware) "
                "or adapter mode for frozen-expert LoRA."
            )


def _validate_fsdp_cfg(fsdp_cfg: FSDPConfig) -> None:
    """Assert the v1-supported FSDPConfig subset (fail fast, actionably)."""
    if str(fsdp_cfg.fsdp_mode).strip().lower() != "full":
        raise ValueError(
            f"VeOmniBackend: fsdp_mode={fsdp_cfg.fsdp_mode!r} unsupported (v1 supports 'full'; "
            "HSDP/hybrid stays on FSDPBackend)."
        )
    if fsdp_cfg.cpu_offload:
        raise ValueError("VeOmniBackend: cpu_offload=true unsupported in v1 (use FSDPBackend).")
    if not fsdp_cfg.mixed_precision:
        raise ValueError("VeOmniBackend: mixed_precision=false unsupported in v1 (bf16-parity mode is fixed).")


def _validate_ep_size(value: object, *, world_size: int) -> int:
    """Return a valid expert-parallel degree without truncating non-integers."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"VeOmniBackend: fsdp_cfg.ep_size must be an integer >= 1, got {value!r}")
    try:
        ep_size = int(value)
    except ValueError as exc:
        raise ValueError(f"VeOmniBackend: fsdp_cfg.ep_size must be an integer >= 1, got {value!r}") from exc
    if ep_size < 1:
        raise ValueError(f"VeOmniBackend: fsdp_cfg.ep_size must be >= 1, got {ep_size}")
    if world_size % ep_size != 0:
        raise ValueError(f"VeOmniBackend: world_size {world_size} not divisible by ep_size {ep_size}")
    return ep_size


__all__ = ["VeOmniBackend"]
