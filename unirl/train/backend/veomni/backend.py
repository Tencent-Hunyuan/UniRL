"""VeOmniBackend — single-track training-state Remote on VeOmni FSDP2.

Drop-in sibling of :class:`unirl.train.backend.fsdp.FSDPBackend`: both subclass
:class:`~unirl.train.backend.base_backend.BaseFSDP2Backend`, which owns the
training step, EMA swap, checkpoint envelope, and memory lifecycle. This leaf
supplies only the constructor lifecycle and the five engine hooks, whose wrap /
grad-clip / offload internals come from VeOmni's distributed layer via the
:mod:`._compat` selective-import shim. Recipes select it purely by ``_target_``.

Lifecycle differences vs FSDPBackend (all internal to construction):

* The default process group is brought up *explicitly* (VeOmni builds its device
  meshes before any ``fully_shard`` call, so torch's lazy auto-init never fires),
  and ``init_parallel_state`` is invoked — one VeOmni-wrapped model per process.
* The trainable module must arrive on the **meta** device (the bundle's
  ``meta_init_transformer`` flag): VeOmni's parallelize materializes it via
  ``to_empty`` and calls its (no-op-stamped) ``init_weights``; the real weights
  load *after* sharding (``eager_ok=False``).
* LoRA/NFT/mirror injection runs on the meta module — exactly the contract
  ``unirl.train.deferred`` documents — and ``apply_deferred_ops`` drains the
  post-materialize resets *after* the weight load.

Checkpointing: ``save``/``load`` are inherited from the base; the optimizer-state
hooks below gather the FULL optimizer state to rank 0 (and broadcast + reshard on
load) — the same DCP path the torch-native FSDP backend uses. The folded
``dp_shard x ulysses`` mesh is a plain 2D DeviceMesh that DCP redistributes across
both dims (already exercised by the ``dcp`` checkpoint format on this mesh).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from unirl.models.types.bundle import Bundle
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig, resolve_trainable_module
from unirl.train.backend.base_backend import BaseFSDP2Backend
from unirl.train.backend.sharded_load import load_trainable_weights
from unirl.train.backend.sharded_state import (
    StateDict,
    gather_optimizer_state_dict,
    load_optimizer_state_dict,
)
from unirl.train.backend.veomni.state import clip_grad_norm, veomni_offload, veomni_onload
from unirl.train.backend.veomni.wrap import veomni_parallelize
from unirl.train.configs import (
    EmaFullConfig,
    EmaLoraConfig,
    FSDPConfig,
    LoraConfig,
)
from unirl.train.deferred import apply_deferred_ops


class VeOmniBackend(BaseFSDP2Backend):
    """Single-track VeOmni-FSDP2 training backend.

    One-shot construction: after ``__init__`` returns the backend is fully usable
    (model wrapped, weights loaded, optimizer/scheduler/EMA built). ``device`` /
    ``rank`` kwargs are accepted for signature parity with :class:`FSDPBackend`
    but resolved from the actor env + process group — backends are constructed
    before ``Remote.setup()`` delivers rank info.
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
        _validate_fsdp_cfg(fsdp_cfg)

        from unirl.train.backend.veomni import _compat

        # 1-3. Distributed bring-up: device binding, default PG, VeOmni parallel
        # state (re-init warns + no-ops, enforcing one VeOmni-wrapped model/process).
        _, _, local_rank = _compat.rank_world_local()
        _compat.ensure_dist_initialized(local_rank)
        import torch.distributed as dist

        self._rank = dist.get_rank() if dist.is_initialized() else int(rank)
        world = dist.get_world_size() if dist.is_initialized() else 1
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _compat.ensure_installed()
        from veomni.distributed.parallel_state import init_parallel_state

        # Ulysses sequence parallelism, folded into the FSDP shard mesh
        # (include_sp_in_fsdp=True default): the shard mesh becomes
        # dp_shard(world//sp) x ulysses(sp), so params shard across the whole
        # world and grads reduce-scatter across SP ranks automatically (verified:
        # docs/usp-derisk/sp_fsdp.py, no manual sp_size compensation). sp_size=1 is
        # a true no-op: dp_size=world, ulysses_size=1 -> the prior 1D dp_shard mesh.
        self._sp_size = int(getattr(fsdp_cfg, "sp_size", 1) or 1)
        if world % self._sp_size != 0:
            raise ValueError(f"VeOmniBackend: world_size {world} not divisible by sp_size {self._sp_size}")
        init_parallel_state(
            dp_size=world // self._sp_size,
            ulysses_size=self._sp_size,
            dp_mode="fsdp2",
            device_type=self._device.type,
        )

        self._bundle = bundle
        model = resolve_trainable_module(bundle, trainable_attr)

        # Structural injection on the meta module (the documented
        # unirl.train.deferred contract: mutate on meta, stamp resets).
        shadow = self._inject_structural(model, lora_cfg, ema_lora_cfg, ema_cfg)

        # Shard + materialize (to_empty; init_weights is a bundle-stamped no-op).
        # Root-wrapped by VeOmni — single-module trainables only.
        veomni_parallelize(
            model,
            block_class_names=tuple(block_class_names),
            param_dtype=fsdp_cfg.param_dtype,
            master_dtype=getattr(fsdp_cfg, "master_dtype", None),
            reshard_after_forward=fsdp_cfg.reshard_after_forward,
            activation_checkpointing=fsdp_cfg.activation_checkpointing,
            use_torch_compile=fsdp_cfg.use_torch_compile,
        )

        # Ulysses sequence parallelism (no-op at sp_size=1): route attention
        # through VeOmni's registered SP attn and wrap the decoder forward to
        # slice the sequence in / gather hidden out. Installed AFTER
        # veomni_parallelize (the GC -> FSDP -> SP order); gated at run time on
        # ulysses_enabled, so safe to call unconditionally.
        from unirl.train.backend.veomni.sp import apply_sequence_parallelism

        apply_sequence_parallelism(model, self._sp_size)

        # Real weights: load into the freshly-sharded module. Meta-init bundles
        # stash a safetensors dir; Pattern-A bundles materialize themselves; eager
        # bundles are rejected (eager_ok=False) — parallelize already to_empty'd,
        # so their weights are gone (FSDPBackend territory).
        load_trainable_weights(
            model,
            bundle,
            device=self._device,
            rank=self._rank,
            with_aux=with_aux,
            eager_ok=False,
        )

        # Post-materialize resets (LoRA adapter init, mirror copies).
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
    # Engine hooks (VeOmni FSDP2)
    # ------------------------------------------------------------------

    def _clip_grad_norm(self, max_grad_norm: float) -> torch.Tensor:
        # VeOmni's clip takes the model (dispatches on EP / cpu-offload attrs).
        return clip_grad_norm(self.model, max_grad_norm)

    def _gather_optimizer_state(self) -> StateDict:
        # Full optimizer state gathered to rank 0 via DCP (full_state_dict=True);
        # the base writes only rank 0's. A plain per-rank optimizer.state_dict()
        # would persist just rank 0's DTensor shard and load it onto every rank,
        # corrupting ranks>0 momentum on resume. The folded dp_shard x ulysses
        # mesh is a plain 2D DeviceMesh DCP gathers across both dims (same path
        # the dcp checkpoint format already uses on this mesh).
        return gather_optimizer_state_dict(self.model, self.optimizer)

    def _load_optimizer_state(self, optimizer_state: StateDict) -> None:
        # Full state on rank 0, broadcast + resharded into each rank's local
        # shard (set_optimizer_state_dict, broadcast_from_rank0=True).
        load_optimizer_state_dict(self.model, self.optimizer, optimizer_state)

    def _onload_model(self) -> None:
        veomni_onload(self.model, self._device)

    def _offload_model(self) -> None:
        veomni_offload(self.model)


# ----------------------------------------------------------------------
# Construction helpers
# ----------------------------------------------------------------------


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


__all__ = ["VeOmniBackend"]
