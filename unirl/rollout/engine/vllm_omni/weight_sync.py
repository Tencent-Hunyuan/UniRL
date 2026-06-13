"""Weight sync — the canonical sync ops + LoRA lifecycle, owned by one component.

``WeightSync`` is a plain object the engine constructs over the seam: it takes
the backend and the LoRA transport choice explicitly and owns ALL sync/LoRA
state (``_lora_loaded`` / ``_weights_released`` / ``_last_lora_*``). Method
names mirror the frozen ``base.py`` surface minus ``track_prefix`` (the
engine's forwards absorb that), so a grep for a trainer-side entry point lands
here. The transports declared are exactly what vllm-omni supports: bucketed
CUDA-IPC, NCCL (init/transfer/destroy), the SGLang-shape tensor bag, and the
two LoRA tensor-bag transports (zero-copy handle vs. TP>1-safe byte copy).

LoRA lifecycle (v1 parity — the *active re-push* model, NOT sglang_diffusion's
passive ``lora_dirty``-only model): every ``set_lora_from_tensors*`` clones the
tensors into ``_last_lora_*`` so :meth:`restore_lora_after_wake` can re-push
after a sleep/wake cycle discards the worker-side adapter pool. The engine's
``sleep()`` fires :meth:`mark_weights_released` (the named weights-released
event); its ``wake_up()`` visibly calls :meth:`restore_lora_after_wake` and
fails fast on a re-push failure — silent base-model rollouts drift the GRPO
ratio invisibly until PPO clip fraction blows up.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from unirl.rollout.engine.vllm_omni.backends import Backend

logger = logging.getLogger(__name__)


class WeightSync:
    """Sync ops + LoRA lifecycle over the seam (one instance per engine)."""

    def __init__(
        self,
        backend: Backend,
        *,
        uses_lora: bool,
        lora_copy_transport: bool,
    ) -> None:
        self._backend = backend
        self._uses_lora = bool(uses_lora)
        # HI3 two-engine stages are TP>1: the wake-time re-push must use the
        # byte-copy transport (the zero-copy handle's one-shot fd pops after
        # the first consumer, crashing ranks 2..N).
        self._lora_copy_transport = bool(lora_copy_transport)
        #: An adapter has been pushed and should be active on generate.
        self._lora_loaded = False
        #: The runtime released its memory since the last push (sleep) — the
        #: worker-side adapter pool may be gone until the wake-time re-push.
        self._weights_released = False
        # Cached adapter state for the wake-time re-push.
        self._last_lora_name: Optional[str] = None
        self._last_lora_tensors: Optional[Dict[str, Any]] = None
        self._last_peft_config: Optional[dict] = None

    @property
    def lora_loaded(self) -> bool:
        """True when a pushed adapter should be activated on generate."""
        return self._lora_loaded

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed."""
        return self._uses_lora and (self._weights_released or not self._lora_loaded)

    # ------------------------------------------------------------------ #
    # Bucketed CUDA-IPC
    # ------------------------------------------------------------------ #

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        replica_rank: Optional[int] = None,
    ) -> None:
        self._backend.update_from_ipc(
            peft_config=peft_config,
            base_sync_done=base_sync_done,
            use_shm=use_shm,
            replica_rank=replica_rank,
        )
        # Phase-2 LoRA sync (peft_config + base_sync_done) has registered the
        # adapter on every worker — flip the activation flag so the next
        # generate attaches a lora_request (without it, vllm-omni's per-request
        # ``set_active_adapter(None)`` deactivates the adapter we just synced
        # and rollout silently runs base weights).
        if peft_config and base_sync_done:
            self._lora_loaded = True
            self._weights_released = False

    # ------------------------------------------------------------------ #
    # NCCL broadcast: init group → transfer bucket → destroy group
    # ------------------------------------------------------------------ #

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        self._backend.init_weights_group(
            master_address=str(master_address),
            master_port=int(master_port),
            rank_offset=int(rank_offset),
            world_size=int(world_size),
            group_name=str(group_name),
            backend=str(backend),
        )

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
    ) -> None:
        self._backend.update_from_distributed(
            names=list(names),
            dtypes=list(dtypes),
            shapes=[list(s) for s in shapes],
            group_name=str(group_name),
            target_modules=list(target_modules) if target_modules else None,
            flush_cache=bool(flush_cache),
        )

    def destroy_weights_update_group(self, *, group_name: str) -> None:
        self._backend.destroy_weights_group(group_name=str(group_name))

    # ------------------------------------------------------------------ #
    # SGLang-shape one-bag tensor payload
    # ------------------------------------------------------------------ #

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        self._backend.update_from_tensor(
            serialized_named_tensors=list(serialized_named_tensors),
            target_modules=list(target_modules) if target_modules else None,
            load_format=load_format,
            flush_cache=bool(flush_cache),
        )

    # ------------------------------------------------------------------ #
    # LoRA tensor bag — two transports; both cache for the wake re-push
    # ------------------------------------------------------------------ #

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Hot-swap the adapter via the zero-copy shm-handle transport."""
        self._cache_lora(adapter_name, lora_tensors, peft_config)
        self._backend.set_lora_handle(adapter_name=adapter_name, lora_tensors=lora_tensors, peft_config=peft_config)
        self._lora_loaded = True
        self._weights_released = False

    def set_lora_from_tensors_copy(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Hot-swap the adapter via the TP>1-safe byte-copy transport."""
        self._cache_lora(adapter_name, lora_tensors, peft_config)
        self._backend.set_lora_copy(adapter_name=adapter_name, lora_tensors=lora_tensors, peft_config=peft_config)
        self._lora_loaded = True
        self._weights_released = False

    def _cache_lora(self, adapter_name: str, lora_tensors: Dict[str, Any], peft_config: Optional[dict]) -> None:
        """Clone the adapter state so a sleep/wake cycle can re-push it."""
        self._last_lora_name = adapter_name
        if isinstance(lora_tensors, dict):
            self._last_lora_tensors = {
                name: t.detach().clone() if isinstance(t, torch.Tensor) else t for name, t in lora_tensors.items()
            }
        else:
            self._last_lora_tensors = lora_tensors
        self._last_peft_config = dict(peft_config or {})

    # ------------------------------------------------------------------ #
    # Post-load value-correctness read-back
    # ------------------------------------------------------------------ #

    def loaded_param_checksums(self, *, names: List[str]) -> dict:
        return self._backend.param_checksums(names=list(names))

    def loaded_lora_checksums(self, *, adapter_id: int, names: Optional[List[str]] = None) -> dict:
        return self._backend.lora_checksums(adapter_id=int(adapter_id), names=names)

    # ------------------------------------------------------------------ #
    # Weights-released event + the wake-time restore
    # ------------------------------------------------------------------ #

    def mark_weights_released(self) -> None:
        """The engine released the runtime memory — the worker-side LoRA pool
        is gone until restored. Unlike sglang_diffusion's passive model (the
        next trainer-driven sync re-pushes), this engine actively re-pushes on
        wake from the cached tensors, so the loaded-intent flag survives the
        release; only the pool-valid bit flips."""
        self._weights_released = True

    def restore_lora_after_wake(self) -> None:
        """Re-push the cached adapter after a wake (v1 parity).

        sleep(level=1) preserves base weights but LoRA adapters may be
        discarded; re-sending after wake ensures rollout uses the adapted
        model. Fail-fast on failure: clears the activation flag and raises so
        the training loop crashes instead of shipping silent base-model
        rollouts (the caller — the engine's ``wake_up`` — stays offloaded).
        """
        if not (self._lora_loaded and self._last_lora_tensors is not None):
            self._weights_released = False
            return
        logger.info(
            "[LoRA-WAKE] Re-loading LoRA after sleep/wake. adapter_name=%s",
            self._last_lora_name,
        )
        try:
            if self._lora_copy_transport:
                self.set_lora_from_tensors_copy(
                    self._last_lora_name,
                    self._last_lora_tensors,
                    peft_config=self._last_peft_config,
                )
            else:
                self.set_lora_from_tensors(
                    self._last_lora_name,
                    self._last_lora_tensors,
                    peft_config=self._last_peft_config,
                )
        except Exception as exc:
            self._lora_loaded = False
            raise RuntimeError(
                f"[LoRA-WAKE] Failed to re-load LoRA adapter "
                f"{self._last_lora_name!r} after sleep/wake; refusing "
                f"to continue serving because rollout would silently "
                f"run the base model, drifting old/new log-probs and "
                f"the GRPO ratio. Original error: {exc!r}"
            ) from exc
        logger.info("[LoRA-WAKE] LoRA re-loaded successfully.")


__all__ = ["WeightSync"]
