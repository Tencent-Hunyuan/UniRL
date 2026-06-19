"""Weight sync — the canonical sync ops + LoRA lifecycle, owned by one component.

``WeightSync`` is a plain object the engine constructs over the seam: it takes
the backend explicitly and owns all sync/LoRA state (``_lora_version`` /
``_lora_loaded`` / ``_active_adapter``). Method names mirror the frozen
``base.py`` surface minus ``track_prefix`` (the engine's forwards absorb that,
along with the per-worker ``Worker.call`` dispatch concern), so a grep for a
trainer-side entry point lands here.

The transports declared are exactly what the predecessor supports: tensor-bag,
NCCL (init/transfer/destroy), and LoRA-from-tensors. Two deliberate divergences
from the ``sglang_diffusion`` component:

- ``target_modules`` is NOT accepted or forwarded — the diffusion-side default
  ``["transformer"]`` doesn't match LLM module naming; omitting the field lets
  the SRT server accept all incoming weights correctly.
- LoRA keys go to the wire RAW (HF-native ``model.layers.*``) — there is no
  ``adapt_lora_for_sglang`` prefix-strip; the SRT LLM LoRA loader consumes the
  PEFT layout directly.

The "weights released" event: the engine's ``sleep()`` calls
:meth:`mark_weights_released` after releasing weights, so ``lora_dirty`` flips
and the next sync re-pushes; :attr:`active_adapter` stops tagging requests until
then (otherwise SRT would reference a freed adapter).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from unirl.rollout.engine.sglang.backends import Backend

logger = logging.getLogger(__name__)


class WeightSync:
    """Sync ops + LoRA lifecycle over the seam (one instance per engine)."""

    def __init__(self, backend: Backend, *, uses_lora: bool) -> None:
        self._backend = backend
        self._uses_lora = bool(uses_lora)
        self._active_adapter: Optional[str] = None
        self._lora_loaded = False
        self._lora_version = 0

    # ------------------------------------------------------------------ #
    # Tensor-bag (SGLang one-bag payload per TP rank)
    # ------------------------------------------------------------------ #

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        if not serialized_named_tensors:
            raise ValueError("serialized_named_tensors must be non-empty")
        self._backend.update_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
        )

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
            master_address=master_address,
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
        flush_cache: bool = True,
    ) -> None:
        if not names:
            raise ValueError("names must be non-empty for distributed update")
        # sglang expects bare dtype strings like "bfloat16", not "torch.bfloat16".
        clean_dtypes = [d.replace("torch.", "") if isinstance(d, str) else d for d in dtypes]
        self._backend.update_from_distributed(
            names=list(names),
            dtypes=clean_dtypes,
            shapes=[list(shape) for shape in shapes],
            group_name=str(group_name),
            flush_cache=flush_cache,
        )

    def destroy_weights_update_group(self, *, group_name: str) -> None:
        self._backend.destroy_weights_group(group_name=str(group_name))

    # ------------------------------------------------------------------ #
    # LoRA tensor bag — versioned-nickname rotation
    # ------------------------------------------------------------------ #

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Push a LoRA adapter from in-memory tensors.

        Rotates to a fresh VERSIONED name (``<name>_v<N>``) each sync — the
        rotation is REQUIRED, not just defensive: upstream sglang hard-rejects
        a duplicate ``lora_name`` (``lora_manager`` raises "already loaded" →
        HTTP 400), so a fresh name is the only way to re-push at all. On the
        legacy fork the failure modes were softer but worse — an explicit
        /unload of the live adapter can stall for minutes under colocate, and
        reusing the name can serve STALE weights, so the rollout policy never
        actually updates (reward stays flat while the FSDP model trains).
        Generation points at the latest version via :attr:`active_adapter`;
        stale versions evict via SRT's LRU (``max_loaded_loras``).
        """
        nickname = self._next_lora_nickname(adapter_name)
        self._backend.set_lora(
            lora_name=nickname,
            lora_tensors=lora_tensors,
            config_dict=peft_config,
        )
        self._active_adapter = nickname
        self._lora_loaded = True
        logger.info(
            "sglang: LoRA adapter %r loaded as %r (%d tensor keys)",
            adapter_name,
            nickname,
            len(lora_tensors),
        )

    def _next_lora_nickname(self, adapter_name: str) -> str:
        self._lora_version += 1
        return f"{adapter_name}_v{self._lora_version}"

    # ------------------------------------------------------------------ #
    # Weights-released event + active-adapter / dirty state
    # ------------------------------------------------------------------ #

    def mark_weights_released(self) -> None:
        """The engine released the runtime weights — the loaded LoRA pool is gone."""
        self._lora_loaded = False

    @property
    def active_adapter(self) -> Optional[str]:
        """The adapter name generation should tag requests with (None = base).

        Only set once an adapter has been pushed and not since invalidated by a
        weight release; otherwise SRT serves the base model.
        """
        return self._active_adapter if self._lora_loaded else None

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed before generate."""
        return self._uses_lora and not self._lora_loaded


__all__ = ["WeightSync"]
