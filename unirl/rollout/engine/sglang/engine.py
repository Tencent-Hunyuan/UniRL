"""``sglang`` engine core — wiring + delegation only.

A thin core over the backend seam: it names no concrete model (the adapter,
picked from the registry by ``config.model_family``, owns the
``RolloutReq``↔``RolloutResp`` conversion) and no concrete transport (the seam
owns the SRT runtime — server subprocess + HTTP, or the in-process Engine,
picked by ``config.backend``). Weight sync is a :class:`WeightSync` component
constructed over the seam; the offload lifecycle (the two staged flags) lives
directly on the engine. The frozen ``base.py`` surface is implemented as thin
forwards here — they must be real class attributes anyway (``Worker.call``
dispatches by name; ``@distributed`` binds the most-derived attribute) — which
also absorbs the surface quirks (``track_prefix``) so the component keeps clean
signatures.

One-shot construction: after ``__init__`` returns, the SRT server is spawned and
healthy and the engine is usable. ``generate`` / ``sleep`` / ``wake_up``
re-apply ``@distributed`` (the decorator is not inherited — see ``base.py``).
No environment mutation happens here — the spawn-scoped env the SRT
subprocesses need is quarantined in the backends' ``boot``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.sglang.adapters import get_adapter
from unirl.rollout.engine.sglang.backends import HTTPBackend, NativeBackend
from unirl.rollout.engine.sglang.config import SGLangEngineConfig, SGLangPorts
from unirl.rollout.engine.sglang.utils import resolve_sampling
from unirl.rollout.engine.sglang.weight_sync import WeightSync
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp

logger = logging.getLogger(__name__)


class SGLangRolloutEngine(BaseRolloutEngine):
    """LLM/VLM rollout engine backed by a SGLang SRT server (v2 layout)."""

    _component_name = "sglang"

    def __init__(
        self,
        config: SGLangEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
        ports: Optional[SGLangPorts] = None,
    ) -> None:
        require(
            isinstance(config, SGLangEngineConfig),
            f"SGLangRolloutEngine requires SGLangEngineConfig; got {type(config).__name__}",
        )
        # LLM engine carries its own model path on the config; the diffusion
        # engine takes it from model_config. Log if a caller supplied one so
        # the divergence is visible.
        if model_config is not None:
            logger.debug(
                "SGLangRolloutEngine: model_config provided but ignored — "
                "LLM engine uses config.pretrained_model_ckpt_path",
            )
        del strategy  # LLM rollout has no SDE strategy

        self.cfg = config
        self.rank = rank
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_offloaded = False
        self._weights_onloaded_for_sync = False

        engine_kwargs: Dict[str, Any] = dict(config.engine_kwargs or {})

        # Tokenizer (+ AutoProcessor for VLM) — the encoding I/O the engine
        # owns, injected into the adapter so its conversion methods stay pure.
        # The processor encodes multimodal prompts the SAME way the trainside
        # replay does (it expands the single image placeholder and emits
        # pixel_values / image_grid_thw), keeping rollout and replay
        # token-for-token aligned.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_ckpt_path, trust_remote_code=True)
        processor = None
        if config.image_token is not None:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(config.pretrained_model_ckpt_path, trust_remote_code=True)

        # Adapter (the only read of a model knob) — owns the conversion.
        self.adapter = get_adapter(config.model_family)(config, model_config, tokenizer=tokenizer, processor=processor)

        logger.info(
            "Initializing sglang engine (rank=%s, model_family=%s, model=%s, tp=%s)",
            rank,
            config.model_family,
            config.pretrained_model_ckpt_path,
            config.tp_size,
        )

        # Ports — engine-reserved on this node at the last moment before the
        # spawn (both backends: nccl_port de-syncs colocated engines). Tests
        # inject a fixed set.
        if ports is None:
            ports = SGLangPorts.reserve()

        # Backend (the seam) — booted from the config-spelled intent.
        intent = config.server_intent(ports=ports, extra=self.adapter.boot_kwargs())
        concurrency = int(engine_kwargs.get("concurrency", config.concurrency))
        if config.backend == "native":
            self._backend = NativeBackend.boot(intent, concurrency=concurrency)
        else:
            # The address peers reach this server at (the bind host is usually
            # the 0.0.0.0 wildcard). Node-identity discovery, not runtime I/O —
            # and HTTP-only: it exists to build the client base_url.
            bind_host = str(engine_kwargs.get("host") or config.host or "0.0.0.0")
            advertise_host = engine_kwargs.get("advertise_host")
            if not advertise_host:
                try:
                    import ray

                    advertise_host = ray.util.get_node_ip_address()
                except Exception:
                    advertise_host = bind_host if bind_host not in ("0.0.0.0", "") else "127.0.0.1"

            self._backend = HTTPBackend.boot(
                intent,
                advertise_host=str(advertise_host),
                concurrency=concurrency,
                health_timeout_s=float(engine_kwargs.get("health_timeout_s", 300.0)),
            )

        # Weight sync — owns all sync/LoRA state, over the live seam.
        self._weight_sync = WeightSync(
            self._backend,
            uses_lora=bool(engine_kwargs.get("enable_lora", False)),
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run text generation against the engine and return a typed response."""
        require(
            int(req.batch_size) > 0,
            "SGLangRolloutEngine.generate requires non-empty req (batch_size > 0)",
        )
        sampling = resolve_sampling(self.cfg, req)
        prepared = self.adapter.build_inputs(req, sampling=sampling)
        # Activate the synced LoRA adapter for these requests — the visible
        # line connecting WeightSync's state to the wire (the adapter and the
        # seam stay unaware of weight sync).
        active_adapter = self._weight_sync.active_adapter
        if active_adapter:
            for payload in prepared.wire:
                payload["lora_path"] = active_adapter
        raw = self._backend.generate(prepared.wire)
        return self.adapter.build_response(req, prepared, raw)

    # ------------------------------------------------------------------ #
    # Lifecycle — the offload flags live here; decorators re-applied
    # (base.py footgun)
    # ------------------------------------------------------------------ #

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self, tags: Optional[List[str]] = None) -> None:
        """Release GPU memory (offload).

        Flushes the cache first; sglang's release only fully frees the KV
        pool when the scheduler has no pending references.

        ``tags`` selects which sglang SRT memory regions to release (e.g.
        ``["weights"]``). ``None`` releases everything. Called again while
        offloaded (post-sync re-offload), it releases the weights that
        ``onload_weights`` restored — or no-ops if they never were.
        """
        release_tags = None if tags is None or len(tags) == 0 else list(tags)
        if release_tags is None and self._is_offloaded:
            if not self._weights_onloaded_for_sync:
                return
            release_tags = ["weights"]
        if release_tags is None or "kv_cache" in release_tags:
            self._backend.flush_cache()
        self._backend.release_memory(tags=release_tags)
        self._is_offloaded = True
        self._weights_onloaded_for_sync = False
        # Releasing weights frees the SRT LoRA pool; the adapter must be
        # re-pushed (set_lora_from_tensors) before it can be referenced again.
        if release_tags is None or "weights" in release_tags:
            self._weight_sync.mark_weights_released()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        """Resume GPU memory.

        Can be called multiple times with different tag subsets for a staged
        resume — e.g. ``wake_up(tags=["weights"])`` to allow weight sync, then
        ``wake_up(tags=["kv_cache", "cuda_graph"])`` before generation.
        """
        full_wake = tags is None or len(tags) == 0
        resume_tags = None if full_wake else list(tags)
        if resume_tags is None:
            if not self._is_offloaded:
                return
            if self._weights_onloaded_for_sync:
                resume_tags = ["kv_cache", "cuda_graph"]
        self._backend.resume_memory(tags=resume_tags)
        if full_wake:
            self._is_offloaded = False
            self._weights_onloaded_for_sync = False
        elif "weights" in resume_tags:
            self._weights_onloaded_for_sync = True

    def onload_weights(self, *, track_prefix: str = "") -> None:
        """Resume only model weights so tensor/NCCL sync can update them."""
        del track_prefix
        if not self._is_offloaded:
            return
        if self._weights_onloaded_for_sync:
            return
        self._backend.resume_memory(tags=["weights"])
        self._weights_onloaded_for_sync = True

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def health_check(self) -> bool:
        if self._is_offloaded:
            return True
        return self._backend.ping()

    def shutdown(self) -> None:
        self._backend.shutdown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Weight sync — frozen base.py surface; thin forwards to the component.
    # Un-decorated: reached per worker via the raw ``Worker.call`` RPC, not
    # through ``@distributed``. ``track_prefix`` is absorbed here.
    # ------------------------------------------------------------------ #

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Update weights from serialized tensors via the seam.

        ``target_modules`` is intentionally NOT forwarded — the diffusion-side
        default ``["transformer"]`` doesn't match LLM module naming. Omitting
        the field lets the SRT server accept all incoming weights correctly.
        """
        del target_modules, track_prefix
        self._weight_sync.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
        )

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
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
        track_prefix: str = "",
    ) -> None:
        """Receive weights via NCCL broadcast from training actors.

        ``target_modules`` is intentionally NOT forwarded (see
        :meth:`update_weights_from_tensor` for rationale).
        """
        del target_modules, track_prefix
        self._weight_sync.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=flush_cache,
        )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.destroy_weights_update_group(group_name=group_name)

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        self._weight_sync.set_lora_from_tensors(adapter_name, lora_tensors, peft_config=peft_config)

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed before generate."""
        return self._weight_sync.lora_dirty

    # ``update_weights_from_ipc`` is deliberately NOT defined — the base raises
    # NotImplementedError (SGLang has no bucketed-IPC receiver).


__all__ = ["SGLangRolloutEngine"]
