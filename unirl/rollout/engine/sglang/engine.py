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
import os
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
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_device_ids: Optional[List[int]] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        ep_rank: int = 0,
        ep_size: int = 1,
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

        # Rollout tensor-parallel layout. Handle injects these per-rank when the
        # rollout Handle carries tp/pp/ep > 1. tp_rank==0 boots a (possibly
        # multi-GPU) SGLang engine spanning ``tp_device_ids``; every other TP
        # rank is a no-op shell that still occupies its Worker for training but
        # holds no SGLang server (SGLang's own scheduler subprocesses own those
        # GPUs). tp_size=1 (the default) reproduces the one-engine-per-GPU path.
        self._tp_rank = int(tp_rank)
        self._tp_size = int(tp_size)
        self._pp_rank = int(pp_rank)
        self._pp_size = int(pp_size)
        self._ep_rank = int(ep_rank)
        self._ep_size = int(ep_size)
        self._tp_device_ids = list(tp_device_ids) if tp_device_ids else None
        self._is_tp_zero = self._tp_rank == 0

        if not self._is_tp_zero:
            # No-op shell: no SGLang server, no ports, no weight sync. All
            # rollout verbs early-return via the ``_is_tp_zero`` guard. The
            # adapter is still built so any pure-Python helpers stay available,
            # but nothing touches the GPU here.
            self.adapter = None
            self._backend = None
            self._weight_sync = None
            logger.info(
                "SGLangRolloutEngine: tp_rank=%d/%d is a no-op shell (rank=%s); "
                "SGLang server hosted by tp_rank=0 of this TP group",
                self._tp_rank,
                self._tp_size,
                rank,
            )
            return

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
            "Initializing sglang engine (rank=%s, model_family=%s, model=%s, tp=%s, tp_group=%s)",
            rank,
            config.model_family,
            config.pretrained_model_ckpt_path,
            self._tp_size,
            self._tp_device_ids,
        )

        # Ports — engine-reserved on this node at the last moment before the
        # spawn (both backends: nccl_port de-syncs colocated engines). Tests
        # inject a fixed set.
        if ports is None:
            ports = SGLangPorts.reserve()

        # Rollout-TP runtime overrides. When a single SGLang engine spans
        # ``tp_size`` GPUs (this worker's + its TP siblings'), expose all of
        # them to the scheduler subprocesses and pin SGLang's device numbering
        # to the local base. We override CUDA_VISIBLE_DEVICES for the spawn (the
        # SGLang schedulers are spawned children that inherit this env), then
        # let SGLang's base_gpu_id/gpu_id_step index within the visible set.
        runtime_overrides: Dict[str, Any] = {}
        if self._tp_size > 1:
            runtime_overrides["tp_size"] = self._tp_size
            runtime_overrides["base_gpu_id"] = 0
            runtime_overrides["gpu_id_step"] = 1
            if self._tp_device_ids:
                self._prev_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in self._tp_device_ids)
                logger.info(
                    "SGLangRolloutEngine tp_rank=0: CUDA_VISIBLE_DEVICES=%s for tp_size=%d",
                    os.environ["CUDA_VISIBLE_DEVICES"],
                    self._tp_size,
                )

        # Backend (the seam) — booted from the config-spelled intent.
        intent = config.server_intent(
            ports=ports,
            extra=self.adapter.boot_kwargs(),
            runtime_overrides=runtime_overrides or None,
        )
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
        """Run text generation against the engine and return a typed response.

        Only tp_rank==0 hosts a SGLang server; other TP ranks in the group are
        no-op shells. The DP_SCATTER collect filters ``tp_rank==0`` results, so
        returning ``None`` from a shell is dropped before the merge.
        """
        if not self._is_tp_zero:
            return None
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
        if not self._is_tp_zero:
            return
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
        resume — e.g.         ``wake_up(tags=["weights"])`` to allow weight sync, then
        ``wake_up(tags=["kv_cache", "cuda_graph"])`` before generation.
        """
        if not self._is_tp_zero:
            return
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
        if not self._is_tp_zero:
            return
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
        if not self._is_tp_zero:
            return True
        if self._is_offloaded:
            return True
        return self._backend.ping()

    def shutdown(self) -> None:
        if not self._is_tp_zero or self._backend is None:
            return
        self._backend.shutdown()
        # Restore CUDA_VISIBLE_DEVICES if we overrode it for a multi-GPU spawn.
        prev = getattr(self, "_prev_cuda_visible_devices", None)
        if self._tp_size > 1 and self._tp_device_ids:
            if prev is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev

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
        if not self._is_tp_zero:
            return
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
        if not self._is_tp_zero:
            return
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
        if not self._is_tp_zero:
            return
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
        if not self._is_tp_zero:
            return
        self._weight_sync.destroy_weights_update_group(group_name=group_name)

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        if not self._is_tp_zero:
            return
        self._weight_sync.set_lora_from_tensors(adapter_name, lora_tensors, peft_config=peft_config)

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed before generate."""
        if not self._is_tp_zero or self._weight_sync is None:
            return False
        return self._weight_sync.lora_dirty

    # ``update_weights_from_ipc`` is deliberately NOT defined — the base raises
    # NotImplementedError (SGLang has no bucketed-IPC receiver).


__all__ = ["SGLangRolloutEngine"]
