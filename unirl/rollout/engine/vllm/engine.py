"""Direct text-only vLLM rollout engine."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.sglang.adapters.text import TextLMAdapter
from unirl.rollout.engine.sglang.utils import resolve_sampling
from unirl.rollout.engine.vllm.config import VLLMEngineConfig
from unirl.rollout.engine.vllm.runtime import engine_process_main
from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


def _resolve_visible_devices(
    tp_size: int,
    tp_visible_devices: Optional[List[str]],
) -> List[str]:
    if tp_visible_devices is not None:
        return list(tp_visible_devices)
    inherited = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
    if inherited:
        return inherited[:tp_size]
    return [str(index) for index in range(tp_size)]


def _resolve_rollout_rank(
    rank: Optional[int],
    tp_size: int,
    visible_devices: List[str],
) -> int:
    if rank is not None:
        return int(rank)
    if tp_size == 1 and visible_devices:
        try:
            physical = int(visible_devices[0])
            base = int(os.environ.get("UNIRL_PHYSICAL_GPU_BASE", "0"))
            return physical - base
        except ValueError:
            pass
    return 0


class VLLMRolloutEngine(BaseRolloutEngine):
    """Run one vLLM instance per TP group in a clean spawned interpreter."""

    _component_name = "vllm"
    _accepts_rollout_tp_kwargs = True

    def __init__(
        self,
        config: VLLMEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Any = None,
        tp_rank: int = 0,
        tp_size: int = 1,
        tp_visible_devices: Optional[List[str]] = None,
        pp_rank: int = 0,
        pp_size: int = 1,
        ep_rank: int = 0,
        ep_size: int = 1,
    ) -> None:
        del strategy, model_config
        require(
            isinstance(config, VLLMEngineConfig),
            f"VLLMRolloutEngine requires VLLMEngineConfig; got {type(config).__name__}",
        )
        require(pp_size == 1 and pp_rank == 0, "direct vLLM rollout currently supports PP=1")
        require(ep_size == 1 and ep_rank == 0, "direct vLLM rollout currently supports EP=1")

        self.cfg = config
        self.rank = rank
        self.device = device
        self._tp_rank = int(tp_rank)
        self._tp_size = int(tp_size)
        self._is_tp_zero = self._tp_rank == 0
        self._is_offloaded = False
        self._partially_awake = False
        self._version = 0
        self._lock = threading.Lock()
        self._process = None
        self._connection = None
        self.adapter = None

        if not self._is_tp_zero:
            return

        # Preserve Ray's physical CUDA token for each TP1 worker. Resetting a
        # spawned child to literal "0" would bypass an outer 4,5,6,7 pin.
        visible = _resolve_visible_devices(self._tp_size, tp_visible_devices)
        require(
            len(visible) == self._tp_size,
            f"direct vLLM expected {self._tp_size} visible devices, got {visible}",
        )
        rollout_rank = _resolve_rollout_rank(rank, self._tp_size, visible)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_ckpt_path,
            trust_remote_code=True,
        )
        self.adapter = TextLMAdapter(config, tokenizer=tokenizer)

        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection = parent
        self._process = context.Process(
            target=engine_process_main,
            kwargs={
                "connection": child,
                "config": {
                    "pretrained_model_ckpt_path": config.pretrained_model_ckpt_path,
                    "tp_size": self._tp_size,
                    "rollout_rank": rollout_rank,
                    "engine_kwargs": dict(config.engine_kwargs or {}),
                },
                "visible_devices": visible,
            },
            name=f"unirl-vllm-tp{self._tp_size}",
            daemon=False,
        )
        self._process.start()
        child.close()
        ready = self._recv(timeout_s=self.cfg.request_timeout_s)
        if ready.get("event") != "ready":
            raise RuntimeError(f"direct vLLM returned invalid startup event: {ready!r}")
        logger.info(
            "Direct vLLM ready: rank=%s tp=%d devices=%s model=%s",
            rank,
            self._tp_size,
            visible,
            config.pretrained_model_ckpt_path,
        )

    @property
    def weight_payload_fanout(self) -> int:
        return self._tp_size

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        if not self._is_tp_zero:
            return None
        require(not self._is_offloaded, "VLLMRolloutEngine.generate called while sleeping")
        sampling = resolve_sampling(self.cfg, sample)
        prepared = self.adapter.build_inputs(sample, sampling=sampling)
        if self.cfg.ignore_eos:
            for payload in prepared.wire:
                payload["sampling_params"]["ignore_eos"] = True
        response = self._request(
            "generate",
            payloads=prepared.wire,
            record_id="batch",
        )
        raw = [SimpleNamespace(**item) for item in response]
        generated = self.adapter.build_response(sample, prepared, raw)
        segment = generated.parts[-1].segment
        if segment is not None and segment.log_probs is not None:
            segment.rollout_log_probs = segment.log_probs.detach().clone()
        return self._stamp_output_version(generated)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self, tags: Optional[List[str]] = None) -> None:
        del tags
        if not self._is_tp_zero or (self._is_offloaded and not self._partially_awake):
            return
        self._request("sleep", level=1)
        self._is_offloaded = True
        self._partially_awake = False

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        if not self._is_tp_zero:
            return
        partial = bool(tags)
        if partial:
            if not self._is_offloaded or self._partially_awake:
                return
        elif not self._is_offloaded and not self._partially_awake:
            return
        request_tags = ["kv_cache"] if not partial and self._partially_awake else tags
        self._request("wake_up", tags=request_tags)
        self._partially_awake = partial
        self._is_offloaded = partial

    def onload_weights(self, *, track_prefix: str = "") -> None:
        del track_prefix
        self.wake_up()

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def health_check(self) -> bool:
        if not self._is_tp_zero:
            return True
        return bool(self._request("health"))

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        del target_modules, track_prefix
        if not self._is_tp_zero:
            return
        self._request(
            "update_weights",
            serialized_named_tensors=list(serialized_named_tensors),
            load_format=load_format,
            flush_cache=bool(flush_cache),
        )
        self._version += 1

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        if not self._is_tp_zero or self._process is None:
            return
        process = self._process
        try:
            if process.is_alive():
                self._request("shutdown")
                process.join(timeout=60)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if self._connection is not None:
                self._connection.close()
            self._process = None
            self._connection = None

    def _request(self, command: str, **payload):
        with self._lock:
            connection = self._require_connection()
            connection.send({"command": command, **payload})
            return self._recv(timeout_s=self.cfg.request_timeout_s).get("result")

    def _recv(self, *, timeout_s: float) -> Dict[str, Any]:
        connection = self._require_connection()
        if not connection.poll(timeout_s):
            raise TimeoutError(f"direct vLLM did not respond within {timeout_s:.0f}s")
        message = connection.recv()
        if not message.get("ok"):
            raise RuntimeError(f"direct vLLM failed: {message.get('error')}\n{message.get('traceback', '')}")
        return message

    def _require_connection(self):
        if self._connection is None:
            raise RuntimeError("direct vLLM runtime is not available on this rank")
        return self._connection


__all__ = ["VLLMRolloutEngine"]
