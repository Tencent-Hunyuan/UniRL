"""Direct text-only vLLM rollout engine."""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import threading
import time
from enum import Enum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.sglang.adapters.text import TextLMAdapter
from unirl.rollout.engine.vllm.config import VLLMEngineConfig
from unirl.rollout.engine.vllm.runtime import engine_process_main
from unirl.rollout.engine.vllm.sampling import resolve_sampling
from unirl.types.sample import Sample

logger = logging.getLogger(__name__)

_PROCESS_STOP_GRACE_S = 5.0


class VLLMConnectionState(str, Enum):
    """Driver-side state of the single synchronous runtime connection."""

    IDLE = "IDLE"
    INFLIGHT = "INFLIGHT"
    BROKEN = "BROKEN"


class ProtocolError(RuntimeError):
    pass


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
        del device, strategy, model_config
        require(
            isinstance(config, VLLMEngineConfig),
            f"VLLMRolloutEngine requires VLLMEngineConfig; got {type(config).__name__}",
        )
        require(
            tp_size == config.tp_size,
            f"vLLM injected tp_size={tp_size} does not match config.tp_size={config.tp_size}",
        )
        require(pp_size == 1 and pp_rank == 0, "vLLM rollout currently supports PP=1")
        require(ep_size == 1 and ep_rank == 0, "vLLM rollout currently supports EP=1")

        self.cfg = config
        self.tp_size = tp_size
        self.is_tp_zero = tp_rank == 0
        self._is_offloaded = False
        self.partially_awake = False
        self.weight_sync_sleep_level = 1
        self.preserve_next_weight_sleep = False
        self.lock = threading.Lock()
        self.process = None
        self.process_group_id = None
        self.connection = None
        self.connection_state = VLLMConnectionState.IDLE
        self.next_request_id = 1
        self.pending_native_publication = None
        self.ipc_worker_device_uuids: List[str] = []
        self.adapter = None

        if not self.is_tp_zero:
            return

        # Preserve Ray's physical CUDA token for each TP1 worker. Resetting a
        # spawned child to literal "0" would bypass an outer 4,5,6,7 pin.
        if tp_visible_devices is not None:
            visible = list(tp_visible_devices)
        else:
            inherited = [
                token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()
            ]
            visible = inherited[: self.tp_size] if inherited else [str(index) for index in range(self.tp_size)]
        require(
            len(visible) == self.tp_size,
            f"vLLM expected {self.tp_size} visible devices, got {visible}",
        )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_ckpt_path,
            revision=config.tokenizer_revision or config.model_revision,
            trust_remote_code=config.trust_remote_code,
        )
        self.adapter = TextLMAdapter(config, tokenizer=tokenizer)

        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(
            target=engine_process_main,
            kwargs={
                "connection": child,
                "config": {
                    "pretrained_model_ckpt_path": config.pretrained_model_ckpt_path,
                    "model_revision": config.model_revision,
                    "tokenizer_revision": config.tokenizer_revision or config.model_revision,
                    "tp_size": self.tp_size,
                    "trust_remote_code": config.trust_remote_code,
                    "dtype": config.dtype,
                    "moe_backend": config.moe_backend,
                    "quantization": config.quantization,
                    "enforce_eager": config.enforce_eager,
                    "enable_prefix_caching": config.enable_prefix_caching,
                    "enable_chunked_prefill": config.enable_chunked_prefill,
                    "logprobs_mode": config.logprobs_mode,
                    "engine_kwargs": dict(config.engine_kwargs),
                },
                "visible_devices": visible,
            },
            name=f"unirl-vllm-tp{self.tp_size}",
            daemon=False,
        )
        try:
            self.process.start()
            self.process_group_id = self.process.pid
            child.close()
            ready = self.request("startup")
            if not isinstance(ready, dict) or ready.get("event") != "ready":
                raise ProtocolError(f"vLLM returned invalid startup result: {ready!r}")
            if ready.get("model_type") != "qwen3_moe":
                raise RuntimeError(
                    f"vLLM native IPC sync currently supports qwen3_moe; got {ready.get('model_type')!r}"
                )
            if ready.get("process_group_id") != self.process.pid:
                raise RuntimeError(f"vLLM did not create its required process group: {ready!r}")
            capabilities = ready.get("worker_capabilities")
            if not isinstance(capabilities, list) or len(capabilities) != self.tp_size:
                raise RuntimeError(f"vLLM returned invalid TP capabilities: {capabilities!r}")
            if any(not isinstance(item, dict) for item in capabilities):
                raise RuntimeError(f"vLLM returned non-dict TP capability entry: {capabilities!r}")
            raw_ranks = [item.get("tp_rank") for item in capabilities]
            if any(type(r) is not int for r in raw_ranks):
                raise RuntimeError(f"vLLM TP capability ranks must be ints: {raw_ranks!r}")
            ranks = sorted(raw_ranks)
            if ranks != list(range(self.tp_size)):
                raise RuntimeError(f"vLLM TP capability ranks {ranks} != {list(range(self.tp_size))}")
            world_sizes = [item.get("tp_world_size") for item in capabilities]
            if any(type(ws) is not int or ws != self.tp_size for ws in world_sizes):
                raise RuntimeError(f"vLLM TP capability world sizes {world_sizes} != {self.tp_size}")
            device_uuids = [item.get("cuda_device_uuid") for item in capabilities]
            if (
                any(not isinstance(value, str) or not value for value in device_uuids)
                or len(set(device_uuids)) != self.tp_size
            ):
                raise RuntimeError(f"vLLM worker CUDA UUIDs are invalid: {device_uuids!r}")
            logger.info("vLLM runtime manifest validated: vllm=%s", ready["vllm_version"])
            self.ipc_worker_device_uuids = [
                item["cuda_device_uuid"]
                for item in sorted(
                    capabilities,
                    key=lambda capability: capability["tp_rank"],
                )
            ]
            self.process_group_id = ready["process_group_id"]
        except BaseException:
            child.close()
            with self.lock:
                self.mark_broken_locked()
            raise
        logger.info(
            "vLLM ready: rank=%s tp=%d devices=%s model=%s",
            rank,
            self.tp_size,
            visible,
            config.pretrained_model_ckpt_path,
        )

    @property
    def weight_payload_fanout(self) -> int:
        return self.tp_size

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        if not self.is_tp_zero:
            return None
        require(not self._is_offloaded, "VLLMRolloutEngine.generate called while sleeping")
        sampling = resolve_sampling(self.cfg, sample)
        prepared = self.adapter.build_inputs(sample, sampling=sampling)
        require(
            len(prepared.wire) == len(prepared.prompt_token_ids),
            "vLLM adapter returned mismatched payload and prompt-token counts",
        )
        max_prompt_length = self.cfg.max_prompt_length
        for index, (payload, prompt_token_ids) in enumerate(zip(prepared.wire, prepared.prompt_token_ids, strict=True)):
            truncated = list(prompt_token_ids[-max_prompt_length:])
            payload["input_ids"] = truncated
            prepared.prompt_token_ids[index] = truncated
        for payload in prepared.wire:
            block = payload["sampling_params"]
            sampling_seed = block.pop("sampling_seed", None)
            if sampling_seed is not None:
                block["seed"] = sampling_seed
            if self.cfg.ignore_eos:
                block["ignore_eos"] = True
        response = self.request(
            "generate",
            payloads=prepared.wire,
            record_id="batch",
        )
        raw = [SimpleNamespace(**item) for item in response]
        generated = self.adapter.build_response(sample, prepared, raw)
        return self._stamp_output_version(generated)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self, tags: Optional[List[str]] = None) -> None:
        del tags
        if not self.is_tp_zero or (self._is_offloaded and not self.partially_awake):
            return
        level = 1 if self.preserve_next_weight_sleep else self.weight_sync_sleep_level
        self.request("sleep", level=level)
        self.preserve_next_weight_sleep = False
        self._is_offloaded = True
        self.partially_awake = False

    def set_weight_sync_sleep_level(self, level: int, *, preserve_next_sleep: bool = False) -> None:
        """Select whether future colocated sleeps offload or discard weights."""
        if level not in (1, 2):
            raise ValueError(f"vLLM sleep level must be 1 or 2, got {level}")
        self.weight_sync_sleep_level = level
        self.preserve_next_weight_sleep = preserve_next_sleep

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def preserve_weights_for_next_sleep(self) -> None:
        """Force the next sleep to retain weights for a wake without sync."""
        self.preserve_next_weight_sleep = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self, tags: Optional[List[str]] = None) -> None:
        if not self.is_tp_zero:
            return
        partial = bool(tags)
        if partial:
            if not self._is_offloaded or self.partially_awake:
                return
        elif not self._is_offloaded and not self.partially_awake:
            return
        request_tags = ["kv_cache"] if not partial and self.partially_awake else tags
        self.request("wake_up", tags=request_tags)
        self.partially_awake = partial
        self._is_offloaded = partial

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def health_check(self) -> bool:
        if not self.is_tp_zero:
            return True
        return bool(self.request("health"))

    def init_native_weight_transfer(self, *, init_info: Dict[str, Any]) -> None:
        """Initialize vLLM's built-in IPC Weight Transfer Engine."""
        if not self.is_tp_zero:
            return
        self.request("init_native_weight_transfer", init_info=dict(init_info))

    def start_native_weight_update(self, *, header: Dict[str, Any]) -> None:
        """Open one fail-stop publication around vLLM's native WTE."""
        if not self.is_tp_zero:
            return
        if self.pending_native_publication is not None:
            raise RuntimeError(f"vLLM native publication already pending: {self.pending_native_publication!r}")
        self.request("start_native_weight_update", header=dict(header))
        self.pending_native_publication = (
            str(header["publication_id"]),
            int(header["model_version"]),
        )

    def update_native_weights(self, *, update_info: Dict[str, Any]) -> None:
        """Forward one native packed IPC chunk to all vLLM TP workers."""
        if not self.is_tp_zero:
            return
        if self.pending_native_publication is None:
            raise RuntimeError("vLLM native weight update requires start first")
        self.request("update_native_weights", update_info=dict(update_info))

    def finish_native_weight_update(
        self,
        *,
        header: Dict[str, Any],
        flush_cache: bool,
        weight_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Finalize native WTE and publish its version after TP attestation."""
        if not self.is_tp_zero:
            return {}
        identity = (str(header["publication_id"]), int(header["model_version"]))
        if self.pending_native_publication != identity:
            raise RuntimeError(
                f"vLLM native finish {identity!r} does not match pending {self.pending_native_publication!r}"
            )
        from unirl.distributed.weight_sync.transfer.vllm_native_protocol import (
            validate_publication_result,
        )

        result = self.request(
            "finish_native_weight_update",
            header=dict(header),
            flush_cache=flush_cache,
            weight_version=weight_version,
        )
        try:
            # Treat the spawned runtime response as untrusted even though it
            # validated receipts before committing its local version.
            validate_publication_result(
                header,
                result,
                fanout=self.tp_size,
                flush_cache=flush_cache,
            )
        except BaseException:
            with self.lock:
                self.mark_broken_locked()
            raise
        self.pending_native_publication = None
        self._version = int(header["model_version"])
        return result

    def release_native_ipc(self) -> None:
        """Collect worker IPC imports after trainer-side export release."""
        if not self.is_tp_zero:
            return
        self.request("release_native_ipc")

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def poison(self, reason: str = "") -> None:
        """Poison and terminate a runtime after a partial weight publication."""
        if not self.is_tp_zero:
            return
        logger.error("Poisoning vLLM runtime: %s", reason or "unspecified failure")
        with self.lock:
            self.mark_broken_locked()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        if not self.is_tp_zero or self.process is None:
            return
        process = self.process
        try:
            if process.is_alive():
                with self.lock:
                    idle = self.connection_state is VLLMConnectionState.IDLE
                if idle:
                    self.request("shutdown")
                process.join(timeout=self.cfg.timeout_for("shutdown"))
        finally:
            with self.lock:
                self.dispose_runtime_locked()

    def request(self, command: str, *, timeout_s: Optional[float] = None, **payload):
        with self.lock:
            if self.connection_state is VLLMConnectionState.BROKEN:
                raise RuntimeError("vLLM runtime connection is broken and cannot be reused")
            if self.connection is None:
                raise RuntimeError("vLLM runtime is not available on this rank")
            connection = self.connection
            if self.connection_state is not VLLMConnectionState.IDLE:
                raise RuntimeError(
                    f"vLLM connection is {self.connection_state.value}; cannot start command {command!r}"
                )
            reserved = {"request_id", "command"}.intersection(payload)
            if reserved:
                raise ValueError(f"vLLM payload uses reserved protocol fields: {sorted(reserved)}")
            request_id = self.next_request_id
            self.next_request_id += 1
            self.connection_state = VLLMConnectionState.INFLIGHT
            req_msg = {
                **payload,
                "request_id": request_id,
                "command": command,
            }
            eff_timeout_s = self.cfg.timeout_for(command) if timeout_s is None else timeout_s
            try:
                connection.send(req_msg)
                if not connection.poll(eff_timeout_s):
                    raise TimeoutError(f"vLLM did not respond within {eff_timeout_s:.0f}s")
                try:
                    response = connection.recv()
                except EOFError as error:
                    raise EOFError("vLLM closed the response pipe") from error
                if not isinstance(response, dict):
                    raise ProtocolError(f"response must be a mapping, got {type(response).__name__}")
                resp_id = response.get("request_id")
                if type(resp_id) is not int or resp_id < 1:
                    raise ProtocolError(f"response has invalid request_id={resp_id!r}")
                resp_cmd = response.get("command")
                if not isinstance(resp_cmd, str) or not resp_cmd:
                    raise ProtocolError(f"response has invalid command={resp_cmd!r}")
                if resp_id != request_id:
                    raise ProtocolError(f"response request_id={resp_id}, expected {request_id}")
                if resp_cmd != command:
                    raise ProtocolError(f"response command={resp_cmd!r}, expected {command!r}")
                if type(response.get("ok")) is not bool:
                    raise ProtocolError(f"response has invalid ok={response.get('ok')!r}")
                if "result" not in response:
                    raise ProtocolError("response omitted result")
                result = response["result"]
                if not response["ok"]:
                    if command in {
                        "init_native_weight_transfer",
                        "start_native_weight_update",
                        "update_native_weights",
                        "finish_native_weight_update",
                        "release_native_ipc",
                    }:
                        if not isinstance(result, dict):
                            raise ProtocolError(f"update result must be a mapping, got {type(result).__name__}")
                        if result.get("status") != "aborted":
                            raise ProtocolError(f"update result status={result.get('status')!r}, expected 'aborted'")
                        if "worker_receipts" not in result:
                            raise ProtocolError("update result omitted worker_receipts")
                    if command in {
                        "sleep",
                        "wake_up",
                        "init_native_weight_transfer",
                        "start_native_weight_update",
                        "update_native_weights",
                        "finish_native_weight_update",
                        "release_native_ipc",
                    }:
                        self.mark_broken_locked()
                    else:
                        self.connection_state = VLLMConnectionState.IDLE
                    raise RuntimeError(
                        f"vLLM {command} request {request_id} failed: "
                        f"{response.get('error')}\n{response.get('traceback', '')}"
                    )
                if command == "startup":
                    if not isinstance(result, dict) or result.get("event") != "ready":
                        raise ProtocolError(f"startup result must be a ready event, got {result!r}")
                elif command == "generate":
                    if not isinstance(result, list):
                        raise ProtocolError(f"generate result must be a list, got {type(result).__name__}")
                elif command in {
                    "init_native_weight_transfer",
                    "start_native_weight_update",
                    "update_native_weights",
                    "release_native_ipc",
                }:
                    if result is not None:
                        raise ProtocolError(f"{command} result must be None, got {result!r}")
                elif command == "finish_native_weight_update":
                    if not isinstance(result, dict) or result.get("status") != "committed":
                        raise ProtocolError("vLLM native IPC finish result status must be 'committed'")
                    if "worker_receipts" not in result:
                        raise ProtocolError("vLLM native IPC finish omitted worker_receipts")
                elif command == "health":
                    if type(result) is not bool:
                        raise ProtocolError(f"health result must be bool, got {type(result).__name__}")
                elif command in {"sleep", "wake_up", "shutdown"} and result is not None:
                    raise ProtocolError(f"{command} result must be None, got {result!r}")
            except BaseException as error:
                if self.connection_state is VLLMConnectionState.INFLIGHT:
                    self.mark_broken_locked()
                if isinstance(error, (TimeoutError, EOFError, BrokenPipeError, OSError, ProtocolError)):
                    raise type(error)(f"vLLM {command} request {request_id} broke the connection: {error}") from error
                raise
            if self.connection_state is not VLLMConnectionState.INFLIGHT:
                raise RuntimeError(
                    f"vLLM {command} request {request_id} completed after runtime became {self.connection_state.value}"
                )
            self.connection_state = VLLMConnectionState.IDLE
            return result

    def mark_broken_locked(self) -> None:
        if self.connection_state is VLLMConnectionState.BROKEN:
            return
        self.connection_state = VLLMConnectionState.BROKEN
        self.dispose_runtime_locked()

    def dispose_runtime_locked(self) -> None:
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

        process, self.process = self.process, None
        process_group_id, self.process_group_id = self.process_group_id, None
        if process is None:
            return
        try:
            alive = process.is_alive()
        except (AssertionError, OSError, ValueError):
            alive = False
        if alive:
            try:
                if process_group_id is None:
                    raise ProcessLookupError
                os.killpg(process_group_id, signal.SIGTERM)
            except (OSError, ValueError):
                try:
                    process.terminate()
                except (OSError, ValueError):
                    pass
            try:
                process.join(timeout=_PROCESS_STOP_GRACE_S)
            except (AssertionError, OSError, ValueError):
                pass
            try:
                alive = process.is_alive()
            except (AssertionError, OSError, ValueError):
                alive = False
            if alive:
                try:
                    if process_group_id is None:
                        raise ProcessLookupError
                    os.killpg(process_group_id, signal.SIGKILL)
                except (AttributeError, OSError, ValueError):
                    try:
                        process.kill()
                    except (AttributeError, OSError, ValueError):
                        pass
                try:
                    process.join(timeout=_PROCESS_STOP_GRACE_S)
                except (AssertionError, OSError, ValueError):
                    pass
        if process_group_id is not None:
            try:
                os.killpg(process_group_id, 0)
            except OSError:
                pass
            else:
                try:
                    os.killpg(process_group_id, signal.SIGTERM)
                    time.sleep(0.5)
                    os.killpg(process_group_id, signal.SIGKILL)
                except OSError:
                    pass
        try:
            process.close()
        except (AttributeError, OSError, ValueError):
            pass


__all__ = ["VLLMConnectionState", "VLLMRolloutEngine"]
