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

_PROTOCOL_VERSION = 1
_PROCESS_STOP_GRACE_S = 5.0


class VLLMConnectionState(str, Enum):
    """Driver-side state of the single synchronous runtime connection."""

    IDLE = "IDLE"
    INFLIGHT = "INFLIGHT"
    BROKEN = "BROKEN"


class _ProtocolError(RuntimeError):
    pass


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
        require(
            int(tp_size) == int(config.tp_size),
            f"vLLM injected tp_size={tp_size} does not match config.tp_size={config.tp_size}",
        )
        require(pp_size == 1 and pp_rank == 0, "vLLM rollout currently supports PP=1")
        require(ep_size == 1 and ep_rank == 0, "vLLM rollout currently supports EP=1")

        self.cfg = config
        self.device = device
        self._tp_rank = int(tp_rank)
        self._tp_size = int(tp_size)
        self._is_tp_zero = self._tp_rank == 0
        self._is_offloaded = False
        self._partially_awake = False
        self._weight_sync_sleep_level = 1
        self._preserve_next_weight_sleep = False
        self._version = 0
        self._lock = threading.Lock()
        self._process = None
        self._process_group_id = None
        self._connection = None
        self._connection_state = VLLMConnectionState.IDLE
        self._next_request_id = 1
        self._pending_native_publication = None
        self._ipc_worker_device_uuids: List[str] = []
        self.adapter = None

        if not self._is_tp_zero:
            return

        # Preserve Ray's physical CUDA token for each TP1 worker. Resetting a
        # spawned child to literal "0" would bypass an outer 4,5,6,7 pin.
        visible = _resolve_visible_devices(self._tp_size, tp_visible_devices)
        require(
            len(visible) == self._tp_size,
            f"vLLM expected {self._tp_size} visible devices, got {visible}",
        )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_ckpt_path,
            revision=config.tokenizer_revision or config.model_revision,
            trust_remote_code=bool(config.trust_remote_code),
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
                    "model_revision": config.model_revision,
                    "tokenizer_revision": config.tokenizer_revision or config.model_revision,
                    "tp_size": self._tp_size,
                    "trust_remote_code": bool(config.trust_remote_code),
                    "dtype": config.dtype,
                    "moe_backend": config.moe_backend,
                    "quantization": config.quantization,
                    "enforce_eager": config.enforce_eager,
                    "enable_prefix_caching": config.enable_prefix_caching,
                    "enable_chunked_prefill": config.enable_chunked_prefill,
                    "logprobs_mode": config.logprobs_mode,
                    "engine_kwargs": dict(config.engine_kwargs or {}),
                },
                "visible_devices": visible,
            },
            name=f"unirl-vllm-tp{self._tp_size}",
            daemon=False,
        )
        try:
            self._process.start()
            self._process_group_id = int(self._process.pid)
            child.close()
            ready = self._request("startup")
            if not isinstance(ready, dict) or ready.get("event") != "ready":
                raise _ProtocolError(f"vLLM returned invalid startup result: {ready!r}")
            self._validate_runtime_manifest(ready)
            self._ipc_worker_device_uuids = [
                str(item["cuda_device_uuid"])
                for item in sorted(
                    ready["worker_capabilities"],
                    key=lambda capability: int(capability["tp_rank"]),
                )
            ]
            self._process_group_id = int(ready["process_group_id"])
        except BaseException:
            child.close()
            self._break_connection()
            raise
        logger.info(
            "vLLM ready: rank=%s tp=%d devices=%s model=%s",
            rank,
            self._tp_size,
            visible,
            config.pretrained_model_ckpt_path,
        )

    @property
    def weight_payload_fanout(self) -> int:
        return self._tp_size

    @property
    def ipc_worker_device_uuids(self) -> List[str]:
        return list(self._ipc_worker_device_uuids)

    @property
    def connection_state(self) -> VLLMConnectionState:
        with self._lock:
            return self._connection_state

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        if not self._is_tp_zero:
            return None
        require(not self._is_offloaded, "VLLMRolloutEngine.generate called while sleeping")
        sampling = resolve_sampling(self.cfg, sample)
        prepared = self.adapter.build_inputs(sample, sampling=sampling)
        self._truncate_prepared_prompts(prepared)
        for payload in prepared.wire:
            block = payload["sampling_params"]
            sampling_seed = block.pop("sampling_seed", None)
            if sampling_seed is not None:
                block["seed"] = int(sampling_seed)
            if self.cfg.ignore_eos:
                block["ignore_eos"] = True
        response = self._request(
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
        if not self._is_tp_zero or (self._is_offloaded and not self._partially_awake):
            return
        level = 1 if self._preserve_next_weight_sleep else self._weight_sync_sleep_level
        self._request("sleep", level=level)
        self._preserve_next_weight_sleep = False
        self._is_offloaded = True
        self._partially_awake = False

    def set_weight_sync_sleep_level(self, level: int, *, preserve_next_sleep: bool = False) -> None:
        """Select whether future colocated sleeps offload or discard weights."""
        if int(level) not in (1, 2):
            raise ValueError(f"vLLM sleep level must be 1 or 2, got {level}")
        self._weight_sync_sleep_level = int(level)
        self._preserve_next_weight_sleep = bool(preserve_next_sleep)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def preserve_weights_for_next_sleep(self) -> None:
        """Force the next sleep to retain weights for a wake without sync."""
        self._preserve_next_weight_sleep = True

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

    def init_native_weight_transfer(self, *, init_info: Dict[str, Any]) -> None:
        """Initialize vLLM's built-in IPC Weight Transfer Engine."""
        if not self._is_tp_zero:
            return
        self._request("init_native_weight_transfer", init_info=dict(init_info))

    def start_native_weight_update(self, *, header: Dict[str, Any]) -> None:
        """Open one fail-stop publication around vLLM's native WTE."""
        if not self._is_tp_zero:
            return
        if self._pending_native_publication is not None:
            raise RuntimeError(f"vLLM native publication already pending: {self._pending_native_publication!r}")
        self._request("start_native_weight_update", header=dict(header))
        self._pending_native_publication = (
            str(header["publication_id"]),
            int(header["model_version"]),
        )

    def update_native_weights(self, *, update_info: Dict[str, Any]) -> None:
        """Forward one native packed IPC chunk to all vLLM TP workers."""
        if not self._is_tp_zero:
            return
        if self._pending_native_publication is None:
            raise RuntimeError("vLLM native weight update requires start first")
        self._request("update_native_weights", update_info=dict(update_info))

    def finish_native_weight_update(
        self,
        *,
        header: Dict[str, Any],
        flush_cache: bool,
        weight_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Finalize native WTE and publish its version after TP attestation."""
        if not self._is_tp_zero:
            return {}
        identity = (str(header["publication_id"]), int(header["model_version"]))
        if self._pending_native_publication != identity:
            raise RuntimeError(
                f"vLLM native finish {identity!r} does not match pending {self._pending_native_publication!r}"
            )
        from unirl.distributed.weight_sync.transfer.vllm_native_protocol import (
            validate_publication_result,
        )

        result = self._request(
            "finish_native_weight_update",
            header=dict(header),
            flush_cache=bool(flush_cache),
            weight_version=weight_version,
        )
        try:
            # Treat the spawned runtime response as untrusted even though it
            # validated receipts before committing its local version.
            validate_publication_result(
                header,
                result,
                fanout=self._tp_size,
                flush_cache=bool(flush_cache),
            )
        except BaseException:
            self._break_connection()
            raise
        self._pending_native_publication = None
        self._version = int(header["model_version"])
        return result

    def release_native_ipc(self) -> None:
        """Collect worker IPC imports after trainer-side export release."""
        if not self._is_tp_zero:
            return
        self._request("release_native_ipc")

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def poison(self, reason: str = "") -> None:
        """Poison and terminate a runtime after a partial weight publication."""
        if not self._is_tp_zero:
            return
        logger.error("Poisoning vLLM runtime: %s", reason or "unspecified failure")
        self._break_connection()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        if not self._is_tp_zero or self._process is None:
            return
        process = self._process
        try:
            if process.is_alive():
                if self.connection_state is VLLMConnectionState.IDLE:
                    self._request("shutdown")
                process.join(timeout=self.cfg.timeout_for("shutdown"))
        finally:
            with self._lock:
                self._dispose_runtime_locked()

    def _truncate_prepared_prompts(self, prepared: Any) -> None:
        require(
            len(prepared.wire) == len(prepared.prompt_token_ids),
            "vLLM adapter returned mismatched payload and prompt-token counts",
        )
        max_prompt_length = int(self.cfg.max_prompt_length)
        for index, (payload, prompt_token_ids) in enumerate(zip(prepared.wire, prepared.prompt_token_ids, strict=True)):
            truncated = [int(token) for token in prompt_token_ids[-max_prompt_length:]]
            payload["input_ids"] = truncated
            prepared.prompt_token_ids[index] = truncated

    def _request(self, command: str, *, timeout_s: Optional[float] = None, **payload):
        with self._lock:
            connection = self._require_connection()
            if self._connection_state is not VLLMConnectionState.IDLE:
                raise RuntimeError(
                    f"vLLM connection is {self._connection_state.value}; cannot start command {command!r}"
                )
            reserved = {"protocol_version", "request_id", "command"}.intersection(payload)
            if reserved:
                raise ValueError(f"vLLM payload uses reserved protocol fields: {sorted(reserved)}")
            request_id = self._next_request_id
            self._next_request_id += 1
            self._connection_state = VLLMConnectionState.INFLIGHT
            request = {
                **payload,
                "protocol_version": _PROTOCOL_VERSION,
                "request_id": request_id,
                "command": command,
            }
            try:
                connection.send(request)
                response = self._recv(
                    connection=connection,
                    timeout_s=self.cfg.timeout_for(command) if timeout_s is None else timeout_s,
                    expected_request_id=request_id,
                    expected_command=command,
                )
                result = response["result"]
                if not response["ok"]:
                    if command in {
                        "init_native_weight_transfer",
                        "start_native_weight_update",
                        "update_native_weights",
                        "finish_native_weight_update",
                        "release_native_ipc",
                    }:
                        self._validate_update_result(result, expected_status="aborted")
                    if command in {
                        "sleep",
                        "wake_up",
                        "init_native_weight_transfer",
                        "start_native_weight_update",
                        "update_native_weights",
                        "finish_native_weight_update",
                        "release_native_ipc",
                    }:
                        self._mark_broken_locked()
                    else:
                        self._connection_state = VLLMConnectionState.IDLE
                    raise RuntimeError(
                        f"vLLM {command} request {request_id} failed: "
                        f"{response.get('error')}\n{response.get('traceback', '')}"
                    )
                self._validate_command_result(command, result)
            except BaseException as error:
                if self._connection_state is VLLMConnectionState.INFLIGHT:
                    self._mark_broken_locked()
                if isinstance(error, (TimeoutError, EOFError, BrokenPipeError, OSError, _ProtocolError)):
                    raise type(error)(f"vLLM {command} request {request_id} broke the connection: {error}") from error
                raise
            if self._connection_state is not VLLMConnectionState.INFLIGHT:
                raise RuntimeError(
                    f"vLLM {command} request {request_id} completed after runtime became {self._connection_state.value}"
                )
            self._connection_state = VLLMConnectionState.IDLE
            return result

    def _recv(
        self,
        *,
        connection,
        timeout_s: float,
        expected_request_id: Optional[int] = None,
        expected_command: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not connection.poll(timeout_s):
            raise TimeoutError(f"vLLM did not respond within {timeout_s:.0f}s")
        try:
            message = connection.recv()
        except EOFError as error:
            raise EOFError("vLLM closed the response pipe") from error
        if not isinstance(message, dict):
            raise _ProtocolError(f"response must be a mapping, got {type(message).__name__}")
        if message.get("protocol_version") != _PROTOCOL_VERSION:
            raise _ProtocolError(
                f"response protocol_version={message.get('protocol_version')!r}, expected {_PROTOCOL_VERSION}"
            )
        request_id = message.get("request_id")
        if type(request_id) is not int or request_id < 1:
            raise _ProtocolError(f"response has invalid request_id={request_id!r}")
        command = message.get("command")
        if not isinstance(command, str) or not command:
            raise _ProtocolError(f"response has invalid command={command!r}")
        if expected_request_id is not None and request_id != expected_request_id:
            raise _ProtocolError(f"response request_id={request_id}, expected {expected_request_id}")
        if expected_command is not None and command != expected_command:
            raise _ProtocolError(f"response command={command!r}, expected {expected_command!r}")
        if type(message.get("ok")) is not bool:
            raise _ProtocolError(f"response has invalid ok={message.get('ok')!r}")
        if "result" not in message:
            raise _ProtocolError("response omitted result")
        return message

    @staticmethod
    def _validate_update_result(result: Any, *, expected_status: str) -> None:
        if not isinstance(result, dict):
            raise _ProtocolError(f"update result must be a mapping, got {type(result).__name__}")
        if result.get("status") != expected_status:
            raise _ProtocolError(f"update result status={result.get('status')!r}, expected {expected_status!r}")
        if "worker_receipts" not in result:
            raise _ProtocolError("update result omitted worker_receipts")

    def _validate_command_result(self, command: str, result: Any) -> None:
        if command == "startup":
            if not isinstance(result, dict) or result.get("event") != "ready":
                raise _ProtocolError(f"startup result must be a ready event, got {result!r}")
        elif command == "generate":
            if not isinstance(result, list):
                raise _ProtocolError(f"generate result must be a list, got {type(result).__name__}")
        elif command in {
            "init_native_weight_transfer",
            "start_native_weight_update",
            "update_native_weights",
            "release_native_ipc",
        }:
            if result is not None:
                raise _ProtocolError(f"{command} result must be None, got {result!r}")
        elif command == "finish_native_weight_update":
            if not isinstance(result, dict) or result.get("status") != "committed":
                raise _ProtocolError("vLLM native IPC finish result status must be 'committed'")
            if "worker_receipts" not in result:
                raise _ProtocolError("vLLM native IPC finish omitted worker_receipts")
        elif command == "health":
            if type(result) is not bool:
                raise _ProtocolError(f"health result must be bool, got {type(result).__name__}")
        elif command in {"sleep", "wake_up", "shutdown"} and result is not None:
            raise _ProtocolError(f"{command} result must be None, got {result!r}")

    def _validate_runtime_manifest(self, ready: Dict[str, Any]) -> None:
        """Validate the vLLM runtime and TP worker capabilities."""
        if ready.get("vllm_version") != "0.27.0":
            raise RuntimeError(f"vLLM rollout requires vllm==0.27.0; got {ready.get('vllm_version')!r}")
        if ready.get("model_type") != "qwen3_moe":
            raise RuntimeError(f"vLLM native IPC sync currently supports qwen3_moe; got {ready.get('model_type')!r}")
        if int(ready.get("process_group_id", -1)) != int(self._process.pid):
            raise RuntimeError(f"vLLM did not create its required process group: {ready!r}")
        capabilities = ready.get("worker_capabilities")
        if not isinstance(capabilities, list) or len(capabilities) != self._tp_size:
            raise RuntimeError(f"vLLM returned invalid TP capabilities: {capabilities!r}")
        ranks = sorted(int(item.get("tp_rank", -1)) for item in capabilities)
        if ranks != list(range(self._tp_size)):
            raise RuntimeError(f"vLLM TP capability ranks {ranks} != {list(range(self._tp_size))}")
        world_sizes = [int(item.get("tp_world_size", -1)) for item in capabilities]
        if any(world_size != self._tp_size for world_size in world_sizes):
            raise RuntimeError(f"vLLM TP capability world sizes {world_sizes} != {self._tp_size}")
        device_uuids = [item.get("cuda_device_uuid") for item in capabilities]
        if (
            any(not isinstance(value, str) or not value for value in device_uuids)
            or len(set(device_uuids)) != self._tp_size
        ):
            raise RuntimeError(f"vLLM worker CUDA UUIDs are invalid: {device_uuids!r}")
        if not all(item.get("vllm_version") == "0.27.0" for item in capabilities):
            raise RuntimeError(f"vLLM worker versions are incompatible: {capabilities!r}")
        logger.info("vLLM runtime manifest validated: vllm=%s", ready["vllm_version"])

    def _require_connection(self):
        if self._connection_state is VLLMConnectionState.BROKEN:
            raise RuntimeError("vLLM runtime connection is broken and cannot be reused")
        if self._connection is None:
            raise RuntimeError("vLLM runtime is not available on this rank")
        return self._connection

    def _break_connection(self) -> None:
        with self._lock:
            if self._connection_state is VLLMConnectionState.BROKEN:
                return
            self._mark_broken_locked()

    def _mark_broken_locked(self) -> None:
        if self._connection_state is VLLMConnectionState.BROKEN:
            return
        self._connection_state = VLLMConnectionState.BROKEN
        self._dispose_runtime_locked()

    def _dispose_runtime_locked(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

        process, self._process = self._process, None
        process_group_id, self._process_group_id = self._process_group_id, None
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
