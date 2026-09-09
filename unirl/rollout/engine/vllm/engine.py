"""Direct text-only vLLM rollout engine."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from enum import Enum
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
        require(
            int(tp_size) == int(config.tp_size),
            f"direct vLLM injected tp_size={tp_size} does not match config.tp_size={config.tp_size}",
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
        self._connection_state = VLLMConnectionState.IDLE
        self._next_request_id = 1
        self._startup_manifest = None
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
            revision=config.model_revision,
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
                    "tp_size": self._tp_size,
                    "rollout_rank": rollout_rank,
                    "trust_remote_code": bool(config.trust_remote_code),
                    "engine_kwargs": dict(config.engine_kwargs or {}),
                },
                "visible_devices": visible,
            },
            name=f"unirl-vllm-tp{self._tp_size}",
            daemon=False,
        )
        try:
            self._process.start()
            child.close()
            ready = self._request("startup")
            if not isinstance(ready, dict) or ready.get("event") != "ready":
                raise _ProtocolError(f"direct vLLM returned invalid startup result: {ready!r}")
            self._startup_manifest = ready.get("plugin_manifest")
        except BaseException:
            child.close()
            self._break_connection()
            raise
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

    @property
    def connection_state(self) -> VLLMConnectionState:
        with self._lock:
            return self._connection_state

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def parity_runtime_manifest(self) -> Optional[Dict[str, Any]]:
        if not self._is_tp_zero:
            return None
        return None if self._startup_manifest is None else dict(self._startup_manifest)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        if not self._is_tp_zero:
            return None
        require(not self._is_offloaded, "VLLMRolloutEngine.generate called while sleeping")
        sampling = resolve_sampling(self.cfg, sample)
        prepared = self.adapter.build_inputs(sample, sampling=sampling)
        self._truncate_prepared_prompts(prepared)
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
        serialized_named_tensors: Optional[List[str]] = None,
        payloads: Optional[List[str]] = None,
        header: Optional[Dict[str, Any]] = None,
        tp_world_size: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> Dict[str, Any]:
        del target_modules, track_prefix
        if not self._is_tp_zero:
            return {}
        wire_payloads = list(payloads if payloads is not None else serialized_named_tensors or ())
        if payloads is not None and serialized_named_tensors is not None:
            if list(payloads) != list(serialized_named_tensors):
                raise ValueError("direct vLLM received conflicting payload aliases")
        expected_tp = self._tp_size if tp_world_size is None else int(tp_world_size)
        if expected_tp != self._tp_size:
            raise ValueError(f"weight update TP world {expected_tp} != rollout TP {self._tp_size}")
        if len(wire_payloads) != self._tp_size:
            raise ValueError(f"weight payload count {len(wire_payloads)} != rollout TP {self._tp_size}")
        if header is not None:
            from unirl.distributed.weight_sync.transfer.fsdp_vllm_protocol import (
                validate_bucket_header,
            )

            validate_bucket_header(header, payload_count=len(wire_payloads), tp_world_size=self._tp_size)
        receipt = self._request(
            "update_weights",
            serialized_named_tensors=wire_payloads,
            payloads=wire_payloads,
            header=header,
            tp_world_size=self._tp_size,
            load_format=load_format,
            flush_cache=bool(flush_cache),
        )
        expected_status = "committed" if header is None or bool(header.get("is_last")) else "staged"
        if receipt.get("status") != expected_status:
            raise _ProtocolError(f"weight update status={receipt.get('status')!r}, expected {expected_status!r}")
        try:
            worker_receipts = receipt.get("worker_receipts")
            if not isinstance(worker_receipts, list) or len(worker_receipts) != self._tp_size:
                raise _ProtocolError(
                    f"weight update worker receipt count must equal TP={self._tp_size}; got {worker_receipts!r}"
                )
            if header is not None:
                from unirl.distributed.weight_sync.transfer.fsdp_vllm_protocol import (
                    validate_receipts,
                )

                validate_receipts(header, receipt, fanout=self._tp_size)
        except BaseException:
            self._break_connection()
            raise
        if expected_status == "committed":
            committed_version = int(header["model_version"]) if header is not None else self._version + 1
            self._version = committed_version
        return receipt

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
            "direct vLLM adapter returned mismatched payload and prompt-token counts",
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
                    f"direct vLLM connection is {self._connection_state.value}; cannot start command {command!r}"
                )
            reserved = {"protocol_version", "request_id", "command"}.intersection(payload)
            if reserved:
                raise ValueError(f"direct vLLM payload uses reserved protocol fields: {sorted(reserved)}")
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
                    timeout_s=self.cfg.timeout_for(command) if timeout_s is None else timeout_s,
                    expected_request_id=request_id,
                    expected_command=command,
                )
                result = response["result"]
                if not response["ok"]:
                    if command == "update_weights":
                        self._validate_update_result(result, expected_status="aborted")
                    if command in {"sleep", "wake_up", "update_weights"}:
                        self._mark_broken_locked()
                    else:
                        self._connection_state = VLLMConnectionState.IDLE
                    raise RuntimeError(
                        f"direct vLLM {command} request {request_id} failed: "
                        f"{response.get('error')}\n{response.get('traceback', '')}"
                    )
                self._validate_command_result(command, result)
            except BaseException as error:
                if self._connection_state is VLLMConnectionState.INFLIGHT:
                    self._mark_broken_locked()
                if isinstance(error, (TimeoutError, EOFError, BrokenPipeError, OSError, _ProtocolError)):
                    raise type(error)(
                        f"direct vLLM {command} request {request_id} broke the connection: {error}"
                    ) from error
                raise
            self._connection_state = VLLMConnectionState.IDLE
            return result

    def _recv(
        self,
        *,
        timeout_s: float,
        expected_request_id: Optional[int] = None,
        expected_command: Optional[str] = None,
    ) -> Dict[str, Any]:
        connection = self._require_connection()
        if not connection.poll(timeout_s):
            raise TimeoutError(f"direct vLLM did not respond within {timeout_s:.0f}s")
        try:
            message = connection.recv()
        except EOFError as error:
            raise EOFError("direct vLLM closed the response pipe") from error
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
        elif command == "update_weights":
            if not isinstance(result, dict) or result.get("status") not in {
                "staged",
                "committed",
            }:
                raise _ProtocolError("update result status must be 'staged' or 'committed'")
            if "worker_receipts" not in result:
                raise _ProtocolError("update result omitted worker_receipts")
        elif command == "health":
            if type(result) is not bool:
                raise _ProtocolError(f"health result must be bool, got {type(result).__name__}")
        elif command in {"sleep", "wake_up", "shutdown"} and result is not None:
            raise _ProtocolError(f"{command} result must be None, got {result!r}")

    def _require_connection(self):
        if self._connection_state is VLLMConnectionState.BROKEN:
            raise RuntimeError("direct vLLM runtime connection is broken and cannot be reused")
        if self._connection is None:
            raise RuntimeError("direct vLLM runtime is not available on this rank")
        return self._connection

    def _break_connection(self) -> None:
        with self._lock:
            self._mark_broken_locked()

    def _mark_broken_locked(self) -> None:
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
        if process is None:
            return
        try:
            alive = process.is_alive()
        except (AssertionError, OSError, ValueError):
            alive = False
        if alive:
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
                    process.kill()
                except (AttributeError, OSError, ValueError):
                    pass
                try:
                    process.join(timeout=_PROCESS_STOP_GRACE_S)
                except (AssertionError, OSError, ValueError):
                    pass
        try:
            process.close()
        except (AttributeError, OSError, ValueError):
            pass


__all__ = ["VLLMConnectionState", "VLLMRolloutEngine"]
