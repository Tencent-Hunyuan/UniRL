"""Spawned vLLM runtime used by the direct rollout engine."""

from __future__ import annotations

import os
import traceback
from multiprocessing.connection import Connection
from typing import Any, Dict, List

_PROTOCOL_VERSION = 1
_SUPPORTED_VLLM_VERSION = "0.28.0"


class _ProtocolError(RuntimeError):
    pass


def _recv_request(connection: Connection, *, last_request_id: int) -> Dict[str, Any]:
    message = connection.recv()
    if not isinstance(message, dict):
        raise _ProtocolError(f"request must be a mapping, got {type(message).__name__}")
    if message.get("protocol_version") != _PROTOCOL_VERSION:
        raise _ProtocolError(
            f"request protocol_version={message.get('protocol_version')!r}, expected {_PROTOCOL_VERSION}"
        )
    request_id = message.get("request_id")
    if type(request_id) is not int or request_id != last_request_id + 1:
        raise _ProtocolError(f"request_id={request_id!r}, expected the next monotonic id {last_request_id + 1}")
    command = message.get("command")
    if not isinstance(command, str) or not command:
        raise _ProtocolError(f"request has invalid command={command!r}")
    return message


def _send_response(
    connection: Connection,
    request: Dict[str, Any],
    *,
    ok: bool,
    result: Any,
    error: BaseException | None = None,
) -> None:
    response = {
        "protocol_version": _PROTOCOL_VERSION,
        "request_id": request["request_id"],
        "command": request["command"],
        "ok": bool(ok),
        "result": result,
    }
    if error is not None:
        response.update(
            {
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
    connection.send(response)


def _shutdown_llm(llm: Any) -> None:
    """Best-effort shutdown of the vLLM engine and all executor workers."""
    if llm is None:
        return
    shutdown = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    shutdown = getattr(shutdown, "shutdown", None)
    if callable(shutdown):
        shutdown(timeout=30.0)


def _sampling_params(payload: Dict[str, Any], *, return_logprob: bool):
    from vllm import SamplingParams

    values = dict(payload)
    if "max_new_tokens" in values:
        values["max_tokens"] = values.pop("max_new_tokens")
    values["logprobs"] = 0 if return_logprob else None
    return SamplingParams(**values)


def _plain_outputs(outputs) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for request in outputs:
        for completion in sorted(request.outputs, key=lambda item: int(item.index)):
            token_ids = [int(token) for token in completion.token_ids]
            if completion.logprobs is None:
                logprobs: List[float] = []
            else:
                if len(completion.logprobs) != len(token_ids):
                    raise RuntimeError(
                        "vLLM returned mismatched token/logprob lengths: "
                        f"{len(token_ids)} vs {len(completion.logprobs)}"
                    )
                logprobs = []
                for token_id, choices in zip(token_ids, completion.logprobs, strict=True):
                    selected = choices.get(token_id)
                    if selected is None:
                        raise RuntimeError(f"vLLM omitted sampled token {token_id} from processed logprobs")
                    logprobs.append(float(selected.logprob))
            flattened.append(
                {
                    "text": str(completion.text or ""),
                    "token_ids": token_ids,
                    "logprobs": logprobs,
                    "finish_reason": str(completion.finish_reason or ""),
                }
            )
    return flattened


def engine_process_main(
    connection: Connection,
    *,
    config: Dict[str, Any],
    visible_devices: List[str],
) -> None:
    """Own the vLLM interpreter and serve synchronous control messages."""
    os.setsid()
    startup_request: Dict[str, Any] | None = None
    llm = None
    try:
        startup_request = _recv_request(connection, last_request_id=0)
        if startup_request["command"] != "startup":
            raise _ProtocolError(f"first vLLM command must be 'startup', got {startup_request['command']!r}")

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # vLLM native IPC handles contain PyTorch reducer objects that cannot
        # cross EngineCore's msgspec channel. Match vLLM's official HTTP IPC
        # client: pickle+base64 inside this trusted local runtime boundary.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        from vllm import LLM
        from vllm import __version__ as vllm_version
        from vllm.config import WeightTransferConfig

        if str(vllm_version) != _SUPPORTED_VLLM_VERSION:
            raise RuntimeError(f"vLLM rollout requires vllm=={_SUPPORTED_VLLM_VERSION}; got {vllm_version}")
        engine_kwargs = dict(config.get("engine_kwargs") or {})
        engine_kwargs["dtype"] = str(config["dtype"])
        engine_kwargs["trust_remote_code"] = bool(config.get("trust_remote_code", False))
        if config.get("model_revision"):
            engine_kwargs["revision"] = str(config["model_revision"])
        if config.get("tokenizer_revision"):
            engine_kwargs["tokenizer_revision"] = str(config["tokenizer_revision"])
        engine_kwargs["distributed_executor_backend"] = "mp"
        engine_kwargs["enable_sleep_mode"] = True
        engine_kwargs["enforce_eager"] = bool(config["enforce_eager"])
        engine_kwargs["enable_prefix_caching"] = bool(config["enable_prefix_caching"])
        engine_kwargs["enable_chunked_prefill"] = bool(config["enable_chunked_prefill"])
        engine_kwargs["moe_backend"] = str(config["moe_backend"])
        engine_kwargs["logprobs_mode"] = str(config["logprobs_mode"])
        engine_kwargs["worker_extension_cls"] = "unirl.rollout.engine.vllm.worker_extension.UniRLWeightSyncExtension"
        engine_kwargs["weight_transfer_config"] = WeightTransferConfig(backend="ipc")
        if config.get("quantization") is not None:
            engine_kwargs["quantization"] = str(config["quantization"])
        llm = LLM(
            model=str(config["pretrained_model_ckpt_path"]),
            tensor_parallel_size=int(config["tp_size"]),
            **engine_kwargs,
        )
        worker_capabilities = llm.collective_rpc("unirl_weight_sync_capabilities")
        model_type = str(getattr(llm.model_config.hf_config, "model_type", ""))
        _send_response(
            connection,
            startup_request,
            ok=True,
            result={
                "event": "ready",
                "vllm_version": str(vllm_version),
                "process_group_id": os.getpgrp(),
                "model_type": model_type,
                "worker_capabilities": worker_capabilities,
            },
        )
    except BaseException as error:
        if startup_request is not None:
            try:
                _send_response(connection, startup_request, ok=False, result=None, error=error)
            except (EOFError, BrokenPipeError, OSError):
                pass
        try:
            _shutdown_llm(llm)
        except BaseException:
            pass
        connection.close()
        return

    last_request_id = int(startup_request["request_id"])
    native_pending_header: Dict[str, Any] | None = None
    try:
        while True:
            try:
                message = _recv_request(connection, last_request_id=last_request_id)
            except EOFError:
                break
            except _ProtocolError:
                break
            last_request_id = int(message["request_id"])
            command = message.get("command")
            worker_receipts: Any = []
            try:
                if command == "generate":
                    payloads = list(message["payloads"])
                    sampling_blocks = [dict(payload["sampling_params"]) for payload in payloads]
                    prompts = [
                        {"prompt_token_ids": [int(token) for token in payload["input_ids"]]} for payload in payloads
                    ]
                    params = [
                        _sampling_params(block, return_logprob=bool(payload.get("return_logprob", True)))
                        for payload, block in zip(payloads, sampling_blocks, strict=True)
                    ]
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    result = _plain_outputs(outputs)
                elif command == "sleep":
                    llm.sleep(level=int(message.get("level", 1)))
                    result = None
                elif command == "wake_up":
                    llm.wake_up(tags=message.get("tags"))
                    result = None
                elif command == "health":
                    result = True
                elif command == "init_native_weight_transfer":
                    init_info = message.get("init_info")
                    if not isinstance(init_info, dict):
                        raise ValueError("vLLM native IPC init requires init_info")
                    llm.init_weight_transfer_engine({"init_info": init_info})
                    result = None
                elif command == "start_native_weight_update":
                    header = message.get("header")
                    if not isinstance(header, dict):
                        raise ValueError("vLLM native IPC start requires a publication header")
                    if native_pending_header is not None:
                        raise RuntimeError("vLLM native IPC cannot replace an active publication")
                    from unirl.distributed.weight_sync.transfer.vllm_native_protocol import (
                        validate_publication_header,
                    )

                    validate_publication_header(
                        header,
                        tp_world_size=int(header.get("tp_world_size", 0)),
                    )
                    llm.start_weight_update()
                    native_pending_header = dict(header)
                    result = None
                elif command == "update_native_weights":
                    if native_pending_header is None:
                        raise RuntimeError("vLLM native IPC update requires start first")
                    update_info = message.get("update_info")
                    if not isinstance(update_info, dict):
                        raise ValueError("vLLM native IPC update requires update_info")
                    llm.update_weights({"update_info": update_info})
                    result = None
                elif command == "finish_native_weight_update":
                    header = message.get("header")
                    if not isinstance(header, dict) or native_pending_header != header:
                        raise RuntimeError("vLLM native IPC finish identifies the wrong publication")
                    llm.finish_weight_update()
                    worker_receipts = llm.collective_rpc(
                        "unirl_native_weight_sync_receipt",
                        kwargs={"header": header},
                    )
                    from unirl.distributed.weight_sync.transfer.vllm_native_protocol import (
                        validate_worker_receipts,
                    )

                    # Gate the worker-side version commit before replying across
                    # the process boundary. The parent validates the reply again.
                    validate_worker_receipts(
                        header,
                        worker_receipts,
                        fanout=int(header["tp_world_size"]),
                    )
                    weight_version = str(message.get("weight_version") or header["model_version"])
                    llm.update_weight_version(weight_version)
                    prefix_cache_reset = False
                    if message.get("flush_cache", True):
                        prefix_cache_reset = bool(llm.reset_prefix_cache(reset_running_requests=True))
                        if not prefix_cache_reset:
                            raise RuntimeError("vLLM native IPC prefix-cache reset failed")
                    result = {
                        "status": "committed",
                        "model_version": int(header["model_version"]),
                        "worker_receipts": worker_receipts,
                        "prefix_cache_reset": prefix_cache_reset,
                    }
                    native_pending_header = None
                elif command == "release_native_ipc":
                    llm.collective_rpc("unirl_native_ipc_collect")
                    result = None
                elif command == "shutdown":
                    result = None
                else:
                    raise ValueError(f"unknown vLLM command {command!r}")
            except BaseException as error:
                result = (
                    {
                        "status": "aborted",
                        "worker_receipts": worker_receipts,
                    }
                    if command
                    in {
                        "init_native_weight_transfer",
                        "start_native_weight_update",
                        "update_native_weights",
                        "finish_native_weight_update",
                        "release_native_ipc",
                    }
                    else None
                )
                _send_response(connection, message, ok=False, result=result, error=error)
                if command in {
                    "init_native_weight_transfer",
                    "start_native_weight_update",
                    "update_native_weights",
                    "finish_native_weight_update",
                    "release_native_ipc",
                }:
                    break
            else:
                _send_response(connection, message, ok=True, result=result)
                if command == "shutdown":
                    break
    finally:
        try:
            _shutdown_llm(llm)
            del llm
        finally:
            connection.close()


__all__ = ["engine_process_main"]
