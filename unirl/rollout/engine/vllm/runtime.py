"""Spawned vLLM runtime used by the direct rollout engine."""

from __future__ import annotations

import os
import traceback
from multiprocessing.connection import Connection
from typing import Any, Dict, List

_PROTOCOL_VERSION = 1


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
    startup_request: Dict[str, Any] | None = None
    try:
        startup_request = _recv_request(connection, last_request_id=0)
        if startup_request["command"] != "startup":
            raise _ProtocolError(f"first direct-vLLM command must be 'startup', got {startup_request['command']!r}")

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        os.environ["UNIRL_ROLLOUT_DP_RANK"] = str(int(config.get("rollout_rank", 0)))
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from vllm import LLM

        engine_kwargs = dict(config.get("engine_kwargs") or {})
        engine_kwargs.setdefault("dtype", "bfloat16")
        engine_kwargs["trust_remote_code"] = bool(config.get("trust_remote_code", False))
        if config.get("model_revision"):
            engine_kwargs.setdefault("revision", str(config["model_revision"]))
            engine_kwargs.setdefault("tokenizer_revision", str(config["model_revision"]))
        engine_kwargs.setdefault("distributed_executor_backend", "mp")
        engine_kwargs.setdefault("enable_sleep_mode", True)
        engine_kwargs.setdefault("enforce_eager", True)
        engine_kwargs.setdefault("enable_prefix_caching", False)
        engine_kwargs.setdefault("enable_chunked_prefill", False)
        engine_kwargs.setdefault("moe_backend", "triton")
        engine_kwargs.setdefault("logprobs_mode", "processed_logprobs")
        engine_kwargs.setdefault(
            "worker_extension_cls",
            "unirl.rollout.engine.vllm.worker_extension.UniRLWeightSyncExtension",
        )
        llm = LLM(
            model=str(config["pretrained_model_ckpt_path"]),
            tensor_parallel_size=int(config["tp_size"]),
            **engine_kwargs,
        )
        plugin_manifest = None
        if os.environ.get("UNIRL_PARITY_ENABLE") == "1":
            from unirl_train_inference_parity_vllm import runtime_manifest

            plugin_manifest = runtime_manifest()
            if not plugin_manifest:
                raise RuntimeError("UNIRL parity was enabled but the vLLM plugin did not install a runtime manifest")
        print(
            "[unirl.vllm.runtime] "
            f"rollout_dp_rank={os.environ['UNIRL_ROLLOUT_DP_RANK']} "
            f"visible_devices={visible_devices}",
            flush=True,
        )
        _send_response(
            connection,
            startup_request,
            ok=True,
            result={"event": "ready", "plugin_manifest": plugin_manifest},
        )
    except BaseException as error:
        if startup_request is not None:
            try:
                _send_response(connection, startup_request, ok=False, result=None, error=error)
            except (EOFError, BrokenPipeError, OSError):
                pass
        connection.close()
        return

    last_request_id = int(startup_request["request_id"])
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
                    if sampling_blocks and any(block != sampling_blocks[0] for block in sampling_blocks[1:]):
                        raise ValueError("one vLLM batch requires identical sampling parameters")
                    prompts = [
                        {"prompt_token_ids": [int(token) for token in payload["input_ids"]]} for payload in payloads
                    ]
                    params = _sampling_params(
                        sampling_blocks[0] if sampling_blocks else {},
                        return_logprob=bool(payloads and payloads[0].get("return_logprob", True)),
                    )
                    outputs = llm.generate(prompts, params, use_tqdm=False)
                    result = _plain_outputs(outputs)
                elif command == "sleep":
                    llm.collective_rpc("unirl_before_sleep")
                    llm.sleep(level=int(message.get("level", 1)))
                    result = None
                elif command == "wake_up":
                    llm.wake_up(tags=message.get("tags"))
                    result = None
                elif command == "health":
                    result = True
                elif command == "update_weights":
                    header = message.get("header")
                    payloads = list(message.get("payloads") or message.get("serialized_named_tensors") or ())
                    tp_world_size = int(message.get("tp_world_size") or len(payloads))
                    worker_receipts = llm.collective_rpc(
                        "unirl_update_weights_from_tensor",
                        kwargs={
                            "serialized_named_tensors": payloads,
                            "payloads": payloads,
                            "header": header,
                            "tp_world_size": tp_world_size,
                            "load_format": message.get("load_format"),
                        },
                    )
                    if header is not None:
                        from unirl.distributed.weight_sync.transfer.fsdp_vllm_protocol import (
                            validate_worker_receipts,
                        )

                        validate_worker_receipts(
                            header,
                            worker_receipts,
                            fanout=tp_world_size,
                        )
                    is_last = header is None or bool(header.get("is_last"))
                    commit_receipts = []
                    if is_last and header is not None:
                        commit_receipts = llm.collective_rpc(
                            "unirl_commit_weight_version",
                            kwargs={
                                "sync_id": str(header["sync_id"]),
                                "model_version": int(header["model_version"]),
                            },
                        )
                    result = {
                        "status": "committed" if is_last else "staged",
                        "worker_receipts": worker_receipts,
                        "commit_receipts": commit_receipts,
                        "committed": is_last,
                    }
                    if header is not None:
                        from unirl.distributed.weight_sync.transfer.fsdp_vllm_protocol import (
                            validate_receipts,
                        )

                        validate_receipts(header, result, fanout=tp_world_size)
                    prefix_cache_reset = False
                    if is_last and message.get("flush_cache", True):
                        llm.reset_prefix_cache(reset_running_requests=True)
                        prefix_cache_reset = True
                    result["prefix_cache_reset"] = prefix_cache_reset
                elif command == "shutdown":
                    result = None
                else:
                    raise ValueError(f"unknown direct-vLLM command {command!r}")
            except BaseException as error:
                result = (
                    {
                        "status": "aborted",
                        "worker_receipts": worker_receipts,
                    }
                    if command == "update_weights"
                    else None
                )
                _send_response(connection, message, ok=False, result=result, error=error)
                if command == "update_weights":
                    break
            else:
                _send_response(connection, message, ok=True, result=result)
                if command == "shutdown":
                    break
    finally:
        try:
            del llm
        finally:
            connection.close()


__all__ = ["engine_process_main"]
