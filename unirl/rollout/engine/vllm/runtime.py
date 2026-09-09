"""Spawned vLLM runtime used by the direct rollout engine."""

from __future__ import annotations

import os
import traceback
from multiprocessing.connection import Connection
from typing import Any, Dict, List


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
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        os.environ["UNIRL_ROLLOUT_DP_RANK"] = str(int(config.get("rollout_rank", 0)))
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from vllm import LLM

        engine_kwargs = dict(config.get("engine_kwargs") or {})
        engine_kwargs.setdefault("dtype", "bfloat16")
        engine_kwargs.setdefault("trust_remote_code", True)
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
        print(
            "[unirl.vllm.runtime] "
            f"rollout_dp_rank={os.environ['UNIRL_ROLLOUT_DP_RANK']} "
            f"visible_devices={visible_devices}",
            flush=True,
        )
        connection.send({"ok": True, "event": "ready"})
    except BaseException as error:
        connection.send(
            {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        )
        connection.close()
        return

    try:
        while True:
            message = connection.recv()
            command = message.get("command")
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
                    result = llm.collective_rpc(
                        "unirl_update_weights_from_tensor",
                        kwargs={
                            "serialized_named_tensors": message["serialized_named_tensors"],
                            "load_format": message.get("load_format"),
                        },
                    )
                    if message.get("flush_cache", True):
                        llm.reset_prefix_cache(reset_running_requests=True)
                elif command == "shutdown":
                    connection.send({"ok": True, "result": None})
                    break
                else:
                    raise ValueError(f"unknown direct-vLLM command {command!r}")
            except BaseException as error:
                connection.send(
                    {
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
            else:
                connection.send({"ok": True, "result": result})
    finally:
        try:
            del llm
        finally:
            connection.close()


__all__ = ["engine_process_main"]
