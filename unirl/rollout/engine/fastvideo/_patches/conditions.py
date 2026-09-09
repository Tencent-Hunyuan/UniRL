"""Carry RL trajectories and text conditions back across FastVideo worker pipes; contract in README.md."""

from __future__ import annotations

from copy import copy
from dataclasses import fields, is_dataclass
from functools import wraps
from typing import Any

import torch


def _to_cpu(value: Any) -> Any:
    """Deep-copy a worker payload onto CPU so no CUDA tensor crosses the IPC boundary."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        result = copy(value)
        for field in fields(value):
            object.__setattr__(result, field.name, _to_cpu(getattr(value, field.name)))
        return result
    return value


_CARRIED = (
    "rl_data",
    "trajectory_latents",
    "trajectory_timesteps",
    "prompt_embeds",
    "negative_prompt_embeds",
    "prompt_attention_mask",
    "negative_attention_mask",
)


class _ForwardResultPipe:
    """Connection proxy that attaches the RL fields upstream's response drops."""

    def __init__(self, connection: Any, owner: Any) -> None:
        self._connection = connection
        self._owner = owner

    def send(self, payload: Any) -> None:
        worker = getattr(self._owner, "worker", None)
        inner = getattr(worker, "worker", worker)
        cached = getattr(inner, "_unirl_last_forward_batch", None) if inner is not None else None
        if isinstance(payload, dict) and "output_batch" in payload and cached is not None:
            for name in _CARRIED:
                payload.setdefault(name, _to_cpu(getattr(cached, name, None)))
            inner._unirl_last_forward_batch = None
        self._connection.send(_to_cpu(payload))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def patch_conditions() -> None:
    """Round-trip RLData and text conditions; upstream's executor rebuilds a bare ForwardBatch."""
    import fastvideo.envs as envs
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.worker.gpu_worker import Worker
    from fastvideo.worker.multiproc_executor import MultiprocExecutor, WorkerMultiprocProc

    # ``WorkerWrapperBase`` has no own ``execute_forward``; it proxies through
    # ``__getattr__``, and the busy loop calls ``self.worker.execute_forward``.
    if not getattr(Worker, "_unirl_fastvideo_forward_cache", False):
        original_execute = Worker.execute_forward

        @wraps(original_execute)
        def execute_forward(self, forward_batch, fastvideo_args):
            output_batch = original_execute(self, forward_batch, fastvideo_args)
            self._unirl_last_forward_batch = output_batch
            return output_batch

        Worker.execute_forward = execute_forward
        Worker._unirl_fastvideo_forward_cache = True

    original_init = WorkerMultiprocProc.__init__
    if not getattr(original_init, "_unirl_fastvideo_conditions", False):

        @wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not isinstance(self.pipe, _ForwardResultPipe):
                self.pipe = _ForwardResultPipe(self.pipe, self)

        init._unirl_fastvideo_conditions = True
        WorkerMultiprocProc.__init__ = init

    if getattr(MultiprocExecutor.execute_forward, "_unirl_fastvideo_conditions", False):
        return

    def execute_forward(self, forward_batch, fastvideo_args):
        responses = self.collective_rpc(
            "execute_forward",
            kwargs={"forward_batch": forward_batch, "fastvideo_args": fastvideo_args},
        )
        if not responses or not isinstance(responses[0], dict):
            raise RuntimeError(f"FastVideo execute_forward returned invalid worker responses: {responses!r}")
        response = responses[0]
        kwargs: dict[str, Any] = {
            "data_type": forward_batch.data_type,
            "output": response.get("output_batch"),
            "logging_info": response.get("logging_info") if envs.FASTVIDEO_STAGE_LOGGING else None,
            "extra": response.get("extra", {}),
            "trajectory_latents": response.get("trajectory_latents"),
            "trajectory_timesteps": response.get("trajectory_timesteps"),
        }
        if response.get("rl_data") is not None:
            kwargs["rl_data"] = response["rl_data"]
        result = ForwardBatch(**kwargs)
        result.prompt_embeds = response.get("prompt_embeds") or []
        result.negative_prompt_embeds = response.get("negative_prompt_embeds")
        result.prompt_attention_mask = response.get("prompt_attention_mask")
        result.negative_attention_mask = response.get("negative_attention_mask")
        return result

    execute_forward._unirl_fastvideo_conditions = True
    MultiprocExecutor.execute_forward = execute_forward


__all__ = ["patch_conditions"]
