"""Carry RL trajectories and exact text conditions across FastVideo worker pipes."""

from __future__ import annotations

from copy import copy
from dataclasses import fields, is_dataclass
from functools import wraps
from typing import Any

import torch


def _cpu_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _cpu_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_value(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu()
    if is_dataclass(value) and not isinstance(value, type):
        result = copy(value)
        for field in fields(value):
            setattr(result, field.name, _cpu_value(getattr(value, field.name)))
        return result
    return value


class _ForwardResultPipe:
    """Connection proxy that enriches only ``execute_forward`` responses."""

    def __init__(self, connection: Any, owner: Any) -> None:
        self._connection = connection
        self._owner = owner

    def send(self, payload: Any) -> None:
        output_batch = getattr(self._owner.worker, "_unirl_last_forward_batch", None)
        if isinstance(payload, dict) and "output_batch" in payload and output_batch is not None:
            payload.setdefault("rl_data", _cpu_value(getattr(output_batch, "rl_data", None)))
            payload.setdefault("trajectory_latents", _cpu_value(getattr(output_batch, "trajectory_latents", None)))
            payload.setdefault("trajectory_timesteps", _cpu_value(getattr(output_batch, "trajectory_timesteps", None)))
            payload.setdefault("prompt_embeds", _cpu_value(getattr(output_batch, "prompt_embeds", None)))
            payload.setdefault(
                "negative_prompt_embeds", _cpu_value(getattr(output_batch, "negative_prompt_embeds", None))
            )
            payload.setdefault(
                "prompt_attention_mask", _cpu_value(getattr(output_batch, "prompt_attention_mask", None))
            )
            payload.setdefault(
                "negative_attention_mask", _cpu_value(getattr(output_batch, "negative_attention_mask", None))
            )
            self._owner.worker._unirl_last_forward_batch = None
        self._connection.send(payload)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def patch_conditions() -> None:
    """Patch the narrow worker and driver seams without replacing busy loops."""

    import fastvideo.envs as envs
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.worker.multiproc_executor import MultiprocExecutor, WorkerMultiprocProc
    from fastvideo.worker.worker_base import WorkerWrapperBase

    if not getattr(WorkerWrapperBase, "_unirl_fastvideo_forward_cache", False):

        def execute_forward(self, forward_batch, fastvideo_args):
            if self.worker is None:
                raise RuntimeError("FastVideo worker is not initialized")
            output_batch = self.worker.execute_forward(forward_batch, fastvideo_args)
            self._unirl_last_forward_batch = output_batch
            return output_batch

        WorkerWrapperBase.execute_forward = execute_forward
        setattr(WorkerWrapperBase, "_unirl_fastvideo_forward_cache", True)

    original_init = WorkerMultiprocProc.__init__
    if not getattr(original_init, "_unirl_fastvideo_conditions", False):

        @wraps(original_init)
        def init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not isinstance(self.pipe, _ForwardResultPipe):
                self.pipe = _ForwardResultPipe(self.pipe, self)

        setattr(init, "_unirl_fastvideo_conditions", True)
        WorkerMultiprocProc.__init__ = init

    current_execute = MultiprocExecutor.execute_forward
    if getattr(current_execute, "_unirl_fastvideo_conditions", False):
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

    setattr(execute_forward, "_unirl_fastvideo_conditions", True)
    MultiprocExecutor.execute_forward = execute_forward


__all__ = ["patch_conditions"]
