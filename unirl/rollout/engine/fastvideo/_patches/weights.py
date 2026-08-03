"""Strict full-parameter checkpoint hot-swap for FastVideo workers."""

from __future__ import annotations

from typing import Any

import torch


def _strip_training_prefix(name: str) -> str:
    for prefix in ("base_model.model.", "module.", "transformer."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _worker_update_transformer_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    transformer = self.pipeline.modules.get("transformer")
    if transformer is None:
        raise RuntimeError("FastVideo worker has no transformer module")

    target_state = transformer.state_dict()
    target_device = next((parameter.device for parameter in transformer.parameters()), self.device)
    target_dtype = next((parameter.dtype for parameter in transformer.parameters()), None)
    stripped: dict[str, torch.Tensor] = {}
    for raw_name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        name = _strip_training_prefix(str(raw_name))
        if target_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=target_dtype)
        stripped[name] = tensor.to(target_device, non_blocking=True)

    if not stripped:
        raise RuntimeError("FastVideo weight update received no tensor entries")

    mapping_dict = getattr(transformer, "param_names_mapping", None)
    if mapping_dict:
        from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict

        mapping_fn = get_param_names_mapping(mapping_dict)
        mapped, _ = hf_to_custom_state_dict(stripped, mapping_fn)
    else:
        mapped = stripped

    target_names = set(target_state)
    mapped_names = set(mapped)
    matched = target_names & mapped_names
    coverage = len(matched) / max(1, len(target_names))
    unexpected_before_load = sorted(mapped_names - target_names)
    if coverage < 0.99 or unexpected_before_load:
        raise RuntimeError(
            "FastVideo transformer key mapping failed closed: "
            f"coverage={coverage:.2%}, matched={len(matched)}/{len(target_names)}, "
            f"unexpected={unexpected_before_load[:10]}"
        )

    missing, unexpected = transformer.load_state_dict(mapped, strict=False, assign=False)
    if missing or unexpected:
        raise RuntimeError(
            "FastVideo transformer load was incomplete: "
            f"missing={list(missing)[:10]}, unexpected={list(unexpected)[:10]}"
        )
    return {
        "status": "transformer_weights_updated",
        "loaded": len(matched),
        "target": len(target_names),
        "coverage": coverage,
    }


setattr(_worker_update_transformer_weights, "_unirl_fastvideo_weights", True)


def patch_weights() -> None:
    """Install additive full-weight APIs on worker, executor, and generator."""

    from fastvideo.entrypoints.video_generator import VideoGenerator
    from fastvideo.worker.gpu_worker import Worker
    from fastvideo.worker.multiproc_executor import MultiprocExecutor

    Worker.update_transformer_weights = _worker_update_transformer_weights

    def executor_update(self, state_dict):
        responses = self.collective_rpc("update_transformer_weights", kwargs={"state_dict": state_dict})
        if len(responses) != int(self.world_size):
            raise RuntimeError(
                f"FastVideo weight update returned {len(responses)} responses for world_size={self.world_size}"
            )
        for rank, response in enumerate(responses):
            if not isinstance(response, dict) or response.get("status") != "transformer_weights_updated":
                raise RuntimeError(f"FastVideo worker {rank} weight update failed: {response!r}")
            if float(response.get("coverage", 0.0)) < 0.99:
                raise RuntimeError(f"FastVideo worker {rank} reported incomplete weight coverage: {response!r}")

    setattr(executor_update, "_unirl_fastvideo_weights", True)
    MultiprocExecutor.update_transformer_weights = executor_update

    def update_from_path(self, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(state_dict, dict):
            raise TypeError(f"FastVideo checkpoint must contain a state-dict mapping; got {type(state_dict).__name__}")
        self.executor.update_transformer_weights(state_dict)

    setattr(update_from_path, "_unirl_fastvideo_weights", True)
    VideoGenerator.update_transformer_weights_from_path = update_from_path


__all__ = ["patch_weights"]
