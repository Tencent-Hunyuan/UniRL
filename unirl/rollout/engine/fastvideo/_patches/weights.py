"""Strict full-parameter checkpoint hot-swap for FastVideo workers."""

from __future__ import annotations

from typing import Any

import torch


def _strip_training_prefix(name: str) -> str:
    for prefix in ("base_model.model.", "module.", "transformer."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _load_transformer_state(
    module: Any,
    state_dict: dict[str, torch.Tensor],
    *,
    label: str,
) -> dict[str, Any]:
    target_state = module.state_dict()
    prepared: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        prepared[str(name)] = tensor.detach().cpu()

    if not prepared:
        raise RuntimeError(f"FastVideo weight update received no tensor entries for {label}")

    mapping_dict = getattr(module, "param_names_mapping", None)
    if mapping_dict:
        from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict

        mapping_fn = get_param_names_mapping(mapping_dict)
        mapped, _ = hf_to_custom_state_dict(prepared, mapping_fn)
    else:
        mapped = prepared

    target_names = set(target_state)
    mapped_names = set(mapped)
    matched = target_names & mapped_names
    coverage = len(matched) / max(1, len(target_names))
    unexpected_before_load = sorted(mapped_names - target_names)
    if coverage < 0.99 or unexpected_before_load:
        raise RuntimeError(
            f"FastVideo {label} key mapping failed closed: "
            f"coverage={coverage:.2%}, matched={len(matched)}/{len(target_names)}, "
            f"unexpected={unexpected_before_load[:10]}"
        )

    layerwise: dict[str, torch.Tensor] = {}
    try:
        from fastvideo.hooks.hooks import ModuleHookManager

        for module_name, child in module.named_modules():
            manager = ModuleHookManager.get_from(child)
            hook = manager.get_forward_hook("LayerwiseOffloadHook") if manager is not None else None
            if hook is None:
                continue
            prefix = f"{module_name}." if module_name else ""
            for parameter_name, cpu_tensor in hook.state.cpu_named_parameters.items():
                layerwise[f"{prefix}{parameter_name}"] = cpu_tensor
    except ImportError:
        pass

    loaded: set[str] = set()
    with torch.no_grad():
        for name in matched:
            source = mapped[name]
            target = layerwise.get(name, target_state[name])
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"FastVideo {label} shape mismatch for {name}: source={tuple(source.shape)} "
                    f"target={tuple(target.shape)}"
                )
            if source.is_floating_point():
                source = source.to(dtype=target.dtype)
            target.copy_(source.to(device=target.device), non_blocking=False)
            loaded.add(name)

    missing = sorted(target_names - loaded)
    unexpected = sorted(mapped_names - target_names)
    if missing or unexpected:
        raise RuntimeError(
            f"FastVideo {label} load was incomplete: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return {
        "label": label,
        "loaded": len(matched),
        "target": len(target_names),
        "coverage": coverage,
    }


def _worker_update_transformer_weights(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    transformer = self.pipeline.modules.get("transformer")
    if transformer is None:
        raise RuntimeError("FastVideo worker has no transformer module")

    stripped = {
        _strip_training_prefix(str(raw_name)): tensor
        for raw_name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
    }
    transformer_2 = self.pipeline.modules.get("transformer_2")
    if transformer_2 is None:
        result = _load_transformer_state(
            transformer,
            stripped,
            label="transformer",
        )
        return {"status": "transformer_weights_updated", **result}

    high_noise = {
        name[len("high_noise.") :]: tensor for name, tensor in stripped.items() if name.startswith("high_noise.")
    }
    low_noise = {
        name[len("low_noise.") :]: tensor for name, tensor in stripped.items() if name.startswith("low_noise.")
    }
    unrelated = sorted(
        set(stripped) - {f"high_noise.{name}" for name in high_noise} - {f"low_noise.{name}" for name in low_noise}
    )
    if not high_noise or not low_noise or unrelated:
        raise RuntimeError(
            "FastVideo dual-transformer update requires only high_noise.* and low_noise.* checkpoint keys: "
            f"high={len(high_noise)}, low={len(low_noise)}, unrelated={unrelated[:10]}"
        )

    results = [
        _load_transformer_state(
            transformer,
            high_noise,
            label="transformer/high_noise",
        ),
        _load_transformer_state(
            transformer_2,
            low_noise,
            label="transformer_2/low_noise",
        ),
    ]
    return {
        "status": "transformer_weights_updated",
        "loaded": sum(int(result["loaded"]) for result in results),
        "target": sum(int(result["target"]) for result in results),
        "coverage": min(float(result["coverage"]) for result in results),
        "modules": results,
    }


setattr(_worker_update_transformer_weights, "_unirl_fastvideo_weights", True)


def _worker_update_transformer_weights_from_path(self, checkpoint_path: str) -> dict[str, Any]:
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise TypeError(f"FastVideo checkpoint must contain a state-dict mapping; got {type(state_dict).__name__}")
    return _worker_update_transformer_weights(self, state_dict)


setattr(_worker_update_transformer_weights_from_path, "_unirl_fastvideo_weights", True)


def patch_weights() -> None:
    """Install additive full-weight APIs on worker, executor, and generator."""

    from fastvideo.entrypoints.video_generator import VideoGenerator
    from fastvideo.worker.gpu_worker import Worker
    from fastvideo.worker.multiproc_executor import MultiprocExecutor

    Worker.update_transformer_weights = _worker_update_transformer_weights
    Worker.update_transformer_weights_from_path = _worker_update_transformer_weights_from_path

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

    def executor_update_from_path(self, checkpoint_path: str) -> None:
        responses = self.collective_rpc(
            "update_transformer_weights_from_path",
            kwargs={"checkpoint_path": checkpoint_path},
        )
        if len(responses) != int(self.world_size):
            raise RuntimeError(
                f"FastVideo weight update returned {len(responses)} responses for world_size={self.world_size}"
            )
        for rank, response in enumerate(responses):
            if not isinstance(response, dict) or response.get("status") != "transformer_weights_updated":
                raise RuntimeError(f"FastVideo worker {rank} weight update failed: {response!r}")
            if float(response.get("coverage", 0.0)) < 0.99:
                raise RuntimeError(f"FastVideo worker {rank} reported incomplete weight coverage: {response!r}")

    setattr(executor_update_from_path, "_unirl_fastvideo_weights", True)
    MultiprocExecutor.update_transformer_weights_from_path = executor_update_from_path

    def update_from_path(self, checkpoint_path: str) -> None:
        self.executor.update_transformer_weights_from_path(checkpoint_path)

    setattr(update_from_path, "_unirl_fastvideo_weights", True)
    VideoGenerator.update_transformer_weights_from_path = update_from_path


__all__ = ["patch_weights"]
