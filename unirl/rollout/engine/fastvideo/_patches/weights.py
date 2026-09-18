"""Strict full-parameter checkpoint hot-swap for FastVideo workers; contract in README.md."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

_MIN_COVERAGE = 0.99
_TRAINING_PREFIXES = ("base_model.model.", "module.", "transformer.")


def _strip_training_prefix(name: str) -> str:
    """Drop the wrapper prefixes FSDP/peft add around the rollout-facing parameter names."""
    for prefix in _TRAINING_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name


def _offloaded_blocks(module: Any) -> List[Tuple[str, Any, Any]]:
    """Return ``(prefix, child, hook)`` for every layerwise-offloaded block, outermost first."""
    try:
        from fastvideo.hooks.hooks import ModuleHookManager
    except ImportError:
        return []

    blocks: List[Tuple[str, Any, Any]] = []
    for module_name, child in module.named_modules():
        manager = ModuleHookManager.get_from(child)
        if manager is None:
            continue
        hook = manager.get_forward_hook("LayerwiseOffloadHook")
        if hook is None:
            continue
        blocks.append((f"{module_name}." if module_name else "", child, hook))
    return blocks


def _copy_into(targets: Dict[str, torch.Tensor], mapped: Dict[str, torch.Tensor], *, label: str) -> set[str]:
    """Copy ``mapped`` into the given already-materialized target tensors."""
    loaded: set[str] = set()
    for name, target in targets.items():
        source = mapped[name]
        if tuple(source.shape) != tuple(target.shape):
            raise RuntimeError(
                f"FastVideo {label} shape mismatch for {name}: "
                f"source={tuple(source.shape)} target={tuple(target.shape)}"
            )
        if source.is_floating_point():
            source = source.to(dtype=target.dtype)
        target.copy_(source.to(device=target.device), non_blocking=False)
        loaded.add(name)
    return loaded


def _load_transformer_state(module: Any, state_dict: Dict[str, torch.Tensor], *, label: str) -> Dict[str, Any]:
    """Copy a full state dict into one FastVideo transformer, failing closed on any key or shape drift."""
    target_state = module.state_dict()
    prepared = {str(name): tensor for name, tensor in state_dict.items() if torch.is_tensor(tensor)}
    if not prepared:
        raise RuntimeError(f"FastVideo weight update received no tensor entries for {label}")

    mapping_dict = getattr(module, "param_names_mapping", None)
    if mapping_dict:
        from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict

        mapped, _ = hf_to_custom_state_dict(prepared, get_param_names_mapping(mapping_dict))
    else:
        mapped = prepared

    target_names = set(target_state)
    mapped_names = set(mapped)
    matched = target_names & mapped_names
    coverage = len(matched) / max(1, len(target_names))
    unexpected = sorted(mapped_names - target_names)
    if coverage < _MIN_COVERAGE or unexpected:
        raise RuntimeError(
            f"FastVideo {label} key mapping failed closed: coverage={coverage:.2%}, "
            f"matched={len(matched)}/{len(target_names)}, unexpected={unexpected[:10]}"
        )

    loaded: set[str] = set()
    with torch.no_grad():
        offloaded = _offloaded_blocks(module)
        claimed: set[str] = set()
        for prefix, child, hook in offloaded:
            names = {f"{prefix}{n}" for n, _ in child.named_parameters()} & matched
            claimed |= names
            if not names:
                continue
            # One block at a time: ``mutate_params_scope`` pulls it onto GPU, then re-seeds
            # both caches from the mutated params on exit. Holding every block's scope open
            # at once would stage the whole expert on GPU, and writing
            # ``state.cpu_named_parameters`` directly would leave the copy prefetched by the
            # previous forward stale (README: layerwise offload).
            with hook.mutate_params_scope():
                targets = {f"{prefix}{n}": p.data for n, p in child.named_parameters() if f"{prefix}{n}" in names}
                loaded |= _copy_into(targets, mapped, label=label)
        rest = {name: target_state[name] for name in matched - claimed}
        loaded |= _copy_into(rest, mapped, label=label)

    missing = sorted(target_names - loaded)
    if missing:
        raise RuntimeError(f"FastVideo {label} load was incomplete: missing={missing[:10]}")
    return {"label": label, "loaded": len(matched), "target": len(target_names), "coverage": coverage}


def _split_experts(stripped: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a WAN 2.2 checkpoint into its boundary-routed high-noise and low-noise experts."""
    high = {name[len("high_noise.") :]: t for name, t in stripped.items() if name.startswith("high_noise.")}
    low = {name[len("low_noise.") :]: t for name, t in stripped.items() if name.startswith("low_noise.")}
    unrelated = sorted(
        set(stripped) - {f"high_noise.{n}" for n in high} - {f"low_noise.{n}" for n in low},
    )
    if not high or not low or unrelated:
        raise RuntimeError(
            "FastVideo dual-transformer update requires only high_noise.* and low_noise.* checkpoint keys: "
            f"high={len(high)}, low={len(low)}, unrelated={unrelated[:10]}"
        )
    return high, low


def _worker_update_transformer_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    transformer = self.pipeline.modules.get("transformer")
    if transformer is None:
        raise RuntimeError("FastVideo worker has no transformer module")

    stripped = {_strip_training_prefix(str(n)): t for n, t in state_dict.items() if torch.is_tensor(t)}
    transformer_2 = self.pipeline.modules.get("transformer_2")
    if transformer_2 is None:
        result = _load_transformer_state(transformer, stripped, label="transformer")
        return {"status": "transformer_weights_updated", **result}

    high, low = _split_experts(stripped)
    results = [
        _load_transformer_state(transformer, high, label="transformer/high_noise"),
        _load_transformer_state(transformer_2, low, label="transformer_2/low_noise"),
    ]
    return {
        "status": "transformer_weights_updated",
        "loaded": sum(int(r["loaded"]) for r in results),
        "target": sum(int(r["target"]) for r in results),
        "coverage": min(float(r["coverage"]) for r in results),
        "modules": results,
    }


def _worker_update_transformer_weights_from_path(self, checkpoint_path: str) -> Dict[str, Any]:
    # CheckpointWeightSync writes a plain ``torch.save`` tensor dict; refuse arbitrary pickles.
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"FastVideo checkpoint must contain a state-dict mapping; got {type(state_dict).__name__}")
    try:
        return _worker_update_transformer_weights(self, state_dict)
    finally:
        del state_dict


def _check_responses(responses: Any, world_size: int) -> None:
    """Fail closed unless every worker reported a complete load."""
    if len(responses) != int(world_size):
        raise RuntimeError(f"FastVideo weight update returned {len(responses)} responses for world_size={world_size}")
    for rank, response in enumerate(responses):
        if not isinstance(response, dict) or response.get("status") != "transformer_weights_updated":
            raise RuntimeError(f"FastVideo worker {rank} weight update failed: {response!r}")
        if float(response.get("coverage", 0.0)) < _MIN_COVERAGE:
            raise RuntimeError(f"FastVideo worker {rank} reported incomplete weight coverage: {response!r}")


def patch_weights() -> None:
    """Install the additive full-weight update API upstream does not provide."""
    from fastvideo.entrypoints.video_generator import VideoGenerator
    from fastvideo.worker.gpu_worker import Worker
    from fastvideo.worker.multiproc_executor import MultiprocExecutor

    if getattr(VideoGenerator, "_unirl_fastvideo_weights", False):
        return

    Worker.update_transformer_weights = _worker_update_transformer_weights
    Worker.update_transformer_weights_from_path = _worker_update_transformer_weights_from_path

    def executor_update_from_path(self, checkpoint_path: str) -> None:
        responses = self.collective_rpc(
            "update_transformer_weights_from_path",
            kwargs={"checkpoint_path": checkpoint_path},
        )
        _check_responses(responses, self.world_size)

    MultiprocExecutor.update_transformer_weights_from_path = executor_update_from_path

    def update_from_path(self, checkpoint_path: str) -> None:
        self.executor.update_transformer_weights_from_path(checkpoint_path)

    VideoGenerator.update_transformer_weights_from_path = update_from_path
    VideoGenerator._unirl_fastvideo_weights = True


__all__ = ["patch_weights"]
