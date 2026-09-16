"""Fail-closed fingerprints of the stock upstream seams every FastVideo patch installs onto."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import os
from typing import Any, Tuple

# The exact upstream surface these patches target; drift fails closed at patch time (README: pin).
PINNED_FORK = "hao-ai-lab/FastVideo@2095477eac7e289c7a7ab13acb367ca60687c304"

SET_TIMESTEPS_PARAMS = (
    "self",
    "num_inference_steps",
    "device",
    "sigmas",
    "mu",
    "shift",
    "use_karras_sigmas",
    "use_kerras_sigma",
)
# Stock upstream stops at ``return_dt_and_std_dev_t``; ``eta``/``sde_type`` are appended by
# ``denoising.patch_denoising``, which must therefore run before this fingerprint is taken.
SDE_STEP_PARAMS = (
    "scheduler",
    "model_output",
    "timestep",
    "sample",
    "prev_sample",
    "generator",
    "deterministic",
    "return_pixel_log_prob",
    "return_dt_and_std_dev_t",
    "eta",
    "sde_type",
)
_STOCK_SDE_STEP_PARAMS = SDE_STEP_PARAMS[:-2]
_COLLECTIVE_RPC_PARAMS = ("self", "method", "timeout", "args", "kwargs")
_EXECUTE_FORWARD_PARAMS = ("self", "forward_batch", "fastvideo_args")
_LOAD_STATE_DICT_PARAMS = (
    "model",
    "full_sd_iterator",
    "device",
    "param_dtype",
    "strict",
    "cpu_offload",
    "param_names_mapping",
    "training_mode",
)
_LOAD_MODULE_PARAMS = ("module_name", "component_model_path", "transformers_or_diffusers", "fastvideo_args")
_RL_DATA_FIELDS = frozenset(
    {
        "enabled",
        "collect_log_probs",
        "store_trajectory",
        "keep_trajectory_on_cpu",
        "sde_step_indices",
        "sde_type",
        "log_probs",
        "trajectory_latents",
        "trajectory_timesteps",
    }
)


def import_fastvideo_module(module_name: str, what: str) -> Any:
    """Import a fastvideo module, distinguishing a missing integration surface from unrelated import errors."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not module_name.startswith(f"{exc.name}.") and exc.name != module_name:
            raise
        raise RuntimeError(f"FastVideo UniPC integration requires {what} (pinned surface: {PINNED_FORK})") from exc


def require_attr(owner: Any, name: str, what: str) -> Any:
    """Fetch ``owner.name`` or fail closed naming the pinned integration surface."""
    value = getattr(owner, name, None)
    if value is None:
        raise RuntimeError(f"FastVideo UniPC integration requires {what} (pinned surface: {PINNED_FORK})")
    return value


def require_signature(fn: Any, expected: Tuple[str, ...], what: str) -> None:
    """Fingerprint a patched callable's parameter list so fork drift fails at patch time, not mid-rollout."""
    actual = tuple(inspect.signature(fn).parameters)
    if actual != expected:
        raise RuntimeError(
            f"FastVideo {what} drifted from the pinned integration surface ({PINNED_FORK}): "
            f"expected parameters {expected}, got {actual}"
        )


def verify_rl_data_surface() -> None:
    """Fail closed unless ``ForwardBatch.RLData`` carries every field the engine-side integration relies on."""
    module = import_fastvideo_module("fastvideo.pipelines.pipeline_batch_info", "ForwardBatch.RLData")
    forward_batch = require_attr(module, "ForwardBatch", "ForwardBatch.RLData")
    rl_data = require_attr(forward_batch, "RLData", "ForwardBatch.RLData")
    names = {f.name for f in dataclasses.fields(rl_data)}
    missing = sorted(_RL_DATA_FIELDS - names)
    if missing:
        raise RuntimeError(
            f"FastVideo ForwardBatch.RLData lacks fields {missing} required by the UniPC "
            f"integration (pinned surface: {PINNED_FORK}); contracts.patch_contracts must run first"
        )


def verify_stock_surface() -> None:
    """Fingerprint the untouched upstream seams before any UniRL patch rewrites them."""
    denoising = import_fastvideo_module(
        "fastvideo.pipelines.stages.denoising", "pipelines.stages.denoising.sde_step_with_logprob"
    )
    original = require_attr(denoising, "sde_step_with_logprob", "sde_step_with_logprob")
    if getattr(original, "_unirl_fastvideo_sde", False):
        return
    require_signature(original, _STOCK_SDE_STEP_PARAMS, "stock sde_step_with_logprob")
    stage = require_attr(denoising, "DenoisingStage", "DenoisingStage")
    forward = require_attr(stage, "forward", "DenoisingStage.forward")
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"cannot inspect FastVideo DenoisingStage.forward: {exc}") from exc
    for marker in ("rl_data", "sde_step_with_logprob", "scheduler.step"):
        if marker not in source:
            raise RuntimeError(
                f"FastVideo DenoisingStage.forward lacks required source marker {marker!r} "
                f"(pinned surface: {PINNED_FORK})"
            )


def verify_weight_surface() -> None:
    """Fail closed unless the worker/executor/generator seams the weight patch installs onto still exist."""
    worker = require_attr(import_fastvideo_module("fastvideo.worker.gpu_worker", "Worker"), "Worker", "Worker")
    require_attr(worker, "execute_forward", "Worker.execute_forward")
    executor = require_attr(
        import_fastvideo_module("fastvideo.worker.multiproc_executor", "MultiprocExecutor"),
        "MultiprocExecutor",
        "MultiprocExecutor",
    )
    require_signature(
        require_attr(executor, "collective_rpc", "MultiprocExecutor.collective_rpc"),
        _COLLECTIVE_RPC_PARAMS,
        "MultiprocExecutor.collective_rpc",
    )
    hooks = import_fastvideo_module("fastvideo.hooks.hooks", "ModuleHookManager")
    manager = require_attr(hooks, "ModuleHookManager", "ModuleHookManager")
    for name in ("get_from", "get_forward_hook"):
        require_attr(manager, name, f"ModuleHookManager.{name}")
    offload = import_fastvideo_module("fastvideo.hooks.layerwise_offload", "LayerwiseOffloadHook")
    require_attr(
        require_attr(offload, "LayerwiseOffloadHook", "LayerwiseOffloadHook"),
        "mutate_params_scope",
        "LayerwiseOffloadHook.mutate_params_scope",
    )
    # The response patch owns these seams; upstream's execute_forward rebuilds a bare
    # ForwardBatch that drops rl_data, the trajectory, and the prompt embeddings.
    require_signature(
        require_attr(executor, "execute_forward", "MultiprocExecutor.execute_forward"),
        _EXECUTE_FORWARD_PARAMS,
        "MultiprocExecutor.execute_forward",
    )
    worker_proc = require_attr(
        import_fastvideo_module("fastvideo.worker.multiproc_executor", "WorkerMultiprocProc"),
        "WorkerMultiprocProc",
        "WorkerMultiprocProc",
    )
    require_attr(worker_proc, "__init__", "WorkerMultiprocProc.__init__")
    require_attr(worker, "execute_forward", "Worker.execute_forward")


def verify_offload_surface() -> None:
    """Fail closed unless the loader seams the offload patch wraps keep their patched-for signatures."""
    fsdp_load = import_fastvideo_module("fastvideo.models.loader.fsdp_load", "fsdp_load")
    state_load = require_attr(
        fsdp_load, "load_model_from_full_model_state_dict", "load_model_from_full_model_state_dict"
    )
    # The patch reads ``cpu_offload`` out of **kwargs; a positional call would silently no-op.
    require_signature(state_load, _LOAD_STATE_DICT_PARAMS, "load_model_from_full_model_state_dict")
    loader = import_fastvideo_module("fastvideo.models.loader.component_loader", "PipelineComponentLoader")
    require_signature(
        require_attr(
            require_attr(loader, "PipelineComponentLoader", "PipelineComponentLoader"),
            "load_module",
            "PipelineComponentLoader.load_module",
        ),
        _LOAD_MODULE_PARAMS,
        "PipelineComponentLoader.load_module",
    )


def require_float_wan_timesteps() -> None:
    """Reject FastVideo's post-scheduler integer cast, which its echo cannot expose."""
    if os.getenv("DIFFUSIONRL_FASTVIDEO_DANCEGRPO_TIMESTEP_LONG", "0").strip().lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "FastVideo canonical UniPC requires floating WAN timesteps; "
            "unset DIFFUSIONRL_FASTVIDEO_DANCEGRPO_TIMESTEP_LONG"
        )


__all__ = [
    "PINNED_FORK",
    "SDE_STEP_PARAMS",
    "SET_TIMESTEPS_PARAMS",
    "import_fastvideo_module",
    "require_attr",
    "require_float_wan_timesteps",
    "require_signature",
    "verify_offload_surface",
    "verify_rl_data_surface",
    "verify_stock_surface",
    "verify_weight_surface",
]
