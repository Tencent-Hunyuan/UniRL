"""FastVideo adapter routing deterministic non-SDE steps through UniPC; integration contract in README.md."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import torch

from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.sde.unipc import UniPCSpec, UniPCStrategy

# The exact upstream surface these patches target; drift fails closed at patch time (README: pin).
_PINNED_FORK = "hao-ai-lab/FastVideo@2095477eac7e289c7a7ab13acb367ca60687c304"

_SET_TIMESTEPS_PARAMS = (
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
# ``_contracts.patch_transition``, which must therefore run before this fingerprint is taken.
_SDE_STEP_PARAMS = (
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
_STOCK_SDE_STEP_PARAMS = _SDE_STEP_PARAMS[:-2]
_COLLECTIVE_RPC_PARAMS = ("self", "method", "timeout", "args", "kwargs")
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


def _import_fastvideo_module(module_name: str, what: str) -> Any:
    """Import a fastvideo module, distinguishing a missing integration surface from unrelated import errors."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not module_name.startswith(f"{exc.name}.") and exc.name != module_name:
            raise
        raise RuntimeError(f"FastVideo UniPC integration requires {what} (pinned surface: {_PINNED_FORK})") from exc


def _require_attr(owner: Any, name: str, what: str) -> Any:
    """Fetch ``owner.name`` or fail closed naming the pinned integration surface."""
    value = getattr(owner, name, None)
    if value is None:
        raise RuntimeError(f"FastVideo UniPC integration requires {what} (pinned surface: {_PINNED_FORK})")
    return value


def _require_signature(fn: Any, expected: Tuple[str, ...], what: str) -> None:
    """Fingerprint a patched callable's parameter list so fork drift fails at patch time, not mid-rollout."""
    actual = tuple(inspect.signature(fn).parameters)
    if actual != expected:
        raise RuntimeError(
            f"FastVideo {what} drifted from the pinned integration surface ({_PINNED_FORK}): "
            f"expected parameters {expected}, got {actual}"
        )


def _verify_rl_data_surface() -> None:
    """Fail closed unless ``ForwardBatch.RLData`` carries every field the engine-side integration relies on."""
    module = _import_fastvideo_module("fastvideo.pipelines.pipeline_batch_info", "ForwardBatch.RLData")
    forward_batch = _require_attr(module, "ForwardBatch", "ForwardBatch.RLData")
    rl_data = _require_attr(forward_batch, "RLData", "ForwardBatch.RLData")
    names = {f.name for f in dataclasses.fields(rl_data)}
    missing = sorted(_RL_DATA_FIELDS - names)
    if missing:
        raise RuntimeError(
            f"FastVideo ForwardBatch.RLData lacks fields {missing} required by the UniPC "
            f"integration (pinned surface: {_PINNED_FORK}); _contracts.patch_contracts must run first"
        )


def _verify_stock_surface() -> None:
    """Fingerprint the untouched upstream seams before any UniRL patch rewrites them."""
    denoising = _import_fastvideo_module(
        "fastvideo.pipelines.stages.denoising", "pipelines.stages.denoising.sde_step_with_logprob"
    )
    original = _require_attr(denoising, "sde_step_with_logprob", "sde_step_with_logprob")
    if getattr(original, "_unirl_fastvideo_sde", False):
        return
    _require_signature(original, _STOCK_SDE_STEP_PARAMS, "stock sde_step_with_logprob")
    stage = _require_attr(denoising, "DenoisingStage", "DenoisingStage")
    forward = _require_attr(stage, "forward", "DenoisingStage.forward")
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError) as exc:
        raise RuntimeError(f"cannot inspect FastVideo DenoisingStage.forward: {exc}") from exc
    for marker in ("rl_data", "sde_step_with_logprob", "scheduler.step"):
        if marker not in source:
            raise RuntimeError(
                f"FastVideo DenoisingStage.forward lacks required source marker {marker!r} "
                f"(pinned surface: {_PINNED_FORK})"
            )


def _verify_weight_surface() -> None:
    """Fail closed unless the worker/executor/generator seams the weight patch installs onto still exist."""
    worker = _require_attr(_import_fastvideo_module("fastvideo.worker.gpu_worker", "Worker"), "Worker", "Worker")
    _require_attr(worker, "execute_forward", "Worker.execute_forward")
    executor = _require_attr(
        _import_fastvideo_module("fastvideo.worker.multiproc_executor", "MultiprocExecutor"),
        "MultiprocExecutor",
        "MultiprocExecutor",
    )
    _require_signature(
        _require_attr(executor, "collective_rpc", "MultiprocExecutor.collective_rpc"),
        _COLLECTIVE_RPC_PARAMS,
        "MultiprocExecutor.collective_rpc",
    )
    hooks = _import_fastvideo_module("fastvideo.hooks.hooks", "ModuleHookManager")
    manager = _require_attr(hooks, "ModuleHookManager", "ModuleHookManager")
    for name in ("get_from", "get_forward_hook"):
        _require_attr(manager, name, f"ModuleHookManager.{name}")
    offload = _import_fastvideo_module("fastvideo.hooks.layerwise_offload", "LayerwiseOffloadHook")
    _require_attr(
        _require_attr(offload, "LayerwiseOffloadHook", "LayerwiseOffloadHook"),
        "mutate_params_scope",
        "LayerwiseOffloadHook.mutate_params_scope",
    )


def _verify_offload_surface() -> None:
    """Fail closed unless the loader seams the offload patch wraps keep their patched-for signatures."""
    fsdp_load = _import_fastvideo_module("fastvideo.models.loader.fsdp_load", "fsdp_load")
    state_load = _require_attr(
        fsdp_load, "load_model_from_full_model_state_dict", "load_model_from_full_model_state_dict"
    )
    # The patch reads ``cpu_offload`` out of **kwargs; a positional call would silently no-op.
    _require_signature(state_load, _LOAD_STATE_DICT_PARAMS, "load_model_from_full_model_state_dict")
    loader = _import_fastvideo_module("fastvideo.models.loader.component_loader", "PipelineComponentLoader")
    _require_signature(
        _require_attr(
            _require_attr(loader, "PipelineComponentLoader", "PipelineComponentLoader"),
            "load_module",
            "PipelineComponentLoader.load_module",
        ),
        _LOAD_MODULE_PARAMS,
        "PipelineComponentLoader.load_module",
    )


def _require_float_wan_timesteps() -> None:
    """Reject FastVideo's post-scheduler integer cast, which its echo cannot expose."""
    if os.getenv("DIFFUSIONRL_FASTVIDEO_DANCEGRPO_TIMESTEP_LONG", "0").strip().lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "FastVideo canonical UniPC requires floating WAN timesteps; "
            "unset DIFFUSIONRL_FASTVIDEO_DANCEGRPO_TIMESTEP_LONG"
        )


def _scheduler_timestep_scale(scheduler: Any) -> float:
    """Return the live scheduler's own timestep scale, which ``index_for_timestep`` matches against."""
    actual = getattr(scheduler.config, "num_train_timesteps", None)
    if not isinstance(actual, (int, float)) or not float(actual) > 0.0:
        raise RuntimeError(f"FastVideo scheduler has no usable num_train_timesteps: {actual!r}")
    return float(actual)


@dataclass(frozen=True)
class FastVideoUniPCPlan:
    """Per-request SDE/UniPC dispatch plan carried into FastVideo workers through ``RLData.sde_type``."""

    sde_type: str
    sde_indices: Tuple[int, ...]
    spec: UniPCSpec = field(default_factory=UniPCSpec)
    timestep_scale: float = 0.0

    def __post_init__(self) -> None:
        canonical = str(self.sde_type).strip().lower()
        if canonical not in {"flow", "dance"}:
            raise ValueError(f"FastVideo UniPC supports SDE type 'flow' or 'dance'; got {self.sde_type!r}")
        indices = tuple(int(i) for i in self.sde_indices)
        if any(i < 0 for i in indices) or tuple(sorted(set(indices))) != indices:
            raise ValueError(f"FastVideo UniPC requires sorted unique non-negative SDE indices; got {indices}")
        if not isinstance(self.spec, UniPCSpec):
            raise ValueError(f"FastVideo UniPC plan requires a UniPCSpec; got {type(self.spec).__name__}")
        scale = float(self.timestep_scale)
        if not scale > 0.0:
            raise ValueError(
                f"FastVideo UniPC requires the model-owned positive timestep_scale; got {self.timestep_scale!r}"
            )
        object.__setattr__(self, "sde_type", canonical)
        object.__setattr__(self, "sde_indices", indices)
        object.__setattr__(self, "timestep_scale", scale)


def _strategy_from_plan(scheduler: Any, plan: FastVideoUniPCPlan) -> UniPCStrategy:
    """Build the canonical strategy from the plan spec after validating the live worker scheduler contract."""
    config = scheduler.config
    prediction_type = str(getattr(config, "prediction_type", ""))
    if prediction_type != "flow_prediction":
        raise RuntimeError(f"FastVideo UniPC requires prediction_type='flow_prediction'; got {prediction_type!r}")
    if not bool(getattr(scheduler, "predict_x0", True)):
        raise RuntimeError("FastVideo UniPC requires predict_x0=True")
    if bool(getattr(config, "thresholding", False)):
        raise RuntimeError("FastVideo UniPC does not support thresholding=True")
    if getattr(scheduler, "solver_p", None) is not None:
        raise RuntimeError("FastVideo UniPC does not support a nested solver_p")
    # The checkpoint solver spec is verified engine-side at init (README: Solver SSOT).
    return UniPCStrategy(config=plan.spec)


def _patch_scheduler_set_timesteps() -> None:
    """Wrap ``FlowUniPCMultistepScheduler.set_timesteps`` to consume canonical sigmas verbatim with float timesteps."""
    module = _import_fastvideo_module(
        "fastvideo.models.schedulers.scheduling_flow_unipc_multistep", "FlowUniPCMultistepScheduler"
    )
    FlowUniPCMultistepScheduler = _require_attr(module, "FlowUniPCMultistepScheduler", "FlowUniPCMultistepScheduler")

    original = FlowUniPCMultistepScheduler.set_timesteps
    if getattr(original, "_unirl_canonical_sigmas", False):
        return
    _require_signature(original, _SET_TIMESTEPS_PARAMS, "FlowUniPCMultistepScheduler.set_timesteps")

    @functools.wraps(original)
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        sigmas: Optional[list[float]] = None,
        mu: Optional[float] = None,
        shift: Optional[float] = None,
        use_karras_sigmas: Optional[bool] = None,
        use_kerras_sigma: Optional[bool] = None,
    ) -> None:
        if sigmas is None:
            original(
                self,
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=None,
                mu=mu,
                shift=shift,
                use_karras_sigmas=use_karras_sigmas,
                use_kerras_sigma=use_kerras_sigma,
            )
            self._unirl_canonical_schedule = False
            self._unirl_timestep_scale = None
            self._unirl_unipc_strategy = None
            self._unirl_device_sigmas = None
            return

        if mu is not None or shift is not None or bool(use_karras_sigmas) or bool(use_kerras_sigma):
            raise ValueError(
                "FastVideo received a canonical external sigma schedule together with a schedule transform"
            )

        external = np.asarray(sigmas, dtype=np.float32)
        if external.ndim != 1 or external.size == 0:
            raise ValueError("FastVideo canonical sigmas must be a non-empty one-dimensional sequence")
        if not np.isfinite(external).all():
            raise ValueError("FastVideo canonical sigmas must all be finite")
        if np.any(external < 0.0) or np.any(external > 1.0):
            raise ValueError("FastVideo canonical sigmas must be normalized to [0, 1]")
        if np.any(external[1:] >= external[:-1]):
            raise ValueError(
                "FastVideo canonical sigmas must be strictly decreasing; duplicate adjacent "
                "sigmas alias to one index_for_timestep slot and break UniPC dispatch"
            )
        if external[-1] == 0.0:
            raise ValueError("FastVideo external sigmas must omit the terminal zero")
        if num_inference_steps is not None and int(num_inference_steps) != int(external.size):
            raise ValueError(
                f"FastVideo num_inference_steps={num_inference_steps} does not match "
                f"the external sigma count {external.size}"
            )
        if str(getattr(self.config, "final_sigmas_type", "zero")) != "zero":
            raise ValueError("FastVideo canonical UniPC requires final_sigmas_type='zero'")

        timestep_scale = _scheduler_timestep_scale(self)
        terminal = np.zeros(1, dtype=np.float32)
        schedule = np.concatenate([external, terminal])
        self.sigmas = torch.from_numpy(schedule).cpu()
        self.timesteps = torch.from_numpy(external * timestep_scale).to(device=device, dtype=torch.float32)
        self.num_inference_steps = int(external.size)

        solver_order = int(self.config.solver_order)
        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

        # The strategy is built lazily at the first UniPC-dispatched denoising
        # call, from the request plan's model-owned spec (README: dispatch).
        self._unirl_canonical_schedule = True
        self._unirl_timestep_scale = timestep_scale
        self._unirl_unipc_strategy = None
        self._unirl_device_sigmas = None

    set_timesteps._unirl_canonical_sigmas = True  # type: ignore[attr-defined]
    FlowUniPCMultistepScheduler.set_timesteps = set_timesteps


def _single_step_index(scheduler: Any, timestep: Any) -> int:
    if torch.is_tensor(timestep):
        if timestep.numel() != 1:
            raise RuntimeError(
                f"FastVideo canonical UniPC expects one timestep per denoising call; got shape {tuple(timestep.shape)}"
            )
        timestep = timestep.reshape(()).item()
    return int(scheduler.index_for_timestep(timestep))


def _patch_denoising_step() -> None:
    """Wrap ``sde_step_with_logprob`` to dispatch plan indices to SDE kernels and all other indices to UniPC."""
    denoising = _import_fastvideo_module(
        "fastvideo.pipelines.stages.denoising", "pipelines.stages.denoising.sde_step_with_logprob"
    )
    original = _require_attr(denoising, "sde_step_with_logprob", "pipelines.stages.denoising.sde_step_with_logprob")
    if getattr(original, "_unirl_unipc_dispatch", False):
        return
    _require_signature(original, _SDE_STEP_PARAMS, "sde_step_with_logprob")

    @functools.wraps(original)
    def sde_step_with_logprob(
        scheduler,
        model_output: torch.Tensor,
        timestep: Any,
        sample: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        deterministic: bool = False,
        return_pixel_log_prob: bool = False,
        return_dt_and_std_dev_t: bool = False,
        eta: Optional[float] = None,
        sde_type: Any = "dance",
    ):
        if not isinstance(sde_type, FastVideoUniPCPlan):
            return original(
                scheduler,
                model_output,
                timestep,
                sample,
                prev_sample=prev_sample,
                generator=generator,
                deterministic=deterministic,
                return_pixel_log_prob=return_pixel_log_prob,
                return_dt_and_std_dev_t=return_dt_and_std_dev_t,
                eta=eta,
                sde_type=sde_type,
            )

        step_index = _single_step_index(scheduler, timestep)
        strategy = getattr(scheduler, "_unirl_unipc_strategy", None)

        if step_index in sde_type.sde_indices:
            if strategy is not None:
                strategy.reset_history()
            return original(
                scheduler,
                model_output,
                timestep,
                sample,
                prev_sample=prev_sample,
                generator=generator,
                deterministic=deterministic,
                return_pixel_log_prob=return_pixel_log_prob,
                return_dt_and_std_dev_t=return_dt_and_std_dev_t,
                eta=eta,
                sde_type=sde_type.sde_type,
            )

        if prev_sample is not None:
            raise RuntimeError("FastVideo canonical UniPC deterministic steps do not support replay")
        if deterministic:
            raise RuntimeError("FastVideo canonical UniPC cannot be combined with deterministic=True")
        if return_pixel_log_prob:
            raise NotImplementedError("Pixel-level log prob is not supported for canonical UniPC")

        if strategy is None:
            if not bool(getattr(scheduler, "_unirl_canonical_schedule", False)):
                raise RuntimeError(
                    "FastVideo worker did not install the canonical UniPC schedule patch before denoising"
                )
            if float(getattr(scheduler, "_unirl_timestep_scale", 0.0)) != float(sde_type.timestep_scale):
                raise RuntimeError(
                    "FastVideo scheduler was pinned with a different model timestep scale than this request's "
                    f"plan: scheduler={getattr(scheduler, '_unirl_timestep_scale', None)!r}, "
                    f"plan={sde_type.timestep_scale!r}"
                )
            strategy = _strategy_from_plan(scheduler, sde_type)
            strategy.init_schedule(scheduler.sigmas)
            scheduler._unirl_unipc_strategy = strategy

        dev_sigmas = getattr(scheduler, "_unirl_device_sigmas", None)
        if dev_sigmas is None or dev_sigmas.device != sample.device:
            dev_sigmas = scheduler.sigmas.to(device=sample.device, dtype=torch.float32)
            scheduler._unirl_device_sigmas = dev_sigmas
        sigma = dev_sigmas[step_index]
        sigma_next = dev_sigmas[step_index + 1]
        result, _, _ = strategy.denoise(
            noise_pred=model_output,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=0.0,
            step_index=step_index,
        )

        # Shape-compatible zero log-prob; the engine slices real SDE columns off (README: Dispatch).
        placeholder_logp = torch.zeros(
            sample.shape[0],
            device=sample.device,
            dtype=torch.float32,
        )
        sqrt_dt = torch.sqrt(torch.clamp(sigma - sigma_next, min=0.0))
        if return_dt_and_std_dev_t:
            return result, placeholder_logp, None, None, sqrt_dt
        return result, placeholder_logp, None, None

    sde_step_with_logprob._unirl_unipc_dispatch = True  # type: ignore[attr-defined]
    denoising.sde_step_with_logprob = sde_step_with_logprob


def _patch_worker_runtime() -> None:
    _require_float_wan_timesteps()
    _verify_stock_surface()
    from unirl.rollout.engine.fastvideo._contracts import patch_contracts, patch_transition
    from unirl.rollout.engine.fastvideo._offload import patch_offload
    from unirl.rollout.engine.fastvideo._weights import patch_weights

    # Contract and transition first: they add the RLData fields and the eta/sde_type
    # parameters that the UniPC fingerprints and dispatch below rely on.
    patch_contracts()
    patch_transition()
    _verify_rl_data_surface()
    _patch_scheduler_set_timesteps()
    _patch_denoising_step()
    patch_offload()
    patch_weights()


def _worker_main_with_unipc(*args, **kwargs):
    """Spawn-safe FastVideo worker entrypoint that installs runtime patches."""
    _patch_worker_runtime()
    from fastvideo.worker.multiproc_executor import WorkerMultiprocProc

    original = getattr(WorkerMultiprocProc, "_unirl_original_worker_main", WorkerMultiprocProc.worker_main)
    if original is _worker_main_with_unipc:
        raise RuntimeError("FastVideo worker entrypoint patch lost the original worker_main")
    return original(*args, **kwargs)


def _patch_worker_entrypoint() -> None:
    module = _import_fastvideo_module("fastvideo.worker.multiproc_executor", "MultiprocExecutor workers")
    WorkerMultiprocProc = _require_attr(module, "WorkerMultiprocProc", "MultiprocExecutor workers")

    current = WorkerMultiprocProc.worker_main
    if current is _worker_main_with_unipc:
        return
    WorkerMultiprocProc._unirl_original_worker_main = current
    WorkerMultiprocProc.worker_main = staticmethod(_worker_main_with_unipc)


def patch_fastvideo_unipc() -> None:
    """Install idempotent parent, worker-entrypoint, and runtime patches after fingerprinting the fork surface."""
    _verify_weight_surface()
    _verify_offload_surface()
    _patch_worker_runtime()
    _patch_worker_entrypoint()


def verify_fastvideo_used_sigmas(
    actual: Any,
    *,
    expected: torch.Tensor,
    sample_index: int,
) -> None:
    """Verify FastVideo's echoed timesteps against the canonical sigma schedule."""
    actual_with_terminal = actual
    if actual is not None:
        actual_t = actual.detach().cpu() if torch.is_tensor(actual) else torch.as_tensor(actual)
        if actual_t.ndim == 1 and int(actual_t.shape[0]) == int(expected.shape[0]) - 1:
            actual_with_terminal = torch.cat([actual_t, torch.zeros(1, dtype=actual_t.dtype)])
    verify_engine_used_sigmas(
        actual_with_terminal,
        expected=expected,
        engine_name=f"fastvideo sample {sample_index}",
    )


__all__ = [
    "FastVideoUniPCPlan",
    "patch_fastvideo_unipc",
    "verify_fastvideo_used_sigmas",
]
