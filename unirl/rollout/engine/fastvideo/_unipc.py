"""FastVideo adapter for UniRL's canonical UniPC solver; see the local README for the runtime patch contract."""

from __future__ import annotations

import functools
import importlib
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch

from unirl.models.wan21.diffusion import WAN21DiffusionStep
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas
from unirl.sde.unipc import UniPCSpec, UniPCStrategy


def _require_float_wan_timesteps() -> None:
    """Reject FastVideo's post-scheduler integer cast, which its echo cannot expose."""
    if os.getenv("DIFFUSIONRL_FASTVIDEO_DANCEGRPO_TIMESTEP_LONG", "0").lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "FastVideo canonical UniPC requires floating WAN timesteps; "
            "unset DIFFUSIONRL_FASTVIDEO_DANCEGRPO_TIMESTEP_LONG"
        )


def _wan_timestep_scale(scheduler: Any) -> float:
    """Return the model-owned scale after validating the worker scheduler contract."""
    expected = float(WAN21DiffusionStep.TIMESTEP_SCALE)
    actual = getattr(scheduler.config, "num_train_timesteps", None)
    if actual != expected:
        raise RuntimeError(
            "FastVideo scheduler num_train_timesteps does not match the WAN21 model "
            f"timestep scale: scheduler={actual!r}, model={expected:g}"
        )
    return expected


@dataclass(frozen=True)
class FastVideoUniPCPlan:
    """Per-request SDE/UniPC dispatch data carried into FastVideo workers."""

    sde_type: str
    sde_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        canonical = str(self.sde_type).strip().lower()
        if canonical not in {"flow", "dance"}:
            raise ValueError(f"FastVideo UniPC supports SDE type 'flow' or 'dance'; got {self.sde_type!r}")
        indices = tuple(int(i) for i in self.sde_indices)
        if any(i < 0 for i in indices) or tuple(sorted(set(indices))) != indices:
            raise ValueError(f"FastVideo UniPC requires sorted unique non-negative SDE indices; got {indices}")
        object.__setattr__(self, "sde_type", canonical)
        object.__setattr__(self, "sde_indices", indices)


def _strategy_from_scheduler(scheduler: Any) -> UniPCStrategy:
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

    return UniPCStrategy(
        config=UniPCSpec(
            solver_order=int(getattr(config, "solver_order", 2)),
            solver_type=str(getattr(config, "solver_type", "bh2")),
            lower_order_final=bool(getattr(config, "lower_order_final", True)),
            disable_corrector=tuple(int(i) for i in getattr(scheduler, "disable_corrector", ())),
        )
    )


def _patch_scheduler_set_timesteps() -> None:
    module_name = "fastvideo.models.schedulers.scheduling_flow_unipc_multistep"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not module_name.startswith(f"{exc.name}.") and exc.name != module_name:
            raise
        raise RuntimeError("FastVideo UniPC integration requires FlowUniPCMultistepScheduler") from exc
    FlowUniPCMultistepScheduler = getattr(module, "FlowUniPCMultistepScheduler", None)
    if FlowUniPCMultistepScheduler is None:
        raise RuntimeError("FastVideo UniPC integration requires FlowUniPCMultistepScheduler")

    original = FlowUniPCMultistepScheduler.set_timesteps
    if getattr(original, "_unirl_canonical_sigmas", False):
        return

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
        if np.any(external[1:] > external[:-1]):
            raise ValueError("FastVideo canonical sigmas must be monotonically non-increasing")
        if external[-1] == 0.0:
            raise ValueError("FastVideo external sigmas must omit the terminal zero")
        if num_inference_steps is not None and int(num_inference_steps) != int(external.size):
            raise ValueError(
                f"FastVideo num_inference_steps={num_inference_steps} does not match "
                f"the external sigma count {external.size}"
            )
        if str(getattr(self.config, "final_sigmas_type", "zero")) != "zero":
            raise ValueError("FastVideo canonical UniPC requires final_sigmas_type='zero'")

        timestep_scale = _wan_timestep_scale(self)
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

        strategy = _strategy_from_scheduler(self)
        strategy.init_schedule(self.sigmas)
        self._unirl_unipc_strategy = strategy

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
    module_name = "fastvideo.pipelines.stages.denoising"
    try:
        denoising = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not module_name.startswith(f"{exc.name}.") and exc.name != module_name:
            raise
        raise RuntimeError(
            "FastVideo UniPC integration requires pipelines.stages.denoising.sde_step_with_logprob"
        ) from exc

    original = getattr(denoising, "sde_step_with_logprob", None)
    if original is None:
        raise RuntimeError("FastVideo UniPC integration requires pipelines.stages.denoising.sde_step_with_logprob")
    if getattr(original, "_unirl_unipc_dispatch", False):
        return

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
        if not isinstance(strategy, UniPCStrategy):
            raise RuntimeError("FastVideo worker did not install the canonical UniPC schedule patch before denoising")

        if step_index in sde_type.sde_indices:
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

        sigmas = scheduler.sigmas.to(device=sample.device, dtype=torch.float32)
        sigma = sigmas[step_index]
        sigma_next = sigmas[step_index + 1]
        result, _, _ = strategy.denoise(
            noise_pred=model_output,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=0.0,
            step_index=step_index,
        )

        # FastVideo stacks one log-prob value per helper invocation. The adapter
        # slices the real SDE columns before building LatentSegment, so use a
        # shape-compatible placeholder for deterministic UniPC columns.
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
    _patch_scheduler_set_timesteps()
    _patch_denoising_step()


def _worker_main_with_unipc(*args, **kwargs):
    """Spawn-safe FastVideo worker entrypoint that installs runtime patches."""
    _patch_worker_runtime()
    from fastvideo.worker.multiproc_executor import WorkerMultiprocProc

    original = getattr(WorkerMultiprocProc, "_unirl_original_worker_main", WorkerMultiprocProc.worker_main)
    if original is _worker_main_with_unipc:
        raise RuntimeError("FastVideo worker entrypoint patch lost the original worker_main")
    return original(*args, **kwargs)


def _patch_worker_entrypoint() -> None:
    module_name = "fastvideo.worker.multiproc_executor"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and not module_name.startswith(f"{exc.name}.") and exc.name != module_name:
            raise
        raise RuntimeError("FastVideo UniPC integration requires MultiprocExecutor workers") from exc
    WorkerMultiprocProc = getattr(module, "WorkerMultiprocProc", None)
    if WorkerMultiprocProc is None:
        raise RuntimeError("FastVideo UniPC integration requires MultiprocExecutor workers")

    current = WorkerMultiprocProc.worker_main
    if current is _worker_main_with_unipc:
        return
    WorkerMultiprocProc._unirl_original_worker_main = current
    WorkerMultiprocProc.worker_main = staticmethod(_worker_main_with_unipc)


def patch_fastvideo_unipc() -> None:
    """Install idempotent parent, worker-entrypoint, and runtime patches."""
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
