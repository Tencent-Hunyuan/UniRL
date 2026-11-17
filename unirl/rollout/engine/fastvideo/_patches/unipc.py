"""Route FastVideo's deterministic non-SDE steps through UniRL's canonical UniPC; contract in README.md."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np
import torch

from unirl.rollout.engine.fastvideo._patches.compat import (
    SDE_STEP_PARAMS,
    SET_TIMESTEPS_PARAMS,
    import_fastvideo_module,
    require_attr,
    require_signature,
)
from unirl.sde.unipc import UniPCSpec, UniPCStrategy


@dataclass(frozen=True)
class FastVideoUniPCPlan:
    """Per-request SDE/UniPC dispatch plan carried into FastVideo workers through ``RLData.sde_type``."""

    sde_type: str
    sde_indices: Tuple[int, ...]
    spec: UniPCSpec = field(default_factory=UniPCSpec)
    timestep_scale: float = 0.0

    def __post_init__(self) -> None:
        canonical = self.sde_type.strip().lower()
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


def patch_unipc() -> None:
    """Pin the scheduler to canonical sigmas, then dispatch non-SDE step indices to UniPC."""
    sched_module = import_fastvideo_module(
        "fastvideo.models.schedulers.scheduling_flow_unipc_multistep", "FlowUniPCMultistepScheduler"
    )
    FlowUniPCMultistepScheduler = require_attr(
        sched_module, "FlowUniPCMultistepScheduler", "FlowUniPCMultistepScheduler"
    )

    orig_set_timesteps = FlowUniPCMultistepScheduler.set_timesteps
    if not getattr(orig_set_timesteps, "_unirl_canonical_sigmas", False):
        require_signature(orig_set_timesteps, SET_TIMESTEPS_PARAMS, "FlowUniPCMultistepScheduler.set_timesteps")

        @functools.wraps(orig_set_timesteps)
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
                orig_set_timesteps(
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

            if mu is not None or shift is not None or use_karras_sigmas or use_kerras_sigma:
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
            if num_inference_steps is not None and num_inference_steps != external.size:
                raise ValueError(
                    f"FastVideo num_inference_steps={num_inference_steps} does not match "
                    f"the external sigma count {external.size}"
                )
            if self.config.final_sigmas_type != "zero":
                raise ValueError("FastVideo canonical UniPC requires final_sigmas_type='zero'")

            raw_scale = self.config.num_train_timesteps
            if not isinstance(raw_scale, (int, float)) or not raw_scale > 0:
                raise RuntimeError(f"FastVideo scheduler has no usable num_train_timesteps: {raw_scale!r}")
            timestep_scale = float(raw_scale)
            terminal = np.zeros(1, dtype=np.float32)
            schedule = np.concatenate([external, terminal])
            self.sigmas = torch.from_numpy(schedule).cpu()
            self.timesteps = torch.from_numpy(external * timestep_scale).to(device=device, dtype=torch.float32)
            self.num_inference_steps = int(external.size)

            solver_order = self.config.solver_order
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

    denoising = import_fastvideo_module(
        "fastvideo.pipelines.stages.denoising", "pipelines.stages.denoising.sde_step_with_logprob"
    )
    original = require_attr(denoising, "sde_step_with_logprob", "pipelines.stages.denoising.sde_step_with_logprob")
    if getattr(original, "_unirl_unipc_dispatch", False):
        return
    require_signature(original, SDE_STEP_PARAMS, "sde_step_with_logprob")

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

        if torch.is_tensor(timestep):
            if timestep.numel() != 1:
                raise RuntimeError(
                    f"FastVideo canonical UniPC expects one timestep per denoising call; got shape {tuple(timestep.shape)}"
                )
            timestep_val = timestep.reshape(()).item()
        else:
            timestep_val = timestep
        step_index = int(scheduler.index_for_timestep(timestep_val))
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
            if not getattr(scheduler, "_unirl_canonical_schedule", False):
                raise RuntimeError(
                    "FastVideo worker did not install the canonical UniPC schedule patch before denoising"
                )
            if scheduler._unirl_timestep_scale != sde_type.timestep_scale:
                raise RuntimeError(
                    "FastVideo scheduler was pinned with a different model timestep scale than this request's "
                    f"plan: scheduler={scheduler._unirl_timestep_scale!r}, "
                    f"plan={sde_type.timestep_scale!r}"
                )
            config = scheduler.config
            if config.prediction_type != "flow_prediction":
                raise RuntimeError(
                    f"FastVideo UniPC requires prediction_type='flow_prediction'; got {config.prediction_type!r}"
                )
            if not scheduler.predict_x0:
                raise RuntimeError("FastVideo UniPC requires predict_x0=True")
            if config.thresholding:
                raise RuntimeError("FastVideo UniPC does not support thresholding=True")
            if scheduler.solver_p is not None:
                raise RuntimeError("FastVideo UniPC does not support a nested solver_p")
            strategy = UniPCStrategy(config=sde_type.spec)
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


__all__ = ["FastVideoUniPCPlan", "patch_unipc"]
