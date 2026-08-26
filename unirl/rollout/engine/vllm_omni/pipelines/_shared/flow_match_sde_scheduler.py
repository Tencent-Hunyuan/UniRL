"""Trajectory-capturing SDE flow-match scheduler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class FlowMatchSDESchedulerOutput(BaseOutput):
    """``return_dict=True`` payload for :class:`FlowMatchSDEDiscreteScheduler`."""

    prev_sample: torch.Tensor
    log_prob: Optional[torch.Tensor]
    prev_sample_mean: torch.Tensor
    std_dev_t: torch.Tensor


class FlowMatchSDEDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """SDE flow-match scheduler with on-instance trajectory capture."""

    _traj_latents: List[torch.Tensor]
    _traj_timesteps: List[torch.Tensor]
    _traj_log_probs: List[torch.Tensor]
    _traj_sde_step_indices: List[int]

    def __init__(self, *args, eta: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if eta < 0.0:
            raise ValueError(f"FlowMatchSDEDiscreteScheduler.eta must be >= 0; got eta={eta!r}.")
        self._eta = float(eta)
        self._traj_latents = []
        self._traj_timesteps = []
        self._traj_log_probs = []
        self._traj_sde_step_indices = []
        self._initial_latent: Optional[torch.Tensor] = None
        self._initial_timestep: Optional[torch.Tensor] = None
        self._sde_indices_set: Optional[frozenset] = None

    def arm(self, *, eta: float, sde_indices: Optional[List[int]] = None) -> None:
        """Per-request arming: SDE strength + the sparse step gate."""
        if eta < 0.0:
            raise ValueError(f"FlowMatchSDEDiscreteScheduler.arm: eta must be >= 0; got eta={eta!r}.")
        self._eta = float(eta)
        self._sde_indices_set = frozenset(int(i) for i in sde_indices) if sde_indices is not None else None

    def set_timesteps(
        self,
        num_inference_steps=None,
        device=None,
        sigmas=None,
        mu=None,
        timesteps=None,
    ):  # type: ignore[override]
        """Reset trajectory buffers and build the sigma schedule with upstream's double static shift neutralized."""
        if sigmas is not None:
            from diffusers.configuration_utils import FrozenDict

            original_internal = self._internal_dict
            original_shift = self._shift
            overrides = dict(original_internal)
            overrides["use_dynamic_shifting"] = False
            overrides["shift_terminal"] = None
            self._internal_dict = FrozenDict(overrides)
            self._shift = 1.0
            try:
                out = super().set_timesteps(
                    num_inference_steps=num_inference_steps,
                    device=device,
                    sigmas=sigmas,
                    mu=mu,
                    timesteps=timesteps,
                )
            finally:
                self._internal_dict = original_internal
                self._shift = original_shift
        else:
            out = super().set_timesteps(
                num_inference_steps=num_inference_steps,
                device=device,
                sigmas=sigmas,
                mu=mu,
                timesteps=timesteps,
            )
        self._traj_latents = []
        self._traj_timesteps = []
        self._traj_log_probs = []
        self._traj_sde_step_indices = []
        self._initial_latent = None
        self._initial_timestep = None
        return out

    def step(  # type: ignore[override]
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = False,
        **_unused,
    ) -> Union[FlowMatchSDESchedulerOutput, Tuple[torch.Tensor, ...]]:
        """SDE Flow-Match transition with trajectory capture."""
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                "FlowMatchSDEDiscreteScheduler.step expects a float-typed timestep "
                "from scheduler.timesteps (not an integer step index)."
            )
        if self.step_index is None:
            self._init_step_index(timestep)

        if self._initial_latent is None:
            self._initial_latent = sample.detach().clone()
            if torch.is_tensor(timestep):
                init_t = timestep.detach().to(sample.device).clone()
            else:
                init_t = torch.as_tensor(float(timestep), device=sample.device)
            self._initial_timestep = init_t.expand(sample.shape[0]).clone()

        original_dtype = sample.dtype
        sample_f32 = sample.to(torch.float32)
        model_output_f32 = model_output.to(torch.float32)

        sigma_idx = self.step_index
        sigma = self.sigmas[sigma_idx]
        sigma_prev = self.sigmas[sigma_idx + 1]
        sigma_max = self.sigmas[1]
        dt = sigma_prev - sigma

        if self._sde_indices_set is None or len(self._sde_indices_set) == 0:
            step_is_sde = False
        else:
            step_is_sde = int(sigma_idx) in self._sde_indices_set

        if step_is_sde:
            # Clamp sigma denominators and reject eta=0 in SDE steps.
            if float(self._eta) <= 0.0:
                raise RuntimeError(
                    f"FlowMatchSDEDiscreteScheduler.step: step_index={int(sigma_idx)} "
                    f"is in _sde_indices_set but scheduler eta={self._eta!r}; "
                    f"eta must be > 0 for SDE steps. Check pipeline "
                    f"_ensure_scheduler_for_eta + driver sde_indices wiring."
                )
            std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * self._eta
            prev_sample_mean = (
                sample_f32 * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + model_output_f32 * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )

            variance_noise = randn_tensor(
                model_output_f32.shape,
                generator=generator,
                device=model_output_f32.device,
                dtype=torch.float32,
            )
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-dt) * variance_noise

            prev_sample = prev_sample.to(original_dtype)
            prev_sample_for_logp = prev_sample.to(torch.float32)

            log_prob_per_elem = (
                -((prev_sample_for_logp.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-dt)) ** 2))
                - torch.log(std_dev_t * torch.sqrt(-dt))
                - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
            )
            log_prob: Optional[torch.Tensor] = log_prob_per_elem.mean(dim=tuple(range(1, log_prob_per_elem.ndim)))
        else:
            # Non-SDE steps must use plain Euler even when the scheduler eta is nonzero.
            std_dev_t = sigma.new_zeros(())
            prev_sample_mean = sample_f32 + model_output_f32 * dt
            prev_sample = prev_sample_mean.to(original_dtype)
            log_prob = None

        self._traj_latents.append(prev_sample.detach().clone())
        if torch.is_tensor(timestep):
            t_for_capture = timestep.detach().to(prev_sample.device).clone()
        else:
            t_for_capture = torch.as_tensor(float(timestep), device=prev_sample.device)
        self._traj_timesteps.append(t_for_capture.expand(prev_sample.shape[0]).clone())
        if log_prob is not None:
            self._traj_log_probs.append(log_prob.detach().clone())
            self._traj_sde_step_indices.append(int(sigma_idx))

        self._step_index += 1

        if return_dict:
            return FlowMatchSDESchedulerOutput(
                prev_sample=prev_sample,
                log_prob=log_prob,
                prev_sample_mean=prev_sample_mean,
                std_dev_t=std_dev_t,
            )
        return (prev_sample, log_prob, prev_sample_mean, std_dev_t)

    @property
    def last_sde_step_indices(self) -> List[int]:
        """Return the list of step indices that ran SDE on the most recent denoise loop."""
        return list(self._traj_sde_step_indices)

    def drain_trajectory(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return ``(latents [B,T+1,...], sigmas [T+1], timesteps [B,T+1], log_probs [B,K])`` or ``None``."""
        if not self._traj_latents:
            return None
        post_latents = torch.stack(self._traj_latents, dim=1)
        post_timesteps = torch.stack(self._traj_timesteps, dim=1)
        if self._traj_log_probs:
            log_probs = torch.stack(self._traj_log_probs, dim=1)
        else:
            B = post_latents.shape[0]
            log_probs = post_latents.new_zeros((B, 0), dtype=torch.float32)

        if self._initial_latent is not None and self._initial_timestep is not None:
            init_lat = self._initial_latent.to(post_latents.dtype).unsqueeze(1)
            init_ts = self._initial_timestep.to(post_timesteps.dtype).unsqueeze(1)
            latents = torch.cat([init_lat, post_latents], dim=1)
            timesteps = torch.cat([init_ts, post_timesteps], dim=1)
        else:
            latents = post_latents
            timesteps = post_timesteps

        T_plus_1 = int(latents.shape[1])
        sigmas = self.sigmas[:T_plus_1].detach().clone().to(latents.device)

        return latents, sigmas, timesteps, log_probs


__all__ = ["FlowMatchSDEDiscreteScheduler", "FlowMatchSDESchedulerOutput"]
