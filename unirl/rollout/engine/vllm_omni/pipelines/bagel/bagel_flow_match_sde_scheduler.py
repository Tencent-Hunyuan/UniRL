"""Trajectory-capturing SDE flow-match scheduler for BAGEL's ``generate_image``."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from diffusers.utils.torch_utils import randn_tensor


@dataclass
class BagelSDEStepOutput:
    """Return payload for :meth:`BagelFlowSDEScheduler.step`."""

    prev_sample: torch.Tensor
    log_prob: Optional[torch.Tensor]
    prev_sample_mean: torch.Tensor


class BagelFlowSDEScheduler:
    """SDE flow-match step for BAGEL ``generate_image`` with trajectory capture."""

    def __init__(self, *, eta: float = 1.0, sigma_max: float = 0.99) -> None:
        if eta < 0.0:
            raise ValueError(f"BagelFlowSDEScheduler.eta must be >= 0; got {eta!r}.")
        self._eta = float(eta)
        self._sigma_max = float(sigma_max)
        self._sde_indices_set: Optional[frozenset] = None
        self._trajectory_dtype: torch.dtype = torch.float32
        self._step_index: int = 0
        self._noise_generator: Optional[torch.Generator] = None
        self._image_token_sizes: Optional[List[int]] = None
        self._traj_latents: List[torch.Tensor] = []
        self._traj_timesteps: List[torch.Tensor] = []
        self._traj_log_probs: List[torch.Tensor] = []
        self._traj_sde_step_indices: List[int] = []
        self._initial_latent: Optional[torch.Tensor] = None
        self._initial_timestep: Optional[torch.Tensor] = None

    def set_for_request(
        self,
        *,
        eta: float,
        sde_indices: Optional[List[int]],
        sigma_max: Optional[float] = None,
        trajectory_dtype: torch.dtype = torch.float32,
        image_token_sizes: Optional[List[int]] = None,
    ) -> None:
        """Arm this request: SDE strength, sparse step gate, σ_max, trajectory dtype."""
        if eta < 0.0:
            raise ValueError(f"BagelFlowSDEScheduler.set_for_request: eta must be >= 0; got {eta!r}.")
        self._eta = float(eta)
        self._sde_indices_set = frozenset(int(i) for i in sde_indices) if sde_indices is not None else None
        if sigma_max is not None:
            self._sigma_max = float(sigma_max)
        self._trajectory_dtype = trajectory_dtype
        self._image_token_sizes = (
            [int(s) for s in image_token_sizes] if image_token_sizes and len(image_token_sizes) > 1 else None
        )
        self._step_index = 0
        self._noise_generator = None
        self._traj_latents = []
        self._traj_timesteps = []
        self._traj_log_probs = []
        self._traj_sde_step_indices = []
        self._initial_latent = None
        self._initial_timestep = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        dt: torch.Tensor,
        **_unused,
    ) -> BagelSDEStepOutput:
        """One SDE flow-match transition; ``timesteps[i]`` is a ``[0, 1]`` σ, NOT a 1000-scale timestep."""
        sigma = timestep if torch.is_tensor(timestep) else torch.as_tensor(float(timestep))
        sigma = sigma.to(device=sample.device, dtype=torch.float32).reshape(())
        dt_passed = dt if torch.is_tensor(dt) else torch.as_tensor(float(dt))
        dt_passed = dt_passed.to(device=sample.device, dtype=torch.float32).reshape(())
        dt_t = -dt_passed
        sigma_next = sigma + dt_t

        step_idx = self._step_index

        if self._initial_latent is None:
            self._initial_latent = sample.detach().to(self._trajectory_dtype).clone()
            self._initial_timestep = sigma.detach().clone()

        original_dtype = sample.dtype
        sample_f32 = sample.to(torch.float32)
        v_t_f32 = model_output.to(torch.float32)

        if self._sde_indices_set is None or len(self._sde_indices_set) == 0:
            step_is_sde = False
        else:
            step_is_sde = int(step_idx) in self._sde_indices_set

        if step_is_sde:
            if float(self._eta) <= 0.0:
                raise RuntimeError(
                    f"BagelFlowSDEScheduler.step: step_index={int(step_idx)} is in the SDE "
                    f"gate but eta={self._eta!r}; eta must be > 0 for SDE steps. Check the "
                    f"adapter's eta / sde_indices wiring."
                )
            clamp_sigma = torch.where(sigma == 1, torch.as_tensor(self._sigma_max, device=sigma.device), sigma)
            std_dev_t = torch.sqrt(sigma / (1 - clamp_sigma)) * self._eta
            prev_sample_mean = (
                sample_f32 * (1 + std_dev_t**2 / (2 * sigma) * dt_t)
                + v_t_f32 * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt_t
            )
            # Draw SDE noise per request; global RNG breaks branch alignment.
            if self._noise_generator is None:
                self._noise_generator = torch.Generator(device=v_t_f32.device)
                self._noise_generator.manual_seed(int.from_bytes(os.urandom(8), "big"))
            noise = randn_tensor(
                v_t_f32.shape,
                generator=self._noise_generator,
                device=v_t_f32.device,
                dtype=torch.float32,
            )
            std_var = std_dev_t * torch.sqrt(-dt_t)
            prev_sample = prev_sample_mean + std_var * noise

            prev_sample = prev_sample.to(self._trajectory_dtype)
            prev_for_logp = prev_sample.to(torch.float32)
            log_prob_elem = (
                -((prev_for_logp.detach() - prev_sample_mean) ** 2) / (2 * std_var**2)
                - torch.log(std_var)
                - 0.5 * math.log(2 * math.pi)
            )
            if self._image_token_sizes is None:
                log_prob: Optional[torch.Tensor] = log_prob_elem.mean()
            else:
                log_prob = torch.stack([chunk.mean() for chunk in log_prob_elem.split(self._image_token_sizes, dim=0)])
        else:
            prev_sample_mean = sample_f32 + v_t_f32 * dt_t
            prev_sample = prev_sample_mean.to(self._trajectory_dtype)
            log_prob = None

        self._traj_latents.append(prev_sample.detach().clone())
        self._traj_timesteps.append(sigma_next.detach().clone())
        if log_prob is not None:
            self._traj_log_probs.append(log_prob.detach().clone())
            self._traj_sde_step_indices.append(int(step_idx))

        self._step_index += 1

        return BagelSDEStepOutput(
            prev_sample=prev_sample.to(original_dtype),
            log_prob=log_prob,
            prev_sample_mean=prev_sample_mean,
        )

    @property
    def last_sde_step_indices(self) -> List[int]:
        """Step indices that ran the SDE branch on the most recent loop."""
        return list(self._traj_sde_step_indices)

    def drain_trajectory(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return ``(latents [N,T+1,seq,C], sigmas [T+1], timesteps [1,T+1], log_probs [N,K])`` or ``None``."""
        if not self._traj_latents:
            return None
        post_latents = torch.stack(self._traj_latents, dim=0)
        post_timesteps = torch.stack(self._traj_timesteps, dim=0)

        if self._initial_latent is not None and self._initial_timestep is not None:
            init_lat = self._initial_latent.to(post_latents.dtype)
            latents = torch.cat([init_lat.unsqueeze(0), post_latents], dim=0)
            sigmas_full = torch.cat([self._initial_timestep.reshape(1), post_timesteps], dim=0)
        else:
            latents = post_latents
            sigmas_full = post_timesteps

        if self._image_token_sizes is None:
            latents = latents.unsqueeze(0)
        else:
            latents = torch.stack(latents.split(self._image_token_sizes, dim=1), dim=0)
        timesteps = sigmas_full.unsqueeze(0)

        if self._traj_log_probs:
            stacked = torch.stack(self._traj_log_probs, dim=0)
            log_probs = stacked.reshape(1, -1) if self._image_token_sizes is None else stacked.t().contiguous()
        else:
            batch = 1 if self._image_token_sizes is None else len(self._image_token_sizes)
            log_probs = latents.new_zeros((batch, 0), dtype=torch.float32)

        return latents, sigmas_full.to(latents.device), timesteps, log_probs


__all__ = ["BagelFlowSDEScheduler", "BagelSDEStepOutput"]
