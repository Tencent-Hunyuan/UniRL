"""SDE kernel registry for shared transition math."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Base class hierarchy
# ---------------------------------------------------------------------------


class StepStrategy(ABC):
    """Base class for all sampling step strategies (SDE and ODE solvers)."""

    @abstractmethod
    def step(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 0.99,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one sampling step."""

    def reset(self) -> None:
        """Reset internal state (no-op for stateless strategies)."""
        pass

    def init_schedule(self, sigmas: torch.Tensor) -> None:
        """Set full sigma schedule (no-op for stateless strategies)."""
        pass

    def denoise(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        *,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run one denoising transition. Returns ``(prev_sample, log_prob, prev_sample_mean)``.

        ``prev_sample=None`` ⇒ sampling; otherwise log-prob replay. ``log_prob``
        is ``None`` for ODE strategies and for SDE strategies with ``eta<1e-7``.
        """
        input_dtype = sample.dtype
        noise_pred = noise_pred.float()
        sample = sample.float()
        if prev_sample is not None:
            prev_sample = prev_sample.float()
        # Ensure sigma/sigma_next are float32 to match sglang's explicit
        # `sigma = self.sigmas[step_indices].to(sample.device).to(sample.dtype)`.
        # Without this, sigma may arrive as float64 (torch.linspace default),
        # causing prev_sample_mean / std_var to compute in float64 while sglang
        # uses float32 — a systematic precision mismatch amplified by 1/(2σ²).
        sigma = sigma.float()
        sigma_next = sigma_next.float()

        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma_next.dim() == 0:
            sigma_next = sigma_next.unsqueeze(0)
        while sigma.dim() < sample.dim():
            sigma = sigma.unsqueeze(-1)
            sigma_next = sigma_next.unsqueeze(-1)

        prev_out, prev_mean, std_var = self.step(
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=eta,
            prev_sample=prev_sample,
            generator=None,  # DONOT PASS GENERATOR HERE - It will hurt diversity and performance
            sigma_max=sigma_max,
            step_index=step_index,
        )
        prev_out, log_prob = self._finalize_logp(
            prev_sample=prev_out,
            prev_sample_mean=prev_mean,
            std_var=std_var,
            eta=eta,
            input_dtype=input_dtype,
        )
        return prev_out, log_prob, prev_mean

    def _finalize_logp(
        self,
        *,
        prev_sample: torch.Tensor,
        prev_sample_mean: Optional[torch.Tensor],
        std_var: Optional[torch.Tensor],
        eta: float,
        input_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Default ODE finalize: no log_prob, no quantization."""
        del prev_sample_mean, std_var, eta, input_dtype
        return prev_sample, None


class SDEStrategy(StepStrategy, ABC):
    """Base class for SDE log probability computation strategies.

    Subclasses implement ``step_with_log_prob()`` which is the **single source
    of truth** for the SDE transition math.  It handles both:

    * **Sampling** (``prev_sample=None``): generates noise, returns new sample
      with log probability evaluated on the (optionally dtype-quantised) result.
    * **Training replay** (``prev_sample`` provided): computes log probability
      of the given transition without generating noise.

    This mirrors Flow-Factory's unified ``scheduler.step()`` pattern where
    ``next_latents is None`` distinguishes the two modes.
    """

    @abstractmethod
    def compute_log_prob(
        self,
        prev_sample: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run elementwise log probability calculation."""

    @abstractmethod
    def step(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 0.99,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Implement StepStrategy.step by delegating to step_with_log_prob or Euler ODE."""

    @abstractmethod
    def _std_dev_t(
        self,
        *,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float,
        sigma_max: float = 0.99,
    ) -> torch.Tensor:
        """Per-step diffusion coefficient ``std_dev_t`` for this SDE.

        Pure function of the schedule + ``eta`` (independent of the model
        output), so it is the single source shared by :meth:`step` (drift /
        noise scaling) and :meth:`transition_std` (KL / log-prob std).
        """

    def transition_std(
        self,
        *,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float,
        sigma_max: float = 0.99,
    ) -> torch.Tensor:
        """Std of the per-step transition Gaussian ``N(mean, std**2)``.

        Equals the ``std_var`` returned by :meth:`step` and used in
        :meth:`compute_log_prob`, and is the correct normalizer for the FlowDPPO
        KL ``(delta_mean)**2 / (2 * std**2)``. Default (Flow / Dance):
        ``std_dev_t * sqrt(-dt)``. CPS overrides it (its noise carries no
        ``sqrt(-dt)`` factor).
        """
        dt = sigma_next - sigma
        std_dev_t = self._std_dev_t(sigma=sigma, sigma_next=sigma_next, eta=eta, sigma_max=sigma_max)
        return std_dev_t * torch.sqrt(-dt)

    def _finalize_logp(
        self,
        *,
        prev_sample: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        std_var: torch.Tensor,
        eta: float,
        input_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """SDE finalize: dtype round-trip on ``prev_sample`` then per-sample log_prob.

        The dtype round-trip simulates trajectory storage precision so replay-time
        log_prob matches sampling-time precision. Skipped for ``eta<1e-7``.
        """
        if eta < 1e-7:
            return prev_sample, None
        prev_sample = prev_sample.to(dtype=input_dtype).float()
        log_prob = self.compute_log_prob(
            prev_sample=prev_sample,
            prev_sample_mean=prev_sample_mean,
            std_var=std_var,
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return prev_sample, log_prob


# ---------------------------------------------------------------------------
# SDE strategy implementations
# ---------------------------------------------------------------------------


class FlowSDEStrategy(SDEStrategy):
    """Standard SDE formulation from FlowGRPO."""

    canonical_name: ClassVar[str] = "flow"

    def __init__(self, *, config: Optional["FlowSpec"] = None) -> None:
        del config  # empty Spec — strategy has no per-instance fields

    def compute_log_prob(
        self,
        prev_sample: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        std_var: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_var**2)
            - torch.log(std_var)
            - 0.5 * math.log(2 * math.pi)
        )
        return log_prob

    def _std_dev_t(
        self,
        *,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float,
        sigma_max: float = 0.99,
    ) -> torch.Tensor:
        return torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta

    def step(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 0.99,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        device = noise_pred.device
        dt = sigma_next - sigma
        std_dev_t = self._std_dev_t(sigma=sigma, sigma_next=sigma_next, eta=eta, sigma_max=sigma_max)

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        if prev_sample is None:
            noise = randn_tensor(noise_pred.shape, generator=generator, device=device, dtype=noise_pred.dtype)
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-dt) * noise

        std_var = std_dev_t * torch.sqrt(-dt)

        return prev_sample, prev_sample_mean, std_var


class CPSSDEStrategy(SDEStrategy):
    """Coefficient-preserving sampling."""

    canonical_name: ClassVar[str] = "cps"

    def __init__(self, *, config: Optional["CPSSpec"] = None) -> None:
        del config

    def compute_log_prob(
        self,
        prev_sample: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return -((prev_sample.detach() - prev_sample_mean) ** 2)

    def _std_dev_t(
        self,
        *,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float,
        sigma_max: float = 0.99,
    ) -> torch.Tensor:
        return sigma_next * math.sin(eta * math.pi / 2)

    def transition_std(
        self,
        *,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float,
        sigma_max: float = 0.99,
    ) -> torch.Tensor:
        # CPS adds noise as std_dev_t * noise (no sqrt(-dt)), so the transition
        # Gaussian std IS std_dev_t -- the KL must not multiply by sqrt(-dt).
        return self._std_dev_t(sigma=sigma, sigma_next=sigma_next, eta=eta, sigma_max=sigma_max)

    def step(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 0.99,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        device = noise_pred.device
        std_dev_t = self._std_dev_t(sigma=sigma, sigma_next=sigma_next, eta=eta, sigma_max=sigma_max)
        pred_original = sample - sigma * noise_pred
        noise_estimate = sample + noise_pred * (1 - sigma)
        prev_sample_mean = pred_original * (1 - sigma_next) + noise_estimate * torch.sqrt(sigma_next**2 - std_dev_t**2)

        if prev_sample is None:
            noise = randn_tensor(noise_pred.shape, generator=generator, device=device, dtype=noise_pred.dtype)
            prev_sample = prev_sample_mean + std_dev_t * noise

        return prev_sample, prev_sample_mean, std_dev_t


class DanceSDEStrategy(SDEStrategy):
    """DanceGRPO SDE formulation."""

    canonical_name: ClassVar[str] = "dance"

    def __init__(self, *, config: Optional["DanceSpec"] = None) -> None:
        del config

    def compute_log_prob(
        self,
        prev_sample: torch.Tensor,
        prev_sample_mean: torch.Tensor,
        std_var: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_var**2)
            - torch.log(std_var)
            - 0.5 * math.log(2 * math.pi)
        )
        return log_prob

    def _std_dev_t(
        self,
        *,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float,
        sigma_max: float = 0.99,
    ) -> torch.Tensor:
        return torch.full_like(sigma, float(eta))

    def step(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 0.99,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        device = noise_pred.device
        dt = sigma_next - sigma
        std_dev_t = self._std_dev_t(sigma=sigma, sigma_next=sigma_next, eta=eta, sigma_max=sigma_max)

        prev_sample_mean = (
            sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
            + noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
        )

        if prev_sample is None:
            noise = randn_tensor(noise_pred.shape, generator=generator, device=device, dtype=noise_pred.dtype)
            prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-dt) * noise

        std_var = std_dev_t * torch.sqrt(-dt)

        return prev_sample, prev_sample_mean, std_var


# ---------------------------------------------------------------------------
# DPM2 deterministic ODE strategy (migrated from sd3_sampler.py)
# ---------------------------------------------------------------------------


@dataclass
class _DPMState:
    order: int
    model_outputs: List[Optional[torch.Tensor]] = None
    lower_order_nums: int = 0

    def __post_init__(self) -> None:
        self.model_outputs = [None] * self.order

    def update(self, model_output: torch.Tensor) -> None:
        for i in range(self.order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

    def update_lower_order(self) -> None:
        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1


def _convert_model_output(
    model_output: torch.Tensor, sample: torch.Tensor, sigmas: torch.Tensor, step_index: int
) -> torch.Tensor:
    compute_device = model_output.device
    if sample.device != compute_device:
        sample = sample.to(compute_device)
    sigma_t = sigmas[step_index].to(device=compute_device, dtype=model_output.dtype)
    x0_pred = sample - sigma_t * model_output
    return x0_pred


def _sigma_to_alpha_sigma_t(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha_t = 1 - sigma
    sigma_t = sigma
    return alpha_t, sigma_t


def _dpm_solver_first_order_update(
    model_output: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    sigma_t, sigma_s = sigmas[step_index + 1], sigmas[step_index]
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = _sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

    h = lambda_t - lambda_s
    x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
    return x_t


def _multistep_dpm_solver_second_order_update(
    model_output_list: List[torch.Tensor],
    sigmas: torch.Tensor,
    step_index: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    sigma_t, sigma_s0, sigma_s1 = (
        sigmas[step_index + 1],
        sigmas[step_index],
        sigmas[step_index - 1],
    )

    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = _sigma_to_alpha_sigma_t(sigma_s1)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

    m0, m1 = model_output_list[-1], model_output_list[-2]

    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)

    x_t = (
        (sigma_t / sigma_s0) * sample
        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
    )
    return x_t


def _dpm_step(
    order: int,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    step_index: int,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    dpm_state: _DPMState,
) -> torch.Tensor:
    lower_order_final = step_index == len(timesteps) - 1
    lower_order_second = (step_index == len(timesteps) - 2) and len(timesteps) < 15

    model_output = _convert_model_output(model_output, sample, sigmas, step_index=step_index)
    dpm_state.update(model_output)

    sample = sample.to(device=model_output.device, dtype=torch.float32)
    local_sigmas = sigmas.to(device=sample.device, dtype=sigmas.dtype)

    if order == 1 or dpm_state.lower_order_nums < 1 or lower_order_final:
        if step_index == 0 or lower_order_final:
            # DDIM update with eta=0
            t, s = local_sigmas[step_index + 1], local_sigmas[step_index]
            noise_pred = (sample - (1 - s) * model_output) / s
            prev_mean = (1 - t) * model_output + torch.sqrt(t**2) * noise_pred
            prev_sample = prev_mean
        else:
            prev_sample = _dpm_solver_first_order_update(
                model_output,
                local_sigmas.to(dtype=torch.float64),
                step_index,
                sample,
            )
    elif order == 2 or dpm_state.lower_order_nums < 2 or lower_order_second:
        prev_sample = _multistep_dpm_solver_second_order_update(
            dpm_state.model_outputs,
            local_sigmas.to(dtype=torch.float64),
            step_index,
            sample,
        )
    else:
        raise ValueError(f"Unsupported DPM order: {order}")

    dpm_state.update_lower_order()
    return prev_sample.to(model_output.dtype)


class DPM2Strategy(StepStrategy):
    """DPM-Solver-2 multi-step ODE strategy (deterministic, stateful)."""

    canonical_name: ClassVar[str] = "dpm2"

    def __init__(self, *, config: Optional["DPM2Spec"] = None) -> None:
        del config
        self._state = _DPMState(order=2)
        self._sigmas: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._state = _DPMState(order=2)
        self._sigmas = None

    def init_schedule(self, sigmas: torch.Tensor) -> None:
        self._sigmas = sigmas
        self._state = _DPMState(order=2)

    def step(
        self,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        eta: float = 1.0,
        prev_sample: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        sigma_max: float = 0.99,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._sigmas is None:
            raise RuntimeError("DPM2Strategy requires init_schedule() before stepping.")
        result = _dpm_step(
            order=2,
            model_output=noise_pred,
            sample=sample,
            step_index=step_index,
            timesteps=self._sigmas[:-1],
            sigmas=self._sigmas,
            dpm_state=self._state,
        )
        return result, None, None


@dataclass
class FlowSpec:
    """Empty Spec: FlowSDEStrategy has no per-strategy config fields."""


@dataclass
class CPSSpec:
    """Empty Spec: CPSSDEStrategy has no per-strategy config fields."""


@dataclass
class DanceSpec:
    """Empty Spec: DanceSDEStrategy has no per-strategy config fields."""


@dataclass
class DPM2Spec:
    """Empty Spec: DPM2Strategy has no per-strategy config fields."""


__all__ = [
    "StepStrategy",
    "SDEStrategy",
    "FlowSDEStrategy",
    "CPSSDEStrategy",
    "DanceSDEStrategy",
    "DPM2Strategy",
    "FlowSpec",
    "CPSSpec",
    "DanceSpec",
    "DPM2Spec",
]
