# SPDX-License-Identifier: Apache-2.0
# Adapted from FastVideo's FlowUniPCMultistepScheduler, which in turn is based
# on Hugging Face Diffusers v0.31.0's UniPCMultistepScheduler.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for UniRL to separate solver state from schedule construction.

"""Canonical UniPC solver for flow-matching trajectories.

The implementation follows the FlowUniPC scheduler used by Wan/FastVideo, but
owns only solver state. Sigma construction remains the responsibility of
``FlowMatchSchedulePolicy`` and is supplied through ``init_schedule``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch

from unirl.sde.kernels import StepStrategy


@dataclass
class UniPCSpec:
    """Configuration for :class:`UniPCStrategy`."""

    solver_order: int = 2
    solver_type: str = "bh2"
    lower_order_final: bool = True
    disable_corrector: Tuple[int, ...] = ()


class UniPCStrategy(StepStrategy):
    """Stateful UniPC predictor/corrector for flow-prediction models.

    This strategy deliberately does not construct timesteps or sigmas. Call
    :meth:`init_schedule` with the canonical ``T + 1`` sigma schedule before
    stepping. Gaps in ``step_index`` clear multistep history automatically,
    which is required when an SDE transition interrupts the deterministic ODE
    path.
    """

    canonical_name: ClassVar[str] = "unipc"

    def __init__(self, *, config: Optional[UniPCSpec] = None) -> None:
        config = config or UniPCSpec()
        self._order = int(config.solver_order)
        self._solver_type = str(config.solver_type)
        self._lower_order_final = bool(config.lower_order_final)
        self._disable_corrector = {int(i) for i in config.disable_corrector}
        if self._order < 1:
            raise ValueError(f"UniPC solver_order must be >= 1; got {self._order}")
        if self._solver_type not in {"bh1", "bh2"}:
            raise ValueError(f"UniPC solver_type must be 'bh1' or 'bh2'; got {self._solver_type!r}")

        self._sigmas: Optional[torch.Tensor] = None
        self._model_outputs: List[Optional[torch.Tensor]] = []
        self._lower_order_nums = 0
        self._last_sample: Optional[torch.Tensor] = None
        self._last_order = 0
        self._last_step_index: Optional[int] = None
        self.reset_history()

    def reset_history(self) -> None:
        """Clear solver history while retaining the initialized schedule."""
        self._model_outputs = [None] * self._order
        self._lower_order_nums = 0
        self._last_sample = None
        self._last_order = 0
        self._last_step_index = None

    def reset(self) -> None:
        self._sigmas = None
        self.reset_history()

    def init_schedule(self, sigmas: torch.Tensor) -> None:
        if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or int(sigmas.shape[0]) < 2:
            raise ValueError("UniPCStrategy.init_schedule requires a one-dimensional T+1 sigma tensor")
        schedule = sigmas.detach()
        if not bool(torch.isfinite(schedule).all()):
            raise ValueError("UniPCStrategy.init_schedule requires finite sigmas")
        if not bool(((schedule >= 0) & (schedule <= 1)).all()):
            raise ValueError("UniPCStrategy.init_schedule requires sigmas normalized to [0, 1]")
        if not bool((schedule[:-1] > schedule[1:]).all()):
            raise ValueError("UniPCStrategy.init_schedule requires strictly decreasing sigmas")
        self._sigmas = schedule
        self.reset_history()

    @staticmethod
    def _alpha_sigma(sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma

    @staticmethod
    def _lambda(alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        return torch.log(torch.clamp(alpha, min=eps)) - torch.log(torch.clamp(sigma, min=eps))

    def _convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        assert self._sigmas is not None
        sigma = self._sigmas[step_index].to(device=model_output.device, dtype=model_output.dtype)
        if sample.device != model_output.device:
            sample = sample.to(model_output.device)
        return sample - sigma * model_output

    @staticmethod
    def _as_scalar_vector(
        values: List[object],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.stack([torch.as_tensor(value, device=device, dtype=dtype) for value in values])

    def _bh_coefficient(self, hh: torch.Tensor) -> torch.Tensor:
        if self._solver_type == "bh1":
            return hh
        return torch.expm1(hh)

    def _predictor(
        self,
        *,
        sample: torch.Tensor,
        step_index: int,
        order: int,
    ) -> torch.Tensor:
        assert self._sigmas is not None
        m0 = self._model_outputs[-1]
        assert m0 is not None

        sigma_t = self._sigmas[step_index + 1].to(device=sample.device, dtype=sample.dtype)
        sigma_s0 = self._sigmas[step_index].to(device=sample.device, dtype=sample.dtype)
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = self._lambda(alpha_t, sigma_t)
        lambda_s0 = self._lambda(alpha_s0, sigma_s0)
        h = lambda_t - lambda_s0
        device = sample.device

        rks: List[object] = []
        d1s: List[torch.Tensor] = []
        for i in range(1, order):
            history_index = step_index - i
            mi = self._model_outputs[-(i + 1)]
            assert mi is not None
            history_sigma = self._sigmas[history_index].to(device=device, dtype=sample.dtype)
            alpha_si, sigma_si = self._alpha_sigma(history_sigma)
            lambda_si = self._lambda(alpha_si, sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks_t = self._as_scalar_vector(rks, device=device, dtype=sample.dtype)

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        b_values: List[torch.Tensor] = []
        r_values: List[torch.Tensor] = []
        factorial_i = 1
        b_h = self._bh_coefficient(hh)
        for i in range(1, order + 1):
            r_values.append(torch.pow(rks_t, i - 1))
            b_values.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        r_matrix = torch.stack(r_values)
        b = torch.stack(b_values)
        d1_stack = torch.stack(d1s, dim=1) if d1s else None
        if d1_stack is None:
            pred_res: object = 0
        else:
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=sample.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(r_matrix[:-1, :-1], b[:-1]).to(device=device, dtype=sample.dtype)
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1_stack)

        result = sigma_t / sigma_s0 * sample - alpha_t * h_phi_1 * m0
        result = result - alpha_t * b_h * pred_res
        return result.to(sample.dtype)

    def _corrector(
        self,
        *,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        step_index: int,
        order: int,
    ) -> torch.Tensor:
        assert self._sigmas is not None
        m0 = self._model_outputs[-1]
        assert m0 is not None

        sigma_t = self._sigmas[step_index].to(device=this_sample.device, dtype=this_sample.dtype)
        sigma_s0 = self._sigmas[step_index - 1].to(device=this_sample.device, dtype=this_sample.dtype)
        alpha_t, sigma_t = self._alpha_sigma(sigma_t)
        alpha_s0, sigma_s0 = self._alpha_sigma(sigma_s0)
        lambda_t = self._lambda(alpha_t, sigma_t)
        lambda_s0 = self._lambda(alpha_s0, sigma_s0)
        h = lambda_t - lambda_s0
        device = this_sample.device

        rks: List[object] = []
        d1s: List[torch.Tensor] = []
        for i in range(1, order):
            history_index = step_index - (i + 1)
            mi = self._model_outputs[-(i + 1)]
            assert mi is not None
            history_sigma = self._sigmas[history_index].to(device=device, dtype=this_sample.dtype)
            alpha_si, sigma_si = self._alpha_sigma(history_sigma)
            lambda_si = self._lambda(alpha_si, sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks_t = self._as_scalar_vector(rks, device=device, dtype=this_sample.dtype)

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        b_values: List[torch.Tensor] = []
        r_values: List[torch.Tensor] = []
        factorial_i = 1
        b_h = self._bh_coefficient(hh)
        for i in range(1, order + 1):
            r_values.append(torch.pow(rks_t, i - 1))
            b_values.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        r_matrix = torch.stack(r_values)
        b = torch.stack(b_values)
        d1_stack = torch.stack(d1s, dim=1) if d1s else None
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=last_sample.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(r_matrix, b).to(device=device, dtype=last_sample.dtype)

        result = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
        corr_res: object = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1_stack) if d1_stack is not None else 0
        d1_t = this_model_output - m0
        result = result - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        return result.to(last_sample.dtype)

    def _push_model_output(self, model_output: torch.Tensor) -> None:
        for i in range(self._order - 1):
            self._model_outputs[i] = self._model_outputs[i + 1]
        self._model_outputs[-1] = model_output

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
        del sigma, sigma_next, eta, prev_sample, generator, sigma_max
        if self._sigmas is None:
            raise RuntimeError("UniPCStrategy requires init_schedule() before stepping.")
        num_steps = int(self._sigmas.shape[0]) - 1
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"UniPC step_index={step_index} outside schedule with {num_steps} steps")

        if self._last_step_index is not None and step_index != self._last_step_index + 1:
            self.reset_history()

        converted = self._convert_model_output(noise_pred, sample, step_index)
        use_corrector = (
            step_index > 0
            and step_index - 1 not in self._disable_corrector
            and self._last_sample is not None
            and self._last_order > 0
        )
        if use_corrector:
            sample = self._corrector(
                this_model_output=converted,
                last_sample=self._last_sample,
                this_sample=sample,
                step_index=step_index,
                order=self._last_order,
            )

        self._push_model_output(converted)
        remaining_order = min(self._order, num_steps - step_index) if self._lower_order_final else self._order
        this_order = min(remaining_order, self._lower_order_nums + 1)
        if this_order < 1:
            raise RuntimeError(f"UniPC computed invalid solver order {this_order} at step {step_index}")

        self._last_sample = sample
        result = self._predictor(sample=sample, step_index=step_index, order=this_order)
        if self._lower_order_nums < self._order:
            self._lower_order_nums += 1
        self._last_order = this_order
        self._last_step_index = step_index
        return result, None, None


__all__ = ["UniPCSpec", "UniPCStrategy"]
