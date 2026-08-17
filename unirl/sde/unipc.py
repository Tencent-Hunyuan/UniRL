# SPDX-License-Identifier: Apache-2.0
# Adapted from FastVideo's FlowUniPCMultistepScheduler, which in turn is based
# on Hugging Face Diffusers v0.31.0's UniPCMultistepScheduler.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified for UniRL to separate solver state from schedule construction.

"""Canonical UniPC predictor-corrector for flow-matching trajectories; solver contract in README.md (UniPC bullets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch

from unirl.sde.kernels import StepStrategy, _convert_model_output, _DPMState, _sigma_to_alpha_sigma_t


def _clamped_lambda(alpha: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Half-log-SNR with the pinned FastVideo fork's 1e-12 numerical-safety clamp."""
    eps = 1e-12
    return torch.log(torch.clamp(alpha, min=eps)) - torch.log(torch.clamp(sigma, min=eps))


@dataclass
class UniPCSpec:
    """Model-owned UniPC solver contract; engine adapters verify the checkpoint scheduler against it."""

    solver_order: int = 2
    solver_type: str = "bh2"
    lower_order_final: bool = True
    disable_corrector: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.solver_order = int(self.solver_order)
        self.solver_type = str(self.solver_type)
        self.lower_order_final = bool(self.lower_order_final)
        self.disable_corrector = tuple(int(i) for i in self.disable_corrector)
        if self.solver_order < 1:
            raise ValueError(f"UniPC solver_order must be >= 1; got {self.solver_order}")
        if self.solver_type not in {"bh1", "bh2"}:
            raise ValueError(f"UniPC solver_type must be 'bh1' or 'bh2'; got {self.solver_type!r}")


class UniPCStrategy(StepStrategy):
    """Stateful UniPC solver over an ``init_schedule``-pinned sigma tensor; step-index gaps clear history."""

    canonical_name: ClassVar[str] = "unipc"

    def __init__(self, *, config: Optional[UniPCSpec] = None) -> None:
        spec = config or UniPCSpec()
        self._order = spec.solver_order
        self._solver_type = spec.solver_type
        self._lower_order_final = spec.lower_order_final
        self._disable_corrector = set(spec.disable_corrector)
        self._sigmas: Optional[torch.Tensor] = None
        self._local_sigmas: Optional[torch.Tensor] = None  # device/dtype cache of _sigmas
        self.reset_history()

    def reset_history(self) -> None:
        """Clear solver history while retaining the initialized schedule."""
        self._state = _DPMState(order=self._order)
        self._last_sample: Optional[torch.Tensor] = None
        self._last_order = 0
        self._last_step_index: Optional[int] = None

    def reset(self) -> None:
        self._sigmas = None
        self._local_sigmas = None
        self.reset_history()

    def init_schedule(self, sigmas: torch.Tensor) -> None:
        if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or int(sigmas.shape[0]) < 2:
            raise ValueError("UniPCStrategy.init_schedule requires a one-dimensional T+1 sigma tensor")
        self._sigmas = sigmas.detach()
        self._local_sigmas = None
        self.reset_history()

    def _schedule_on(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        assert self._sigmas is not None
        cached = self._local_sigmas
        if cached is None or cached.device != device or cached.dtype != dtype:
            cached = self._sigmas.to(device=device, dtype=dtype)
            self._local_sigmas = cached
        return cached

    def _solver_coefficients(
        self,
        *,
        anchor: int,
        order: int,
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Assemble the schedule-derived UniPC terms shared by predictor (anchor=step) and corrector (anchor=step-1)."""
        sigmas = self._schedule_on(ref.device, ref.dtype)
        m0 = self._state.model_outputs[-1]
        assert m0 is not None

        alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigmas[anchor + 1])
        alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigmas[anchor])
        lambda_t = _clamped_lambda(alpha_t, sigma_t)
        lambda_s0 = _clamped_lambda(alpha_s0, sigma_s0)
        h = lambda_t - lambda_s0

        rks: List[object] = []
        d1s: List[torch.Tensor] = []
        for i in range(1, order):
            mi = self._state.model_outputs[-(i + 1)]
            assert mi is not None
            alpha_si, sigma_si = _sigma_to_alpha_sigma_t(sigmas[anchor - i])
            lambda_si = _clamped_lambda(alpha_si, sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk)
        rks.append(1.0)
        rks_t = torch.stack([torch.as_tensor(value, device=ref.device, dtype=ref.dtype) for value in rks])

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        b_h = hh if self._solver_type == "bh1" else torch.expm1(hh)
        r_rows: List[torch.Tensor] = []
        b_values: List[torch.Tensor] = []
        factorial_i = 1
        for i in range(1, order + 1):
            r_rows.append(torch.pow(rks_t, i - 1))
            b_values.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i
        r_matrix = torch.stack(r_rows)
        b = torch.stack(b_values)
        d1_stack = torch.stack(d1s, dim=1) if d1s else None
        return sigma_t, sigma_s0, alpha_t, h_phi_1, b_h, r_matrix, b, d1_stack, m0

    def _predictor(self, *, sample: torch.Tensor, step_index: int, order: int) -> torch.Tensor:
        sigma_t, sigma_s0, alpha_t, h_phi_1, b_h, r_matrix, b, d1_stack, m0 = self._solver_coefficients(
            anchor=step_index, order=order, ref=sample
        )
        if d1_stack is None:
            pred_res: object = 0
        else:
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=sample.dtype, device=sample.device)
            else:
                rhos_p = torch.linalg.solve(r_matrix[:-1, :-1], b[:-1]).to(device=sample.device, dtype=sample.dtype)
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
        sigma_t, sigma_s0, alpha_t, h_phi_1, b_h, r_matrix, b, d1_stack, m0 = self._solver_coefficients(
            anchor=step_index - 1, order=order, ref=this_sample
        )
        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=last_sample.dtype, device=this_sample.device)
        else:
            rhos_c = torch.linalg.solve(r_matrix, b).to(device=this_sample.device, dtype=last_sample.dtype)

        result = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
        corr_res: object = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1_stack) if d1_stack is not None else 0
        d1_t = this_model_output - m0
        result = result - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        return result.to(last_sample.dtype)

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
        """One deterministic UniPC transition at ``step_index``; sigma args are unused (schedule-owned)."""
        del sigma, sigma_next, eta, prev_sample, generator, sigma_max
        if self._sigmas is None:
            raise RuntimeError("UniPCStrategy requires init_schedule() before stepping.")
        num_steps = int(self._sigmas.shape[0]) - 1
        if step_index < 0 or step_index >= num_steps:
            raise IndexError(f"UniPC step_index={step_index} outside schedule with {num_steps} steps")

        if self._last_step_index is not None and step_index != self._last_step_index + 1:
            self.reset_history()

        converted = _convert_model_output(noise_pred, sample, self._sigmas, step_index=step_index)
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

        self._state.update(converted)
        remaining_order = min(self._order, num_steps - step_index) if self._lower_order_final else self._order
        this_order = min(remaining_order, self._state.lower_order_nums + 1)
        if this_order < 1:
            raise RuntimeError(f"UniPC computed invalid solver order {this_order} at step {step_index}")

        self._last_sample = sample
        result = self._predictor(sample=sample, step_index=step_index, order=this_order)
        self._state.update_lower_order()
        self._last_order = this_order
        self._last_step_index = step_index
        if step_index == num_steps - 1:
            self.reset_history()  # final step: release retained latents; the schedule stays pinned
        return result, None, None


__all__ = ["UniPCSpec", "UniPCStrategy"]
