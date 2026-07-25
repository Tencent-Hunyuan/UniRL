"""Advantage computation helpers for generated parts.

GRPO-style group normalization lives on :meth:`Part.compute_advantages` in
:mod:`unirl.types.sample`. GAE and other per-step estimators live here as pure
tensor utilities consumed by trainers before the train step.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns from step rewards and value predictions.

    For each step ``t`` (backward in time):

        δ_t = r_t + γ V_{t+1} - V_t
        A_t = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...

    After the last valid step, ``V_{T}`` bootstraps with zero (episodic terminal).

    Args:
        rewards: Step rewards ``[T]`` or ``[B, T]``.
        values: Critic predictions ``V_t``, same shape as ``rewards``.
        gamma: Discount factor γ.
        gae_lambda: GAE smoothing λ.
        mask: Optional validity mask (1 = include, 0 = padding / ignore).
            When provided, GAE does not carry across zero-mask positions.

    Returns:
        ``(advantages, returns)`` with the same shape as ``rewards``.
        ``returns = advantages + values`` (standard GAE-return convention).

    Raises:
        ValueError: On shape mismatch or invalid hyperparameters.

    Reference: Schulman et al., "High-Dimensional Continuous Control Using
    Generalized Advantage Estimation" (2016).
    """
    if rewards.shape != values.shape:
        raise ValueError(
            f"compute_gae_advantages: rewards shape {tuple(rewards.shape)} != values shape {tuple(values.shape)}"
        )
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"compute_gae_advantages: gamma must be in [0, 1], got {gamma}")
    if not (0.0 <= gae_lambda <= 1.0):
        raise ValueError(f"compute_gae_advantages: gae_lambda must be in [0, 1], got {gae_lambda}")
    if mask is not None and mask.shape != rewards.shape:
        raise ValueError(
            f"compute_gae_advantages: mask shape {tuple(mask.shape)} != rewards shape {tuple(rewards.shape)}"
        )

    if rewards.ndim == 1:
        advantages = _gae_1d(rewards, values, gamma=gamma, gae_lambda=gae_lambda, mask=mask)
    elif rewards.ndim == 2:
        advantages = _gae_2d(rewards, values, gamma=gamma, gae_lambda=gae_lambda, mask=mask)
    else:
        raise ValueError(f"compute_gae_advantages: expected 1D or 2D tensors, got ndim={rewards.ndim}")

    returns = advantages + values
    return advantages, returns


def scatter_terminal_rewards(
    rewards_per_sample: torch.Tensor,
    *,
    lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Place each sample's scalar reward on its last response token in packed layout.

    Args:
        rewards_per_sample: Per-trajectory rewards ``[B]``.
        lengths: Response token counts per sample ``[B]``.
        cu_seqlens: Packed cumulative offsets ``[B + 1]`` (``TextSegment.cu_seqlens``).

    Returns:
        Packed per-token rewards ``[total_tokens]`` (zero except terminal positions).
    """
    if rewards_per_sample.ndim != 1:
        raise ValueError(
            f"scatter_terminal_rewards: rewards_per_sample must be 1D, got shape {tuple(rewards_per_sample.shape)}"
        )
    batch_size = int(lengths.shape[0])
    if int(rewards_per_sample.shape[0]) != batch_size:
        raise ValueError(
            f"scatter_terminal_rewards: rewards batch ({int(rewards_per_sample.shape[0])}) "
            f"!= lengths batch ({batch_size})"
        )
    if int(cu_seqlens.shape[0]) != batch_size + 1:
        raise ValueError(
            f"scatter_terminal_rewards: cu_seqlens length ({int(cu_seqlens.shape[0])}) "
            f"!= batch_size + 1 ({batch_size + 1})"
        )
    total = int(cu_seqlens[-1].item())
    out = rewards_per_sample.new_zeros(total)
    cu = [int(c) for c in cu_seqlens.tolist()]
    for b in range(batch_size):
        n = int(lengths[b].item())
        if n <= 0:
            continue
        out[cu[b] + n - 1] = rewards_per_sample[b]
    return out


def _gae_1d(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    t_steps = int(rewards.shape[0])
    next_values = torch.cat([values[1:], values.new_zeros(1)])
    if mask is not None:
        valid = mask.to(dtype=values.dtype)
        next_valid = torch.cat([valid[1:], valid.new_zeros(1)])
        next_values = next_values * next_valid
    deltas = rewards + gamma * next_values - values

    advantages = rewards.new_zeros(t_steps)
    gae = rewards.new_zeros(())
    for t in range(t_steps - 1, -1, -1):
        gae = deltas[t] + gamma * gae_lambda * gae
        if mask is not None:
            gae = gae * mask[t].to(dtype=gae.dtype)
        advantages[t] = gae
    return advantages


def _gae_2d(
    rewards: torch.Tensor,
    values: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    batch, t_steps = rewards.shape
    next_values = torch.cat([values[:, 1:], values.new_zeros(batch, 1)], dim=1)
    if mask is not None:
        valid = mask.to(dtype=values.dtype)
        next_valid = torch.cat([valid[:, 1:], valid.new_zeros(batch, 1)], dim=1)
        next_values = next_values * next_valid
    deltas = rewards + gamma * next_values - values

    advantages = rewards.new_zeros(batch, t_steps)
    gae = rewards.new_zeros(batch)
    for t in range(t_steps - 1, -1, -1):
        if mask is not None:
            step_mask = mask[:, t].to(dtype=deltas.dtype)
            gae = deltas[:, t] + gamma * gae_lambda * gae
            gae = gae * step_mask
        else:
            gae = deltas[:, t] + gamma * gae_lambda * gae
        advantages[:, t] = gae
    return advantages
