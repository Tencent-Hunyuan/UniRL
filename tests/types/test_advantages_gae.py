"""CPU unit tests for GAE advantage computation."""

from __future__ import annotations

import math

import pytest
import torch

from unirl.types.advantages import compute_gae_advantages


def test_gae_hand_computed_lambda_one() -> None:
    """Sparse terminal reward; λ=1; hand-computed backward pass."""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.2, 0.5, 0.8])
    advantages, returns = compute_gae_advantages(rewards, values, gamma=1.0, gae_lambda=1.0)
    # δ = [0.3, 0.3, 0.2]; backward GAE with λ=1 → [0.8, 0.5, 0.2]
    expected_adv = torch.tensor([0.8, 0.5, 0.2])
    assert torch.allclose(advantages, expected_adv, atol=1e-6)
    assert torch.allclose(returns, expected_adv + values, atol=1e-6)


def test_gae_lambda_zero_td_only() -> None:
    """λ=0 → advantages equal TD errors only."""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.2, 0.5, 0.8])
    advantages, _ = compute_gae_advantages(rewards, values, gamma=1.0, gae_lambda=0.0)
    expected = torch.tensor([0.3, 0.3, 0.2])
    assert torch.allclose(advantages, expected, atol=1e-6)


def test_gae_flat_critic_sparse_reward() -> None:
    """Perfect flat critic on sparse terminal reward → zero advantages."""
    rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
    values = torch.tensor([1.0, 1.0, 1.0, 1.0])
    advantages, _ = compute_gae_advantages(rewards, values, gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(advantages, torch.zeros(4), atol=1e-6)


def test_gae_uniform_reward_zero_advantage() -> None:
    """Constant rewards and matching finite-horizon values → zero TD errors."""
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([3.0, 2.0, 1.0])
    advantages, _ = compute_gae_advantages(rewards, values, gamma=1.0, gae_lambda=0.95)
    assert torch.allclose(advantages, torch.zeros(3), atol=1e-6)


def test_gae_batched_independent_rows() -> None:
    """Batched [B, T] matches row-wise 1D computation."""
    rewards = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    values = torch.tensor([[0.1, 0.4], [0.2, 0.5]])
    batched_adv, _ = compute_gae_advantages(rewards, values, gamma=1.0, gae_lambda=1.0)
    for b in range(2):
        row_adv, _ = compute_gae_advantages(rewards[b], values[b], gamma=1.0, gae_lambda=1.0)
        assert torch.allclose(batched_adv[b], row_adv, atol=1e-6)


def test_gae_mask_resets_carry() -> None:
    """Padding mask prevents GAE from crossing sequence boundaries."""
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    values = torch.tensor([[0.2, 0.5, 0.8, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    advantages, _ = compute_gae_advantages(rewards, values, gamma=1.0, gae_lambda=1.0, mask=mask)
    unmasked_adv, _ = compute_gae_advantages(rewards[0, :3], values[0, :3], gamma=1.0, gae_lambda=1.0)
    assert torch.allclose(advantages[0, :3], unmasked_adv, atol=1e-6)
    assert advantages[0, 3].item() == 0.0


@pytest.mark.parametrize("batched", [False, True])
def test_gae_mask_zeros_bootstrap_from_padding_values(batched: bool) -> None:
    """The last valid step is terminal even when padding contains nonzero values."""
    rewards = torch.tensor([0.0, 0.0, 0.0])
    values = torch.tensor([0.25, 123.0, 456.0])
    mask = torch.tensor([1.0, 0.0, 0.0])
    if batched:
        rewards = rewards.unsqueeze(0)
        values = values.unsqueeze(0)
        mask = mask.unsqueeze(0)

    advantages, _ = compute_gae_advantages(
        rewards,
        values,
        gamma=1.0,
        gae_lambda=1.0,
        mask=mask,
    )

    expected = torch.tensor([-0.25, 0.0, 0.0])
    if batched:
        expected = expected.unsqueeze(0)
    assert torch.allclose(advantages, expected, atol=1e-6)


def test_gae_gamma_discount() -> None:
    """γ < 1 discounts bootstrap from future values."""
    rewards = torch.tensor([0.0, 0.0, 1.0])
    values = torch.tensor([0.0, 0.0, 0.0])
    gamma = 0.9
    advantages, _ = compute_gae_advantages(rewards, values, gamma=gamma, gae_lambda=1.0)
    # δ2=1, δ1=0, δ0=0 → backward: A2=1, A1=0.9, A0=0.81
    expected = torch.tensor([gamma**2, gamma, 1.0])
    assert torch.allclose(advantages, expected, atol=1e-6)


def test_gae_shape_mismatch_raises() -> None:
    rewards = torch.tensor([0.0, 1.0])
    values = torch.tensor([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        compute_gae_advantages(rewards, values)


def test_gae_invalid_hyperparams_raise() -> None:
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.5])
    with pytest.raises(ValueError, match="gamma"):
        compute_gae_advantages(rewards, values, gamma=1.1)
    with pytest.raises(ValueError, match="gae_lambda"):
        compute_gae_advantages(rewards, values, gae_lambda=-0.1)


def test_gae_single_step() -> None:
    """Degenerate T=1: advantage = r + γ·0 - V."""
    rewards = torch.tensor([1.0])
    values = torch.tensor([0.25])
    advantages, returns = compute_gae_advantages(rewards, values, gamma=1.0, gae_lambda=0.95)
    assert math.isclose(float(advantages.item()), 0.75, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(float(returns.item()), 1.0, rel_tol=0, abs_tol=1e-6)
