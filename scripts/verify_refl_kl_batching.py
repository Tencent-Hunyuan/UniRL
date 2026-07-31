#!/usr/bin/env python3
"""Verify KL DP-batching semantics for the refl recipe (CPU, no Ray, no GPU).

Regression verification for the P1 review finding on PR #210: the original
``REFLGenerated.kl_loss`` was a per-shard *scalar* ``shared_field`` — DP
collect kept only rank 0's KL and re-broadcast it to every actor rank, which
happened to "work" only on the verified ``batch_size == actor_dp == 8``
topology (logs duplicated rank 0; other B/dp splits risked a hard shape
mismatch between the routed KL grad and each rank's saved scalar).

The fix makes ``kl_loss`` a batch-aligned per-sample ``[B]`` concat field.
This script pins the invariants at the exact wire layer DP dispatch uses
(``pytree_chunk`` / ``pytree_cat`` / ``infer_batch_size``) across the
topologies called out in review: B == dp, B > dp, non-power-of-two, dp == 1,
and unequal actor/reward dp.

Standalone by design (repo policy after #99/#267 is no unenforced test tree):
run it directly whenever the refl recipe's KL/reward wire types change::

    python scripts/verify_refl_kl_batching.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from experimental.refl.roles import REFLGenerated, REFLLossMetrics  # noqa: E402
from unirl.distributed.tensor.pytree import infer_batch_size, pytree_cat, pytree_chunk  # noqa: E402

# (batch_size, dp) — B == dp (the only previously-verified shape), B > dp,
# non-power-of-two, and the degenerate dp == 1.
TOPOLOGIES = [(8, 8), (8, 2), (6, 3), (12, 4), (4, 1)]


def _generated(batch: int) -> REFLGenerated:
    decoded = torch.arange(batch * 6, dtype=torch.float32).reshape(batch, 2, 3)
    kl = torch.arange(batch, dtype=torch.float32) + 1.0  # distinct per sample
    return REFLGenerated(decoded=decoded, kl_loss=kl)


def check_refl_generated_chunk_cat_roundtrip(batch: int, dp: int) -> None:
    """Each DP shard sees exactly its own KL rows; merge restores the batch."""
    gen = _generated(batch)
    assert infer_batch_size((gen,), {}) == batch

    shards = pytree_chunk(gen, dp, batch)
    assert len(shards) == dp
    per = batch // dp
    for rank, shard in enumerate(shards):
        expect = gen.kl_loss[rank * per : (rank + 1) * per]
        assert torch.equal(shard.kl_loss, expect), f"rank {rank} got foreign KL rows"
        assert torch.equal(shard.decoded, gen.decoded[rank * per : (rank + 1) * per])

    merged = pytree_cat(shards)
    assert torch.equal(merged.kl_loss, gen.kl_loss)
    assert torch.equal(merged.decoded, gen.decoded)


def check_forward_backward_payload_alignment(batch: int, dp: int) -> None:
    """rewards and kl_loss chunk in lockstep — the forward_backward_loss wire.

    The old scalar-shared KL made this payload rewards=[B] + kl=[1]; chunking
    by the inferred batch could not keep the two aligned off the B == dp
    topology. Per-sample KL makes both first-class batch columns.
    """
    kwargs = {
        "rewards": torch.randn(batch),
        "kl_loss": torch.arange(batch, dtype=torch.float32),
    }
    assert infer_batch_size((), kwargs) == batch
    shards = pytree_chunk(kwargs, dp, batch)
    per = batch // dp
    for rank, shard in enumerate(shards):
        assert shard["rewards"].shape == (per,)
        assert shard["kl_loss"].shape == (per,)
        assert torch.equal(shard["kl_loss"], kwargs["kl_loss"][rank * per : (rank + 1) * per])


def check_per_shard_backward_grad_shape(batch: int, dp: int) -> None:
    """The KL grad each rank produces matches its saved generate output rows.

    Mirrors ReflActorRole.forward_backward_loss on one shard: the KL input is
    a grad leaf of shape [B/dp]; after backward its .grad must be the same
    shape, because GradContext routes it as out_grads onto the SAME rank's
    saved kl tensor from generate_samples. With the old scalar KL this pairing
    was [broadcast scalar] vs [rank-local scalar] and only lined up by luck.
    """
    per = batch // dp
    for _rank in range(dp):
        rewards = torch.randn(per, requires_grad=True)
        kl = torch.rand(per, requires_grad=True)
        reward_loss = (-(rewards.to(torch.bfloat16) - 0.5) / 0.25 * 1.0).mean()
        loss = reward_loss + 1.0 * kl.float().mean()
        loss.backward()
        assert kl.grad is not None and kl.grad.shape == kl.shape
        assert rewards.grad is not None and rewards.grad.shape == rewards.shape


def check_unequal_actor_reward_dp_stays_aligned() -> None:
    """B-length columns survive reward-dp merge → actor-dp re-chunk.

    The recipe colocates actor and reward (equal dp by construction), but the
    wire contract must not depend on that: scoring merged at reward dp=2 and
    re-scattered at actor dp=4 must hand every actor rank the reward/KL rows
    of its own samples.
    """
    batch, rdp, adp = 8, 2, 4
    rewards = torch.arange(batch, dtype=torch.float32)
    reward_shards = pytree_chunk({"r": rewards}, rdp, batch)
    merged = pytree_cat(reward_shards)["r"]
    assert torch.equal(merged, rewards)

    kwargs = {"rewards": merged, "kl_loss": torch.arange(batch, dtype=torch.float32) * 10.0}
    actor_shards = pytree_chunk(kwargs, adp, batch)
    per = batch // adp
    for rank, shard in enumerate(actor_shards):
        assert torch.equal(shard["rewards"], rewards[rank * per : (rank + 1) * per])
        assert torch.equal(shard["kl_loss"], kwargs["kl_loss"][rank * per : (rank + 1) * per])


def check_per_sample_kl_equals_legacy_scalar_mean() -> None:
    """Per-sample reduction then batch-mean == the legacy global scalar mean."""
    torch.manual_seed(0)
    kl_pred = torch.randn(3, 4, 2, 5, 5)
    ref = torch.randn(3, 4, 2, 5, 5)
    sigma = torch.tensor(0.7)
    per_sample = ((kl_pred - ref) ** 2 / (2.0 * sigma**2)).flatten(1).mean(dim=1)
    legacy = ((kl_pred - ref) ** 2 / (2.0 * sigma**2)).mean()
    assert per_sample.shape == (3,)
    assert torch.allclose(per_sample.mean(), legacy, atol=1e-6)


def check_loss_metrics_concat_keeps_every_shard() -> None:
    """REFLLossMetrics concat lists must surface every shard's scalars."""
    shards = [
        REFLLossMetrics(loss=[0.1], reward_loss=[0.2], kl_loss=[0.3], reward_mean=[0.4]),
        REFLLossMetrics(loss=[1.1], reward_loss=[1.2], kl_loss=[1.3], reward_mean=[1.4]),
    ]
    merged = pytree_cat(shards)
    assert merged.loss == [0.1, 1.1]
    assert merged.kl_loss == [0.3, 1.3]


def main() -> int:
    for batch, dp in TOPOLOGIES:
        check_refl_generated_chunk_cat_roundtrip(batch, dp)
        check_forward_backward_payload_alignment(batch, dp)
        check_per_shard_backward_grad_shape(batch, dp)
        print(f"[ok] topology B={batch} dp={dp}: chunk/cat, payload lockstep, backward shapes")
    check_unequal_actor_reward_dp_stays_aligned()
    print("[ok] unequal actor/reward dp (rdp=2 → adp=4) stays row-aligned")
    check_per_sample_kl_equals_legacy_scalar_mean()
    print("[ok] per-sample KL batch-mean equals legacy scalar mean")
    check_loss_metrics_concat_keeps_every_shard()
    print("[ok] REFLLossMetrics keeps every shard's scalars")
    print("verify-refl-kl-batching: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
