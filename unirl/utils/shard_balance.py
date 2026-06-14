"""Token-balanced sharding of a flat sample list across equal-size partitions.

Pure-Python helpers for the driver-side seqlen-balancing path (verl
``trainer.balance_batch`` parity): reorder a batch so that, when it is split into
``num_shards`` equal contiguous chunks (the layout ``DP_SCATTER`` produces), each
chunk carries a similar total token count. No torch / framework deps, so the
partition logic is unit-testable in isolation.
"""

from __future__ import annotations


def shard_token_spread(lengths: list[int], num_shards: int) -> float:
    """Return the relative token gap between the heaviest and lightest shard.

    Splits ``lengths`` into ``num_shards`` equal contiguous chunks and returns
    ``(max - min) / mean`` of the per-shard token sums — a unitless imbalance
    measure (``0.0`` is perfectly balanced; also ``0.0`` when the mean is 0).

    Args:
        lengths: Per-sample token counts, in current batch order.
        num_shards: Number of equal contiguous shards. Must divide ``len(lengths)``.

    Returns:
        The per-shard token-sum spread as a fraction of the mean.
    """
    per_shard = len(lengths) // num_shards
    sums = [sum(lengths[s * per_shard : (s + 1) * per_shard]) for s in range(num_shards)]
    mean = sum(sums) / num_shards
    return (max(sums) - min(sums)) / mean if mean else 0.0


def lpt_shard_permutation(lengths: list[int], num_shards: int) -> list[int]:
    """Return a permutation that token-balances ``num_shards`` equal-size shards.

    Greedy longest-processing-time under an equal-count constraint: assign the
    longest sample first into the shard with the smallest running token sum that
    still has room (each shard caps at ``len(lengths) // num_shards`` samples,
    because ``DP_SCATTER`` splits into equal contiguous chunks). Reading the
    returned permutation in ``num_shards`` equal contiguous blocks yields the
    balanced shards.

    Args:
        lengths: Per-sample token counts, in current batch order.
        num_shards: Number of equal contiguous shards. Must divide ``len(lengths)``.

    Returns:
        A permutation of ``range(len(lengths))``.
    """
    per_shard = len(lengths) // num_shards
    shards: list[list[int]] = [[] for _ in range(num_shards)]
    sums = [0] * num_shards
    for i in sorted(range(len(lengths)), key=lambda j: (-lengths[j], j)):
        target = min((s for s in range(num_shards) if len(shards[s]) < per_shard), key=lambda s: sums[s])
        shards[target].append(i)
        sums[target] += lengths[i]
    return [i for shard in shards for i in shard]
