"""Token-balanced sharding of a flat sample list across equal-size partitions."""

from __future__ import annotations


def shard_token_spread(lengths: list[int], num_shards: int) -> float:
    """Return the relative token gap between the heaviest and lightest shard."""
    per_shard = len(lengths) // num_shards
    sums = [sum(lengths[s * per_shard : (s + 1) * per_shard]) for s in range(num_shards)]
    mean = sum(sums) / num_shards
    return (max(sums) - min(sums)) / mean if mean else 0.0


def lpt_shard_permutation(lengths: list[int], num_shards: int) -> list[int]:
    """Return a permutation that token-balances ``num_shards`` equal-size shards."""
    per_shard = len(lengths) // num_shards
    shards: list[list[int]] = [[] for _ in range(num_shards)]
    sums = [0] * num_shards
    for i in sorted(range(len(lengths)), key=lambda j: (-lengths[j], j)):
        target = min((s for s in range(num_shards) if len(shards[s]) < per_shard), key=lambda s: sums[s])
        shards[target].append(i)
        sums[target] += lengths[i]
    return [i for shard in shards for i in shard]
