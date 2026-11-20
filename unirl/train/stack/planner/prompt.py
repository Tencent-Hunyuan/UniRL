"""Prompt-stratified, rank-balanced membership for two disjoint optimizer updates."""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.distributed as dist

from unirl.types.sample import Part
from unirl.types.sample_id import ancestor_id


def prompt_half_order(
    coarse: Part,
    target: Part,
    *,
    prompt_groups: tuple[int, int | None],
    dp_rank: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Order local lineage blocks so both updates contain half of each global prompt and DP shard."""
    group_size, layer = prompt_groups
    local = (dp_rank, coarse.sample_ids if coarse is not target else None, target.sample_ids)
    metadata = [local]
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        metadata = [None] * dist.get_world_size()
        dist.all_gather_object(metadata, local)

    shards = {}
    for rank, ancestors, ids in metadata:
        if shards.setdefault(rank, (ancestors, ids)) != (ancestors, ids):
            raise ValueError("Prompt-stratified updates require identical row IDs on replicas of each DP shard.")
    graph = defaultdict(list)
    seen = set()
    local_rows = []
    shard_shape = None
    for rank, (ancestors, ids) in sorted(shards.items()):
        blocks = len(ancestors) if ancestors is not None else len(ids)
        if not blocks or blocks % 2 or not ids or len(ids) % blocks:
            raise ValueError("Prompt-stratified updates need an even per-rank batch of equal-size lineage blocks.")
        if shard_shape is not None and shard_shape != (blocks, len(ids)):
            raise ValueError("Prompt-stratified updates require equal batch sizes on every DP shard.")
        shard_shape = (blocks, len(ids))
        fanout = len(ids) // blocks
        if group_size % (2 * fanout):
            raise ValueError("A prompt cannot split evenly across two updates while keeping its lineage blocks intact.")
        if any(not isinstance(sid, str) or not sid or sid in seen for sid in ids) or len(set(ids)) != len(ids):
            raise ValueError("Prompt-stratified updates require globally unique, nonempty sample IDs.")
        seen.update(ids)
        depth = max(ids[0].count("/") - 1, 0) if layer is None else layer
        rows = torch.randperm(blocks, generator=generator).tolist() if generator is not None else range(blocks)
        if rank == dp_rank:
            local_rows = rows
        for row in rows:
            children = ids[row * fanout : (row + 1) * fanout]
            prompt = ancestor_id(children[0], depth)
            if ancestors is not None and any(
                (sid != ancestors[row] and not sid.startswith(f"{ancestors[row]}/"))
                or ancestor_id(sid, depth) != prompt
                for sid in children
            ):
                raise ValueError(
                    "Prompt-stratified updates require a contiguous single-prompt expansion of each ancestor."
                )
            edge = (rank, row)
            graph[rank].append((prompt, edge))
            graph[prompt].append((rank, edge))

    if any(len(edges) * fanout != group_size for node, edges in graph.items() if isinstance(node, str)):
        raise ValueError(f"Prompt-stratified updates need complete groups of samples_per_prompt={group_size}.")

    # All vertex degrees are even, so each trail is closed and enters/leaves each vertex equally often.
    used, first_update = set(), set()
    for start in graph:
        node = start
        while graph[node]:
            other, edge = graph[node].pop()
            if edge in used:
                continue
            used.add(edge)
            if node == dp_rank:
                first_update.add(edge[1])
            node = other

    return torch.tensor(sorted(local_rows, key=lambda row: row not in first_update), dtype=torch.long)
