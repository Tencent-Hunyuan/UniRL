"""Conditioning embedding stage protocols, prompt deduplication, and bounded CPU LRU caching."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from typing import Protocol, Tuple, TypeVar, runtime_checkable

import torch

from unirl.config.require import require
from unirl.types.conditions import TextEmbedCondition

P = TypeVar("P")
ImageP = TypeVar("ImageP")
C = TypeVar("C")
K = TypeVar("K", bound=Hashable)

_PROMPT_CACHE_SIZE = 8


def dedup_prompts(items: Sequence[K]) -> tuple[list[K], list[int] | None]:
    """Deduplicate prompt keys in first-seen order and return ``(uniq, inverse_or_none)``."""
    index_of: dict[K, int] = {}
    inverse: list[int] = []
    uniq: list[K] = []
    for item in items:
        i = index_of.get(item)
        if i is None:
            i = len(uniq)
            index_of[item] = i
            uniq.append(item)
        inverse.append(i)
    return uniq, inverse if len(uniq) != len(items) else None


def _to_device(cond: TextEmbedCondition, device: torch.device) -> TextEmbedCondition:
    return TextEmbedCondition(
        embeds=cond.embeds.to(device=device) if cond.embeds is not None else None,
        pooled=cond.pooled.to(device=device) if cond.pooled is not None else None,
        attn_mask=cond.attn_mask.to(device=device) if cond.attn_mask is not None else None,
    )


def _cpu_batch(cond: TextEmbedCondition) -> TextEmbedCondition:
    return TextEmbedCondition(
        embeds=cond.embeds.detach().to("cpu") if cond.embeds is not None else None,
        pooled=cond.pooled.detach().to("cpu") if cond.pooled is not None else None,
        attn_mask=cond.attn_mask.detach().to("cpu") if cond.attn_mask is not None else None,
    )


def _row_view(cond: TextEmbedCondition, index: int) -> TextEmbedCondition:
    return TextEmbedCondition(
        embeds=cond.embeds[index : index + 1] if cond.embeds is not None else None,
        pooled=cond.pooled[index : index + 1] if cond.pooled is not None else None,
        attn_mask=cond.attn_mask[index : index + 1] if cond.attn_mask is not None else None,
    )


def _cpu_clone_row(cond: TextEmbedCondition, index: int) -> TextEmbedCondition:
    return TextEmbedCondition(
        embeds=cond.embeds[index : index + 1].clone() if cond.embeds is not None else None,
        pooled=cond.pooled[index : index + 1].clone() if cond.pooled is not None else None,
        attn_mask=cond.attn_mask[index : index + 1].clone() if cond.attn_mask is not None else None,
    )


def embed_with_cpu_cache(
    prompts: Sequence[str],
    cache: OrderedDict[str, TextEmbedCondition],
    encode_fn: Callable[[list[str]], TextEmbedCondition],
    *,
    device: torch.device,
) -> TextEmbedCondition:
    """Encode fixed-length prompts with in-batch deduplication and bounded CPU LRU caching."""
    require(len(prompts) > 0, "embed_with_cpu_cache: no prompts to embed")
    uniq, inverse = dedup_prompts(prompts)
    cached_cpu = {p: cache[p] for p in uniq if p in cache}
    missing = [p for p in uniq if p not in cached_cpu]
    encoded: TextEmbedCondition | None = None
    if missing:
        encoded = encode_fn(missing)
        cpu_batch = _cpu_batch(encoded)
        for i, p in enumerate(missing):
            cached_cpu[p] = _cpu_clone_row(cpu_batch, i)

    for p in uniq:
        if p in cache:
            cache.move_to_end(p)
        else:
            cache[p] = cached_cpu[p]
            if len(cache) > _PROMPT_CACHE_SIZE:
                cache.popitem(last=False)

    if not (len(cached_cpu) - len(missing)) and encoded is not None:
        return encoded if inverse is None else encoded.select(inverse)

    on_device: dict[str, TextEmbedCondition] = {
        p: _to_device(cond, device) for p, cond in cached_cpu.items() if p not in missing
    }
    if encoded is not None:
        for i, p in enumerate(missing):
            on_device[p] = _row_view(encoded, i)

    if len(prompts) == 1:
        return on_device[prompts[0]]
    return TextEmbedCondition.concat([on_device[p] for p in prompts])


def embed_pair_with_cpu_cache(
    prompts: Sequence[str],
    cache: OrderedDict[str, Tuple[TextEmbedCondition, TextEmbedCondition]],
    encode_fn: Callable[[list[str]], Tuple[TextEmbedCondition, TextEmbedCondition]],
    *,
    device: torch.device,
) -> Tuple[TextEmbedCondition, TextEmbedCondition]:
    """Encode coupled two-stream fixed-length prompts with in-batch deduplication and bounded CPU LRU caching."""
    require(len(prompts) > 0, "embed_pair_with_cpu_cache: no prompts to embed")
    uniq, inverse = dedup_prompts(prompts)
    cached_cpu = {p: cache[p] for p in uniq if p in cache}
    missing = [p for p in uniq if p not in cached_cpu]
    encoded_pair: Tuple[TextEmbedCondition, TextEmbedCondition] | None = None
    if missing:
        first_batch, second_batch = encode_fn(missing)
        encoded_pair = (first_batch, second_batch)
        cpu_first = _cpu_batch(first_batch)
        cpu_second = _cpu_batch(second_batch)
        for i, p in enumerate(missing):
            cached_cpu[p] = (_cpu_clone_row(cpu_first, i), _cpu_clone_row(cpu_second, i))

    for p in uniq:
        if p in cache:
            cache.move_to_end(p)
        else:
            cache[p] = cached_cpu[p]
            if len(cache) > _PROMPT_CACHE_SIZE:
                cache.popitem(last=False)

    if not (len(cached_cpu) - len(missing)) and encoded_pair is not None:
        first_batch, second_batch = encoded_pair
        if inverse is None:
            return first_batch, second_batch
        return first_batch.select(inverse), second_batch.select(inverse)

    on_device: dict[str, Tuple[TextEmbedCondition, TextEmbedCondition]] = {
        p: (_to_device(first, device), _to_device(second, device))
        for p, (first, second) in cached_cpu.items()
        if p not in missing
    }
    if encoded_pair is not None:
        first_batch, second_batch = encoded_pair
        for i, p in enumerate(missing):
            on_device[p] = (_row_view(first_batch, i), _row_view(second_batch, i))

    if len(prompts) == 1:
        return on_device[prompts[0]]
    return (
        TextEmbedCondition.concat([on_device[p][0] for p in prompts]),
        TextEmbedCondition.concat([on_device[p][1] for p in prompts]),
    )


@runtime_checkable
class EmbedStage(Protocol[P, C]):
    """Embed a primitive into its condition form (e.g. text → text-condition)."""

    def embed(self, p: P) -> C: ...


@runtime_checkable
class ImageConditionedEmbedStage(Protocol[P, ImageP, C]):
    """Embed a primitive with optional image context into its condition form."""

    def embed(self, p: P, images: ImageP | None = None) -> C: ...


__all__ = [
    "EmbedStage",
    "ImageConditionedEmbedStage",
    "dedup_prompts",
    "embed_pair_with_cpu_cache",
    "embed_with_cpu_cache",
]
