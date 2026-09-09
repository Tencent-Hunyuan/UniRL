"""Trajectory → chat-conversation rendering (pure, tokenizer-free)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from unirl.models.types.conversations import group_consecutive_roles, system_prefix
from unirl.types.primitives import Images
from unirl.types.sample import Sample

Conversation = List[Dict[str, Any]]


def unique_group_indices(group_ids: List[str]) -> Tuple[List[int], int]:
    """Representative row index per fan-out group + the uniform repeat ``k``."""
    n = len(group_ids)
    if n == 0:
        return [], 1

    rep_indices: List[int] = []
    sizes: Dict[str, int] = {}
    for i, gid in enumerate(group_ids):
        if gid not in sizes:
            sizes[gid] = 0
            rep_indices.append(i)
        sizes[gid] += 1

    k_values = set(sizes.values())
    if len(k_values) != 1:
        return list(range(n)), 1
    return rep_indices, next(iter(k_values))


def build_text_conversations(
    sample: Sample,
    system_instruction: Any = None,
) -> Tuple[List[Conversation], int]:
    """The all-text trajectory as ``P`` unique chat conversations + fan-out ``k``."""
    turns = sample.text_conditioning()
    rep, k = unique_group_indices(sample.parts[-1].group_ids)
    roles = [t.role for t in turns]
    cols = [t.content.texts for t in turns]
    prefix = system_prefix(system_instruction, roles)

    conversations = [prefix + [{"role": roles[j], "content": cols[j][row]} for j in range(len(turns))] for row in rep]
    return conversations, k


def build_vision_conversations(
    sample: Sample,
    system_instruction: Any = None,
) -> Tuple[List[Conversation], List[List[Any]], int]:
    """The text+image trajectory as ``P`` conversations, their per-conv PIL"""
    turns, _ = sample.vision_conditioning()
    rep, k = unique_group_indices(sample.parts[-1].group_ids)
    roles = [t.role for t in turns]
    is_image = [isinstance(t.content, Images) for t in turns]
    cols = [t.content.to_pils() if im else t.content.texts for t, im in zip(turns, is_image)]
    role_groups = group_consecutive_roles(roles)
    prefix = system_prefix(system_instruction, roles)

    conversations: List[Conversation] = []
    images_list: List[List[Any]] = []
    for row in rep:
        messages: Conversation = list(prefix)
        conv_images: List[Any] = []
        for role, idxs in role_groups:
            image_blocks: List[Dict[str, Any]] = []
            text_blocks: List[Dict[str, Any]] = []
            for j in idxs:
                if is_image[j]:
                    image_blocks.append({"type": "image"})
                    conv_images.append(cols[j][row])
                else:
                    text_blocks.append({"type": "text", "text": cols[j][row]})
            messages.append({"role": role, "content": image_blocks + text_blocks})
        conversations.append(messages)
        images_list.append(conv_images)
    return conversations, images_list, k


__all__ = [
    "Conversation",
    "build_text_conversations",
    "build_vision_conversations",
    "unique_group_indices",
]
