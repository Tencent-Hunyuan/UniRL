"""Trajectory → chat-conversation rendering for trainside encoders (pure)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from unirl.types.primitives import Images
from unirl.types.sample import Turn

Conversation = List[Dict[str, Any]]


def system_prefix(system_instruction: Optional[str], roles: List[str]) -> Conversation:
    """The config ``system_instruction`` as a leading message, unless the trajectory"""
    if system_instruction and "system" not in roles:
        return [{"role": "system", "content": system_instruction}]
    return []


def group_consecutive_roles(roles: List[str]) -> List[Tuple[str, List[int]]]:
    """Group consecutive turn indices that share a role → ``[(role, [idx…]), …]``."""
    groups: List[Tuple[str, List[int]]] = []
    for j, role in enumerate(roles):
        if groups and groups[-1][0] == role:
            groups[-1][1].append(j)
        else:
            groups.append((role, [j]))
    return groups


def build_text_messages(
    turns: List[Turn],
    system_instruction: Optional[str] = None,
) -> List[Conversation]:
    """One all-text chat conversation per frontier row (no de-expand)."""
    if not turns:
        return []
    roles = [t.role for t in turns]
    cols = [t.content.texts for t in turns]
    prefix = system_prefix(system_instruction, roles)
    n_rows = len(turns[0].content)
    return [prefix + [{"role": roles[j], "content": cols[j][row]} for j in range(len(turns))] for row in range(n_rows)]


def build_vision_messages(
    turns: List[Turn],
    system_instruction: Optional[str] = None,
) -> List[Conversation]:
    """One text+image chat conversation per frontier row (no de-expand)."""
    if not turns:
        return []
    roles = [t.role for t in turns]
    is_image = [isinstance(t.content, Images) for t in turns]
    cols = [t.content.to_pils() if im else t.content.texts for t, im in zip(turns, is_image)]
    role_groups = group_consecutive_roles(roles)
    prefix = system_prefix(system_instruction, roles)
    n_rows = len(turns[0].content)

    conversations: List[Conversation] = []
    for row in range(n_rows):
        messages: Conversation = list(prefix)
        for role, idxs in role_groups:
            image_blocks: List[Dict[str, Any]] = []
            text_blocks: List[Dict[str, Any]] = []
            for j in idxs:
                if is_image[j]:
                    image_blocks.append({"type": "image", "image": cols[j][row]})
                else:
                    text_blocks.append({"type": "text", "text": cols[j][row]})
            messages.append({"role": role, "content": image_blocks + text_blocks})
        conversations.append(messages)
    return conversations


__all__ = [
    "Conversation",
    "build_text_messages",
    "build_vision_messages",
    "group_consecutive_roles",
    "system_prefix",
]
