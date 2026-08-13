"""Trajectory → chat-conversation rendering for trainside encoders (pure).

The trainside in-process AR encoders (``qwen3`` / ``qwen_vl`` chat-template stages)
must build one chat conversation per frontier sample from the role-aware trajectory
view :meth:`Sample.turns` (turn-major, frontier-aligned). These helpers transpose
that into per-sample message lists.

Unlike the sglang engine's equivalent (``rollout/engine/sglang/utils/conversations.py``,
which de-expands the ``*n`` wire fan-out), the trainside embeds **every** frontier
sample per-row — so there is NO de-expand here; one conversation per row.

Pure (only :mod:`unirl.types`): no tokenizer, no processor, no engine.

Two renders live here because both have several model consumers: string content
(:func:`build_text_messages` — qwen3, qwen3_5) and inline-PIL parts
(:func:`build_vision_messages` — qwen_vl, qwen3_5). They share two composition
rules, :func:`system_prefix` (the config ``system_instruction`` is a default; an
explicit ``system`` turn in the data wins) and :func:`group_consecutive_roles`,
which the sglang wire render now imports instead of mirroring byte-identical
private copies.

A render with a single model consumer stays in that model's package (Qwen3-Omni's
URI-media render lives in ``unirl.models.qwen3_omni.media``); it moves here when a
second consumer actually exists, not before.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from unirl.types.agent_trace import system_instruction_wins
from unirl.types.primitives import Images
from unirl.types.sample import Turn

Conversation = List[Dict[str, Any]]


def system_prefix(system_instruction: Optional[str], roles: List[str]) -> Conversation:
    """The Turn-dialect form of the one system rule (see
    :func:`unirl.types.agent_trace.system_instruction_wins`): the config
    ``system_instruction`` leads unless the trajectory carries its own ``system``
    turn."""
    if system_instruction_wins(system_instruction, roles):
        return [{"role": "system", "content": system_instruction}]
    return []


def group_consecutive_roles(roles: List[str]) -> List[Tuple[str, List[int]]]:
    """Group consecutive turn indices that share a role → ``[(role, [idx…]), …]``.

    Multi-input modalities ride as separate same-role turns (e.g. it2i is a text
    ``user`` turn + an image ``user`` turn); a chat message holds one role, so
    consecutive same-role turns fuse into one message.
    """
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
    """One all-text chat conversation per frontier row (no de-expand).

    Transposes :meth:`Sample.text_conditioning` (turn-major, frontier-aligned) into
    one message list per frontier sample. Degenerates to a single ``user`` message
    when no roles are set, so it is byte-identical on single-turn workloads.

    Structured agent channels ride through when present: a turn's
    :attr:`Turn.tool_calls` rows emit OpenAI-style ``tool_calls`` entries (dict
    ``arguments`` — the HF template form), a calls-only turn emits
    ``content: None``, and a tool turn emits ``tool_call_id`` / ``name``. Plain
    chat turns emit exactly ``{"role", "content"}``.
    """
    if not turns:
        return []
    roles = [t.role for t in turns]
    cols = [t.content.texts if t.content is not None else None for t in turns]
    prefix = system_prefix(system_instruction, roles)
    n_rows = next(len(t.content if t.content is not None else t.tool_calls) for t in turns)

    def _message(j: int, row: int) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": roles[j], "content": cols[j][row] if cols[j] is not None else None}
        calls = turns[j].tool_calls.rows[row] if turns[j].tool_calls is not None else []
        if calls:
            message["tool_calls"] = [call.to_message_entry() for call in calls]
        for key, channel in (("tool_call_id", turns[j].tool_call_ids), ("name", turns[j].tool_names)):
            if channel is not None and channel[row] is not None:
                message[key] = channel[row]
        return message

    return [prefix + [_message(j, row) for j in range(len(turns))] for row in range(n_rows)]


def build_vision_messages(
    turns: List[Turn],
    system_instruction: Optional[str] = None,
) -> List[Conversation]:
    """One text+image chat conversation per frontier row (no de-expand).

    Same transpose as :func:`build_text_messages`, but consecutive same-role turns
    fuse into one message whose content is ``[image blocks…, text blocks…]`` —
    image-before-text — with the PIL image inlined (``{"type":"image","image":pil}``),
    matching ``QwenVLChatTemplateStage``'s processor input (the processor reads PILs
    from the message content). One image per row (callers guard).
    """
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
