"""Trajectory → chat-conversation rendering for trainside encoders (pure).

The trainside in-process AR encoders (``qwen3`` / ``qwen_vl`` chat-template stages)
must build one chat conversation per frontier sample from the role-aware trajectory
view :meth:`Sample.turns` (turn-major, frontier-aligned). These helpers transpose
that into per-sample message lists.

Unlike the sglang engine's equivalent (``rollout/engine/sglang/utils/conversations.py``,
which de-expands the ``*n`` wire fan-out), the trainside embeds **every** frontier
sample per-row — so there is NO de-expand here; one conversation per row.

Pure (only :mod:`unirl.types`): no tokenizer, no processor, no engine.

**This module is the repo's single conversation-composition layer.** Rules that
decide how a conversation is assembled — the system-instruction precedence, role
fusion, structured tool-call emission, message normalization — live here and
ONLY here, in two dialects: Turn-based helpers (rollout/replay trajectories) and
message-based helpers (OpenAI-style manifests / agent traces). Per-model chat
stages own the *encoding* of a composed conversation (chat template, processor,
Bagel splits), never the composition rules. The sglang wire render
(``rollout/engine/sglang/utils/conversations.py``) imports these helpers too.

The one system rule, both dialects: **the config ``system_instruction`` is a
default; an explicit ``system`` turn in the data wins.** Paths that want
"data is the sole source of truth" (e.g. faithful trace SFT) express that by
not setting ``system_instruction`` in the recipe — never by a code branch.

The canonical external format for conversation data is the OpenAI messages
shape (``{"messages": [{role, content|null, tool_calls}], "tools": [...]}``);
``tool_calls.function.arguments`` is normalized to a dict at ingest
(:func:`normalize_tool_arguments`) because HF chat templates expect dicts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from unirl.types.primitives import Images, Texts
from unirl.types.sample import TURN_ROLES, Turn
from unirl.types.tool_calls import ToolCall, ToolCalls

Conversation = List[Dict[str, Any]]


def system_prefix(system_instruction: Optional[str], roles: List[str]) -> Conversation:
    """The config ``system_instruction`` as a leading message, unless the trajectory
    already carries an explicit ``system`` turn (which wins)."""
    if system_instruction and "system" not in roles:
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


def _turn_batch_size(turns: List[Turn]) -> int:
    for t in turns:
        if t.content is not None:
            return len(t.content)
        if t.tool_calls is not None:
            return len(t.tool_calls)
    raise ValueError("conversations: every turn is empty (no content, no tool_calls).")


def build_text_messages(
    turns: List[Turn],
    system_instruction: Optional[str] = None,
) -> List[Conversation]:
    """One all-text chat conversation per frontier row (no de-expand).

    Transposes :meth:`Sample.text_conditioning` (turn-major, frontier-aligned) into
    one message list per frontier sample. Degenerates to a single ``user`` message
    when no roles are set, so it is byte-identical on single-turn workloads.

    Structured agent channels ride through: a turn's :attr:`Turn.tool_calls`
    rows emit OpenAI-style ``tool_calls`` entries (dict ``arguments`` — the HF
    template form) on the assistant message, a calls-only turn emits
    ``content: None``, and a tool turn's :attr:`Turn.tool_call_ids` emits
    ``tool_call_id``. Plain chat turns emit exactly ``{"role", "content"}``.
    """
    if not turns:
        return []
    roles = [t.role for t in turns]
    cols = [t.content.texts if t.content is not None else None for t in turns]
    prefix = system_prefix(system_instruction, roles)
    n_rows = _turn_batch_size(turns)

    def _message(j: int, row: int) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": roles[j], "content": cols[j][row] if cols[j] is not None else None}
        calls = turns[j].tool_calls.rows[row] if turns[j].tool_calls is not None else []
        if calls:
            message["tool_calls"] = [call.to_message_entry() for call in calls]
        ids = turns[j].tool_call_ids
        if ids is not None and ids[row] is not None:
            message["tool_call_id"] = ids[row]
        names = turns[j].tool_names
        if names is not None and names[row] is not None:
            message["name"] = names[row]
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
    if any(t.content is None for t in turns):
        raise ValueError(
            "build_vision_messages: calls-only turns (content=None) are unsupported on the "
            "vision render; structured tool calls currently render via build_text_messages."
        )
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


def ensure_system_message(
    messages: Sequence[Dict[str, Any]],
    system_instruction: Optional[str],
) -> List[Dict[str, Any]]:
    """The message-dialect twin of :func:`system_prefix` — same one rule.

    Prepends the config ``system_instruction`` as a leading ``system`` message
    unless the history already carries an explicit ``system`` message (which
    wins). Returns a new list; never mutates ``messages``.
    """
    out = list(messages)
    if system_instruction and not any(m.get("role") == "system" for m in out):
        return [{"role": "system", "content": system_instruction}, *out]
    return out


def normalize_tool_arguments(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize every assistant ``tool_calls`` entry to dict ``arguments``.

    OpenAI serializes ``function.arguments`` as a JSON-encoded string; HF chat
    templates expect a dict and mis-render the string form (transformers docs
    warn about this explicitly). Call this once at manifest ingest so trace
    data from either dialect renders identically. Returns new message dicts
    where normalization applies; non-assistant / call-free messages pass
    through unchanged.
    """
    out: List[Dict[str, Any]] = []
    for message in messages:
        calls = message.get("tool_calls")
        if not calls:
            out.append(message)
            continue
        normalized = [ToolCall.from_message_entry(entry, context="normalize_tool_arguments") for entry in calls]
        rebuilt = dict(message)
        rebuilt["tool_calls"] = [call.to_message_entry() for call in normalized]
        out.append(rebuilt)
    return out


def turns_from_messages(messages: Sequence[Dict[str, Any]]) -> List[Turn]:
    """One OpenAI-style message history → batch-1 :class:`Turn` list.

    The manifest→trajectory half of the round trip whose other half is
    :func:`build_text_messages`, so an SFT path over trace manifests can render
    through exactly the same composition code as rollout. Supported today:
    string (or ``None``) content, assistant ``tool_calls`` (either arguments
    dialect — normalized to dicts), and ``tool``-role ``tool_call_id`` pairing.
    Content-part lists (interleaved media) are not yet representable as Turns
    and raise.
    """
    turns: List[Turn] = []
    for i, message in enumerate(messages):
        role = message.get("role")
        if role not in TURN_ROLES:
            raise ValueError(f"turns_from_messages: message {i} role {role!r} not in {TURN_ROLES}.")
        content = message.get("content")
        raw_calls = message.get("tool_calls") or []
        if raw_calls and role != "assistant":
            raise ValueError(f"turns_from_messages: message {i} ({role}) carries tool_calls; only assistant may.")
        calls = [ToolCall.from_message_entry(entry, context=f"turns_from_messages[{i}]") for entry in raw_calls]
        tool_calls = ToolCalls.from_rows([calls]) if calls else None
        if isinstance(content, list):
            raise NotImplementedError(
                f"turns_from_messages: message {i} has content-part list; interleaved media "
                "parts are not yet representable as Turns (text/tool histories only)."
            )
        if content is None:
            if not calls:
                raise ValueError(f"turns_from_messages: message {i} ({role}) has neither content nor tool_calls.")
            turns.append(Turn(role=role, content=None, tool_calls=tool_calls))
            continue
        if not isinstance(content, str):
            raise TypeError(
                f"turns_from_messages: message {i} content must be str/None/list, got {type(content).__name__}."
            )
        call_id = message.get("tool_call_id")
        name = message.get("name")
        turns.append(
            Turn(
                role=role,
                content=Texts(texts=[content]),
                tool_calls=tool_calls,
                tool_call_ids=[call_id] if role == "tool" and call_id is not None else None,
                tool_names=[name] if role == "tool" and name is not None else None,
            )
        )
    return turns


__all__ = [
    "Conversation",
    "build_text_messages",
    "build_vision_messages",
    "ensure_system_message",
    "group_consecutive_roles",
    "normalize_tool_arguments",
    "system_prefix",
    "turns_from_messages",
]
