"""The canonical external conversation format: OpenAI-style messages.

Manifests and converted agent traces are
``{"messages": [{role, content|null, tool_calls}], "tools": [...]}`` — the
trainer-facing shape OpenAI/Azure fine-tuning, HF ``apply_chat_template``, TRL,
axolotl, LLaMA-Factory and verl all speak. Per-source converters
(``datasets/<name>/convert_*.py``) may accept anything; what they *emit* is this
schema, unextended, with RL metadata (rewards, outcome labels, trace ids) as
sibling fields next to ``messages`` rather than inside it.

This module owns the *format*: normalization and the messages → :class:`Turn`
parse. Rendering ``Turn`` s back into model-consumable messages is the job of
``unirl.models.types.conversations`` (and of a model package when only one model
consumes that dialect) — format lives with the types, rendering lives with the
models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from unirl.types.media import MediaRef, MediaRefs, SUPPORTED_MEDIA_MODALITIES
from unirl.types.primitives import Texts
from unirl.types.sample import TURN_ROLES, Turn
from unirl.types.tool_calls import ToolCall, ToolCalls

Message = Dict[str, Any]


def system_instruction_wins(system_instruction: Any, roles: Sequence[str]) -> bool:
    """The one system rule: the config default applies unless the data carries a
    ``system`` turn, which wins.

    Both dialects go through this predicate — :func:`ensure_system_message` for
    message histories and ``conversations.system_prefix`` for ``Turn`` lists — so
    the precedence is defined once. A path that wants "data is the sole source of
    truth" leaves ``system_instruction`` unset in the recipe; it is never a code
    branch.
    """
    return bool(system_instruction) and "system" not in roles


def ensure_system_message(messages: Sequence[Message], system_instruction: Any) -> List[Message]:
    """Prepend the config ``system_instruction`` unless the history has its own."""
    out = list(messages)
    if system_instruction_wins(system_instruction, [m.get("role") for m in out]):
        return [{"role": "system", "content": system_instruction}, *out]
    return out


def normalize_tool_arguments(messages: Sequence[Message]) -> List[Message]:
    """Normalize every assistant ``tool_calls`` entry to dict ``arguments``.

    OpenAI serializes ``function.arguments`` as a JSON-encoded string; HF chat
    templates expect a dict and mis-render the string form (the transformers docs
    warn about exactly this). Run this once at manifest ingest so trace data from
    either dialect renders identically. Messages without calls pass through
    unchanged (same object).
    """
    out: List[Message] = []
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


def _media_ref_from_part(part: Message, *, context: str) -> MediaRef:
    """One content part → prompt ``MediaRef``, tolerant of the common dialects.

    Accepts ``{"type": X, X: uri}`` (canonical / Qwen), ``{"type": X, "url"|"path":
    uri}`` (HF), and ``{"type": "image_url", "image_url": {"url": uri}}`` (OpenAI).
    Re-rendering emits the canonical first form.
    """
    ptype = part.get("type")
    if ptype == "image_url":
        uri = (part.get("image_url") or {}).get("url")
        if not uri:
            raise ValueError(f"{context}: image_url part carries no url: {part!r}.")
        return MediaRef(modality="image", role="prompt", uri=uri)
    if ptype in SUPPORTED_MEDIA_MODALITIES:
        uri = part.get(ptype) or part.get("url") or part.get("path")
        if not isinstance(uri, str) or not uri:
            raise ValueError(f"{context}: {ptype} part carries no usable URI: {part!r}.")
        return MediaRef(modality=ptype, role="prompt", uri=uri)
    raise ValueError(f"{context}: unsupported content part type {ptype!r}.")


def turns_from_messages(messages: Sequence[Message]) -> List[Turn]:
    """One OpenAI-style message history → batch-1 :class:`Turn` list.

    The manifest half of the round trip whose render half lives with the model
    dialects. Supported content: strings, ``None`` (calls-only assistant), and
    content-part lists carrying URI media parts plus at most one text part; parts
    normalize to canonical media-before-text order, so parse→render→parse is
    stable in canonical form (it does not preserve arbitrary input part order).

    ``Turn`` carries one primitive, so a message with both media and text becomes
    two turns; the renderer fuses them back by consecutive role. A ``tool``
    message is the one case where that is lossy — its ``tool_call_id`` belongs to
    the whole message, and the renderer never fuses tool turns (each result keeps
    its own pairing) — so a media-bearing tool result raises instead of silently
    dropping the link. Representing it needs composite ``Turn`` content, which is
    a typed-representation change, not a parser workaround.
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
        call_id = message.get("tool_call_id")
        name = message.get("name")
        pairing = {
            "tool_call_ids": [call_id] if role == "tool" and call_id is not None else None,
            "tool_names": [name] if role == "tool" and name is not None else None,
        }

        if isinstance(content, list):
            text_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "text"]
            media_parts = [p for p in content if isinstance(p, dict) and p.get("type") != "text"]
            if len(text_parts) > 1:
                raise ValueError(
                    f"turns_from_messages: message {i} has {len(text_parts)} text parts; one message "
                    "carries at most one text part — join the texts in the converter."
                )
            if media_parts and role == "tool":
                raise NotImplementedError(
                    f"turns_from_messages: message {i} is a tool result carrying media parts. A Turn holds "
                    "one primitive, so this would split into two turns and the tool_call_id link would be "
                    "lost on the media half; composite Turn content is required first."
                )
            refs = [_media_ref_from_part(p, context=f"turns_from_messages[{i}]") for p in media_parts]
            if not refs and not text_parts and not calls:
                raise ValueError(f"turns_from_messages: message {i} ({role}) has an empty content-part list.")
            if refs:
                turns.append(Turn(role=role, content=MediaRefs.from_rows([refs])))
            if text_parts or calls:
                text = text_parts[0]["text"] if text_parts else None
                turns.append(
                    Turn(
                        role=role,
                        content=Texts(texts=[text]) if text is not None else None,
                        tool_calls=tool_calls,
                        **pairing,
                    )
                )
            continue

        if content is None:
            if not calls:
                raise ValueError(f"turns_from_messages: message {i} ({role}) has neither content nor tool_calls.")
            turns.append(Turn(role=role, content=None, tool_calls=tool_calls))
            continue
        if not isinstance(content, str):
            raise TypeError(
                f"turns_from_messages: message {i} content must be str/None/list, got {type(content).__name__}."
            )
        turns.append(Turn(role=role, content=Texts(texts=[content]), tool_calls=tool_calls, **pairing))
    return turns


__all__ = [
    "Message",
    "ensure_system_message",
    "normalize_tool_arguments",
    "system_instruction_wins",
    "turns_from_messages",
]
