"""Trajectory → chat-conversation rendering for trainside encoders (pure).

The trainside in-process AR encoders (``qwen3`` / ``qwen_vl`` chat-template stages)
must build one chat conversation per frontier sample from the role-aware trajectory
view :meth:`Sample.turns` (turn-major, frontier-aligned). These helpers transpose
that into per-sample message lists.

Unlike the sglang engine's equivalent (``rollout/engine/sglang/utils/conversations.py``,
which de-expands the ``*n`` wire fan-out), the trainside embeds **every** frontier
sample per-row — so there is NO de-expand here; one conversation per row.

Pure (only :mod:`unirl.types`): no tokenizer, no processor, no engine. The trainside
cannot import the sglang engine's helper (sglang is an optional dependency whose
package import pulls the backend), so the small transpose is mirrored here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from unirl.types.primitives import Audios, Images, Texts, Videos
from unirl.types.sample import Turn

# One sample's chat conversation: an ordered list of role-tagged messages.
Conversation = List[Dict[str, Any]]


def _system_prefix(system_instruction: Optional[str], roles: List[str]) -> Conversation:
    """The config ``system_instruction`` as a leading message, unless the trajectory
    already carries an explicit ``system`` turn (which wins)."""
    if system_instruction and "system" not in roles:
        return [{"role": "system", "content": system_instruction}]
    return []


def _group_consecutive_roles(roles: List[str]) -> List[Tuple[str, List[int]]]:
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
    """
    if not turns:
        return []
    roles = [t.role for t in turns]
    cols = [t.content.texts for t in turns]
    prefix = _system_prefix(system_instruction, roles)
    n_rows = len(turns[0].content)
    return [prefix + [{"role": roles[j], "content": cols[j][row]} for j in range(len(turns))] for row in range(n_rows)]


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
    # Convert each image turn's PILs once (not per row).
    cols = [t.content.to_pils() if im else t.content.texts for t, im in zip(turns, is_image)]
    role_groups = _group_consecutive_roles(roles)
    prefix = _system_prefix(system_instruction, roles)
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


def _video_rows(videos: Videos) -> List[Any]:
    """Return one URI or packed-frame tensor per row of a ``Videos`` batch."""
    if videos.uris is not None:
        if videos.frames is not None:
            raise ValueError("build_video_messages: Videos cannot carry both uris and packed frames.")
        if len(videos.uris) != len(videos):
            raise ValueError(f"build_video_messages: video URI count {len(videos.uris)} != video batch {len(videos)}.")
        return list(videos.uris)

    rows = videos.to_list()
    if len(rows) != len(videos):
        raise ValueError("build_video_messages: Videos carries neither batch-aligned uris nor valid packed frames.")
    return [row.frames for row in rows]


def build_video_messages(
    turns: List[Turn],
    system_instruction: Optional[str] = None,
) -> List[Conversation]:
    """One text+optional-video conversation per frontier row.

    Qwen3-Omni's current agentic contract permits one persistent source-video
    turn plus arbitrary role-aware text turns. Consecutive same-role turns are
    fused, with the video block placed before text blocks so the initial
    ``text input Part -> video input child`` dataset shape renders to the
    checkpoint's canonical ``[video, text]`` user message.
    """
    if not turns:
        return []

    unsupported = [type(turn.content).__name__ for turn in turns if not isinstance(turn.content, (Texts, Videos))]
    if unsupported:
        raise ValueError(
            f"build_video_messages: Qwen3-Omni requires text/video turns only; got unsupported content {unsupported}."
        )

    video_turns = sum(isinstance(turn.content, Videos) for turn in turns)
    if video_turns > 1:
        raise ValueError(
            "build_video_messages: Qwen3-Omni currently supports at most one persistent "
            f"source-video turn per trajectory, got {video_turns}."
        )

    roles = [turn.role for turn in turns]
    is_video = [isinstance(turn.content, Videos) for turn in turns]
    cols = [_video_rows(turn.content) if video else list(turn.content.texts) for turn, video in zip(turns, is_video)]
    n_rows = len(turns[0].content)
    mismatched = [i for i, turn in enumerate(turns) if len(turn.content) != n_rows]
    if mismatched:
        raise ValueError(
            f"build_video_messages: frontier-aligned turns must share batch size {n_rows}; "
            f"mismatched turn indices {mismatched}."
        )

    role_groups = _group_consecutive_roles(roles)
    prefix = _system_prefix(system_instruction, roles)
    conversations: List[Conversation] = []
    for row in range(n_rows):
        messages: Conversation = list(prefix)
        for role, indices in role_groups:
            video_blocks: List[Dict[str, Any]] = []
            text_blocks: List[Dict[str, Any]] = []
            for index in indices:
                if is_video[index]:
                    video_blocks.append({"type": "video", "video": cols[index][row]})
                else:
                    text_blocks.append({"type": "text", "text": cols[index][row]})
            messages.append({"role": role, "content": video_blocks + text_blocks})
        conversations.append(messages)
    return conversations


def build_omni_messages(
    turns: List[Turn],
    system_instruction: Optional[str] = None,
    *,
    prompt_media_refs: Optional[List[List[Any]]] = None,
) -> List[Conversation]:
    """Render Qwen3-Omni text/audio/video conversations per row.

    ``prompt_media_refs`` is the sparse request-side channel used by mixed
    batches. Each row contains normalized refs with ``modality`` and ``uri``;
    refs are prepended to the first user message without forcing a rectangular
    media primitive Part.
    """
    if not turns:
        return []
    supported = (Texts, Videos, Audios)
    unsupported = [type(turn.content).__name__ for turn in turns if not isinstance(turn.content, supported)]
    if unsupported:
        raise ValueError(f"build_omni_messages: unsupported turn content {unsupported}.")

    n_rows = len(turns[0].content)
    mismatched = [i for i, turn in enumerate(turns) if len(turn.content) != n_rows]
    if mismatched:
        raise ValueError(
            f"build_omni_messages: frontier-aligned turns must share batch size {n_rows}; "
            f"mismatched turn indices {mismatched}."
        )
    refs_by_row = prompt_media_refs if prompt_media_refs is not None else [[] for _ in range(n_rows)]
    if len(refs_by_row) != n_rows:
        raise ValueError(f"build_omni_messages: media-ref rows {len(refs_by_row)} != text rows {n_rows}.")

    roles = [turn.role for turn in turns]
    role_groups = _group_consecutive_roles(roles)
    prefix = _system_prefix(system_instruction, roles)
    columns: List[List[Any]] = []
    for turn in turns:
        if isinstance(turn.content, Texts):
            columns.append(list(turn.content.texts))
        elif isinstance(turn.content, Videos):
            columns.append(_video_rows(turn.content))
        else:
            columns.append([audio.waveform for audio in turn.content.to_list()])

    conversations: List[Conversation] = []
    for row in range(n_rows):
        messages: Conversation = list(prefix)
        for role, indices in role_groups:
            media_blocks: List[Dict[str, Any]] = []
            text_blocks: List[Dict[str, Any]] = []
            for index in indices:
                content = turns[index].content
                value = columns[index][row]
                if isinstance(content, Texts):
                    text_blocks.append({"type": "text", "text": value})
                elif isinstance(content, Videos):
                    media_blocks.append({"type": "video", "video": value})
                else:
                    media_blocks.append({"type": "audio", "audio": value})
            messages.append({"role": role, "content": media_blocks + text_blocks})

        sparse_blocks: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for ref in refs_by_row[row] or []:
            modality = str(ref.get("modality") if isinstance(ref, dict) else getattr(ref, "modality", "")).lower()
            uri = ref.get("uri") if isinstance(ref, dict) else getattr(ref, "uri", None)
            if modality not in {"audio", "video"}:
                raise ValueError(f"build_omni_messages: unsupported prompt media modality {modality!r}.")
            if modality in seen:
                raise ValueError(f"build_omni_messages: row {row} has more than one {modality} prompt input.")
            seen.add(modality)
            sparse_blocks.append({"type": modality, modality: uri})
        if sparse_blocks:
            user_message = next((message for message in messages if message.get("role") == "user"), None)
            if user_message is None:
                messages.append({"role": "user", "content": sparse_blocks})
            elif isinstance(user_message.get("content"), list):
                user_message["content"] = sparse_blocks + user_message["content"]
            else:
                user_message["content"] = sparse_blocks + [{"type": "text", "text": str(user_message["content"])}]
        conversations.append(messages)
    return conversations


__all__ = [
    "Conversation",
    "build_text_messages",
    "build_omni_messages",
    "build_video_messages",
    "build_vision_messages",
]
