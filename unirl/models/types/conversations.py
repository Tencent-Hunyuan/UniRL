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

from unirl.types.primitives import Images, Texts, Videos
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


__all__ = [
    "Conversation",
    "build_text_messages",
    "build_video_messages",
    "build_vision_messages",
]
