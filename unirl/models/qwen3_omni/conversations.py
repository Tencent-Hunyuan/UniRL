"""Qwen3-Omni trajectory → chat-conversation rendering (model-owned policy).

Qwen3-Omni's prompt-media contract (URI-only ``MediaRefs``, at most one prompt
input per modality per row) is this model's policy, so it lives here rather
than in the shared :mod:`unirl.models.types.conversations` layer — that module
keeps only the multi-model transpose helpers this one builds on.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from unirl.models.types.conversations import (
    Conversation,
    _group_consecutive_roles,
    _system_prefix,
)
from unirl.types.media import MediaRef, MediaRefs
from unirl.types.primitives import Audios, Images, Texts, Videos
from unirl.types.sample import Turn


def _uri_videos_to_media_refs(videos: Videos, *, context: str) -> MediaRefs:
    """Compat shim: URI-backed ``Videos`` → sparse ``MediaRefs``.

    Decoded frame tensors remain unsupported for Qwen3-Omni prompts.
    """
    if videos.uris is None:
        raise ValueError(
            f"{context}: decoded Videos frames are unsupported for Qwen3-Omni prompts; "
            "use URI-backed MediaRefs (or deprecated Videos.from_uris)."
        )
    if videos.frames is not None:
        raise ValueError(f"{context}: Videos cannot carry both frames and uris for Qwen3-Omni prompts.")
    rows: List[List[MediaRef]] = []
    for uri in videos.uris:
        if uri is None or (isinstance(uri, str) and not uri.strip()):
            rows.append([])
            continue
        rows.append([MediaRef(modality="video", role="prompt", uri=uri)])
    return MediaRefs.from_rows(rows)


def build_omni_messages(
    turns: List[Turn],
    system_instruction: Optional[str] = None,
) -> List[Conversation]:
    """Render typed Qwen3-Omni text/image/audio/video conversations per row.

    Canonical prompt media is ``MediaRefs``. URI-backed ``Videos.from_uris`` is
    accepted as a temporary compatibility shim and normalized to video
    ``MediaRef`` rows; decoded ``Images`` / ``Videos`` / ``Audios`` remain
    rejected so waveform and frame tensors cannot bypass the URI contract.
    """
    if not turns:
        return []
    decoded = [
        type(turn.content).__name__
        for turn in turns
        if isinstance(turn.content, (Images, Audios))
        or (isinstance(turn.content, Videos) and turn.content.uris is None)
    ]
    if decoded:
        raise ValueError(
            "build_omni_messages: Qwen3-Omni prompt media must use URI-backed MediaRefs; "
            f"decoded primitive inputs are unsupported: {decoded}."
        )
    supported = (Texts, MediaRefs, Videos)
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

    roles = [turn.role for turn in turns]
    role_groups = _group_consecutive_roles(roles)
    prefix = _system_prefix(system_instruction, roles)
    columns: List[List[Any]] = []
    for turn in turns:
        if isinstance(turn.content, Texts):
            columns.append(list(turn.content.texts))
        elif isinstance(turn.content, MediaRefs):
            columns.append([list(refs) for refs in turn.content.rows])
        elif isinstance(turn.content, Videos):
            warnings.warn(
                "build_omni_messages: URI-backed Videos prompt inputs are deprecated; "
                "migrate to Part.primitives['media'] = MediaRefs.",
                DeprecationWarning,
                stacklevel=2,
            )
            columns.append(
                [list(refs) for refs in _uri_videos_to_media_refs(turn.content, context="build_omni_messages").rows]
            )
        else:  # pragma: no cover - guarded by the supported-type check above.
            raise AssertionError(type(turn.content).__name__)

    conversations: List[Conversation] = []
    for row in range(n_rows):
        messages: Conversation = list(prefix)
        seen_modalities: set[str] = set()
        for role, indices in role_groups:
            media_blocks: List[Dict[str, Any]] = []
            text_blocks: List[Dict[str, Any]] = []
            for index in indices:
                content = turns[index].content
                value = columns[index][row]
                if isinstance(content, Texts):
                    text_blocks.append({"type": "text", "text": value})
                else:
                    for ref in value:
                        if ref.role != "prompt":
                            raise ValueError(
                                f"build_omni_messages: row {row} MediaRefs only supports role='prompt', "
                                f"got {ref.role!r}."
                            )
                        modality = ref.modality
                        if modality not in {"image", "audio", "video"}:
                            raise ValueError(f"build_omni_messages: unsupported prompt media modality {modality!r}.")
                        if modality in seen_modalities:
                            raise ValueError(
                                f"build_omni_messages: row {row} has more than one {modality} prompt input."
                            )
                        seen_modalities.add(modality)
                        media_blocks.append({"type": modality, modality: ref.uri})
            if media_blocks or text_blocks:
                messages.append({"role": role, "content": media_blocks + text_blocks})
        conversations.append(messages)
    return conversations


__all__ = ["build_omni_messages"]
