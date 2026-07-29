"""Trajectory → chat-conversation rendering (pure, tokenizer-free).

The AR adapters encode one chat conversation per *prompt*, but the trajectory
view :meth:`Sample.text_conditioning` / :meth:`Sample.vision_conditioning`
surfaces it turn-major and **frontier-aligned**: ``T`` turns, each holding one
row per frontier sample (``P`` prompts × ``n`` fan-out = ``P*n`` rows). These
helpers transpose that into ``P`` per-sample message lists and de-expand the
``*n`` fan-out back to the unique conversations the backend fans out server-side.

Pure (only :mod:`unirl.types`): no tokenizer, no processor, no engine state — so
the message-shape logic unit-tests with canned ``Sample``s. The *model-specific*
encode (``apply_chat_template`` / processor) stays in the adapter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from unirl.types.primitives import Images
from unirl.types.sample import Sample

# One sample's chat conversation: an ordered list of role-tagged messages.
Conversation = List[Dict[str, Any]]


def unique_group_indices(group_ids: List[str]) -> Tuple[List[int], int]:
    """Representative row index per fan-out group + the uniform repeat ``k``.

    The frontier is group-by-parent **contiguous** (siblings adjacent — the
    ``fork`` contract), so the first occurrence of each parent id yields the
    group-order representatives ``[0, k, 2k, …]``. That order is the one
    ``TextLMAdapter.build_response`` expects for ``raw`` (``[conv0×k, conv1×k,
    …]``), so the de-expanded ``wire`` lines up with the filled gen rows.

    Returns ``(rep_indices, k)``. Falls back to ``(range(n), 1)`` — one
    conversation per row, no de-expand — when groups are non-uniform or empty
    (e.g. a tree-shaped trajectory with branch>1 at an intermediate turn), which
    degrades safely rather than mis-collapsing.
    """
    n = len(group_ids)
    if n == 0:
        return [], 1

    rep_indices: List[int] = []
    sizes: Dict[str, int] = {}
    for i, gid in enumerate(group_ids):
        if gid not in sizes:
            sizes[gid] = 0
            rep_indices.append(i)  # first occurrence == group-order representative
        sizes[gid] += 1

    k_values = set(sizes.values())
    if len(k_values) != 1:
        return list(range(n)), 1
    return rep_indices, next(iter(k_values))


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


def _system_prefix(system_instruction: Any, roles: List[str]) -> Conversation:
    """The config ``system_instruction`` as a leading message, unless the
    trajectory already carries an explicit ``system`` turn (which wins)."""
    if system_instruction and "system" not in roles:
        return [{"role": "system", "content": system_instruction}]
    return []


def build_text_conversations(
    sample: Sample,
    system_instruction: Any = None,
) -> Tuple[List[Conversation], int]:
    """The all-text trajectory as ``P`` unique chat conversations + fan-out ``k``.

    Transposes :meth:`Sample.text_conditioning` (turn-major, frontier-aligned)
    into one message list per unique prompt, then de-expands the ``*n`` siblings.
    Degenerates to today's single ``user`` message when no roles are set, so it
    is byte-identical on single-turn workloads.
    """
    turns = sample.text_conditioning()
    rep, k = unique_group_indices(sample.parts[-1].group_ids)
    roles = [t.role for t in turns]
    cols = [t.content.texts for t in turns]
    prefix = _system_prefix(system_instruction, roles)

    conversations = [prefix + [{"role": roles[j], "content": cols[j][row]} for j in range(len(turns))] for row in rep]
    return conversations, k


def build_vision_conversations(
    sample: Sample,
    system_instruction: Any = None,
) -> Tuple[List[Conversation], List[List[Any]], int]:
    """The text+image trajectory as ``P`` conversations, their per-conv PIL
    images, and fan-out ``k``.

    Same transpose as :func:`build_text_conversations`, but consecutive same-role
    turns fuse into one message whose content is ``[image blocks…, text blocks…]``
    — image-before-text, matching ``VLMAdapter.encode_mm`` and the trainside
    ``QwenVLChatTemplateStage`` ordering. ``images_list[g]`` are that
    conversation's PILs in placeholder order, for the processor.
    """
    turns, _ = sample.vision_conditioning()
    rep, k = unique_group_indices(sample.parts[-1].group_ids)
    roles = [t.role for t in turns]
    is_image = [isinstance(t.content, Images) for t in turns]
    # Convert each image turn's PILs once (not per representative row).
    cols = [t.content.to_pils() if im else t.content.texts for t, im in zip(turns, is_image)]
    role_groups = _group_consecutive_roles(roles)
    prefix = _system_prefix(system_instruction, roles)

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
