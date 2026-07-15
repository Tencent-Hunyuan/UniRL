#!/usr/bin/env python3
"""CPU oracle for the sglang AR trajectory→conversation encode (LIN-503, gap C).

Exercises the PURE message-builders the sglang text/VLM adapters now encode
through — :func:`unique_group_indices`, :func:`build_text_conversations`,
:func:`build_vision_conversations` — with fabricated Samples (no tokenizer, no
processor, no backend, no GPU). It pins the contracts the adapters rely on: the
single-turn byte-shape (behavior-preserving), multi-turn lineage order, the
``*n`` fan-out de-expand (first-occurrence/group order, matching
``build_response``'s ``raw`` ordering), the system-instruction precedence rule,
and the VLM image-before-text fusion (``encode_mm`` parity).

Run in the engine venv (imports the sglang adapter package's utils):

    python scripts/check_ar_conversations.py

Standalone ``main() -> int``; exits non-zero on the first failed contract.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Tuple

import torch

from unirl.rollout.engine.sglang.utils.conversations import (
    build_text_conversations,
    build_vision_conversations,
    unique_group_indices,
)
from unirl.types.primitives import Image, Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _images(n: int) -> Images:
    return Images.from_list([Image(pixels=torch.zeros(3, 8, 8)) for _ in range(n)])


def _ar() -> ARSamplingParams:
    return ARSamplingParams()


def check_unique_group_indices() -> None:
    """First-occurrence representatives + uniform ``k``; safe fallback otherwise."""
    rep, k = unique_group_indices(["a", "a", "b", "b", "c", "c"])
    _check(rep == [0, 2, 4] and k == 2, "uniform contiguous groups -> first-occurrence reps + k")
    rep, k = unique_group_indices(["a", "b", "c"])
    _check(rep == [0, 1, 2] and k == 1, "singleton groups -> identity reps, k=1")
    rep, k = unique_group_indices(["a", "a", "b"])
    _check(rep == [0, 1, 2] and k == 1, "non-uniform groups -> (range, 1) safe fallback")
    rep, k = unique_group_indices([])
    _check(rep == [] and k == 1, "empty -> ([], 1)")


def check_single_turn_text() -> None:
    """No roles set ⇒ each prompt renders a lone ``user`` message (byte-identical to
    the pre-gap-C hardcoded shape); fan-out de-expands to P unique conversations."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a cat", "a dog"])})  # no role
    sample = Sample(parts=[inp, inp.fork(2, sampling_params=_ar())])  # branch 2
    convs, k = build_text_conversations(sample)
    _check(k == 2, "k == fork branch (2)")
    _check(
        convs
        == [
            [{"role": "user", "content": "a cat"}],
            [{"role": "user", "content": "a dog"}],
        ],
        "P=2 unique single-user-message conversations, group order",
    )


def check_single_turn_system_rule() -> None:
    """Config ``system_instruction`` is prepended only when no explicit ``system``
    turn exists; an explicit system turn wins."""
    inp = Part.input(["p0"], primitives={"text": Texts(texts=["q"])})
    sample = Sample(parts=[inp, inp.fork(1, sampling_params=_ar())])
    convs, _ = build_text_conversations(sample, "BE NICE")
    _check(
        convs == [[{"role": "system", "content": "BE NICE"}, {"role": "user", "content": "q"}]],
        "config system_instruction prepended when no system turn",
    )

    sys_in = Part.input(["s0"], primitives={"text": Texts(texts=["SYS"])}, role="system")
    usr = sys_in.input_child({"text": Texts(texts=["q"])}, role="user")
    sample2 = Sample(parts=[sys_in, usr, usr.fork(1, sampling_params=_ar())])
    convs2, _ = build_text_conversations(sample2, "CONFIG SYS")
    _check([m["role"] for m in convs2[0]] == ["system", "user"], "explicit system turn rendered")
    _check(convs2[0][0]["content"] == "SYS", "explicit system turn wins; config not prepended")


def check_multi_turn_text_order() -> None:
    """user → assistant → tool renders three messages in lineage order."""
    inp = Part.input(["p0"], primitives={"text": Texts(texts=["what is 2+2?"])}, role="user")
    asst = inp.fork(1, sampling_params=_ar()).fill(primitives={"text": Texts(texts=["let me compute"])})
    tool = asst.input_child({"text": Texts(texts=["4"])}, role="tool")
    sample = Sample(parts=[inp, asst, tool, tool.fork(1, sampling_params=_ar())])
    convs, k = build_text_conversations(sample)
    _check(k == 1, "branch-1 chain -> k=1")
    _check(
        convs
        == [
            [
                {"role": "user", "content": "what is 2+2?"},
                {"role": "assistant", "content": "let me compute"},
                {"role": "tool", "content": "4"},
            ]
        ],
        "multi-turn conversation in lineage order",
    )


def check_multi_turn_dexpand_fanout() -> None:
    """A final-turn branch>1 over a multi-turn prefix collapses to one conversation
    (the shared prefix), ``k`` = the final branch."""
    inp = Part.input(["p0"], primitives={"text": Texts(texts=["q"])}, role="user")
    asst = inp.fork(1, sampling_params=_ar()).fill(primitives={"text": Texts(texts=["a"])})
    tool = asst.input_child({"text": Texts(texts=["t"])}, role="tool")
    sample = Sample(parts=[inp, asst, tool, tool.fork(3, sampling_params=_ar())])
    convs, k = build_text_conversations(sample)
    _check(k == 3, "k == final fork branch (3)")
    _check(len(convs) == 1, "single group -> one unique conversation")
    _check([m["role"] for m in convs[0]] == ["user", "assistant", "tool"], "prefix rendered once")


def check_two_prompt_fanout() -> None:
    """Two prompts × branch 2 → two conversations, representatives in group order."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["x", "y"])}, role="user")
    sample = Sample(parts=[inp, inp.fork(2, sampling_params=_ar())])
    convs, k = build_text_conversations(sample)
    _check(k == 2 and len(convs) == 2, "2 prompts x branch 2 -> 2 convs, k=2")
    _check([c[0]["content"] for c in convs] == ["x", "y"], "reps pick the right unique prompt, group order")


def check_vlm_fusion() -> None:
    """it2i [text(user), image(user)] fuses into one user message with the image
    block BEFORE the text block — byte-identical to ``encode_mm``."""
    text = Part.input(["p0"], primitives={"text": Texts(texts=["edit"])}, role="user")
    img = text.input_child({"image": _images(1)}, role="user")
    sample = Sample.request(text, img).fork(1, sampling_params=_ar())
    convs, images_list, k = build_vision_conversations(sample)
    _check(k == 1, "branch-1 -> k=1")
    _check(
        convs == [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "edit"}]}]],
        "one fused user message: image block before text block",
    )
    _check(len(images_list) == 1 and len(images_list[0]) == 1, "one PIL image bundled for the conversation")


def check_vlm_dexpand() -> None:
    """Two image+text prompts × branch 2 → two conversations + two image bundles."""
    text = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])}, role="user")
    img = text.input_child({"image": _images(2)}, role="user")
    sample = Sample.request(text, img).fork(2, sampling_params=_ar())
    convs, images_list, k = build_vision_conversations(sample)
    _check(k == 2 and len(convs) == 2 and len(images_list) == 2, "2 prompts x branch 2 -> 2 convs/images, k=2")
    _check([c[0]["content"][1]["text"] for c in convs] == ["a", "b"], "reps pick the right prompt per group")
    _check(all(len(im) == 1 for im in images_list), "one image per conversation")


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_unique_group_indices,
    check_single_turn_text,
    check_single_turn_system_rule,
    check_multi_turn_text_order,
    check_multi_turn_dexpand_fanout,
    check_two_prompt_fanout,
    check_vlm_fusion,
    check_vlm_dexpand,
)


def main() -> int:
    failures: List[str] = []
    for check in _CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — the oracle reports, doesn't crash
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok  {check.__name__}")
    if failures:
        print("check-ar-conversations: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"check-ar-conversations: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
