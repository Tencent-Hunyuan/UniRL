#!/usr/bin/env python3
"""CPU oracle for the trainside trajectory→conversation encode (LIN-503, gap C conjugate).

Exercises the PURE message-builders the trainside AR/VLM chat-template stages now
encode through — ``unirl.models.types.conversations.build_text_messages`` /
``build_vision_messages`` — with fabricated Samples (no tokenizer, no processor, no
model, no GPU). Pins the trainside contracts: NO de-expand (one conversation per
frontier row, the key difference from the sglang wire), single-turn byte-shape,
multi-turn lineage order, system-instruction precedence, and the VLM image-before-text
fusion with inline PIL.

Run in the engine/train venv:

    python scripts/check_trainside_conversations.py

Standalone ``main() -> int``; exits non-zero on the first failed contract.
"""

from __future__ import annotations

import sys
from typing import Callable, List, Tuple

import torch

from unirl.models.types.conversations import build_text_messages, build_vision_messages
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


def check_single_turn_text() -> None:
    """No roles ⇒ a lone ``user`` message per sample (byte-identical to pre-fix shape)."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a cat", "a dog"])})  # no role
    sample = Sample(parts=[inp, inp.fork(1, sampling_params=_ar())])
    msgs = build_text_messages(sample.text_conditioning())
    _check(
        msgs
        == [
            [{"role": "user", "content": "a cat"}],
            [{"role": "user", "content": "a dog"}],
        ],
        "single-turn: one lone user message per sample",
    )


def check_no_dexpand_text() -> None:
    """THE trainside contract: every frontier sample gets its own conversation — NO
    de-expand (the sglang wire would collapse to 2 + fan out n; trainside keeps 4)."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["x", "y"])})
    sample = Sample(parts=[inp, inp.fork(2, sampling_params=_ar())])  # 2 prompts x branch 2
    msgs = build_text_messages(sample.text_conditioning())
    _check(len(msgs) == 4, f"no de-expand: 2 prompts x branch 2 -> 4 conversations, got {len(msgs)}")
    _check(
        [m[0]["content"] for m in msgs] == ["x", "x", "y", "y"],
        "each frontier row carries its prompt (siblings replicated, group-by-parent)",
    )


def check_system_rule() -> None:
    """Config ``system_instruction`` prepended iff no explicit ``system`` turn."""
    inp = Part.input(["p0"], primitives={"text": Texts(texts=["q"])})
    sample = Sample(parts=[inp, inp.fork(1, sampling_params=_ar())])
    msgs = build_text_messages(sample.text_conditioning(), "BE NICE")
    _check(
        msgs == [[{"role": "system", "content": "BE NICE"}, {"role": "user", "content": "q"}]],
        "config system_instruction prepended when no system turn",
    )

    sys_in = Part.input(["s0"], primitives={"text": Texts(texts=["SYS"])}, role="system")
    usr = sys_in.input_child({"text": Texts(texts=["q"])}, role="user")
    sample2 = Sample(parts=[sys_in, usr, usr.fork(1, sampling_params=_ar())])
    msgs2 = build_text_messages(sample2.text_conditioning(), "CONFIG SYS")
    _check([m["role"] for m in msgs2[0]] == ["system", "user"], "explicit system turn rendered")
    _check(msgs2[0][0]["content"] == "SYS", "explicit system turn wins; config not prepended")


def check_multi_turn_order() -> None:
    """user → assistant → tool renders three messages per sample, in lineage order."""
    inp = Part.input(["p0"], primitives={"text": Texts(texts=["what is 2+2?"])}, role="user")
    asst = inp.fork(1, sampling_params=_ar()).fill(primitives={"text": Texts(texts=["let me compute"])})
    tool = asst.input_child({"text": Texts(texts=["4"])}, role="tool")
    sample = Sample(parts=[inp, asst, tool, tool.fork(1, sampling_params=_ar())])
    msgs = build_text_messages(sample.text_conditioning())
    _check(
        msgs
        == [
            [
                {"role": "user", "content": "what is 2+2?"},
                {"role": "assistant", "content": "let me compute"},
                {"role": "tool", "content": "4"},
            ]
        ],
        "multi-turn conversation in lineage order",
    )


def check_vlm_fusion_inline_pil() -> None:
    """it2i [text(user), image(user)] fuses into one user message; image block BEFORE
    text, with the PIL inlined (the trainside processor format)."""
    text = Part.input(["p0"], primitives={"text": Texts(texts=["edit"])}, role="user")
    img = text.input_child({"image": _images(1)}, role="user")
    sample = Sample.request(text, img).fork(1, sampling_params=_ar())
    turns, _ = sample.vision_conditioning()
    msgs = build_vision_messages(turns)
    _check(len(msgs) == 1 and len(msgs[0]) == 1 and msgs[0][0]["role"] == "user", "one fused user message")
    content = msgs[0][0]["content"]
    _check(content[0]["type"] == "image" and "image" in content[0], "image block first, inline PIL key")
    _check(not isinstance(content[0]["image"], str), "inline PIL object (not a placeholder string)")
    _check(content[1] == {"type": "text", "text": "edit"}, "text block after image")


def check_no_dexpand_vlm() -> None:
    """VLM also keeps one conversation per frontier row (no de-expand)."""
    text = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])}, role="user")
    img = text.input_child({"image": _images(2)}, role="user")
    sample = Sample.request(text, img).fork(2, sampling_params=_ar())
    turns, _ = sample.vision_conditioning()
    msgs = build_vision_messages(turns)
    _check(len(msgs) == 4, f"VLM no de-expand: 2 x branch 2 -> 4 conversations, got {len(msgs)}")
    _check([m[0]["content"][1]["text"] for m in msgs] == ["a", "a", "b", "b"], "text per frontier row")


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_single_turn_text,
    check_no_dexpand_text,
    check_system_rule,
    check_multi_turn_order,
    check_vlm_fusion_inline_pil,
    check_no_dexpand_vlm,
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
        print("check-trainside-conversations: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"check-trainside-conversations: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
