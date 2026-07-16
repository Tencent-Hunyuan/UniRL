#!/usr/bin/env python3
"""CPU oracle for the role-aware conditioning translation layer (LIN-503, Phase 1).

Exercises the PURE Sample methods the agent-trajectory translator rests on — the
turn-role primitive (``Part.role`` / ``Part.resolved_role``), the role-tagged
``Sample.turns`` walk, the ``conditioning`` role-stripped view, the
``text_conditioning`` / ``vision_conditioning`` consumer renderers, and the
``replace_frontier`` / ``with_filled_frontier`` write-back — with fabricated
Samples (no GPU, no backend, no vllm/sglang). It guards the structural contracts
so a broken role derivation / turn walk / fail-loud surfaces here, not mid-rollout.

Run in the engine venv:

    python scripts/check_conditioning_contracts.py

Standalone ``main() -> int``; exits non-zero on the first failed contract.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Callable, List, Tuple

import torch

from unirl.types.primitives import Image, Images, Texts, Video, Videos
from unirl.types.sample import Part, Sample, Turn
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _expect_raises(fn: Callable[[], object], exc: type, msg: str) -> None:
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{msg}: expected {exc.__name__} but none was raised")


def _images(n: int) -> Images:
    return Images.from_list([Image(pixels=torch.zeros(3, 8, 8)) for _ in range(n)])


def _videos(n: int) -> Videos:
    return Videos.from_list([Video(frames=torch.zeros(2, 3, 8, 8)) for _ in range(n)])


def _diff() -> DiffusionSamplingParams:
    return DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)


def _roles(turns: List[Turn]) -> List[str]:
    return [t.role for t in turns]


def check_single_turn_backward_compat() -> None:
    """No roles set ⇒ derivation reproduces today's single-`user`-turn shape, and
    ``conditioning()`` stays the role-stripped view of ``turns()`` (behavior-preserving)."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a cat", "a dog"])})  # no role set
    sample = Sample(parts=[inp, inp.fork(1, sampling_params=_diff())])

    ts = sample.turns()
    _check(_roles(ts) == ["user"], "single-turn derives a lone 'user' turn (input, no role set)")
    _check(
        isinstance(ts[0].content, Texts) and list(ts[0].content.texts) == ["a cat", "a dog"],
        "the derived user turn carries the prompt Texts",
    )
    cond = sample.conditioning()
    _check(
        len(cond) == 1 and isinstance(cond[0], Texts) and list(cond[0].texts) == ["a cat", "a dog"],
        "conditioning() is the role-stripped view of turns() (same primitives, same order)",
    )
    _check(_roles(sample.text_conditioning()) == ["user"], "text_conditioning returns the lone user turn")


def check_vlm_turns() -> None:
    """A text + image trajectory renders as two `user` turns; vision_conditioning
    returns the turns + a 1-element image collection; text_conditioning rejects it."""
    text = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["edit the cat", "edit the dog"])}, role="user")
    img_in = text.input_child({"image": _images(2)}, role="user")
    sample = Sample.request(text, img_in).fork(1, sampling_params=_diff())

    ts, images = sample.vision_conditioning()
    _check(_roles(ts) == ["user", "user"], "vision turns: text + image both 'user'")
    _check(
        isinstance(ts[0].content, Texts) and isinstance(ts[1].content, Images),
        "turn order is [Texts, Images] (lineage order)",
    )
    _check(len(images) == 1 and isinstance(images[0], Images), "images is a collection (1 element today)")
    _expect_raises(sample.text_conditioning, ValueError, "text_conditioning must reject an image turn")


def check_recaption_turns() -> None:
    """The unified two-text recaption is a 2-turn conversation: [user prompt,
    assistant recaption] — the assistant role DERIVED from the gen part's params."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a cat", "a dog"])}, role="user")
    ar = inp.fork(1, sampling_params=ARSamplingParams()).fill(
        primitives={"text": Texts(texts=["a fluffy cat", "a happy dog"])}
    )
    sample = Sample(parts=[inp, ar, ar.fork(1, sampling_params=_diff())])

    ts = sample.text_conditioning()
    _check(_roles(ts) == ["user", "assistant"], "recaption: [user prompt, assistant recaption] (assistant derived)")
    _check(
        [list(t.content.texts) for t in ts] == [["a cat", "a dog"], ["a fluffy cat", "a happy dog"]],
        "turn contents are prompt then recaption, in lineage order",
    )


def check_agent_text_turns() -> None:
    """A user → assistant → tool agent trajectory renders 3 role-tagged text turns."""
    inp = Part.input(["p0"], primitives={"text": Texts(texts=["what is 2+2?"])}, role="user")
    asst = inp.fork(1, sampling_params=ARSamplingParams()).fill(primitives={"text": Texts(texts=["let me compute"])})
    tool = asst.input_child({"text": Texts(texts=["4"])}, role="tool")
    gen = tool.fork(1, sampling_params=ARSamplingParams())
    sample = Sample(parts=[inp, asst, tool, gen])

    ts = sample.text_conditioning()
    _check(_roles(ts) == ["user", "assistant", "tool"], "agent text: 3 role-tagged turns in lineage order")
    _check(
        [list(t.content.texts) for t in ts] == [["what is 2+2?"], ["let me compute"], ["4"]],
        "turn contents in chronological order",
    )


def check_frontier_alignment() -> None:
    """Every turn's content is frontier-aligned: a branch>1 fan-out expands each
    ancestor to one row per gen sample (the per-sample conversation)."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["look", "see"])}, role="user")
    img = inp.input_child({"image": _images(2)}, role="user")
    sample = Sample.request(inp, img).fork(2, sampling_params=_diff())  # 2 prompts x branch 2 = 4

    n = sample.parts[-1].batch_size
    _check(n == 4, "fan-out gen frontier holds 4 samples (2 prompts x branch 2)")
    ts, images = sample.vision_conditioning()
    _check(_roles(ts) == ["user", "user"], "fan-out VLM: text + image turns")
    _check(all(len(t.content) == n for t in ts), "every turn is frontier-aligned (one row per gen sample)")
    _check(len(images[0]) == n, "the image collection is expanded to the frontier fan-out")


def check_fail_loud() -> None:
    """The renderers fail loud on what their consumer cannot ingest."""
    # text + image + video: text_conditioning rejects non-text; vision rejects the extra modality.
    inp = Part.input(["p0"], primitives={"text": Texts(texts=["q"])}, role="user")
    img = inp.input_child({"image": _images(1)}, role="user")
    vid = img.input_child({"video": _videos(1)}, role="tool")
    mixed = Sample.request(inp, img, vid).fork(1, sampling_params=_diff())
    _expect_raises(mixed.text_conditioning, ValueError, "text_conditioning rejects image/video turns")
    _expect_raises(mixed.vision_conditioning, ValueError, "vision_conditioning rejects an extra (video) modality")

    # text-only trajectory: vision_conditioning rejects a no-image request.
    text_inp = Part.input(["p0"], primitives={"text": Texts(texts=["q"])}, role="user")
    text_only = Sample(parts=[text_inp, text_inp.fork(1, sampling_params=ARSamplingParams())])
    _expect_raises(text_only.vision_conditioning, ValueError, "vision_conditioning rejects a no-image trajectory")


def check_writeback_preserves_intermediates() -> None:
    """``with_filled_frontier`` / ``replace_frontier`` preserve EVERY intermediate
    part (the sglang_diffusion part-drop bug) and ``reward_compute_s``."""
    inp = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])}, role="user")
    ar = inp.fork(1, sampling_params=ARSamplingParams()).fill(primitives={"text": Texts(texts=["r0", "r1"])})
    gen = ar.fork(1, sampling_params=_diff())
    sample = Sample(parts=[inp, ar, gen], reward_compute_s=1.5)

    seg, dec = SimpleNamespace(tag="seg"), _images(2)
    out = sample.with_filled_frontier(segment=seg, primitives={"image": dec})
    _check(len(out.parts) == 3, "with_filled_frontier preserves the part count (no drop)")
    _check(out.parts[0] is inp and out.parts[1] is ar, "intermediate parts preserved (the dropped-part bug)")
    _check(
        out.parts[-1].segment is seg and out.parts[-1].primitives["image"] is dec,
        "frontier filled with outputs",
    )
    _check(out.parts[-1].sampling_params is gen.sampling_params, "frontier sampling_params preserved through fill")
    _check(out.reward_compute_s == 1.5, "with_filled_frontier preserves reward_compute_s")

    new_gen = gen.fill(primitives={"text": Texts(texts=["x", "y"])})
    out2 = sample.replace_frontier(new_gen)
    _check(
        out2.parts[0] is inp and out2.parts[1] is ar and out2.parts[-1] is new_gen,
        "replace_frontier swaps only the frontier, keeping parts[:-1] whole",
    )
    _check(out2.reward_compute_s == 1.5, "replace_frontier preserves reward_compute_s")


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_single_turn_backward_compat,
    check_vlm_turns,
    check_recaption_turns,
    check_agent_text_turns,
    check_frontier_alignment,
    check_fail_loud,
    check_writeback_preserves_intermediates,
)


def main() -> int:
    failures = []
    for check in _CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — the oracle reports, doesn't crash
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok  {check.__name__}")
    if failures:
        print("check-conditioning-contracts: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"check-conditioning-contracts: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
