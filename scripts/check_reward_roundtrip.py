#!/usr/bin/env python3
"""Runtime oracle for the reward-adapter Sample → Sample migration (LIN-481).

Guards the PURE reward-assembly logic the migrated ``RewardService`` relies on —
``primitive_modality_key``, ``Sample.root_metadata``, the ``_build_reward_request``
builder (conditioning → primitives keying, nearest-ancestor caption, root-metadata
alignment, the modality guard), and the full ``score_and_attach`` path driven by a
stub backend (a ``@distributed`` method runs inline when called directly, so no
worker group / GPU is needed). It does NOT boot a real scorer.

Only needs the unirl runtime types (torch). Run in the engine venv:

    python scripts/check_reward_roundtrip.py

Standalone ``main() -> int``; exits non-zero on the first failed contract.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import torch

from unirl.reward.service import RewardService, _build_reward_request
from unirl.types.primitives import Image, Images, Texts, primitive_modality_key
from unirl.types.reward import RewardRequest, RewardResponse
from unirl.types.sample import Part, Sample, _part_with_field
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


def _diff_params() -> DiffusionSamplingParams:
    return DiffusionSamplingParams(num_inference_steps=4, height=256, width=256)


class _StubBackend:
    """Minimal duck-typed RewardBackend: returns fixed scores, no model."""

    def __init__(
        self,
        preferred_input_kind: str = "image",
        rewards: Optional[List[float]] = None,
        components: Optional[dict] = None,
        successes: Optional[List[bool]] = None,
        errors: Optional[List[Optional[str]]] = None,
    ) -> None:
        self.preferred_input_kind = preferred_input_kind
        self._rewards = rewards
        self._components = components or {}
        self._successes = successes
        self._errors = errors

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        n = request.batch_size
        rewards = list(self._rewards) if self._rewards is not None else [0.0] * n
        successes = list(self._successes) if self._successes is not None else [True] * len(rewards)
        errors = list(self._errors) if self._errors is not None else [None] * len(rewards)
        return RewardResponse(
            rewards=rewards,
            component_rewards={k: list(v) for k, v in self._components.items()},
            successes=successes,
            errors=errors,
            compute_time=0.0,
        )

    def get_model_name(self) -> str:
        return "stub"

    def is_available(self) -> bool:
        return True

    def offload(self) -> None:
        pass

    def onload(self) -> None:
        pass

    def dispose(self) -> None:
        pass


def _t2i_sample(n_prompts: int = 2, branch: int = 2, metadata: Optional[list] = None) -> Sample:
    """``[root(P), image(P*branch)]`` with the image Part filled — a scorable T2I Sample."""
    root = Part.input(
        [f"p{i}" for i in range(n_prompts)],
        primitives={"text": Texts(texts=[f"prompt {i}" for i in range(n_prompts)])},
        metadata=metadata,
    )
    image = root.fork(branch, sampling_params=_diff_params()).fill(primitives={"image": _images(n_prompts * branch)})
    return Sample(parts=[root, image], reward_compute_s=1.5)


def _pe_sample() -> Sample:
    """``[root(orig text), ar(rewrite text), image]`` — both ancestor texts populated."""
    root = Part.input(["p0"], primitives={"text": Texts(texts=["original prompt"])})
    ar = root.fork(1, sampling_params=ARSamplingParams()).fill(
        primitives={"text": Texts(texts=["the rewritten prompt"])}
    )
    image = ar.fork(1, sampling_params=_diff_params()).fill(primitives={"image": _images(1)})
    return Sample(parts=[root, ar, image])


def check_primitive_modality_key() -> None:
    """Maps each batched primitive to its modality slot; rejects non-primitives."""
    _check(primitive_modality_key(Texts(texts=["x"])) == "text", "Texts -> 'text'")
    _check(primitive_modality_key(_images(1)) == "image", "Images -> 'image'")
    _expect_raises(lambda: primitive_modality_key(object()), TypeError, "non-primitive must raise TypeError")


def check_root_metadata() -> None:
    """``Sample.root_metadata`` projects the root prompt's metadata onto a descendant
    Part's rows (lineage walk); a metadata-free root yields all-None."""
    md = [{"vqa": "cat"}, {"vqa": "dog"}]
    root = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a cat", "a dog"])}, metadata=md)
    ar = root.fork(2, sampling_params=ARSamplingParams())
    image = ar.fork(1, sampling_params=_diff_params())
    sample = Sample(parts=[root, ar, image])

    _check(list(image.sample_ids) == ["p0/0/0", "p0/1/0", "p1/0/0", "p1/1/0"], "image fan-out ids")
    _check(
        sample.root_metadata(2) == [{"vqa": "cat"}, {"vqa": "cat"}, {"vqa": "dog"}, {"vqa": "dog"}],
        "root metadata aligned to each image's ROOT prompt",
    )
    _check(sample.root_metadata(-1) == sample.root_metadata(2), "default part_index=-1 is the frontier")
    _check(sample.root_metadata(0) == md, "root part: metadata is itself (one per prompt)")

    no_md_root = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])})
    no_md = Sample(parts=[no_md_root, no_md_root.fork(2, sampling_params=_diff_params())])
    _check(no_md.root_metadata(-1) == [None, None, None, None], "metadata-free root -> all-None, frontier length")


def check_build_reward_request() -> None:
    """``_build_reward_request`` keys conditioning into ``primitives`` (nearest ancestor
    wins), pairs the frontier ``primitives`` map as ``generated``, aligns root metadata, and
    guards the backend/frontier modality match."""
    # T2I: single text ancestor, row-aligned to the 4 images; metadata expands per image.
    sample = _t2i_sample(n_prompts=2, branch=2, metadata=[{"m": 0}, {"m": 1}])
    req = _build_reward_request(sample, "image")
    _check(set(req.primitives) == {"text"}, "T2I primitives keyed by modality slot")
    _check(
        list(req.primitives["text"].texts) == ["prompt 0", "prompt 0", "prompt 1", "prompt 1"],
        "conditioning text row-aligned to the frontier images",
    )
    _check(set(req.generated) == {"image"} and len(req.generated["image"]) == 4, "frontier primitive is 'generated'")
    _check(req.metadata == [{"m": 0}, {"m": 0}, {"m": 1}, {"m": 1}], "root metadata aligned per image")
    _check(list(req.sample_ids) == list(sample.parts[-1].sample_ids), "sample_ids come from the frontier")

    # Modality guard: a text-consuming backend on an image frontier must fail loud.
    _expect_raises(
        lambda: _build_reward_request(sample, "text"), ValueError, "backend/frontier modality mismatch raises"
    )

    # PE: two text ancestors (original + rewrite) — the NEAREST (rewrite) wins the slot.
    pe = _pe_sample()
    pe_req = _build_reward_request(pe, "image")
    _check(
        list(pe_req.primitives["text"].texts) == ["the rewritten prompt"], "caption = nearest text ancestor (rewrite)"
    )


def check_score_and_attach_t2i() -> None:
    """End-to-end: rewards + component_rewards land on the frontier Part; the input
    Part and ``reward_compute_s`` pass through untouched."""
    sample = _t2i_sample(n_prompts=2, branch=2)
    backend = _StubBackend("image", rewards=[0.1, 0.2, 0.3, 0.4], components={"clip": [1.0, 2.0, 3.0, 4.0]})
    service = RewardService(backend)

    out = service.score_and_attach(sample)
    frontier = out.parts[-1]
    _check(
        frontier.rewards is not None and torch.allclose(frontier.rewards, torch.tensor([0.1, 0.2, 0.3, 0.4])),
        "rewards on frontier",
    )
    _check(
        frontier.component_rewards is not None
        and torch.allclose(frontier.component_rewards["clip"], torch.tensor([1.0, 2.0, 3.0, 4.0])),
        "component_rewards on frontier",
    )
    _check(out.parts[0] is sample.parts[0], "input Part passes through unchanged")
    _check(sample.parts[-1].rewards is None, "input Sample is not mutated (returns a copy)")
    _check(out.reward_compute_s == 1.5, "reward_compute_s preserved through with_parts")


def check_score_and_attach_ar_truncation() -> None:
    """AR truncation shaping fires only on an AR frontier: a sample that hit
    ``max_new_tokens`` is zeroed under ``truncated_reward='zero'`` and kept under 'keep'."""
    root = Part.input(["p0", "p1"], primitives={"text": Texts(texts=["a", "b"])})
    ar = root.fork(1, sampling_params=ARSamplingParams(max_new_tokens=10)).fill(
        primitives={"text": Texts(texts=["o0", "o1"])},
        segment=SimpleNamespace(lengths=torch.tensor([10, 5])),  # sample 0 hit the cap
    )
    sample = Sample(parts=[root, ar])
    backend = _StubBackend("text", rewards=[1.0, 1.0])

    out_zero = RewardService(backend, truncated_reward="zero").score_and_attach(sample)
    _check(torch.allclose(out_zero.parts[-1].rewards, torch.tensor([0.0, 1.0])), "truncated (len>=cap) sample zeroed")

    out_keep = RewardService(backend, truncated_reward="keep").score_and_attach(sample)
    _check(torch.allclose(out_keep.parts[-1].rewards, torch.tensor([1.0, 1.0])), "'keep' leaves the raw score")


def check_score_and_attach_guards() -> None:
    """Fail-fast: precomputed rewards, an unfilled frontier, and per-sample failures all raise."""
    service = RewardService(_StubBackend("image", rewards=[0.0, 0.0, 0.0, 0.0]))

    scored = _t2i_sample()
    scored = scored.with_parts([scored.parts[0], _part_with_field(scored.parts[-1], "rewards", torch.zeros(4))])
    _expect_raises(lambda: service.score_and_attach(scored), RuntimeError, "precomputed rewards on frontier must raise")

    root = Part.input(["p0"], primitives={"text": Texts(texts=["x"])})
    unfilled = Sample(parts=[root, root.fork(1, sampling_params=_diff_params())])  # empty primitive map
    _expect_raises(lambda: service.score_and_attach(unfilled), ValueError, "unfilled frontier must raise")

    failing = RewardService(_StubBackend("image", rewards=[0.0, 0.0, 0.0, 0.0], successes=[True, False, True, True]))
    _expect_raises(lambda: failing.score_and_attach(_t2i_sample()), RuntimeError, "a flagged failure must raise")


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_primitive_modality_key,
    check_root_metadata,
    check_build_reward_request,
    check_score_and_attach_t2i,
    check_score_and_attach_ar_truncation,
    check_score_and_attach_guards,
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
        print("check-reward-roundtrip: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"check-reward-roundtrip: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
