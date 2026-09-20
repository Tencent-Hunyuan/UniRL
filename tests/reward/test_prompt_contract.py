from __future__ import annotations

from typing import cast

import pytest
import torch
from PIL import Image

from unirl.reward.base import RewardBackend
from unirl.reward.local.llm_judge import LLMJudgeRewardScorer, LLMJudgeSpec
from unirl.reward.local.per_domain import _select_request
from unirl.reward.remote import RemoteRewardBackend, RemoteRewardSpec
from unirl.reward.service import _build_reward_request
from unirl.types.primitives import Images, Texts
from unirl.types.reward import PromptSource, RewardRequest, RewardResponse
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


def _images(batch_size: int) -> Images:
    return Images.from_dense(torch.zeros(batch_size, 3, 2, 2))


def test_composed_request_keeps_distinct_prompts_with_nonuniform_lineage() -> None:
    root = Part.input(
        sample_ids=["p0", "p1"],
        primitives={"text": Texts(texts=["original-0", "original-1"])},
        metadata=[{"explanation": "e0"}, {"explanation": "e1"}],
    )
    rewrite = Part(
        sample_ids=["p0/0", "p1/0"],
        sampling_params=ARSamplingParams(),
        primitives={"text": Texts(texts=["rewrite-0", "rewrite-1"])},
    )
    image = Part(
        sample_ids=["p0/0/0", "p1/0/0", "p1/0/1", "p1/0/2"],
        sampling_params=DiffusionSamplingParams(),
        primitives={"image": _images(4)},
    )

    request = _build_reward_request(Sample(parts=[root, rewrite, image]), "image")

    assert request.original_prompt is not None
    assert request.original_prompt.texts == ["original-0", "original-1", "original-1", "original-1"]
    assert request.generation_prompt is not None
    assert request.generation_prompt.texts == ["rewrite-0", "rewrite-1", "rewrite-1", "rewrite-1"]
    assert request.metadata == [
        {"explanation": "e0"},
        {"explanation": "e1"},
        {"explanation": "e1"},
        {"explanation": "e1"},
    ]


def test_it2i_request_keeps_source_image_in_conditioning() -> None:
    root = Part.input(sample_ids=["p"], primitives={"text": Texts(texts=["edit the image"])})
    image_input = root.input_child({"image": _images(1)})
    sample = Sample.request(root, image_input).fork(2, sampling_params=DiffusionSamplingParams())

    request = _build_reward_request(sample.with_filled_frontier(primitives={"image": _images(2)}), "image")

    assert request.original_prompt is not None
    assert request.original_prompt.texts == ["edit the image", "edit the image"]
    assert request.generation_prompt is not None
    assert request.generation_prompt.texts == request.original_prompt.texts
    assert len(request.conditioning["image"]) == 2


def test_reward_request_rejects_text_conditioning_and_misaligned_fields() -> None:
    with pytest.raises(ValueError, match="non-text"):
        RewardRequest(
            generated={"image": _images(1)},
            conditioning={"text": Texts(texts=["bad"])},
        )

    with pytest.raises(ValueError, match="share one batch size"):
        RewardRequest(
            generated={"image": _images(2)},
            original_prompt=Texts(texts=["one"]),
        )


def test_per_domain_selection_preserves_prompt_and_conditioning_alignment() -> None:
    request = RewardRequest(
        generated={"image": _images(2)},
        conditioning={"image": _images(2)},
        original_prompt=Texts(texts=["original-0", "original-1"]),
        generation_prompt=Texts(texts=["generation-0", "generation-1"]),
        metadata=[{"domain": "a"}, {"domain": "b"}],
        sample_ids=["s0", "s1"],
        group_ids=["g0", "g1"],
    )

    selected = _select_request(request, [1])

    assert selected.original_prompt is not None
    assert selected.original_prompt.texts == ["original-1"]
    assert selected.generation_prompt is not None
    assert selected.generation_prompt.texts == ["generation-1"]
    assert len(selected.conditioning["image"]) == 1
    assert selected.metadata == [{"domain": "b"}]
    assert selected.sample_ids == ["s1"]
    assert selected.group_ids == ["g1"]


class _PromptBackend(RewardBackend):
    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("original", ["original"]),
        ("generation", ["rewrite"]),
    ],
)
def test_backend_selects_declared_prompt_source(source: PromptSource, expected: list[str]) -> None:
    request = RewardRequest(
        generated={"image": _images(1)},
        original_prompt=Texts(texts=["original"]),
        generation_prompt=Texts(texts=["rewrite"]),
    )
    backend = _PromptBackend(prompt_source=source)

    assert backend.prompts(request) == expected


def test_local_backend_receives_prompt_source_from_spec() -> None:
    request = RewardRequest(
        generated={"text": Texts(texts=["answer"])},
        original_prompt=Texts(texts=["original"]),
        generation_prompt=Texts(texts=["rewrite"]),
    )
    backend = LLMJudgeRewardScorer(
        config=LLMJudgeSpec(
            endpoint="http://judge.invalid",
            prompt_source="original",
        ),
        base_device="cpu",
    )

    assert backend.prompts(request) == ["original"]


def test_remote_payload_uses_configured_prompt_source(monkeypatch: pytest.MonkeyPatch) -> None:
    request = RewardRequest(
        generated={"image": _images(1)},
        conditioning={"image": _images(1)},
        original_prompt=Texts(texts=["original"]),
        generation_prompt=Texts(texts=["rewrite"]),
        sample_ids=["sample"],
        group_ids=["group"],
    )
    monkeypatch.setattr(
        RewardRequest,
        "images",
        property(lambda self: [Image.new("RGB", (2, 2))]),
    )
    backend = RemoteRewardBackend(
        config=RemoteRewardSpec(
            base_url="http://reward.invalid",
            required_rewards=("wise",),
            prompt_source="original",
        ),
        base_device="cpu",
    )
    monkeypatch.setattr(backend, "_get_condition_images", lambda _request: [Image.new("RGB", (2, 2))])
    try:
        payload = backend._build_score_payload(request)
    finally:
        backend.dispose()

    history = cast(list[dict[str, object]], payload["requests"][0]["history"])
    assert len(history) == 2
    assert history[0]["text"] == "original"
    assert history[1]["text"] == "original"


def test_remote_backend_rejects_unknown_prompt_source() -> None:
    with pytest.raises(ValueError, match="prompt_source"):
        RemoteRewardBackend(
            config=RemoteRewardSpec(
                base_url="http://reward.invalid",
                required_rewards=("wise",),
                prompt_source=cast(PromptSource, "nearest"),
            ),
            base_device="cpu",
        )
