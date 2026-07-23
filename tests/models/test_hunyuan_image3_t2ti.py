"""t2ti must condition its AR pass at the AR frontier, not the final image frontier.

``Sample.conditioning`` is frontier-aligned by contract: it gathers one row per
``parts[-1]`` sample. Given the hierarchical ``[input, ar(P*N), image(P*N*M)]`` request
that ``UnifiedModelTrainer._build_request_sample`` builds, reading the frontier view for
the AR pass runs it at ``P*N*M`` width and then fills a ``P*N``-row Part.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.models.hunyuan_image3.modes import t2ti
from unirl.types.primitives import Image, Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams

PROMPTS, AR_BRANCH, IMAGE_BRANCH = 1, 2, 3
AR_ROWS = PROMPTS * AR_BRANCH
IMAGE_ROWS = AR_ROWS * IMAGE_BRANCH


def _request() -> Sample:
    root = Part.input(
        [f"r0:prompt:{i}:sample:0" for i in range(PROMPTS)],
        primitives={"text": Texts(texts=[f"prompt-{i}" for i in range(PROMPTS)])},
    )
    return (
        Sample.request(root)
        .fork(AR_BRANCH, sampling_params=ARSamplingParams(samples_per_prompt=AR_BRANCH))
        .fork(
            IMAGE_BRANCH,
            sampling_params=DiffusionSamplingParams(
                samples_per_prompt=IMAGE_BRANCH,
                height=64,
                width=64,
                sigmas=torch.linspace(1.0, 0.0, 5),
            ),
        )
    )


def test_conditioning_at_resolves_an_interior_stage_at_its_own_width() -> None:
    sample = _request()

    assert len(sample.conditioning()[0].texts) == IMAGE_ROWS
    assert len(sample.conditioning_at(1)[0].texts) == AR_ROWS
    assert len(sample.conditioning_at(-2)[0].texts) == AR_ROWS


def test_conditioning_at_rejects_an_out_of_range_index() -> None:
    with pytest.raises(IndexError, match="out of range"):
        _request().conditioning_at(9)


class _FakePipeline:
    """Enough of HunyuanImage3Pipeline to exercise t2ti's width plumbing on CPU."""

    def __init__(self, seen: dict) -> None:
        self.seen = seen
        self.bundle = SimpleNamespace(device=torch.device("cpu"), transformer=None)
        self.text_embed = SimpleNamespace(
            embed_for_ar=self._embed_for_ar,
            embed_for_gen_image=self._embed_for_gen_image,
        )
        self.ar = SimpleNamespace(autoregress=lambda *a, **k: None)
        self.diffusion = SimpleNamespace(diffuse=lambda *a, **k: None)
        self.vae_decode = SimpleNamespace(decode=self._decode)

    def _embed_for_ar(self, texts, *, bot_task, system_prompt):
        self.seen["ar_texts"] = len(texts.texts)
        self.seen["ar_system_prompt"] = len(system_prompt) if system_prompt else None
        return {"fused": None, "tokenizer_output": None}

    def _embed_for_gen_image(self, texts, *, cfg, height, width, bot_task, cot_text, system_prompt):
        self.seen["image_texts"] = len(texts.texts)
        self.seen["image_cots"] = list(cot_text)
        self.seen["image_system_prompt"] = len(system_prompt) if system_prompt else None
        return {"fused": None, "tokenizer_output": None}

    def _decode(self, segment):
        return Images.from_list(
            [Image(pixels=torch.zeros(3, 8, 8)) for _ in range(self.seen["image_texts"])]
        )

    def _detokenize_text_segment(self, segment, *, skip_special_tokens):
        # The AR stage emits one CoT per AR-frontier row.
        return Texts(texts=[f"<recaption>cot-{i}</recaption>" for i in range(self.seen["ar_texts"])])


class _FakeConditions:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def to_dict(self) -> dict:
        return {}


@pytest.fixture()
def stubbed_t2ti(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(t2ti, "_resolve_system_prompt", lambda *a, **k: "SYSTEM")
    monkeypatch.setattr(t2ti, "_cot_stop_tokens", lambda *a, **k: [1])
    monkeypatch.setattr(t2ti, "HunyuanImage3ARConditions", _FakeConditions)
    monkeypatch.setattr(t2ti, "HunyuanImage3DiffusionConditions", _FakeConditions)
    return {}


def test_generate_runs_the_ar_pass_at_ar_width_and_diffusion_at_image_width(stubbed_t2ti: dict) -> None:
    seen = stubbed_t2ti

    out = t2ti.generate(_FakePipeline(seen), _request())

    assert seen["ar_texts"] == AR_ROWS
    assert seen["ar_system_prompt"] == AR_ROWS
    assert seen["image_texts"] == IMAGE_ROWS
    assert seen["image_system_prompt"] == IMAGE_ROWS
    assert len(seen["image_cots"]) == IMAGE_ROWS
    assert out.parts[-2].batch_size == AR_ROWS
    assert out.parts[-1].batch_size == IMAGE_ROWS


def test_each_ar_trajectory_conditions_exactly_its_own_images(stubbed_t2ti: dict) -> None:
    seen = stubbed_t2ti

    t2ti.generate(_FakePipeline(seen), _request())

    # The filled AR Part is an ancestor of the image shell, so the conditioning walk
    # broadcasts each CoT onto its IMAGE_BRANCH children — contiguously, group-by-parent.
    cots = seen["image_cots"]
    for row in range(AR_ROWS):
        window = cots[row * IMAGE_BRANCH : (row + 1) * IMAGE_BRANCH]
        assert len(set(window)) == 1
    assert len(set(cots)) == AR_ROWS
