"""Sample-native rollout-adapter response assembly tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.adapters.video import VideoAdapter
from unirl.rollout.engine.vllm_omni.adapters import bagel as bagel_module
from unirl.rollout.engine.vllm_omni.adapters import dit as dit_module
from unirl.rollout.engine.vllm_omni.adapters.bagel import BagelOutputAdapter
from unirl.rollout.engine.vllm_omni.adapters.hi3 import Hi3ImageOutputAdapter
from unirl.rollout.engine.vllm_omni.adapters.hv15 import Hv15VideoOutputAdapter
from unirl.rollout.engine.vllm_omni.adapters.qwen_image import QwenImageOutputAdapter
from unirl.rollout.engine.vllm_omni.adapters.sd3 import Sd3OutputAdapter
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.primitives import Images, Texts, Video, Videos
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment, make_image_segment


def _diffusion_output(*, stage_id: int, custom_output: dict | None = None):
    return SimpleNamespace(
        request_id="0_test",
        stage_id=stage_id,
        final_output_type="image",
        request_output=None,
        prompt_token_ids=None,
        images=[object()],
        trajectory_latents=torch.zeros(1, 3, 1, 1, 1),
        trajectory_timesteps=torch.tensor([1.0, 0.5, 0.0]),
        trajectory_log_probs=torch.empty(1, 0),
        custom_output=dict(custom_output or {}),
    )


def _fake_images(items) -> Images:
    return Images(pixels=torch.zeros(len(items), 3, 2, 2))


def test_bagel_output_fills_only_the_typed_diffusion_part(monkeypatch) -> None:
    monkeypatch.setattr(bagel_module, "pils_to_images", _fake_images)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    root = Part.input(["p"], primitives={"text": Texts(texts=["hello"])})
    params = BagelDiffusionParams(
        num_inference_steps=2,
        sigmas=sigmas,
        height=2,
        width=2,
    )
    sample = Sample.request(root).fork(1, sampling_params=params)

    output = BagelOutputAdapter("bagel_t2i").build(sample, [[_diffusion_output(stage_id=0)]])

    assert output.parts[0] is root
    assert len(output.gen_parts()) == 1
    generated = output.gen_part(BagelDiffusionParams)
    assert generated.sample_ids == sample.gen_part(BagelDiffusionParams).sample_ids
    assert generated.sampling_params is params
    assert isinstance(generated.segment, LatentSegment)
    assert set(generated.primitives) == {"image"}
    assert set(generated.conditions) == {"bagel"}
    assert generated.conditions["bagel"].prompts == ["hello"]


@pytest.mark.parametrize(
    ("adapter_type", "primitive_kind"),
    [
        (Hv15VideoOutputAdapter, "video"),
        (QwenImageOutputAdapter, "image"),
        (Sd3OutputAdapter, "image"),
    ],
)
def test_single_stage_output_families_share_the_direct_part_fill_contract(
    monkeypatch,
    adapter_type,
    primitive_kind: str,
) -> None:
    root = Part.input(["p"], primitives={"text": Texts(texts=["hello"])})
    params = DiffusionSamplingParams(sigmas=torch.tensor([1.0, 0.0]))
    sample = Sample.request(root).fork(1, sampling_params=params)
    segment = make_image_segment(
        latents=torch.zeros(1, 2, 1, 1, 1),
        sigmas=torch.tensor([1.0, 0.0]),
        indices=torch.tensor([0, 1]),
    )
    decoded = (
        Videos.from_list([Video(frames=torch.zeros(1, 3, 2, 2))])
        if primitive_kind == "video"
        else Images(pixels=torch.zeros(1, 3, 2, 2))
    )
    condition = TextEmbedCondition(embeds=torch.zeros(1, 1, 1))
    adapter = adapter_type("test")
    monkeypatch.setattr(adapter, "build_segment", lambda *_: segment)
    monkeypatch.setattr(adapter, "build_decoded", lambda *_: decoded)
    monkeypatch.setattr(adapter, "build_conditions", lambda *_: {"text": condition})

    output = adapter.build(sample, [[object()]])

    assert output.parts[0] is root
    generated = output.gen_part(DiffusionSamplingParams)
    assert generated.sampling_params is params
    assert generated.segment is segment
    assert set(generated.primitives) == {primitive_kind}
    assert generated.primitives[primitive_kind] is decoded
    assert set(generated.conditions) == {"text"}
    assert generated.conditions["text"] is condition


def test_hi3_two_stage_output_keeps_ar_and_diffusion_conditions_separate(monkeypatch) -> None:
    monkeypatch.setattr(dit_module, "pils_to_images", _fake_images)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    root = Part.input(["p"], primitives={"text": Texts(texts=["hello"])})
    ar_params = ARSamplingParams(max_new_tokens=2)
    diffusion_params = DiffusionSamplingParams(
        num_inference_steps=2,
        sigmas=sigmas,
        height=2,
        width=2,
    )
    sample = Sample.request(root).fork(1, sampling_params=ar_params).fork(1, sampling_params=diffusion_params)

    ar_output = SimpleNamespace(
        request_id="0_test",
        stage_id=0,
        final_output_type="text",
        request_output=SimpleNamespace(outputs=[SimpleNamespace(token_ids=[5, 6], logprobs=None, text="thinking")]),
        prompt_token_ids=[1, 2],
        images=None,
        trajectory_latents=None,
        trajectory_timesteps=None,
        trajectory_log_probs=None,
        custom_output={},
    )
    fused_capture = {
        "input_ids": torch.tensor([[11, 12, 13]]),
        "attention_mask": torch.ones(1, 1, 3, 3, dtype=torch.bool),
        "position_ids": torch.tensor([[0, 1, 2]]),
        "gen_image_mask": torch.tensor([[False, True, True]]),
        "gen_timestep_scatter_index": torch.tensor([[1]]),
        "rope_cache": (torch.zeros(1, 3, 1), torch.zeros(1, 3, 1)),
    }
    diffusion_output = _diffusion_output(
        stage_id=1,
        custom_output={"fused_mm_capture": fused_capture},
    )

    output = Hi3ImageOutputAdapter("hi3_t2i").build(sample, [[ar_output, diffusion_output]])

    assert output.parts[0] is root
    ar_part = output.gen_part(ARSamplingParams)
    diffusion_part = output.gen_part(DiffusionSamplingParams)
    assert ar_part.segment.tokens.tolist() == [5, 6]
    assert ar_part.primitives["text"].texts == ["thinking"]
    assert isinstance(diffusion_part.segment, LatentSegment)
    assert set(diffusion_part.primitives) == {"image"}

    ar_fused = ar_part.conditions["fused"]
    diffusion_fused = diffusion_part.conditions["fused"]
    assert ar_fused is not diffusion_fused
    assert ar_fused.input_ids.tolist() == [[1, 2]]
    assert ar_fused.prompt_lengths.tolist() == [2]
    assert ar_fused.gen_image_mask is None
    assert diffusion_fused.input_ids.tolist() == [[11, 12, 13]]
    assert diffusion_fused.prompt_lengths is None
    assert diffusion_fused.gen_image_mask.tolist() == [[False, True, True]]


@pytest.mark.parametrize(
    ("adapter_type", "decoded", "primitive_kind"),
    [
        (ImageAdapter, Images(pixels=torch.zeros(1, 3, 2, 2)), "image"),
        (VideoAdapter, Videos.from_list([Video(frames=torch.zeros(1, 3, 2, 2))]), "video"),
    ],
)
def test_sglang_diffusion_fills_the_canonical_decoded_modality(
    monkeypatch,
    adapter_type,
    decoded,
    primitive_kind: str,
) -> None:
    sigmas = torch.tensor([1.0, 0.0])
    root = Part.input(["p"], primitives={"text": Texts(texts=["hello"])})
    params = DiffusionSamplingParams(num_inference_steps=1, sigmas=sigmas)
    sample = Sample.request(root).fork(1, sampling_params=params)
    segment = make_image_segment(
        latents=torch.zeros(1, 2, 1, 1, 1),
        sigmas=sigmas,
        indices=torch.tensor([0, 1]),
    )
    adapter = adapter_type.__new__(adapter_type)
    adapter.cfg = SimpleNamespace(populate_conditions=False)
    monkeypatch.setattr(adapter, "build_segment", lambda *_args, **_kwargs: segment)
    monkeypatch.setattr(adapter, "build_decoded", lambda *_args, **_kwargs: decoded)

    output = adapter.build_response(sample, [object()])

    generated = output.gen_part(DiffusionSamplingParams)
    assert generated.segment is segment
    assert set(generated.primitives) == {primitive_kind}
    assert generated.primitives[primitive_kind] is decoded
