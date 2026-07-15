from types import SimpleNamespace

import pytest
import torch

from unirl.rollout.engine.sglang_diffusion.adapters.sd3 import SD3Adapter
from unirl.rollout.engine.sglang_diffusion.engine import SGLangDiffusionRolloutEngine
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, _grouped_texts_from_sample
from unirl.rollout.engine.vllm_omni.adapters.qwen_image import QwenImageGroupedInputAdapter, QwenImageOutputAdapter
from unirl.rollout.engine.vllm_omni.adapters.sd3 import Sd3InputAdapter, Sd3OutputAdapter
from unirl.rollout.engine.vllm_omni.utils.noise import pack_initial_noise_extra_args
from unirl.rollout.engine.vllm_omni.utils.tracks import assemble_sample
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import make_image_segment


def _diffusion_sample(*, prompts: list[str], spp: int = 2) -> Sample:
    params = DiffusionSamplingParams(
        samples_per_prompt=spp,
        num_inference_steps=2,
        guidance_scale=1.0,
        height=8,
        width=8,
        num_frames=1,
        seed=7,
        eta=0.0,
        sde_indices=[],
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        init_noise_latent_shape=[1, 2, 2],
    )
    root = Part.input(
        [f"p{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=prompts)},
    )
    return Sample.request(root).fork(spp, sampling_params=params)


@pytest.mark.parametrize("spp", [1, 2])
def test_grouped_texts_from_sample_collapses_complete_groups(spp: int) -> None:
    sample = _diffusion_sample(prompts=["a", "b"], spp=spp)

    grouped, actual_spp = _grouped_texts_from_sample(sample, caller="test")

    assert grouped == ["a", "b"]
    assert actual_spp == spp


def test_grouped_texts_from_sample_rejects_interleaved_groups() -> None:
    sample = _diffusion_sample(prompts=["a", "b"], spp=2)
    params = sample.parts[-1].sampling_params
    interleaved = Part(
        sample_ids=["p0/0", "p1/0", "p0/1", "p1/1"],
        sampling_params=params,
    )

    with pytest.raises(RuntimeError, match="not repeated|not one contiguous group"):
        _grouped_texts_from_sample(sample.replace_frontier(interleaved), caller="test")


def test_vllm_negative_prompts_use_sampler_kwargs_and_model_defaults() -> None:
    sample = _diffusion_sample(prompts=["a"], spp=1)
    params = sample.parts[-1].sampling_params
    params.guidance_scale = 2.0

    assert DitInputAdapter("base").build_prompts(sample)[0]["negative_prompt"] == ""
    assert Sd3InputAdapter("sd3").build_prompts(sample)[0]["negative_prompt"] == ""
    assert QwenImageGroupedInputAdapter("qwen").build_prompts(sample)[0]["negative_prompt"] == " "

    params.sampler_kwargs = {"negative_prompt": "no blur"}
    assert DitInputAdapter("base").build_prompts(sample)[0]["negative_prompt"] == "no blur"
    assert Sd3InputAdapter("sd3").build_prompts(sample)[0]["negative_prompt"] == "no blur"
    assert QwenImageGroupedInputAdapter("qwen").build_prompts(sample)[0]["negative_prompt"] == "no blur"


def test_qwen_capture_is_not_expanded_twice() -> None:
    sample = _diffusion_sample(prompts=["a", "b"], spp=2)
    per_request = []
    for _ in range(2):
        capture = {
            "prompt_embeds": torch.zeros(2, 3, 4),
            "prompt_embeds_mask": torch.ones(2, 3),
            "negative_prompt_embeds": None,
        }
        output = SimpleNamespace(
            final_output_type="image",
            custom_output={"text_capture": capture},
            images=[object(), object()],
            trajectory_latents=torch.zeros(2, 3, 1, 1, 1),
        )
        per_request.append([output])

    conditions = QwenImageOutputAdapter("qwen").build_conditions(sample, per_request)

    assert conditions["text"].embeds.shape[0] == 4


def test_sd3_prompt_level_capture_expands_to_trajectory_batch() -> None:
    sample = _diffusion_sample(prompts=["a", "b"], spp=2)
    per_request = []
    for _ in range(2):
        capture = {
            "prompt_embeds": torch.zeros(1, 3, 4),
            "pooled_prompt_embeds": torch.zeros(1, 5),
        }
        output = SimpleNamespace(
            final_output_type="image",
            custom_output={"text_capture": capture},
            images=[object(), object()],
            trajectory_latents=torch.zeros(2, 3, 1, 1, 1),
        )
        per_request.append([output])

    conditions = Sd3OutputAdapter("sd3").build_conditions(sample, per_request)

    assert conditions["text"].embeds.shape[0] == 4
    assert conditions["text"].pooled.shape[0] == 4


def test_sglang_initial_noise_and_denoise_seeds_are_gated_together() -> None:
    sample = _diffusion_sample(prompts=["a"], spp=2)
    cfg = SimpleNamespace(populate_conditions=False, target_modules=())
    model_cfg = SimpleNamespace(
        pretrained_model_ckpt_path="dummy",
        shift=1.0,
        weight_sync_param_name_prefix="",
    )
    adapter = SD3Adapter(cfg, model_cfg)

    engine_owned = adapter.build_inputs(sample, initial_noise=None)
    driver_owned = adapter.build_inputs(sample, initial_noise=torch.zeros(2, 1, 2, 2))

    assert "initial_noise" not in engine_owned
    assert "denoise_seeds" not in engine_owned
    assert driver_owned["initial_noise"].shape[0] == 2
    assert driver_owned["denoise_seeds"] == sample.parts[-1].sample_ids


def test_disable_driver_xt_skips_all_engine_transports() -> None:
    sample = _diffusion_sample(prompts=["a"], spp=1)
    params = sample.parts[-1].sampling_params
    params.disable_driver_xt = True
    sample = sample.replace_frontier(
        Part(
            sample_ids=sample.parts[-1].sample_ids,
            segment=make_image_segment(initial_latents=torch.ones(1, 1, 2, 2)),
            sampling_params=params,
        )
    )

    extra_args = {}
    pack_initial_noise_extra_args(extra_args, sample.parts[-1], params, caller="test")
    assert extra_args == {}

    engine = object.__new__(SGLangDiffusionRolloutEngine)
    engine.cfg = SimpleNamespace(init_same_noise=True)
    assert engine._resolve_initial_noise(sample) is None


def test_assemble_sample_preserves_reward_compute_time() -> None:
    sample = _diffusion_sample(prompts=["a"], spp=1)
    sample.reward_compute_s = 3.5
    segment = make_image_segment(latents=torch.zeros(1, 1, 1, 1, 1))

    assembled = assemble_sample(
        sample,
        segments_for_track={"image": segment},
        decoded_for_track={"image": None},
        conditions={},
    )

    assert assembled.reward_compute_s == 3.5
