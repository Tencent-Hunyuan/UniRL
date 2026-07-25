from types import SimpleNamespace

import pytest
import torch

from unirl.models.hunyuan_image3.conditions import (
    HunyuanImage3DiffusionConditions,
    HunyuanImage3FusedMultimodalCondition,
)
from unirl.models.hunyuan_image3.diffusion import (
    HunyuanImage3DiffusionStage,
    HunyuanImage3DiffusionStep,
)
from unirl.models.hunyuan_image3.modes.it2i import _encode_cond_images_per_sample
from unirl.models.hunyuan_image3.vae import HunyuanImage3VAEDecodeStage
from unirl.rollout.engine.vllm_omni.adapters.hi3 import Hi3InputAdapter
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments import LatentSegment


def _hi3_request(noise_group_ids):
    return RolloutReq(
        sample_ids=["sample-0", "sample-1"],
        group_ids=["group-0", "group-1"],
        primitives={"text": Texts(texts=["first", "second"])},
        sampling_params={
            "ar": ARSamplingParams(),
            "diffusion": DiffusionSamplingParams(
                num_inference_steps=2,
                guidance_scale=2.5,
                height=64,
                width=64,
                seed=17,
                sde_indices=[0],
            ),
        },
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        init_noise_group_ids=list(noise_group_ids),
    )


def _hi3_input_adapter():
    return Hi3InputAdapter(
        "hi3_t2i",
        tokenize_fn=lambda text, **_: [len(text)],
        task_key="t2i_think",
        output_modalities=("image",),
        stages=("ar", "dit"),
        carries_target_size=True,
    )


def test_hi3_input_adapter_slices_noise_recipe_per_worker_request():
    calls = _hi3_input_adapter().build(_hi3_request(["noise-0", "noise-1"]))

    assert len(calls) == 2
    assert [call.prompts[0]["prompt"] for call in calls] == ["first", "second"]
    for call, expected_gid in zip(calls, ["noise-0", "noise-1"]):
        assert len(call.prompts) == 1
        assert call.group_by_request_id is True
        assert call.sampling[1].kwargs["extra_args"]["init_noise_group_ids"] == [expected_gid]
        assert call.sampling[1].kwargs["extra_args"]["init_noise_seed"] == 17


def test_hi3_input_adapter_rejects_misaligned_noise_recipe():
    with pytest.raises(ValueError, match="gid count 1 != prompt count 2"):
        _hi3_input_adapter().build(_hi3_request(["noise-0"]))


def test_hi3_fused_concat_pads_stacked_rope_cache():
    short = HunyuanImage3FusedMultimodalCondition(
        input_ids=torch.ones(1, 2, dtype=torch.long),
        attention_mask=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        position_ids=torch.arange(2).unsqueeze(0),
        rope_cache=torch.ones(1, 2, 2, 4),
        gen_image_mask=torch.ones(1, 2, dtype=torch.bool),
        gen_timestep_scatter_index=torch.zeros(1, 1, dtype=torch.long),
    )
    long = HunyuanImage3FusedMultimodalCondition(
        input_ids=torch.full((1, 3), 2, dtype=torch.long),
        attention_mask=torch.ones(1, 1, 3, 3, dtype=torch.bool),
        position_ids=torch.arange(3).unsqueeze(0),
        rope_cache=torch.full((1, 2, 3, 4), 2.0),
        gen_image_mask=torch.ones(1, 3, dtype=torch.bool),
        gen_timestep_scatter_index=torch.zeros(1, 1, dtype=torch.long),
    )

    merged = HunyuanImage3FusedMultimodalCondition.concat([short, long])

    assert merged.rope_cache.shape == (2, 2, 3, 4)
    torch.testing.assert_close(merged.rope_cache[0, :, :2], torch.ones(2, 2, 4))
    torch.testing.assert_close(merged.rope_cache[0, :, 2], torch.zeros(2, 4))
    torch.testing.assert_close(merged.rope_cache[1], torch.full((2, 3, 4), 2.0))


def test_hi3_fused_condition_rejects_legacy_rope_tuple():
    with pytest.raises(TypeError, match=r"stacked \[B, 2, L, D\] tensor"):
        HunyuanImage3FusedMultimodalCondition.from_dict(
            {
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "rope_cache": (torch.ones(1, 2, 4), torch.ones(1, 2, 4)),
            }
        )


class _FakeTransformer:
    def __init__(self):
        self.training = False
        self.vision_model = SimpleNamespace(training=False)
        self.vae = SimpleNamespace(training=False)
        self.cached_rope = SimpleNamespace(
            cos_cache=None,
            sin_cache=None,
            seq_len=None,
            rope_image_info="unset",
        )
        self.seen_timesteps = None

    def _check_inputs(self, *args, **kwargs):
        raise AssertionError("predict_noise must temporarily bypass the upstream checker")

    def __call__(self, **kwargs):
        self.seen_timesteps = kwargs["timesteps"].detach().clone()
        return {"diffusion_prediction": kwargs["images"].clone()}


def _fused_condition(rope_value):
    batch, length, rope_dim = 2, 2, 4
    image_mask = torch.zeros(batch, length, dtype=torch.bool)
    image_mask[:, 1] = True
    return HunyuanImage3FusedMultimodalCondition(
        input_ids=torch.ones(batch, length, dtype=torch.long),
        attention_mask=torch.ones(batch, 1, length, length, dtype=torch.bool),
        position_ids=torch.arange(length).expand(batch, -1),
        rope_cache=torch.full((batch, 2, length, rope_dim), rope_value),
        gen_image_mask=image_mask,
        gen_timestep_scatter_index=torch.zeros(batch, 1, dtype=torch.long),
    )


def _guided_stage():
    transformer = _FakeTransformer()
    stage = HunyuanImage3DiffusionStage(
        model=SimpleNamespace(transformer=transformer),
        step=HunyuanImage3DiffusionStep(),
        strategy=object(),
    )
    conditions = HunyuanImage3DiffusionConditions(
        fused=_fused_condition(1.0),
        fused_uncond=_fused_condition(2.0),
    )
    return stage, transformer, conditions


@pytest.mark.parametrize(
    ("sigma_values", "expected_timesteps"),
    [
        ([0.1, 0.8], [100.0, 800.0, 100.0, 800.0]),
        ([0.3], [300.0, 300.0, 300.0, 300.0]),
        ([0.1, 0.8, 0.2, 0.9], [100.0, 800.0, 200.0, 900.0]),
    ],
)
def test_guided_forward_accepts_batched_sigma_layouts(sigma_values, expected_timesteps):
    stage, transformer, conditions = _guided_stage()

    prediction = stage.predict_noise_at_step(
        conditions,
        sample=torch.zeros(2, 1, 1, 1),
        sigma=torch.tensor(sigma_values),
        params=DiffusionSamplingParams(guidance_scale=2.5),
    )

    torch.testing.assert_close(
        transformer.seen_timesteps,
        torch.tensor(expected_timesteps),
    )
    assert prediction.shape == (2, 1, 1, 1)
    assert transformer.training is True
    assert transformer.vision_model.training is False
    assert transformer.vae.training is False
    assert transformer.cached_rope.cos_cache.shape == (4, 2, 4)
    assert transformer.cached_rope.sin_cache.shape == (4, 2, 4)


def test_guided_forward_rejects_misaligned_sigma_batch():
    stage, _, conditions = _guided_stage()

    with pytest.raises(ValueError, match=r"sigma must be scalar, \[B\], or \[2B\]"):
        stage.predict_noise_at_step(
            conditions,
            sample=torch.zeros(2, 1, 1, 1),
            sigma=torch.tensor([0.1, 0.2, 0.3]),
            params=DiffusionSamplingParams(guidance_scale=2.5),
        )


def test_it2i_condition_vae_uses_matching_generator_per_sample():
    class FakeEncoder:
        def __init__(self):
            self.calls = []

        def _encode_cond_image(self, images, *, cfg_factor, generator):
            self.calls.append((images, cfg_factor, generator))
            value = float(len(self.calls))
            return (
                torch.full((1, 2), value),
                torch.full((1,), value),
                [f"vit-{int(value)}"],
            )

    encoder = FakeEncoder()
    generators = [
        torch.Generator(device="cpu").manual_seed(11),
        torch.Generator(device="cpu").manual_seed(22),
    ]

    cond_vae, cond_timestep, cond_vit = _encode_cond_images_per_sample(
        encoder,
        ["image-0", "image-1"],
        generators,
    )

    assert [call[0] for call in encoder.calls] == [["image-0"], ["image-1"]]
    assert [call[1] for call in encoder.calls] == [1, 1]
    assert encoder.calls[0][2] == [generators[0]]
    assert encoder.calls[1][2] == [generators[1]]
    torch.testing.assert_close(cond_vae, torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    torch.testing.assert_close(cond_timestep, torch.tensor([1.0, 2.0]))
    assert cond_vit == ["vit-1", "vit-2"]


def test_hi3_vae_decode_masks_distributed_rank_discard(monkeypatch):
    class FakeVAE:
        config = SimpleNamespace(scaling_factor=1.0)

        def __init__(self):
            self.distributed_was_initialized = None

        def to(self, _dtype):
            return self

        def decode(self, latents):
            self.distributed_was_initialized = torch.distributed.is_initialized()
            batch, _, _, height, width = latents.shape
            return SimpleNamespace(sample=torch.zeros(batch, 3, 1, height, width))

    fake_vae = FakeVAE()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    stage = HunyuanImage3VAEDecodeStage(SimpleNamespace(vae=fake_vae))

    images = stage.decode(LatentSegment(latents=torch.zeros(2, 1, 4, 2, 2)))

    assert fake_vae.distributed_was_initialized is False
    assert torch.distributed.is_initialized() is True
    assert images.pixels.shape == (2, 3, 2, 2)
