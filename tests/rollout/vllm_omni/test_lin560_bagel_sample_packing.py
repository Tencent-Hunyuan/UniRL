"""Sample-native BAGEL prompt packing and fallback layout checks."""

from __future__ import annotations

import pytest
import torch

from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.rollout.engine.vllm_omni.adapters.bagel import BagelInputAdapter
from unirl.rollout.engine.vllm_omni.pipelines.bagel.bagel_flow_match_sde_scheduler import (
    BagelFlowSDEScheduler,
)
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample


def _bagel_sample(
    *,
    spp: int,
    cfg_text_scale: float = 1.0,
    cfg_img_scale: float = 1.0,
    with_image: bool = False,
) -> Sample:
    primitives = {"text": Texts(texts=["alpha", "beta"])}
    if with_image:
        primitives["image"] = Images(pixels=torch.zeros(2, 3, 4, 4))
    root = Part.input(["p0", "p1"], primitives=primitives)
    params = BagelDiffusionParams(
        samples_per_prompt=spp,
        num_inference_steps=2,
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        init_noise_latent_shape=[2, 4],
        seed=17,
    )
    return Sample.request(root).fork(max(spp, 1), sampling_params=params)


@pytest.mark.parametrize(
    ("spp", "cfg_text", "cfg_img", "expected_prompts", "expected_outputs"),
    [
        (1, 1.0, 1.0, ["alpha", "beta"], 1),
        (3, 1.0, 1.0, ["alpha", "beta"], 3),
        (3, 2.0, 1.0, ["alpha"] * 3 + ["beta"] * 3, 1),
        (3, 1.0, 1.5, ["alpha"] * 3 + ["beta"] * 3, 1),
        (3, 0.5, 0.5, ["alpha", "beta"], 3),
    ],
)
def test_bagel_sample_layout_packs_only_cfg_off_groups(
    spp: int,
    cfg_text: float,
    cfg_img: float,
    expected_prompts: list[str],
    expected_outputs: int,
) -> None:
    sample = _bagel_sample(spp=spp, cfg_text_scale=cfg_text, cfg_img_scale=cfg_img)
    call = BagelInputAdapter("bagel_t2i").build(sample)[0]

    assert [prompt["prompt"] for prompt in call.prompts] == expected_prompts
    kwargs = call.sampling[0].kwargs
    assert kwargs["num_outputs_per_prompt"] == expected_outputs
    assert len(call.prompts) * kwargs["num_outputs_per_prompt"] == len(sample.gen_part(BagelDiffusionParams).sample_ids)
    # x_T remains authored for every expanded Sample row even when prompts pack.
    assert kwargs["extra_args"]["init_noise_group_ids"] == sample.gen_part(BagelDiffusionParams).sample_ids


def test_bagel_adapter_rejects_image_conditioning_before_packing() -> None:
    sample = _bagel_sample(spp=2, with_image=True)
    with pytest.raises(ValueError, match="does not accept image conditioning"):
        BagelInputAdapter("bagel_t2i").build_prompts(sample)


def test_bagel_adapter_rejects_non_positive_samples_per_prompt() -> None:
    sample = _bagel_sample(spp=0)
    with pytest.raises(ValueError, match="samples_per_prompt must be >= 1"):
        BagelInputAdapter("bagel_t2i").build(sample)


def test_bagel_adapter_rejects_interleaved_sample_groups() -> None:
    valid = _bagel_sample(spp=2)
    gen_part = valid.gen_part(BagelDiffusionParams)
    interleaved = Part(
        sample_ids=["p0/0", "p1/0", "p0/1", "p1/1"],
        sampling_params=gen_part.sampling_params,
    )
    malformed = valid.with_parts([valid.parts[0], interleaved])

    with pytest.raises(RuntimeError, match="not repeated within their generation group"):
        BagelInputAdapter("bagel_t2i").build(malformed)


def test_bagel_scheduler_splits_packed_ode_trajectory_back_to_samples() -> None:
    scheduler = BagelFlowSDEScheduler()
    scheduler.set_for_request(
        eta=0.0,
        sde_indices=[],
        image_token_sizes=[2, 2],
    )
    scheduler.step(
        model_output=torch.ones(4, 3),
        timestep=torch.tensor(0.8),
        sample=torch.zeros(4, 3),
        dt=torch.tensor(0.1),
    )

    trajectory = scheduler.drain_trajectory()
    assert trajectory is not None
    latents, sigmas, timesteps, log_probs = trajectory
    assert latents.shape == (2, 2, 2, 3)
    assert sigmas.shape == (2,)
    assert timesteps.shape == (1, 2)
    assert log_probs.shape == (2, 0)


def test_bagel_scheduler_reduces_packed_sde_log_prob_per_sample() -> None:
    scheduler = BagelFlowSDEScheduler()
    scheduler.set_for_request(
        eta=0.25,
        sde_indices=[0],
        image_token_sizes=[2, 2],
    )
    out = scheduler.step(
        model_output=torch.zeros(4, 3),
        timestep=torch.tensor(0.8),
        sample=torch.zeros(4, 3),
        dt=torch.tensor(0.1),
    )

    assert out.log_prob is not None
    std_var = (torch.sqrt(torch.tensor(0.8 / 0.2)) * 0.25) * torch.sqrt(torch.tensor(0.1))
    elem = (
        -((out.prev_sample.detach().float() - out.prev_sample_mean) ** 2) / (2 * std_var**2)
        - torch.log(std_var)
        - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
    )
    expected = torch.stack([chunk.mean() for chunk in elem.split([2, 2], dim=0)])
    torch.testing.assert_close(out.log_prob, expected)

    trajectory = scheduler.drain_trajectory()
    assert trajectory is not None
    latents, _, _, log_probs = trajectory
    assert latents.shape == (2, 2, 2, 3)
    assert log_probs.shape == (2, 1)
    torch.testing.assert_close(log_probs[:, 0], expected)
