"""CPU contracts for the SenseNova vLLM-Omni driver adapter."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from unirl.models.sensenova_u1.conditions import SenseNovaU1Conditions
from unirl.models.sensenova_u1.diffusion import SenseNovaU1DiffusionParams
from unirl.rollout.engine.vllm_omni.adapters.sensenova_u1 import (
    SenseNovaU1OutputAdapter,
    SenseNovaU1T2IAdapter,
)
from unirl.rollout.engine.vllm_omni.pipelines.sensenova_u1.weight_names import (
    missing_weight_sync_names,
)
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample


def _request_sample() -> Sample:
    root = Part.input(
        ["prompt-a", "prompt-b"],
        primitives={"text": Texts(texts=["a lighthouse", "a mountain"])},
    )
    params = SenseNovaU1DiffusionParams(
        num_inference_steps=4,
        guidance_scale=3.5,
        cfg_norm="channel",
        cfg_interval=(0.1, 0.9),
        t_eps=0.03,
        height=32,
        width=64,
        eta=0.7,
        samples_per_prompt=2,
        seed=7,
        sde_indices=[1, 3],
        sigmas=torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0]),
        init_noise_latent_shape=[3, 32, 64],
    )
    return Sample.request(root).fork(2, sampling_params=params)


def test_sensenova_adapter_groups_prompt_fanout_and_forwards_flow_controls() -> None:
    model_config = SimpleNamespace(timestep_shift=3.0)
    adapter = SenseNovaU1T2IAdapter(
        SimpleNamespace(model_path="unused"),
        model_config,
    )

    calls = adapter.build_inputs(_request_sample())

    assert len(calls) == 2
    assert calls[0].prompts == [{"prompt": "a lighthouse"}]
    assert calls[1].prompts == [{"prompt": "a mountain"}]
    kwargs = calls[0].sampling[0].kwargs
    assert kwargs["num_outputs_per_prompt"] == 2
    torch.testing.assert_close(
        torch.tensor(kwargs["sigmas"]),
        torch.tensor([1.0, 0.8, 0.5, 0.2]),
    )
    assert kwargs["guidance_scale"] == 3.5
    assert kwargs["eta"] == 0.7
    assert kwargs["extra_args"]["batch_size"] == 2
    assert kwargs["extra_args"]["cfg_scale"] == 3.5
    assert kwargs["extra_args"]["cfg_norm"] == "channel"
    assert kwargs["extra_args"]["cfg_interval"] == [0.1, 0.9]
    assert kwargs["extra_args"]["timestep_shift"] == 3.0
    assert kwargs["extra_args"]["t_eps"] == 0.03
    assert kwargs["extra_args"]["sde_indices"] == [1, 3]
    torch.testing.assert_close(
        torch.tensor(kwargs["extra_args"]["unirl_sigmas"]),
        torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0]),
    )
    assert kwargs["extra_args"]["init_noise_group_ids"] == [
        "prompt-a/0",
        "prompt-a/1",
    ]
    assert kwargs["extra_args"]["sde_seed"] == 7
    assert calls[1].sampling[0].kwargs["extra_args"]["init_noise_group_ids"] == [
        "prompt-b/0",
        "prompt-b/1",
    ]
    assert calls[1].sampling[0].kwargs["extra_args"]["sde_seed"] == 1_000_010


def test_sensenova_adapter_uses_checkpoint_time_shift_for_schedule() -> None:
    adapter = SenseNovaU1T2IAdapter(
        SimpleNamespace(model_path="unused"),
        SimpleNamespace(timestep_shift=3.0),
    )

    actual = adapter.schedule_policy().compute_sigma(
        num_inference_steps=4,
        height=32,
        width=64,
    )
    raw = torch.linspace(1.0, 0.0, 5)
    expected = 3.0 * raw / (1.0 + 2.0 * raw)

    torch.testing.assert_close(actual, expected)


def test_sensenova_output_adapter_flattens_grouped_images_and_prefix_caches() -> None:
    sample = _request_sample()
    params = sample.frontier_gen_part(SenseNovaU1DiffusionParams).sampling_params

    def result(prompt: str):
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=torch.ones(1), values=torch.zeros(1))])
        capture = {
            "prompts": [prompt, prompt],
            "condition_caches": [cache, cache],
            "uncondition_caches": [cache, cache],
            "condition_image_indexes": [torch.zeros(3, 1, dtype=torch.long)] * 2,
            "uncondition_image_indexes": [torch.zeros(3, 1, dtype=torch.long)] * 2,
            "image_shapes": [(32, 64)] * 2,
        }
        return SimpleNamespace(
            final_output_type="image",
            stage_id=0,
            images=[Image.new("RGB", (64, 32)), Image.new("RGB", (64, 32))],
            trajectory_latents=torch.zeros(2, 4, 1, 1),
            trajectory_timesteps=params.sigmas,
            trajectory_log_probs=torch.zeros(2, 2),
            custom_output={
                "sde_step_indices": [1, 3],
                "trajectory_indices": [1, 2, 3, 4],
                "sensenova_u1_capture": capture,
            },
        )

    output = SenseNovaU1OutputAdapter("sensenova_u1_t2i").build(
        sample,
        [[result("a lighthouse")], [result("a mountain")]],
    )

    frontier = output.frontier_gen_part(SenseNovaU1DiffusionParams)
    conditions = SenseNovaU1Conditions.from_dict(frontier.conditions)
    assert conditions.prompts == ["a lighthouse", "a lighthouse", "a mountain", "a mountain"]
    assert conditions.image_shapes == [(32, 64)] * 4
    assert len(frontier.primitives["image"]) == 4
    assert frontier.segment.latents.shape[0] == 4
    assert frontier.segment.indices.tolist() == [1, 2, 3, 4]


def test_sensenova_weight_sync_names_cover_fused_worker_layout() -> None:
    parameter_names = {
        "language_model.model.layers.0.self_attn.qkv_proj_mot_gen.weight",
        "language_model.model.layers.0.mlp_mot_gen.gate_up_proj.weight",
        "fm_modules.fm_head.conv1.weight",
    }
    incoming = [
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        "language_model.model.layers.0.self_attn.k_proj_mot_gen.weight",
        "language_model.model.layers.0.self_attn.v_proj_mot_gen.weight",
        "language_model.model.layers.0.mlp_mot_gen.gate_proj.weight",
        "language_model.model.layers.0.mlp_mot_gen.up_proj.weight",
        "fm_modules.fm_head.conv1.weight",
    ]

    assert missing_weight_sync_names(incoming, parameter_names) == []
    assert missing_weight_sync_names([*incoming, "unknown.weight"], parameter_names) == ["unknown.weight"]
