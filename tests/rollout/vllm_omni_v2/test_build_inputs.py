"""Adapter ``build_inputs`` — RolloutReq → GenerateCall translation, per shape.

Locks the v1 request.py behavior: prompt-dict shapes, per-stage sampling
kwargs, the σ T-slice, raw eta/sde_indices pass-through, the initial-noise
variants, and the dit_recaption per-prompt seeded fan-out.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.rollout.vllm_omni_v2.conftest import make_adapter, make_req, sigmas_for
from unirl.rollout.engine.vllm_omni_v2.backends import STAGE_KIND_AR, STAGE_KIND_DIFFUSION
from unirl.rollout.engine.vllm_omni_v2.utils import seed_from_sample_id
from unirl.types.primitives import Images, Texts


def test_sd35_t2i_single_call_shape(model_config):
    adapter = make_adapter("sd35_t2i", model_config)
    req = make_req(2, sigmas=sigmas_for(2))
    calls = adapter.build_inputs(req)
    assert len(calls) == 1
    call = calls[0]
    assert call.group_by_request_id
    assert call.prompts == [
        {"prompt": "prompt 0", "negative_prompt": ""},
        {"prompt": "prompt 1", "negative_prompt": ""},
    ]
    (dit,) = call.sampling
    assert dit.kind == STAGE_KIND_DIFFUSION
    kw = dit.kwargs
    assert kw["height"] == 64 and kw["width"] == 64
    assert kw["num_inference_steps"] == 2 and kw["guidance_scale"] == 4.0
    assert kw["guidance_scale_provided"] is True and kw["eta"] == 1.0
    assert kw["return_trajectory_latents"] is True and kw["return_trajectory_decoded"] is False
    assert kw["num_outputs_per_prompt"] == 1 and kw["seed"] == 7
    # σ shape contract: T values, terminal 0 sliced off.
    assert kw["sigmas"] == pytest.approx([1.0, 0.5])
    assert "extra_args" not in kw  # no SDE indices, no driver noise


def test_sd35_t2i_sde_and_materialized_noise(model_config):
    adapter = make_adapter("sd35_t2i", model_config)
    req = make_req(2, sigmas=sigmas_for(2))
    req.sampling_params.sde_indices = [1, 0, 1]
    noise = torch.randn(2, 4, 8, 8)
    req.request_conditions = {"initial_latents": SimpleNamespace(latents=noise)}
    (call,) = adapter.build_inputs(req)
    ea = call.sampling[0].kwargs["extra_args"]
    assert ea["sde_indices"] == [0, 1]  # sorted, deduped
    assert ea["initial_noise_batch"] is noise

    # Batch-dim mismatch fails fast.
    req.request_conditions = {"initial_latents": SimpleNamespace(latents=torch.randn(3, 4, 8, 8))}
    with pytest.raises(RuntimeError, match="initial_latents.shape"):
        adapter.build_inputs(req)


def test_sd35_t2i_noise_recipe_path(model_config):
    adapter = make_adapter("sd35_t2i", model_config)
    req = make_req(
        2,
        sigmas=sigmas_for(2),
        init_noise_group_ids=["r0:a", "r0:b"],
        init_noise_latent_shape=[4, 8, 8],
    )
    (call,) = adapter.build_inputs(req)
    ea = call.sampling[0].kwargs["extra_args"]
    assert ea["init_noise_group_ids"] == ["r0:a", "r0:b"]
    assert ea["init_noise_latent_shape"] == [4, 8, 8]
    assert ea["init_noise_seed"] == 7


def test_sd35_t2i_rejects_image(model_config):
    adapter = make_adapter("sd35_t2i", model_config)
    req = make_req(2, sigmas=sigmas_for(2), primitives={"image": Images(pixels=torch.zeros(2, 3, 8, 8))})
    with pytest.raises(ValueError, match="does not accept"):
        adapter.build_inputs(req)
    with pytest.raises(ValueError, match="rejects image-bearing"):
        adapter.validate_request(req)


def test_t2v_adds_num_frames(model_config):
    adapter = make_adapter("t2v", model_config)
    req = make_req(2, sigmas=sigmas_for(2))
    req.sampling_params.num_frames = 9
    (call,) = adapter.build_inputs(req)
    assert all(p["num_frames"] == 9 for p in call.prompts)
    assert call.sampling[0].kwargs["num_frames"] == 9


def test_t2i_two_stage_call(model_config):
    adapter = make_adapter("t2i", model_config)
    req = make_req(2, modality_params="composed", sigmas=sigmas_for(2))
    (call,) = adapter.build_inputs(req)
    assert [s.kind for s in call.sampling] == [STAGE_KIND_AR, STAGE_KIND_DIFFUSION]
    ar_kw = call.sampling[0].kwargs
    # ARSamplingParams fields land via getattr; logprobs=1 always.
    assert ar_kw == {"temperature": 0.5, "top_p": 0.9, "top_k": 50, "max_tokens": 16, "logprobs": 1}
    # Prompt entries carry the official end2end.py base fields + t2i h/w.
    entry = call.prompts[0]
    assert entry["prompt"] == "prompt 0" and entry["use_system_prompt"] == "en_unified"
    assert entry["modalities"] == ["image"]
    assert entry["height"] == 64 and entry["width"] == 64
    # fake_tokenize is length-keyed: text/task/sys_type lengths.
    assert entry["prompt_token_ids"] == [len("prompt 0"), len("t2i_think"), len("en_unified")]


def test_t2i_rejects_materialized_noise_and_ships_recipe(model_config):
    adapter = make_adapter("t2i", model_config)
    req = make_req(2, modality_params="composed", sigmas=sigmas_for(2))
    req.request_conditions = {"initial_latents": SimpleNamespace(latents=torch.zeros(2, 4, 8, 8))}
    with pytest.raises(NotImplementedError, match="AR-dynamic"):
        adapter.build_inputs(req)

    req2 = make_req(
        2, modality_params="composed", sigmas=sigmas_for(2), init_noise_group_ids=["a", "b"]
    )
    (call,) = adapter.build_inputs(req2)
    ea = call.sampling[1].kwargs["extra_args"]
    assert ea["init_noise_group_ids"] == ["a", "b"] and ea["init_noise_seed"] == 7
    assert "init_noise_latent_shape" not in ea  # AR-dynamic shape stays in-worker


def test_it2i_attaches_image_and_pil_dims(model_config):
    adapter = make_adapter("it2i", model_config)
    req = make_req(
        2,
        modality_params="composed",
        sigmas=sigmas_for(2),
        primitives={"image": Images(pixels=torch.zeros(2, 3, 8, 8))},
    )
    (call,) = adapter.build_inputs(req)
    entry = call.prompts[0]
    assert "image" in entry["multi_modal_data"]
    assert entry["height"] == 8 and entry["width"] == 8  # PIL dims, not cfg

    # And it requires the image.
    with pytest.raises(ValueError, match="requires req.primitives"):
        adapter.build_inputs(make_req(2, modality_params="composed", sigmas=sigmas_for(2)))


def test_t2t_single_ar_stage_no_dims(model_config):
    adapter = make_adapter("t2t", model_config)
    req = make_req(2, modality_params="ar")
    (call,) = adapter.build_inputs(req)
    assert [s.kind for s in call.sampling] == [STAGE_KIND_AR]
    assert "height" not in call.prompts[0] and "multi_modal_data" not in call.prompts[0]


def test_ar_recaption_carries_dims(model_config):
    adapter = make_adapter("ar_recaption", model_config)
    req = make_req(2, modality_params="composed")
    (call,) = adapter.build_inputs(req)
    assert [s.kind for s in call.sampling] == [STAGE_KIND_AR]
    assert call.prompts[0]["height"] == 64 and call.prompts[0]["width"] == 64
    # task defaults to t2i_think (the think/recaption producer).
    assert call.prompts[0]["prompt_token_ids"][1] == len("t2i_think")


def test_dit_recaption_per_prompt_seeded_calls(model_config):
    adapter = make_adapter("dit_recaption", model_config)
    req = make_req(
        3,
        sigmas=sigmas_for(2),
        primitives={"cot_text": Texts(texts=["recap 0", "recap 1", "recap 2"])},
        init_noise_group_ids=["g:a", "g:b", "g:c"],
    )
    calls = adapter.build_inputs(req)
    assert len(calls) == 3
    seeds = []
    for idx, call in enumerate(calls):
        assert not call.group_by_request_id  # flat list IS the group
        (prompt,) = call.prompts
        assert prompt["extra"]["ar_generated_text"] == f"recap {idx}"
        assert prompt["use_system_prompt"] == "en_unified"
        kw = call.sampling[0].kwargs
        assert kw["seed"] == seed_from_sample_id(req.sample_ids[idx])
        seeds.append(kw["seed"])
        # Per-call gid slice: ONLY this sample's recipe gid.
        assert kw["extra_args"]["init_noise_group_ids"] == [req.init_noise_group_ids[idx]]
        assert kw["extra_args"]["init_noise_seed"] == 7
    assert len(set(seeds)) == 3  # distinct noise per image — GRPO group diversity


def test_dit_recaption_requires_cot_text(model_config):
    adapter = make_adapter("dit_recaption", model_config)
    with pytest.raises(TypeError, match="cot_text"):
        adapter.build_inputs(make_req(2, sigmas=sigmas_for(2)))
    # Misaligned counts fail too.
    req = make_req(2, sigmas=sigmas_for(2), primitives={"cot_text": Texts(texts=["only one"])})
    with pytest.raises(ValueError, match="cot_text count"):
        adapter.build_inputs(req)


def test_sigmas_length_contract(model_config):
    adapter = make_adapter("sd35_t2i", model_config)
    req = make_req(2, sigmas=sigmas_for(5))  # wrong length for 2 steps
    with pytest.raises(Exception, match="num_inference_steps"):
        adapter.build_inputs(req)
