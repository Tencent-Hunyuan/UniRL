"""Pure adapter tests — registry, build_inputs, build_response, Klein unpack.

Canned data, no SGLang, no GPU. Adapters are constructed with ``SimpleNamespace``
stand-ins for the engine config + model config (they only read a handful of fields).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.rollout.engine.sglang_diffusion import adapters
from unirl.rollout.engine.sglang_diffusion.adapters.base import get_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.flux import Flux2KleinAdapter
from unirl.rollout.engine.sglang_diffusion.adapters.image_dit import ImageDiTAdapter
from unirl.rollout.engine.sglang_diffusion.adapters.sd3 import SD3Adapter
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp
from unirl.types.sampling import DiffusionSamplingParams


class _Flow:
    canonical_name = "flow"


class _Cps:
    canonical_name = "cps"


class _Dance:
    canonical_name = "dance"


def _cfg(*, populate_conditions=True, logprob_source="replay", target_modules=None):
    return SimpleNamespace(
        populate_conditions=populate_conditions,
        logprob_source=logprob_source,
        target_modules=target_modules,
    )


def _model_config(**over):
    base = dict(
        pretrained_model_ckpt_path="/fake/ckpt",
        shift=3.0,
        weight_sync_param_name_prefix="transformer.",
        use_lora=False,
        lora_target_modules=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _req(prompts, group_ids=None, *, num_inference_steps=2, sde_indices=None, sampler_kwargs=None,
         height=256, width=256):
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    sp = DiffusionSamplingParams(
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        seed=42,
        eta=1.0,
        sde_indices=sde_indices,
        sampler_kwargs=sampler_kwargs or {},
    )
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(len(prompts))],
        group_ids=group_ids if group_ids is not None else [f"g{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=list(prompts))},
        sampling_params=sp,
        sigmas=sigmas,
    )


def _result(latents, sigmas, **kw):
    base = dict(
        trajectory_latents=latents,
        trajectory_timesteps=sigmas,
        trajectory_log_probs=None,
        samples=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        encoder_attention_mask=None,
        negative_prompt_embeds=None,
        neg_pooled_prompt_embeds=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_all_families_registered():
    assert set(adapters.registered_adapters()) == {
        "sd3", "flux", "flux2_klein", "mochi", "hunyuan_video"
    }


def test_get_adapter_unknown_raises():
    with pytest.raises(Exception, match="unknown model_family"):
        get_adapter("does_not_exist")


def test_register_sets_model_family():
    assert SD3Adapter.model_family == "sd3"
    assert get_adapter("sd3") is SD3Adapter


# --------------------------------------------------------------------------- #
# build_inputs
# --------------------------------------------------------------------------- #


def test_build_inputs_basic_no_sde():
    a = SD3Adapter(_cfg(), _model_config(), strategy=None)
    req = _req(["a photo"], num_inference_steps=2)
    kw = a.build_inputs(req, initial_noise=None)
    assert kw["prompt"] == "a photo"
    assert kw["num_inference_steps"] == 2
    assert kw["sigmas"] == req.sigmas.tolist()[:-1]  # interior T, terminal 0 dropped
    assert kw["seed"] == 42
    assert kw["return_trajectory_latents"] is True
    assert "rollout" not in kw  # no SDE branch without sde_indices


def test_build_inputs_sde_branch():
    a = SD3Adapter(_cfg(), _model_config(), strategy=_Flow())
    req = _req(["p"], num_inference_steps=2, sde_indices=[0, 1])
    kw = a.build_inputs(req, initial_noise=None)
    assert kw["rollout"] is True
    assert kw["rollout_sde_type"] == "sde"
    assert kw["rollout_sde_indices"] == [0, 1]
    assert kw["rollout_noise_level"] == 1.0


def test_build_inputs_deexpands_groups():
    a = SD3Adapter(_cfg(), _model_config(), strategy=None)
    req = _req(["a", "a"], group_ids=["g", "g"])
    kw = a.build_inputs(req, initial_noise=None)
    assert kw["prompt"] == "a"  # collapsed
    assert kw["num_outputs_per_prompt"] == 2


def test_build_inputs_negative_prompt_invariant():
    a = SD3Adapter(_cfg(), _model_config(), strategy=None)
    req = _req(["p"], sampler_kwargs={"negative_prompt": "bad"})
    with pytest.raises(Exception, match="return_negative_prompt_embeds"):
        a.build_inputs(req, initial_noise=None)


def test_build_inputs_sde_without_label_raises():
    # sde_indices set but strategy=None → no sde_label → fail fast.
    a = SD3Adapter(_cfg(), _model_config(), strategy=None)
    req = _req(["p"], sde_indices=[0])
    with pytest.raises(Exception, match="requires an sde_label"):
        a.build_inputs(req, initial_noise=None)


# --------------------------------------------------------------------------- #
# build_response
# --------------------------------------------------------------------------- #


def _image_results(req, *, with_neg=False):
    sigmas = req.sigmas
    traj = torch.randn(1, sigmas.shape[0], 2, 4, 4)
    kw = dict(
        samples=torch.rand(3, 4, 4),
        prompt_embeds=torch.zeros(1, 3, 8),
        pooled_prompt_embeds=torch.zeros(1, 8),
    )
    if with_neg:
        kw["negative_prompt_embeds"] = torch.zeros(1, 3, 8)
    return [_result(traj, sigmas, **kw)]


def test_build_response_image_track():
    a = SD3Adapter(_cfg(populate_conditions=True), _model_config(), strategy=None)
    req = _req(["p"])
    resp = a.build_response(req, _image_results(req, with_neg=True))
    assert isinstance(resp, RolloutResp)
    assert set(resp.tracks) == {"image"}
    track = resp.tracks["image"]
    assert track.segment is not None and track.segment.latents.shape[0] == 1
    assert isinstance(track.decoded, Images)
    assert "text" in track.conditions and "negative_text" in track.conditions
    assert track.sample_ids == ["s0"] and track.parent_ids == ["g0"]


def test_build_response_populate_conditions_false():
    a = SD3Adapter(_cfg(populate_conditions=False), _model_config(), strategy=None)
    req = _req(["p"])
    resp = a.build_response(req, _image_results(req))
    assert resp.tracks["image"].conditions == {}


# --------------------------------------------------------------------------- #
# boilerplate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "strategy,expected",
    [(_Flow(), "sde"), (_Cps(), "cps"), (_Dance(), "dance"), (None, None)],
)
def test_resolve_sde_label(strategy, expected):
    assert ImageDiTAdapter.resolve_sde_label(strategy) == expected


def test_resolve_sde_label_unknown_raises():
    class _Weird:
        canonical_name = "weird"

    with pytest.raises(ValueError, match="supports sde_type"):
        ImageDiTAdapter.resolve_sde_label(_Weird())


def test_lora_spec_defaults():
    a = SD3Adapter(_cfg(), _model_config(weight_sync_param_name_prefix="transformer."), strategy=None)
    prefix, targets = a.lora_spec()
    assert prefix == "transformer."
    assert targets == ["transformer"]


def test_validate_requires_shift():
    with pytest.raises(Exception, match="requires model_config.shift"):
        SD3Adapter(_cfg(), SimpleNamespace(pretrained_model_ckpt_path="/x"), strategy=None)


# --------------------------------------------------------------------------- #
# trajectory unpack (base rejects 4-D; Klein handles it)
# --------------------------------------------------------------------------- #


def test_base_image_adapter_rejects_4d_trajectory():
    a = SD3Adapter(_cfg(), _model_config(), strategy=None)
    with pytest.raises(ValueError, match="expected a 5-D image-form trajectory"):
        a.unpack_trajectory(torch.randn(1, 3, 4, 8), _req(["p"]))


def test_klein_passes_through_5d():
    mc = _model_config(shift=1.0, build_schedule_policy=lambda: None)
    a = Flux2KleinAdapter(_cfg(), mc, strategy=_Dance())
    traj = torch.randn(1, 3, 2, 4, 4)
    assert a.unpack_trajectory(traj, _req(["p"])) is traj


def test_klein_unpacks_packed_4d():
    mc = _model_config(shift=1.0, build_schedule_policy=lambda: None)
    a = Flux2KleinAdapter(_cfg(), mc, strategy=_Dance())
    # height=width=32 → h_pat=w_pat=2 → S=4; C_packed=8.
    req = _req(["p"], height=32, width=32)
    B, T, S, C = 1, 3, 4, 8
    traj = torch.randn(B, T, S, C)
    out = a.unpack_trajectory(traj, req)
    assert out.shape == (B, T, C, 2, 2)


def test_klein_rejects_token_count_mismatch():
    mc = _model_config(shift=1.0, build_schedule_policy=lambda: None)
    a = Flux2KleinAdapter(_cfg(), mc, strategy=_Dance())
    req = _req(["p"], height=32, width=32)  # expects S = 4
    traj = torch.randn(1, 3, 9, 8)  # S = 9 ≠ 4
    with pytest.raises(ValueError, match="packed token count"):
        a.unpack_trajectory(traj, req)


def test_klein_validate_requires_build_schedule_policy():
    with pytest.raises(Exception, match="build_schedule_policy"):
        Flux2KleinAdapter(_cfg(), _model_config(shift=1.0), strategy=_Dance())
