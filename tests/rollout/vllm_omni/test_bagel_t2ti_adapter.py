from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from unirl.models.bagel.conditions import (
    BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT,
    BagelARConditions,
    BagelT2TIDiffusionConditions,
)
from unirl.models.bagel.diffusion import BagelDiffusionParams
from unirl.rollout.engine.vllm_omni.adapters import registered_adapters
from unirl.rollout.engine.vllm_omni.adapters.bagel import (
    GEN_THINK_SYSTEM_PROMPT,
    BagelT2TIAdapter,
)
from unirl.rollout.engine.vllm_omni.backends import STAGE_KIND_AR, STAGE_KIND_DIFFUSION
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import ARSamplingParams


def _adapter() -> BagelT2TIAdapter:
    return BagelT2TIAdapter(
        SimpleNamespace(model_path="unused"),
        SimpleNamespace(shift=3.0, use_lora=False),
    )


def _params(*, n_ar: int = 2, n_images: int = 1, cfg_text: float = 1.0):
    ar = ARSamplingParams(
        samples_per_prompt=n_ar,
        seed=23,
        temperature=1.0,
        top_p=0.92,
        top_k=17,
        max_new_tokens=31,
        stop_token_id=151645,
    )
    diffusion = BagelDiffusionParams(
        samples_per_prompt=n_images,
        num_inference_steps=2,
        guidance_scale=1.0,
        height=8,
        width=8,
        seed=17,
        eta=0.7,
        sde_indices=[0],
        trajectory_precision="fp32",
        cfg_text_scale=cfg_text,
        cfg_img_scale=1.0,
    )
    return ar, diffusion


def _request(
    prompts: list[str],
    *,
    n_ar: int = 2,
    n_images: int = 1,
    cfg_text: float = 1.0,
    noise_ids: list[str] | None = None,
) -> RolloutReq:
    ar, diffusion = _params(n_ar=n_ar, n_images=n_images, cfg_text=cfg_text)
    sample_ids = [f"p{i}" for i in range(len(prompts))]
    return RolloutReq(
        sample_ids=sample_ids,
        group_ids=[f"g{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=prompts)},
        sampling_params={"ar": ar, "diffusion": diffusion},
        stage_config={"rollout_id": 7},
        sigmas=torch.tensor([1.0, 0.4, 0.0], dtype=torch.float32),
        init_noise_group_ids=noise_ids or [],
        init_noise_latent_shape=[4, 2],
    )


def test_registered_and_builds_prompt_major_seeded_single_prompt_calls() -> None:
    assert "bagel_t2ti" in registered_adapters()
    req = _request(["a cat", "a bridge"], noise_ids=["r7:p0", "r7:p1"])
    adapter = _adapter()
    adapter.validate_request(req)

    calls = adapter.build_inputs(req)
    assert len(calls) == 4
    assert all(len(call.prompts) == 1 and call.group_by_request_id is False for call in calls)
    assert [[stage.kind for stage in call.sampling] for call in calls] == [[STAGE_KIND_AR, STAGE_KIND_DIFFUSION]] * 4

    exact_first_prompt = f"<|im_start|>{GEN_THINK_SYSTEM_PROMPT}<|im_end|><|im_start|>a cat<|im_end|><|im_start|>"
    assert calls[0].prompts[0]["prompt"] == exact_first_prompt
    assert calls[1].prompts[0]["prompt"] == exact_first_prompt
    assert calls[2].prompts[0]["prompt"].endswith("<|im_start|>a bridge<|im_end|><|im_start|>")
    assert calls[0].prompts[0]["modalities"] == ["image"]
    assert calls[0].prompts[0]["mm_processor_kwargs"] == {
        "target_h": 8,
        "target_w": 8,
        "modalities": ["image"],
    }

    ar_seeds = [call.sampling[0].kwargs["seed"] for call in calls]
    image_seeds = [call.sampling[1].kwargs["seed"] for call in calls]
    sde_seeds = [call.sampling[1].kwargs["extra_args"]["sde_seed"] for call in calls]
    assert len(set(ar_seeds + image_seeds + sde_seeds)) == 12
    assert [call.sampling[0].kwargs["n"] for call in calls] == [1] * 4
    assert [call.sampling[0].kwargs["logprobs"] for call in calls] == [1] * 4
    assert [call.sampling[0].kwargs["stop_token_ids"] for call in calls] == [[151645]] * 4

    expected_noise_ids = ["r7:p0/a0/i0", "r7:p0/a1/i0", "r7:p1/a0/i0", "r7:p1/a1/i0"]
    for call, expected_gid in zip(calls, expected_noise_ids):
        diff_kwargs = call.sampling[1].kwargs
        assert diff_kwargs["num_inference_steps"] == 3
        assert diff_kwargs["num_outputs_per_prompt"] == 1
        assert diff_kwargs["extra_args"]["init_noise_group_ids"] == [expected_gid]
        assert diff_kwargs["extra_args"]["init_noise_latent_shape"] == [4, 2]
        assert diff_kwargs["extra_args"]["init_noise_seed"] == 17
        assert diff_kwargs["extra_args"]["cfg_text_scale"] == 1.0
        assert diff_kwargs["extra_args"]["cfg_img_scale"] == 1.0
        assert diff_kwargs["extra_args"]["sde_seed"] != diff_kwargs["seed"]

    repeat = adapter.build_inputs(req)
    assert [call.sampling[0].kwargs["seed"] for call in repeat] == ar_seeds
    assert [call.sampling[1].kwargs["seed"] for call in repeat] == image_seeds
    assert [call.sampling[1].kwargs["extra_args"]["sde_seed"] for call in repeat] == sde_seeds

    req.sampling_params["ar"] = replace(req.sampling_params["ar"], seed=24)
    reseeded = adapter.build_inputs(req)
    assert [call.sampling[0].kwargs["seed"] for call in reseeded] != ar_seeds
    assert [call.sampling[1].kwargs["seed"] for call in reseeded] == image_seeds


def test_adapter_rejects_lora_configuration() -> None:
    with pytest.raises(ValueError, match="full-weight training only"):
        BagelT2TIAdapter(
            SimpleNamespace(model_path="unused"),
            SimpleNamespace(shift=3.0, use_lora=True),
        )


def test_init_same_noise_shares_only_xt_across_thought_branches() -> None:
    req = _request(["a cat"], noise_ids=["shared-root"])
    req.sampling_params["diffusion"] = replace(req.sampling_params["diffusion"], init_same_noise=True)

    calls = _adapter().build_inputs(req)
    assert [call.sampling[1].kwargs["extra_args"]["init_noise_group_ids"] for call in calls] == [
        ["shared-root"],
        ["shared-root"],
    ]
    image_seeds = [call.sampling[1].kwargs["seed"] for call in calls]
    sde_seeds = [call.sampling[1].kwargs["extra_args"]["sde_seed"] for call in calls]
    assert len(set(image_seeds)) == 2
    assert len(set(sde_seeds)) == 2
    assert set(image_seeds).isdisjoint(sde_seeds)


@pytest.mark.parametrize(
    ("req", "match"),
    [
        (_request(["x"], n_images=2), "samples_per_prompt == 1"),
        (_request(["x"], cfg_text=2.0), "CFG scales == 1"),
        (_request(["x"], n_ar=1), "ar.samples_per_prompt >= 2"),
    ],
)
def test_strict_unigrpo_validation(req: RolloutReq, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _adapter().validate_request(req)


def _raw_pair(index: int, *, include_replay: bool = True):
    # The exact rendered prompt starts and ends with the same <|im_start|> ID.
    prompt_ids = [99, 10 + index, 11 + index, 99]
    sampled_ids = [20 + index, 21 + index]
    completion = SimpleNamespace(
        token_ids=sampled_ids,
        text=f"<think>plan {index}</think>",
        # Deliberately put a higher-probability non-sampled token first.
        logprobs=[
            {777: SimpleNamespace(logprob=-0.01), sampled_ids[0]: SimpleNamespace(logprob=-0.2 - index)},
            {778: SimpleNamespace(logprob=-0.02), sampled_ids[1]: SimpleNamespace(logprob=-0.3 - index)},
        ],
    )
    ar_output = SimpleNamespace(
        request_id=f"{index}_ar",
        stage_id=0,
        final_output_type="text",
        request_output=SimpleNamespace(outputs=[completion]),
        prompt_token_ids=prompt_ids,
    )

    # Final vLLM KV contains every prompt token and sampled token except the
    # last token, which was emitted but never consumed by another decode step.
    trace = prompt_ids + sampled_ids[:-1]
    replay_payload = {
        "cache_input_ids": trace,
        # Scheduler chunking splits the prompt; the adapter must validate the
        # full trace prefix rather than assuming chunk 0 is the whole prompt.
        "chunk_offsets": [0, 2, 5],
        "kv_length": 5,
        "ropes": [5],
        "received_kv_length": 5,
        "received_ropes": [5],
        "image_shape": [8, 8],
    }
    custom_output = {"sde_step_indices": [0]}
    if include_replay:
        custom_output[BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT] = replay_payload
    image_output = SimpleNamespace(
        request_id=f"{index}_image",
        stage_id=1,
        final_output_type="image",
        images=[object()],
        trajectory_latents=torch.full((1, 3, 2, 2), float(index)),
        trajectory_timesteps=torch.tensor([1.0, 0.4, 0.0], dtype=torch.float32),
        trajectory_log_probs=torch.tensor([[-1.0 - index]], dtype=torch.float32),
        custom_output=custom_output,
    )
    return [ar_output, image_output]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda output: setattr(output, "trajectory_latents", None), "trajectory_latents"),
        (lambda output: setattr(output, "trajectory_log_probs", None), "trajectory_log_probs"),
        (lambda output: setattr(output, "trajectory_timesteps", torch.tensor([1.0, 0.5, 0.0])), "sigmas"),
        (lambda output: output.custom_output.__setitem__("sde_step_indices", [1]), "SDE indices"),
        (lambda output: setattr(output, "images", [object(), object()]), "expected one"),
    ],
)
def test_output_rejects_misaligned_image_trajectory(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    match: str,
) -> None:
    req = _request(["one prompt"], n_ar=2)
    per_request = [_raw_pair(0), _raw_pair(1)]
    mutate(per_request[1][1])
    monkeypatch.setattr(
        "unirl.rollout.engine.vllm_omni.adapters.bagel.pils_to_images",
        lambda images: Images(pixels=torch.zeros(len(images), 3, 8, 8)),
    )

    with pytest.raises(RuntimeError, match=match):
        _adapter().build_response(req, per_request)


def test_output_builds_linked_tracks_and_native_replay_conditions(monkeypatch: pytest.MonkeyPatch) -> None:
    req = _request(["one prompt"], n_ar=2)
    adapter = _adapter()
    per_request = [_raw_pair(0), _raw_pair(1)]

    # torchvision is intentionally not a dependency of this CPU contract test.
    monkeypatch.setattr(
        "unirl.rollout.engine.vllm_omni.adapters.bagel.pils_to_images",
        lambda images: Images(pixels=torch.zeros(len(images), 3, 8, 8)),
    )
    resp = adapter.build_response(req, per_request)

    ar_track = resp.tracks["ar"]
    image_track = resp.tracks["image"]
    assert ar_track.sample_ids == ["p0/a0", "p0/a1"]
    assert ar_track.parent_ids == ["p0", "p0"]
    assert ar_track.parent_track is None
    assert image_track.sample_ids == ["p0/a0/i0", "p0/a1/i0"]
    assert image_track.parent_ids == ar_track.sample_ids
    assert image_track.parent_track == "ar"

    assert set(ar_track.conditions) == {"bagel_ar"}
    ar_conditions = BagelARConditions.from_dict(ar_track.conditions)
    assert ar_conditions.prompt_splits[0][0]["ids"].tolist() == [99, 10, 11]
    assert ar_conditions.prompt_splits[1][0]["ids"].tolist() == [99, 11, 12]

    assert set(image_track.conditions) == {"bagel_t2ti"}
    image_conditions = BagelT2TIDiffusionConditions.from_dict(image_track.conditions)
    assert image_conditions.replay_specs[0].cache_input_ids == (99, 10, 11, 99, 20)
    assert image_conditions.replay_specs[0].chunks() == ((99, 10), (11, 99, 20))
    assert image_conditions.replay_specs[1].cache_input_ids == (99, 11, 12, 99, 21)

    # Log-probs must come from the emitted token-id entry, not dict insertion order.
    assert torch.allclose(
        ar_track.segment.log_probs,
        torch.tensor([-0.2, -0.3, -1.2, -1.3], dtype=torch.float32),
    )
    assert ar_track.decoded.texts == ["<think>plan 0</think>", "<think>plan 1</think>"]
    assert tuple(image_track.decoded.pixels.shape) == (2, 3, 8, 8)
    assert tuple(image_track.segment.latents.shape) == (2, 3, 2, 2)


def test_output_rejects_missing_native_replay_metadata() -> None:
    req = _request(["one prompt"], n_ar=1)
    with pytest.raises(RuntimeError, match=BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT):
        _adapter().build_response(req, [_raw_pair(0, include_replay=False)])


def test_output_rejects_incomplete_native_kv_boundary() -> None:
    req = _request(["one prompt"], n_ar=1)
    pair = _raw_pair(0)
    payload = pair[1].custom_output[BAGEL_T2TI_REPLAY_CUSTOM_OUTPUT]
    payload["cache_input_ids"] = payload["cache_input_ids"][:-1]
    payload["chunk_offsets"] = [0, 2, 4]
    payload["kv_length"] = 4
    payload["ropes"] = [4]
    payload["received_kv_length"] = 4
    payload["received_ropes"] = [4]
    with pytest.raises(RuntimeError, match=r"sampled token_ids\[:-1\]"):
        _adapter().build_response(req, [pair])


@pytest.mark.parametrize(
    "bad_logprobs",
    [
        None,
        [],
        [None, None],
        [
            {20: SimpleNamespace(logprob=float("nan"))},
            {21: SimpleNamespace(logprob=-0.3)},
        ],
    ],
)
def test_output_rejects_missing_malformed_or_nonfinite_ar_logprobs(bad_logprobs) -> None:
    req = _request(["one prompt"], n_ar=1)
    pair = _raw_pair(0)
    pair[0].request_output.outputs[0].logprobs = bad_logprobs
    with pytest.raises(RuntimeError, match="logprob"):
        _adapter().build_response(req, [pair])


def test_output_rejects_unproven_trailing_assistant_start_token() -> None:
    req = _request(["one prompt"], n_ar=1)
    pair = _raw_pair(0)
    pair[0].prompt_token_ids[-1] = 123
    with pytest.raises(RuntimeError, match=r"<\|im_start\|>"):
        _adapter().build_response(req, [pair])


def test_validation_uses_modified_dataclass_values() -> None:
    req = _request(["x"])
    req.sampling_params["diffusion"] = replace(req.sampling_params["diffusion"], cfg_img_scale=1.5)
    with pytest.raises(ValueError, match="cfg_img_scale"):
        _adapter().validate_request(req)


def test_validation_rejects_tempered_ar_logprob_mismatch() -> None:
    req = _request(["x"])
    req.sampling_params["ar"] = replace(req.sampling_params["ar"], temperature=0.6)
    with pytest.raises(ValueError, match="ar.temperature == 1.0"):
        _adapter().validate_request(req)
