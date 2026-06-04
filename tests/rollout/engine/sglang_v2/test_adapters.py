"""Pure adapter conversion tests — canned wire data, stub tokenizer/processor.

Covers the sampling-resolution precedence (the predecessor's exact rules,
incl. the top_k translation and the ``samples_pre_expanded`` n-logic), the
chat-template encode paths, and the response packing (TextSegment, sample-id
mangling, prompt-condition right-padding, think-tag stripping, VLM condition
replication).
"""

from __future__ import annotations

import pytest
import torch
from conftest import StubProcessor, StubTokenizer, make_raw

from unirl.rollout.engine.sglang_v2.adapters import TextLMAdapter, VLMAdapter
from unirl.rollout.engine.sglang_v2.config import SGLangV2EngineConfig
from unirl.rollout.engine.sglang_v2.utils import resolve_sampling, split_thinking_tags
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import ARSamplingParams


def make_req(prompts, *, sampling_params=None, stage_config=None, images=None):
    primitives = {"text": Texts(texts=list(prompts))}
    if images is not None:
        primitives["image"] = images
    return RolloutReq(
        sample_ids=[f"p{i}" for i in range(len(prompts))],
        group_ids=[f"g{i}" for i in range(len(prompts))],
        primitives=primitives,
        sampling_params=sampling_params,
        stage_config=dict(stage_config or {}),
    )


def make_text_adapter(config=None, tokenizer=None) -> TextLMAdapter:
    config = config or SGLangV2EngineConfig(pretrained_model_ckpt_path="stub/model")
    return TextLMAdapter(config, None, tokenizer=tokenizer or StubTokenizer())


# ---------------------------------------------------------------------------
# Sampling resolution — the 3-source precedence
# ---------------------------------------------------------------------------


def test_resolve_sampling_typed_ar_wins_over_config_defaults():
    cfg = SGLangV2EngineConfig(pretrained_model_ckpt_path="m", temperature=0.7, top_p=0.9, max_new_tokens=512)
    ar = ARSamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_new_tokens=8192)
    sampling = resolve_sampling(cfg, make_req(["a"], sampling_params=ar))
    assert sampling.block["temperature"] == 1.0
    assert sampling.block["top_p"] == 0.95
    assert sampling.block["max_new_tokens"] == 8192


def test_resolve_sampling_config_defaults_without_ar_params():
    cfg = SGLangV2EngineConfig(pretrained_model_ckpt_path="m", temperature=0.3, top_p=0.8, max_new_tokens=64)
    sampling = resolve_sampling(cfg, make_req(["a"]))
    assert sampling.block["temperature"] == 0.3
    assert sampling.block["top_p"] == 0.8
    assert sampling.block["max_new_tokens"] == 64


def test_resolve_sampling_top_k_translation():
    cfg = SGLangV2EngineConfig(pretrained_model_ckpt_path="m")
    # Positive top_k passes through.
    ar = ARSamplingParams(top_k=20)
    assert resolve_sampling(cfg, make_req(["a"], sampling_params=ar)).block["top_k"] == 20
    # The trainer's 0 (unrestricted, HF convention) maps to SGLang's -1.
    ar0 = ARSamplingParams(top_k=0)
    assert resolve_sampling(cfg, make_req(["a"], sampling_params=ar0)).block["top_k"] == -1
    # No typed AR params: the predecessor sent -1 (disabled), NOT cfg.top_k.
    assert resolve_sampling(cfg, make_req(["a"])).block["top_k"] == -1


def test_resolve_sampling_n_logic():
    ar = ARSamplingParams(samples_per_prompt=8)
    # Pre-expanded: the caller already fanned P -> P*N; emit one per entry.
    pre = SGLangV2EngineConfig(pretrained_model_ckpt_path="m", samples_pre_expanded=True)
    assert resolve_sampling(pre, make_req(["a"], sampling_params=ar)).n == 1
    # Unexpanded: the engine fans out samples_per_prompt itself.
    raw = SGLangV2EngineConfig(pretrained_model_ckpt_path="m")
    assert resolve_sampling(raw, make_req(["a"], sampling_params=ar)).n == 8
    # No typed params: stage_config['ar']['n'] then 1.
    assert resolve_sampling(raw, make_req(["a"], stage_config={"ar": {"n": 3}})).n == 3
    assert resolve_sampling(raw, make_req(["a"])).n == 1


def test_resolve_sampling_stage_config_extras():
    cfg = SGLangV2EngineConfig(pretrained_model_ckpt_path="m", system_instruction="/cfg")
    stage = {
        "ar": {
            "return_logprob": False,
            "system_instruction": "/stage",
            "stop": ["</s>"],
            "stop_token_ids": [2],
            "skip_special_tokens": False,
        }
    }
    sampling = resolve_sampling(cfg, make_req(["a"], stage_config=stage))
    assert sampling.return_logprob is False
    assert sampling.system_instruction == "/stage"  # stage wins over config
    assert sampling.block["stop"] == ["</s>"]
    assert sampling.block["stop_token_ids"] == [2]
    assert sampling.block["skip_special_tokens"] is False
    # Config fallback when stage carries none.
    fallback = resolve_sampling(cfg, make_req(["a"]))
    assert fallback.system_instruction == "/cfg"
    assert fallback.return_logprob is True
    assert "stop" not in fallback.block


# ---------------------------------------------------------------------------
# build_inputs — chat-template encode paths
# ---------------------------------------------------------------------------


def test_build_inputs_chat_template_path():
    tok = StubTokenizer()
    cfg = SGLangV2EngineConfig(pretrained_model_ckpt_path="m", chat_template_kwargs={"enable_thinking": False})
    adapter = make_text_adapter(cfg, tok)
    sampling = resolve_sampling(cfg, make_req(["hi"], stage_config={"ar": {"system_instruction": "/sys"}}))
    prepared = adapter.build_inputs(make_req(["hi"]), sampling=sampling)

    assert len(prepared.wire) == 1
    payload = prepared.wire[0]
    assert payload["input_ids"] == prepared.prompt_token_ids[0]
    assert "text" not in payload
    assert payload["return_logprob"] is True
    assert payload["logprob_start_len"] == 0
    assert payload["sampling_params"]["n"] == sampling.n
    # The system message reached the template and chat_template_kwargs forwarded.
    call = tok.template_calls[0]
    assert call["messages"][0] == {"role": "system", "content": "/sys"}
    assert call["kwargs"] == {"enable_thinking": False}


def test_build_inputs_raw_text_fallback_when_template_fails():
    tok = StubTokenizer(fail_template=True)
    adapter = make_text_adapter(tokenizer=tok)
    cfg = adapter.cfg
    sampling = resolve_sampling(cfg, make_req(["fallback prompt"]))
    prepared = adapter.build_inputs(make_req(["fallback prompt"]), sampling=sampling)

    payload = prepared.wire[0]
    assert payload["text"] == "fallback prompt"
    assert "input_ids" not in payload
    # The replay prompt condition still carries ids (tokenizer.encode fallback).
    assert prepared.prompt_token_ids[0] == tok.encode("fallback prompt")


def test_build_inputs_rejects_images_in_text_mode():
    adapter = make_text_adapter()
    req = make_req(["a"], images=Images(pixels=torch.zeros(1, 3, 4, 4)))
    sampling = resolve_sampling(adapter.cfg, req)
    with pytest.raises(ValueError, match="text-only mode"):
        adapter.build_inputs(req, sampling=sampling)


def test_build_inputs_sampling_block_is_copied_per_payload():
    adapter = make_text_adapter()
    sampling = resolve_sampling(adapter.cfg, make_req(["a", "b"]))
    prepared = adapter.build_inputs(make_req(["a", "b"]), sampling=sampling)
    prepared.wire[0]["sampling_params"]["temperature"] = 99.0
    assert prepared.wire[1]["sampling_params"]["temperature"] != 99.0


# ---------------------------------------------------------------------------
# build_response — packing
# ---------------------------------------------------------------------------


def make_prepared(adapter, prompts, *, n=1):
    cfg = adapter.cfg
    req = make_req(prompts)
    sampling = resolve_sampling(cfg, req)
    prepared = adapter.build_inputs(req, sampling=sampling)
    prepared.resolved_n = n
    return req, prepared


def test_build_response_single_sample_shape():
    adapter = make_text_adapter()
    req, prepared = make_prepared(adapter, ["a", "b"])
    raw = [make_raw("x", [1, 2], [-0.5, -0.6]), make_raw("y", [3], [-0.7])]
    resp = adapter.build_response(req, prepared, raw)

    track = resp.tracks["ar"]
    assert track.sample_ids == ["p0", "p1"]  # no mangling at n=1
    assert track.parent_ids == ["g0", "g1"]
    assert track.decoded.texts == ["x", "y"]
    assert track.segment.tokens.tolist() == [1, 2, 3]
    assert torch.allclose(track.segment.log_probs, torch.tensor([-0.5, -0.6, -0.7], dtype=torch.float32))


def test_build_response_mangles_sample_ids_when_n_gt_1():
    adapter = make_text_adapter()
    req, prepared = make_prepared(adapter, ["a", "b"], n=2)
    raw = [make_raw(t) for t in ("a0", "a1", "b0", "b1")]
    resp = adapter.build_response(req, prepared, raw)

    track = resp.tracks["ar"]
    assert track.sample_ids == ["p0#0", "p0#1", "p1#0", "p1#1"]
    # Group membership stays intact across siblings.
    assert track.parent_ids == ["g0", "g0", "g1", "g1"]
    # Prompt-major order preserved end-to-end.
    assert track.decoded.texts == ["a0", "a1", "b0", "b1"]


def test_build_response_rejects_count_mismatch():
    adapter = make_text_adapter()
    req, prepared = make_prepared(adapter, ["a"], n=2)
    with pytest.raises(ValueError, match="expected 2 candidates"):
        adapter.build_response(req, prepared, [make_raw()])


def test_build_response_prompt_condition_right_padded():
    tok = StubTokenizer()
    adapter = make_text_adapter(tokenizer=tok)
    req, prepared = make_prepared(adapter, ["a", "bb"], n=2)
    # Force different prompt lengths.
    prepared.prompt_token_ids = [[11, 12, 13], [21]]
    raw = [make_raw() for _ in range(4)]
    resp = adapter.build_response(req, prepared, raw)

    cond = resp.tracks["ar"].conditions["prompt"]
    assert cond.input_ids.shape == (4, 3)  # 2 prompts × n=2 rows, padded to max 3
    # Prompt ids replicated per sibling; pad uses the tokenizer's pad id (7).
    assert cond.input_ids[0].tolist() == [11, 12, 13]
    assert cond.input_ids[1].tolist() == [11, 12, 13]
    assert cond.input_ids[2].tolist() == [21, 7, 7]
    assert cond.attention_mask[2].tolist() == [1, 0, 0]
    assert cond.attention_mask[0].tolist() == [1, 1, 1]


def test_build_response_strips_thinking_tags_into_decoded():
    adapter = make_text_adapter()
    req, prepared = make_prepared(adapter, ["a", "b", "c"])
    raw = [
        make_raw("<think>plan</think>answer"),
        make_raw("before<think>truncated reasoning"),  # unclosed: content before tag
        make_raw("<think>only thoughts</think>"),  # empty content -> raw-text fallback
    ]
    resp = adapter.build_response(req, prepared, raw)
    texts = resp.tracks["ar"].decoded.texts
    assert texts[0] == "answer"
    assert texts[1] == "before"
    # Predecessor's `content or text`: an all-think output decodes as the RAW
    # text (tags intact), never as the empty string.
    assert texts[2] == "<think>only thoughts</think>"


def test_split_thinking_tags_forms():
    assert split_thinking_tags("<think>r</think>c") == ("c", "r")
    assert split_thinking_tags("c<think>r") == ("c", "r")
    assert split_thinking_tags("plain") == ("plain", "")


# ---------------------------------------------------------------------------
# VLM adapter — narrowest overrides
# ---------------------------------------------------------------------------


def make_vlm_adapter():
    cfg = SGLangV2EngineConfig(pretrained_model_ckpt_path="m", image_token="<img>")
    return VLMAdapter(cfg, None, tokenizer=StubTokenizer(), processor=StubProcessor())


def make_vlm_req(prompts):
    return make_req(prompts, images=Images(pixels=torch.rand(len(prompts), 3, 8, 8)))


def test_vlm_validate_requires_processor_and_image_token():
    cfg = SGLangV2EngineConfig(pretrained_model_ckpt_path="m", image_token="<img>")
    with pytest.raises(ValueError, match="AutoProcessor"):
        VLMAdapter(cfg, None, tokenizer=StubTokenizer(), processor=None)
    cfg_text = SGLangV2EngineConfig(pretrained_model_ckpt_path="m", model_family="vlm")
    with pytest.raises(ValueError, match="image_token"):
        VLMAdapter(cfg_text, None, tokenizer=StubTokenizer(), processor=StubProcessor())


def test_vlm_build_inputs_sends_templated_text_plus_image_data():
    adapter = make_vlm_adapter()
    req = make_vlm_req(["look at this"])
    sampling = resolve_sampling(adapter.cfg, req)
    prepared = adapter.build_inputs(req, sampling=sampling)

    payload = prepared.wire[0]
    # The single-placeholder TEXT goes to SRT (never the expanded input_ids).
    assert payload["text"].startswith("templated:")
    assert "input_ids" not in payload
    assert payload["image_data"].startswith("data:image/png;base64,")
    # The replay prompt is the processor's EXPANDED ids.
    assert prepared.prompt_token_ids[0] == prepared.mm[0].input_ids
    assert len(prepared.prompt_token_ids[0]) >= 8


def test_vlm_build_inputs_requires_images():
    adapter = make_vlm_adapter()
    req = make_req(["no image"])
    sampling = resolve_sampling(adapter.cfg, req)
    with pytest.raises(ValueError, match="primitives\\['image'\\]"):
        adapter.build_inputs(req, sampling=sampling)


def test_vlm_conditions_replicate_per_sibling():
    adapter = make_vlm_adapter()
    req = make_vlm_req(["a", "b"])
    sampling = resolve_sampling(adapter.cfg, req)
    prepared = adapter.build_inputs(req, sampling=sampling)
    prepared.resolved_n = 3
    raw = [make_raw() for _ in range(6)]
    resp = adapter.build_response(req, prepared, raw)

    conds = resp.tracks["ar"].conditions
    assert "prompt" in conds
    assert len(conds["pixel_values"]) == 6
    assert len(conds["image_grid_thw"]) == 6
    # Rows 0-2 carry prompt 0's encoding; rows 3-5 prompt 1's.
    assert conds["pixel_values"][0] is prepared.mm[0].pixel_values
    assert conds["pixel_values"][2] is prepared.mm[0].pixel_values
    assert conds["pixel_values"][3] is prepared.mm[1].pixel_values


def test_vlm_conditions_absent_without_mm():
    # The text base never emits multimodal conditions.
    adapter = make_text_adapter()
    req, prepared = make_prepared(adapter, ["a"])
    resp = adapter.build_response(req, prepared, [make_raw()])
    conds = resp.tracks["ar"].conditions
    assert "pixel_values" not in conds and "image_grid_thw" not in conds
