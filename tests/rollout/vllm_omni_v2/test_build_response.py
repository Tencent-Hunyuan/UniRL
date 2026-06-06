"""Adapter ``build_response`` — grouped wire results → RolloutResp, per shape.

Locks the v1 response.py behavior: two-track lineage (``parent_track="ar"``),
condition replication across tracks, missing-capture raises (not silent
empties), σ round-trip verification, and the best-effort AR text path.
"""

from __future__ import annotations

import pytest
import torch

from tests.rollout.vllm_omni_v2.conftest import (
    fake_ar_output,
    fake_dit_output,
    fake_fused_capture,
    fake_hv15_capture,
    fake_qwen_capture,
    fake_sd3_capture,
    make_adapter,
    make_req,
    sigmas_for,
)


def test_t2i_two_tracks_with_lineage(model_config):
    adapter = make_adapter("hi3_t2i", model_config)
    sig = sigmas_for(2)
    req = make_req(2, modality_params="composed", sigmas=sig)
    per_request = [
        [fake_ar_output(i), fake_dit_output(i, sigmas=sig, custom_capture={"fused_mm_capture": fake_fused_capture()})]
        for i in range(2)
    ]
    resp = adapter.build_response(req, per_request)

    assert set(resp.tracks) == {"ar", "image"}
    image, ar = resp.tracks["image"], resp.tracks["ar"]
    # HI3 think_recaption lineage: image is generated from AR output 1-to-1.
    assert image.parent_track == "ar" and image.parent_ids == list(req.sample_ids)
    assert ar.parent_track is None and ar.parent_ids == list(req.group_ids)
    # Legacy resp-wide conditions replicated onto every track.
    assert "fused" in image.conditions and "fused" in ar.conditions
    # Decoded payloads: image pixels [B, C, H, W] in [0, 1]; AR text best-effort.
    assert image.decoded.pixels.shape == (2, 3, 8, 8)
    assert ar.decoded.texts == ["ar text", "ar text"]
    # Segment: dense latents [B, T+1, ...] + the σ echo.
    assert image.segment.latents.shape[0] == 2 and image.segment.latents.shape[1] == 3
    assert ar.segment is not None


def test_t2i_missing_fused_capture_raises(model_config):
    adapter = make_adapter("hi3_t2i", model_config)
    sig = sigmas_for(2)
    req = make_req(1, modality_params="composed", sigmas=sig)
    per_request = [[fake_ar_output(0), fake_dit_output(0, sigmas=sig)]]  # no fused_mm_capture
    with pytest.raises(RuntimeError, match="fused_mm_capture"):
        adapter.build_response(req, per_request)


def test_t2i_missing_dit_output_raises(model_config):
    adapter = make_adapter("hi3_t2i", model_config)
    req = make_req(1, modality_params="composed", sigmas=sigmas_for(2))
    with pytest.raises(RuntimeError, match="did the DiT stage fail"):
        adapter.build_response(req, [[fake_ar_output(0)]])


def test_sigma_echo_mismatch_raises(model_config):
    adapter = make_adapter("sd3_t2i", model_config)
    sig = sigmas_for(2)
    req = make_req(1, sigmas=sig)
    bad = fake_dit_output(0, sigmas=torch.tensor([0.9, 0.4, 0.0]), stage_id=0, custom_capture=fake_sd3_capture())
    with pytest.raises(Exception):
        adapter.build_response(req, [[bad]])


def test_sd35_text_condition_and_empty_input_guard(model_config):
    adapter = make_adapter("sd3_t2i", model_config)
    sig = sigmas_for(2)
    req = make_req(2, sigmas=sig)
    per_request = [[fake_dit_output(i, sigmas=sig, stage_id=0, custom_capture=fake_sd3_capture())] for i in range(2)]
    resp = adapter.build_response(req, per_request)
    assert set(resp.tracks) == {"image"}
    cond = resp.tracks["image"].conditions["text"]
    assert cond.embeds.shape == (2, 6, 8) and cond.pooled.shape == (2, 8)
    assert resp.tracks["image"].parent_track is None  # no AR sibling → root

    with pytest.raises(RuntimeError, match="text_capture"):
        adapter.build_response(req, [[fake_dit_output(i, sigmas=sig, stage_id=0)] for i in range(2)])
    with pytest.raises(ValueError, match="empty per-request"):
        adapter.build_response(req, [[], []])


def test_qwen_text_condition_cfg_off(model_config):
    adapter = make_adapter("qwen_image_t2i", model_config)
    sig = sigmas_for(2)
    req = make_req(2, sigmas=sig)
    per_request = [[fake_dit_output(i, sigmas=sig, stage_id=0, custom_capture=fake_qwen_capture())] for i in range(2)]
    resp = adapter.build_response(req, per_request)
    assert set(resp.tracks) == {"image"}
    track = resp.tracks["image"]
    cond = track.conditions["text"]
    # Qwen-Image conditioning: token embeds + mask, NO pooled vector.
    assert cond.embeds.shape == (2, 6, 8) and cond.attn_mask.shape == (2, 6)
    assert cond.pooled is None
    assert "negative_text" not in track.conditions  # CFG-off rollout
    assert track.parent_track is None

    with pytest.raises(RuntimeError, match="text_capture"):
        adapter.build_response(req, [[fake_dit_output(i, sigmas=sig, stage_id=0)] for i in range(2)])


def test_qwen_ragged_pad_cat_across_requests(model_config):
    """Per-request L differs after the chat-template prefix strip — embeds
    pad with zero rows, masks with zeros, so mask sums echo true lengths."""
    adapter = make_adapter("qwen_image_t2i", model_config)
    sig = sigmas_for(2)
    req = make_req(2, sigmas=sig)
    per_request = [
        [fake_dit_output(0, sigmas=sig, stage_id=0, custom_capture=fake_qwen_capture(L=5))],
        [fake_dit_output(1, sigmas=sig, stage_id=0, custom_capture=fake_qwen_capture(L=7))],
    ]
    resp = adapter.build_response(req, per_request)
    cond = resp.tracks["image"].conditions["text"]
    assert cond.embeds.shape == (2, 7, 8) and cond.attn_mask.shape == (2, 7)
    assert cond.attn_mask.sum(dim=1).tolist() == [5, 7]


def test_qwen_cfg_on_emits_negative_text(model_config):
    adapter = make_adapter("qwen_image_t2i", model_config)
    sig = sigmas_for(2)
    req = make_req(2, sigmas=sig)
    per_request = [
        [fake_dit_output(i, sigmas=sig, stage_id=0, custom_capture=fake_qwen_capture(with_negative=True))]
        for i in range(2)
    ]
    resp = adapter.build_response(req, per_request)
    neg = resp.tracks["image"].conditions["negative_text"]
    assert neg.embeds.shape == (2, 4, 8) and neg.pooled is None

    # Mixed capture (CFG fired for one request only) is fatal, not silent.
    mixed = [
        [fake_dit_output(0, sigmas=sig, stage_id=0, custom_capture=fake_qwen_capture(with_negative=True))],
        [fake_dit_output(1, sigmas=sig, stage_id=0, custom_capture=fake_qwen_capture())],
    ]
    with pytest.raises(RuntimeError, match="uniform"):
        adapter.build_response(req, mixed)


def test_qwen_sigma_echo_and_nft_k0(model_config):
    adapter = make_adapter("qwen_image_t2i", model_config)
    sig = sigmas_for(2)
    req = make_req(1, sigmas=sig)
    bad = fake_dit_output(0, sigmas=torch.tensor([0.9, 0.4, 0.0]), stage_id=0, custom_capture=fake_qwen_capture())
    with pytest.raises(Exception):
        adapter.build_response(req, [[bad]])

    # K=0 NFT path: clean-latents segment, sde fields dropped.
    nft = fake_dit_output(0, sigmas=sig, stage_id=0, K=0, sde_step_indices=[], custom_capture=fake_qwen_capture())
    resp = adapter.build_response(req, [[nft]])
    seg = resp.tracks["image"].segment
    assert seg.sde_logp is None and seg.sde_indices is None
    assert seg.indices.tolist() == [0, 1, 2]


def test_t2v_video_track_and_dual_stream_conditions(model_config):
    adapter = make_adapter("hv15_t2v", model_config)
    sig = sigmas_for(2)
    req = make_req(2, sigmas=sig)
    per_request = [
        [
            fake_dit_output(
                i,
                sigmas=sig,
                stage_id=0,
                final_output_type="video",
                custom_capture=fake_hv15_capture(),
                num_images=3,
            )
        ]
        for i in range(2)
    ]
    resp = adapter.build_response(req, per_request)
    assert set(resp.tracks) == {"video"}
    video = resp.tracks["video"]
    # Per-prompt frame groups → Videos packed varlen along T: [total_T, C, H, W]
    # with cu_frames boundaries per sample.
    assert video.decoded.frames.shape == (6, 3, 8, 8)
    assert video.decoded.cu_frames.tolist() == [0, 3, 6]
    assert {"text_mllm", "text_glyph"} <= set(video.conditions)

    # Missing dual-stream capture is fatal.
    with pytest.raises(RuntimeError, match="text_capture"):
        adapter.build_response(
            req,
            [[fake_dit_output(i, sigmas=sig, stage_id=0, final_output_type="video", num_images=3)] for i in range(2)],
        )


def test_ar_only_track(model_config):
    adapter = make_adapter("hi3_t2t", model_config)
    req = make_req(2, modality_params="ar")
    resp = adapter.build_response(req, [[fake_ar_output(i, text=f"out {i}")] for i in range(2)])
    assert set(resp.tracks) == {"ar"}
    ar = resp.tracks["ar"]
    assert ar.decoded.texts == ["out 0", "out 1"]
    assert ar.conditions == {}
    assert ar.segment is not None and ar.parent_track is None


def test_ar_recaption_fused_prompt_condition(model_config):
    adapter = make_adapter("hi3_ar_recaption", model_config)
    req = make_req(2, modality_params="composed")
    per_request = [
        [fake_ar_output(0, prompt_token_ids=[1, 2, 3])],
        [fake_ar_output(1, prompt_token_ids=[4, 5])],
    ]
    resp = adapter.build_response(req, per_request)
    fused = resp.tracks["ar"].conditions["fused"]
    # Right-padded [B, max_len] + true lengths (the replay slices by length).
    assert fused.input_ids.tolist() == [[1, 2, 3], [4, 5, 0]]
    assert fused.prompt_lengths.tolist() == [3, 2]


def test_ar_text_extraction_is_best_effort_for_two_stage(model_config, monkeypatch):
    """t2i: a broken AR-text path must not break the rollout (v1:671-675)."""
    adapter = make_adapter("hi3_t2i", model_config)
    sig = sigmas_for(2)
    req = make_req(1, modality_params="composed", sigmas=sig)
    per_request = [
        [fake_ar_output(0), fake_dit_output(0, sigmas=sig, custom_capture={"fused_mm_capture": fake_fused_capture()})]
    ]
    import unirl.rollout.engine.vllm_omni_v2.adapters.hi3 as hi3_mod

    def boom(_):
        raise RuntimeError("text extraction broke")

    monkeypatch.setattr(hi3_mod, "decoded_text_from_ar", boom)
    resp = adapter.build_response(req, per_request)
    assert resp.tracks["ar"].decoded is None  # dropped, not raised
