"""Pure-helper mechanics: segments, grouping, seeds, σ slicing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.rollout.vllm_omni.conftest import fake_ar_output, fake_dit_output, make_req, sigmas_for
from unirl.rollout.engine.vllm_omni.backends.native import (
    _group_by_request,
    _tp_from_stage_configs,
)
from unirl.rollout.engine.vllm_omni.utils import (
    build_ar_segment,
    build_image_segment,
    pils_to_images,
    seed_from_sample_id,
    sigmas_list_from_req,
)


def test_build_image_segment_sparse_sde():
    sig = sigmas_for(4)
    outs = [fake_dit_output(i, sigmas=sig, K=2, sde_step_indices=[1, 3]) for i in range(2)]
    seg = build_image_segment(outs, expected_sigmas=sig)
    assert seg.latents.shape == (2, 5, 4, 2, 2)  # [B, T+1, ...]
    assert seg.sde_logp.shape == (2, 2)
    assert seg.sde_indices.tolist() == [1, 3]
    assert seg.indices.tolist() == [0, 1, 2, 3, 4]  # dense storage slots


def test_build_image_segment_k_zero_nft_path():
    sig = sigmas_for(3)
    outs = [fake_dit_output(0, sigmas=sig, K=0, sde_step_indices=[])]
    seg = build_image_segment(outs, expected_sigmas=sig)
    # Clean-latents path: indices kept, sde fields dropped (no [B, 0] stub).
    assert seg.sde_logp is None and seg.sde_indices is None
    assert seg.indices.tolist() == [0, 1, 2, 3]


def test_build_image_segment_sde_count_mismatch_raises():
    sig = sigmas_for(4)
    outs = [fake_dit_output(0, sigmas=sig, K=2, sde_step_indices=[1])]  # 1 != K=2
    with pytest.raises(RuntimeError, match="inconsistent"):
        build_image_segment(outs, expected_sigmas=sig)


def test_build_image_segment_sparse_without_echo_raises():
    """Legacy fallback is only safe when K == T; sparse K < T must raise."""
    sig = sigmas_for(4)
    out = fake_dit_output(0, sigmas=sig, K=2)
    out.custom_output = {}  # no sde_step_indices echo
    with pytest.raises(RuntimeError, match="sde_step_indices"):
        build_image_segment([out], expected_sigmas=sig)


def test_pils_to_images_range():
    from PIL import Image as PILImage

    imgs = pils_to_images([PILImage.new("RGB", (4, 4), color=(255, 0, 0))])
    assert imgs.pixels.shape == (1, 3, 4, 4)
    assert float(imgs.pixels.max()) <= 1.0 and float(imgs.pixels.min()) >= 0.0


def test_build_ar_segment_packs_and_drops_ragged_logp():
    outs_ok = [[fake_ar_output(0, token_ids=[1, 2])], [fake_ar_output(1, token_ids=[3, 4, 5])]]
    seg = build_ar_segment(outs_ok)
    assert seg is not None and seg.log_probs is not None

    # All-or-nothing: one token-bearing row missing logp drops the field.
    outs_ragged = [
        [fake_ar_output(0, token_ids=[1, 2])],
        [fake_ar_output(1, token_ids=[3, 4], with_logprobs=False)],
    ]
    seg2 = build_ar_segment(outs_ragged)
    assert seg2 is not None and seg2.log_probs is None

    # No stage-0 anywhere → None (single-DiT shapes).
    assert build_ar_segment([[SimpleNamespace(stage_id=1)]]) is None


def test_group_by_request_orders_by_rid_prefix():
    """Group index i ↔ request i, regardless of arrival order (risk #3)."""
    outs = [
        SimpleNamespace(request_id="2_zz"),
        SimpleNamespace(request_id="0_aa"),
        SimpleNamespace(request_id="1_bb"),
        SimpleNamespace(request_id="0_cc"),  # second final stage of request 0
        SimpleNamespace(request_id="garbage"),  # unparseable → dropped
    ]
    grouped = _group_by_request(outs, 3)
    assert [len(g) for g in grouped] == [2, 1, 1]
    assert grouped[0][0].request_id == "0_aa" and grouped[0][1].request_id == "0_cc"
    assert grouped[2][0].request_id == "2_zz"


def test_tp_from_stage_configs_reads_both_stage_kinds():
    """LLM stages: flat tensor_parallel_size; diffusion stages: nested under
    parallel_config; absent → 1. Sourced from the runtime's merged configs
    (``omni.stage_configs``) — mapping- and attr-style entries both work."""
    tp = _tp_from_stage_configs(
        [
            {"stage_id": 0, "engine_args": {"tensor_parallel_size": 4}},
            {"stage_id": 1, "engine_args": {"parallel_config": {"tensor_parallel_size": 2}}},
            {"stage_id": 2, "engine_args": {}},
        ]
    )
    assert tp == {0: 4, 1: 2, 2: 1}
    # Attr-style entries (OmegaConf-ish objects without .get).
    tp2 = _tp_from_stage_configs([SimpleNamespace(stage_id=0, engine_args=SimpleNamespace(tensor_parallel_size=8))])
    assert tp2 == {0: 8}


def test_seed_from_sample_id_deterministic_and_distinct():
    a, b = seed_from_sample_id("p0/a0/i3"), seed_from_sample_id("p0/a0/i4")
    assert a == seed_from_sample_id("p0/a0/i3")
    assert a != b
    assert 0 <= a < 2**31


def test_sigmas_list_from_req_slices_terminal_zero():
    req = make_req(1, sigmas=sigmas_for(4))
    assert sigmas_list_from_req(req, 4) == pytest.approx([1.0, 0.75, 0.5, 0.25])
    assert sigmas_list_from_req(make_req(1), 4) is None  # no σ pinned → worker schedule
    with pytest.raises(Exception, match="num_inference_steps"):
        sigmas_list_from_req(req, 3)
