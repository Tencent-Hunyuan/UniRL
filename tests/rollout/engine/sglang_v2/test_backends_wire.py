"""HTTP wire tests — the seam's pure deserialization, CPU-only.

The gating test: ``backends.http`` must import without sglang installed (the
sglang import is lazy, inside ``boot`` / ``set_lora`` only). The rest exercises
``parse_generate_response`` — the dict-vs-list absorption, the logprob-pair
parsing with its fallbacks, and finish_reason normalization.
"""

from __future__ import annotations

import importlib.util

import pytest


def test_http_module_imports_without_sglang():
    """If this fails, the seam leaked a top-level runtime import."""
    assert importlib.util.find_spec("sglang") is None, (
        "this CPU suite assumes sglang is NOT installed — the import-hygiene assertion below would be vacuous otherwise"
    )
    import unirl.rollout.engine.sglang_v2.backends.http  # noqa: F401


from unirl.rollout.engine.sglang_v2.backends.http import parse_generate_response  # noqa: E402


def candidate(text="hi", logprob_pairs=None, finish="stop", **meta_extra):
    meta = {"finish_reason": finish, **meta_extra}
    if logprob_pairs is not None:
        meta["output_token_logprobs"] = logprob_pairs
    return {"text": text, "meta_info": meta}


def test_single_dict_response_is_one_result():
    results = parse_generate_response(candidate("a", [[-0.5, 11], [-0.6, 12]]))
    assert len(results) == 1
    assert results[0].text == "a"
    assert results[0].token_ids == [11, 12]
    assert results[0].logprobs == [-0.5, -0.6]
    assert results[0].finish_reason == "stop"


def test_list_response_n_gt_1_keeps_order():
    results = parse_generate_response([candidate("a"), candidate("b"), candidate("c")])
    assert [r.text for r in results] == ["a", "b", "c"]


def test_bare_float_logprobs_fall_back_to_output_token_ids():
    results = parse_generate_response(candidate("a", [-0.5, -0.6], output_token_ids=[7, 8]))
    assert results[0].logprobs == [-0.5, -0.6]
    assert results[0].token_ids == [7, 8]


def test_finish_reason_dict_and_missing_forms():
    assert parse_generate_response(candidate(finish={"type": "length"}))[0].finish_reason == "length"
    assert parse_generate_response({"text": "x"})[0].finish_reason == "unknown"
    assert parse_generate_response(candidate(finish="abort"))[0].finish_reason == "abort"


def test_length_mismatch_warns_but_still_parses(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        results = parse_generate_response(candidate("a", [-0.5], output_token_ids=[7, 8]))
    assert "MISMATCH" in caplog.text
    assert results[0].token_ids == [7, 8]
    assert results[0].logprobs == [-0.5]


def test_no_logprobs_requested_yields_empty_lists():
    results = parse_generate_response(candidate("a"))
    assert results[0].token_ids == []
    assert results[0].logprobs == []


def test_unexpected_response_type_raises():
    with pytest.raises(RuntimeError, match="Unexpected sglang response type"):
        parse_generate_response("not a dict")
