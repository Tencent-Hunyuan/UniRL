"""HTTP wire tests — the seam's pure helpers, CPU-only.

The gating test: ``backends.http`` must import without sglang installed (the
sglang import is lazy, inside ``_import_sglang_runtime`` only). The rest
exercises ``parse_generate_response`` — the dict-vs-list absorption, the
logprob-item parsing (pairs and upstream's 3-tuples), finish_reason
normalization — and ``asdict_drop_none`` (the io_struct → wire view).
"""

from __future__ import annotations

import importlib.util

import pytest


def test_http_module_imports_without_sglang():
    """If this fails, the seam leaked a top-level runtime import."""
    assert importlib.util.find_spec("sglang") is None, (
        "this CPU suite assumes sglang is NOT installed — the import-hygiene assertion below would be vacuous otherwise"
    )
    import unirl.rollout.engine.sglang.backends.http  # noqa: F401


from unirl.rollout.engine.sglang.backends.http import (  # noqa: E402
    asdict_drop_none,
    parse_generate_response,
)


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


def test_upstream_triple_items_parse():
    """Upstream builds (logprob, token_id, token_text|None) 3-tuples — same parse."""
    results = parse_generate_response(candidate("a", [[-0.5, 11, None], [-0.6, 12, "tok"]]))
    assert results[0].token_ids == [11, 12]
    assert results[0].logprobs == [-0.5, -0.6]


def test_finish_reason_dict_and_missing_forms():
    assert parse_generate_response(candidate(finish={"type": "length"}))[0].finish_reason == "length"
    assert parse_generate_response({"text": "x"})[0].finish_reason == "unknown"
    assert parse_generate_response(candidate(finish="abort"))[0].finish_reason == "abort"


def test_no_logprobs_requested_yields_empty_lists():
    results = parse_generate_response(candidate("a"))
    assert results[0].token_ids == []
    assert results[0].logprobs == []


def test_unexpected_response_type_raises():
    with pytest.raises(RuntimeError, match="Unexpected sglang response type"):
        parse_generate_response("not a dict")


def test_asdict_drop_none_keeps_falsy_non_none():
    """None fields drop off the wire; False/0/empty containers must survive."""
    from dataclasses import dataclass, field
    from typing import List, Optional

    @dataclass
    class StubReq:
        tags: Optional[List[str]] = None
        flush_cache: bool = False
        rank_offset: int = 0
        names: List[str] = field(default_factory=list)
        load_format: Optional[str] = None

    assert asdict_drop_none(StubReq()) == {
        "flush_cache": False,
        "rank_offset": 0,
        "names": [],
    }
    assert asdict_drop_none(StubReq(tags=["weights"], load_format="auto")) == {
        "tags": ["weights"],
        "flush_cache": False,
        "rank_offset": 0,
        "names": [],
        "load_format": "auto",
    }
