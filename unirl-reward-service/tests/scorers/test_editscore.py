from __future__ import annotations

import sys
import types
from collections.abc import Callable

import pytest
from PIL import Image
from reward_service.scorers.base import ScoreItem
from reward_service.scorers.editscore import EditScoreScorer


def _bare_scorer(model=None) -> EditScoreScorer:
    scorer = object.__new__(EditScoreScorer)
    scorer.sub_metric_names = (
        "prompt_following",
        "consistency",
        "perceptual_quality",
        "overall",
    )
    scorer._max_image_side = None
    scorer._batched = True
    scorer._num_pass = 1
    scorer._score_range = 25
    scorer._seed = 42
    scorer.es = types.SimpleNamespace(
        model=model,
        SC_prompt="SC <instruction>",
        PQ_prompt="PQ",
    )
    return scorer


def _install_parser(monkeypatch: pytest.MonkeyPatch, parser: Callable) -> None:
    utils = types.ModuleType("editscore.utils")
    utils.mllm_output_to_dict = parser
    monkeypatch.setitem(sys.modules, "editscore.utils", utils)


def test_sleep_mode_forces_vllm_caches_off(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def original_llm(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace()

    editscore_package = types.ModuleType("editscore")
    editscore_package.__path__ = []
    mllm_tools = types.ModuleType("editscore.mllm_tools")
    mllm_tools.__path__ = []
    backbone = types.ModuleType("editscore.mllm_tools.qwen3vl_vllm")
    backbone.LLM = original_llm

    class FakeEditScore:
        def __init__(self, **_kwargs):
            self.model = types.SimpleNamespace(
                model=backbone.LLM(
                    enable_prefix_caching=True,
                    mm_processor_cache_gb=4,
                )
            )

    editscore_package.EditScore = FakeEditScore
    monkeypatch.setitem(sys.modules, "editscore", editscore_package)
    monkeypatch.setitem(sys.modules, "editscore.mllm_tools", mllm_tools)
    monkeypatch.setitem(
        sys.modules,
        "editscore.mllm_tools.qwen3vl_vllm",
        backbone,
    )

    EditScoreScorer(
        model_name_or_path="fake",
        enable_sleep_mode=True,
        extra_llm_kwargs={
            "enable_prefix_caching": True,
            "mm_processor_cache_gb": 2,
        },
    )

    assert captured["enable_sleep_mode"] is True
    assert captured["enable_prefix_caching"] is False
    assert captured["mm_processor_cache_gb"] == 0
    assert backbone.LLM is original_llm


def test_batched_scoring_retries_only_failed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    sc_attempt = 0

    class FakeModel:
        @staticmethod
        def prepare_input(images, _prompt):
            return "sc" if isinstance(images, list) else "pq"

        def batch_inference(self, messages, seed):
            nonlocal sc_attempt
            assert seed == 42
            kind = messages[0]
            calls.append((kind, len(messages)))
            if kind == "sc":
                sc_attempt += 1
                if sc_attempt == 1:
                    return ["bad", "sc-good"]
                return ["sc-good"] * len(messages)
            return ["pq-good"] * len(messages)

    def parser(text, *, give_up_parsing, text_prompt, score_range):
        assert text_prompt in {"first", "second"}
        assert score_range == 25
        if text == "bad":
            assert give_up_parsing is False
            return False
        if text == "sc-good":
            return {"score": [20, 25]}
        return {"score": [16]}

    _install_parser(monkeypatch, parser)
    scorer = _bare_scorer(FakeModel())
    image = Image.new("RGB", (8, 8))
    rows = [
        (0, "first", image, image),
        (1, "second", image, image),
    ]

    results = scorer._score_rows_batched(rows)

    assert calls == [("sc", 2), ("pq", 2), ("sc", 1), ("pq", 1)]
    assert results[0] is not None
    assert results[1] is not None
    assert results[0]["prompt_following"] == pytest.approx(8.0)
    assert results[0]["consistency"] == pytest.approx(10.0)
    assert results[0]["perceptual_quality"] == pytest.approx(6.4)


def test_persistent_batched_parse_failure_falls_back_to_evaluate() -> None:
    scorer = _bare_scorer()
    scorer._score_rows_batched = lambda _rows: [None]
    scorer.es.evaluate = lambda _images, _prompt: {
        "prompt_following": 1.0,
        "consistency": 2.0,
        "perceptual_quality": 3.0,
        "overall": 4.0,
    }
    image = Image.new("RGB", (8, 8))
    item = ScoreItem(history=[("edit it", image), ("edit it", image)])

    assert scorer.score([item]) == [
        {
            "prompt_following": 1.0,
            "consistency": 2.0,
            "perceptual_quality": 3.0,
            "overall": 4.0,
        }
    ]


def test_batched_oom_is_not_retried_per_item() -> None:
    class FakeOOM(RuntimeError):
        pass

    scorer = _bare_scorer()

    def raise_oom(_rows):
        raise FakeOOM("out of memory")

    scorer._score_rows_batched = raise_oom
    scorer._is_oom_error = lambda exc: isinstance(exc, FakeOOM)
    scorer.es.evaluate = lambda _images, _prompt: pytest.fail("OOM must not fall back to per-item scoring")
    image = Image.new("RGB", (8, 8))
    item = ScoreItem(history=[("edit it", image), ("edit it", image)])

    with pytest.raises(FakeOOM):
        scorer.score([item])
