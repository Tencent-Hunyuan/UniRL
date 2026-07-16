"""CPU tests for the M2 deep-research web tools + LLM-judge (LIN-519).

No network: HTTP is monkeypatched. Validates each tool's ``json_schema`` and
argument handling, the Serper/Jina response parsing, and the judge's
correct/incorrect verdict parsing.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load
pytest.importorskip("requests")

from unirl.reward.local.llm_judge import LLMJudgeRewardScorer, LLMJudgeSpec  # noqa: E402
from unirl.rollout.loop.tools import SearchTool, VisitTool  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.reward import RewardRequest  # noqa: E402


class _Resp:
    def __init__(self, payload=None, text=""):
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):  # noqa: D401
        pass

    def json(self):
        return self._payload


def _chat(content):
    return _Resp({"choices": [{"message": {"content": content}}]})


# --------------------------------------------------------------------------- #
# SearchTool
# --------------------------------------------------------------------------- #


def test_search_schema_and_validation():
    tool = SearchTool()
    assert tool.name == "search"
    fn = tool.json_schema()["function"]
    assert fn["name"] == "search"
    assert fn["parameters"]["required"] == ["query"]
    with pytest.raises(ValueError):
        tool.execute({})
    with pytest.raises(ValueError):
        tool.execute({"query": []})


def test_search_parses_serper(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    def fake_post(url, json=None, headers=None, timeout=None):
        assert json["q"] == "who won"
        return _Resp({"organic": [{"title": "T", "link": "http://x", "snippet": "S"}]})

    monkeypatch.setattr(mod.requests, "post", fake_post)
    out = SearchTool().execute({"query": ["who won"]})
    assert "T" in out and "http://x" in out and "S" in out


def test_search_error_is_text(monkeypatch):
    import unirl.rollout.loop.tools.search as mod

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("net down")))
    out = SearchTool().execute({"query": "x"})
    assert out.startswith("[search] error")  # surfaced to the model, not raised


# --------------------------------------------------------------------------- #
# VisitTool
# --------------------------------------------------------------------------- #


def test_visit_schema_and_raw_fallback(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod

    monkeypatch.delenv("SUMMARY_URL", raising=False)
    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp(text="PAGE BODY"))
    tool = VisitTool()  # no summarizer endpoint -> returns raw content
    assert tool.json_schema()["function"]["name"] == "visit"
    out = tool.execute({"url": "http://x", "goal": "g"})
    assert "PAGE BODY" in out


def test_visit_summarizes(monkeypatch):
    import unirl.rollout.loop.tools.visit as mod

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: _Resp(text="LONG PAGE"))
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _chat("SUMMARY"))
    out = VisitTool(endpoint="http://judge/v1/chat/completions", model="m").execute({"url": "http://x", "goal": "g"})
    assert "SUMMARY" in out


# --------------------------------------------------------------------------- #
# LLMJudgeRewardScorer
# --------------------------------------------------------------------------- #


def _judge():
    return LLMJudgeRewardScorer(
        config=LLMJudgeSpec(endpoint="http://j/v1/chat/completions", model="m"), base_device="cpu"
    )


def _req(prediction, answer):
    return RewardRequest(
        primitives={"text": Texts(texts=["q"])},
        generated={"text": Texts(texts=[prediction])},
        metadata=[{"answer": answer}],
    )


def test_judge_verdict_parsing(monkeypatch):
    import unirl.reward.local.llm_judge as mod

    scorer = _judge()
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _chat("Correct."))
    assert scorer._compute_model_rewards(_req("42", "42")) == [1.0]
    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: _chat("Incorrect"))
    assert scorer._compute_model_rewards(_req("41", "42")) == [0.0]


def test_judge_missing_answer_scores_zero():
    scorer = _judge()
    req = RewardRequest(
        primitives={"text": Texts(texts=["q"])},
        generated={"text": Texts(texts=["x"])},
        metadata=[None],
    )
    assert scorer._compute_model_rewards(req) == [0.0]
