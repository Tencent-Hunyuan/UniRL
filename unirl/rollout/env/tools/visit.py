"""VisitTool — read webpage(s) and summarize toward a goal."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from unirl.rollout.env.tools.base import Tool

_JINA_READ = "https://r.jina.ai/"
_EXTRACT_PROMPT = (
    "Process the following webpage content and extract the information relevant "
    "to the goal.\n\n"
    "## Webpage content\n{content}\n\n"
    "## Goal\n{goal}\n\n"
    "## Task\n"
    "1. evidence: extract the most relevant facts, figures, dates, and quotes "
    "from the content — keep the full original context where possible.\n"
    "2. summary: organize it into a concise paragraph, judging its contribution "
    "to the goal.\n\n"
    'Output ONLY a JSON object with string keys "evidence" and "summary".'
)


def _extract_json(raw: str) -> Optional[dict]:
    """Best-effort JSON parse: strip code fences, else grab the outer ``{...}``."""
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        left, right = raw.find("{"), raw.rfind("}")
        if left != -1 and right != -1 and left <= right:
            try:
                return json.loads(raw[left : right + 1])
            except Exception:
                return None
    return None


class VisitTool(Tool):
    """Visit URL(s) via the Jina reader and summarize the content toward a goal."""

    name = "visit"

    def __init__(
        self,
        *,
        endpoint: str = "",
        model: str = "",
        timeout: float = 60.0,
        max_content_chars: int = 90000,
        max_read_retries: int = 3,
        max_summary_retries: int = 2,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._timeout = float(timeout)
        self._max_content_chars = int(max_content_chars)
        self._max_read_retries = max(1, int(max_read_retries))
        self._max_summary_retries = max(1, int(max_summary_retries))

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Visit webpage(s) and return a summary of the content relevant to a goal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": ["string", "array"],
                            "items": {"type": "string"},
                            "description": "The URL, or an array of URLs, to visit.",
                        },
                        "goal": {
                            "type": "string",
                            "description": "The specific information to extract from the page(s).",
                        },
                    },
                    "required": ["url", "goal"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        url = arguments.get("url")
        goal = str(arguments.get("goal", ""))
        urls: List[str] = [url] if isinstance(url, str) else list(url or [])
        if not urls:
            raise ValueError("visit requires a 'url' string or array")
        return "\n=======\n".join(self._visit_one(str(u), goal) for u in urls)

    def _visit_one(self, url: str, goal: str) -> str:
        content = self._read(url)
        if content.startswith("[visit] "):
            return f"The useful information in {url} for goal {goal}:\n{content}"
        summary = self._summarize(content[: self._max_content_chars], goal)
        return f"The useful information in {url} for goal {goal}:\n{summary}"

    def _read(self, url: str) -> str:
        """Fetch page text via Jina, retrying on transient failures."""
        key = os.environ.get("JINA_API_KEYS", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        last = f"[visit] failed to read {url}"
        for _ in range(self._max_read_retries):
            try:
                resp = requests.get(_JINA_READ + url, headers=headers, timeout=self._timeout)
                resp.raise_for_status()
                text = resp.text
                if text and text.strip():
                    return text
                last = f"[visit] empty content for {url}"
            except Exception as exc:  # noqa: BLE001 — surfaced to the model as text, not raised
                last = f"[visit] failed to read {url}: {exc}"
            time.sleep(0.5)
        return last

    def _summarize(self, content: str, goal: str) -> str:
        """Summarize toward the goal via the out-of-band LLM into a structured evidence/summary block."""
        endpoint = os.environ.get("SUMMARY_URL", self._endpoint)
        model = os.environ.get("SUMMARY_MODEL", self._model)
        if not endpoint:
            return content
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("SUMMARY_API_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": _EXTRACT_PROMPT.format(goal=goal, content=content)}],
            "temperature": 0.2,
        }
        for _ in range(self._max_summary_retries):
            try:
                resp = requests.post(endpoint, json=payload, headers=headers, timeout=self._timeout)
                resp.raise_for_status()
                raw = str(resp.json()["choices"][0]["message"]["content"]).strip()
                obj = _extract_json(raw)
                if obj is not None:
                    evidence = str(obj.get("evidence", "")).strip()
                    summary = str(obj.get("summary", "")).strip()
                    if evidence or summary:
                        return f"Evidence:\n{evidence}\n\nSummary:\n{summary}"
                if raw:
                    return raw
            except Exception:  # noqa: BLE001 — retry, then fall back to raw content
                pass
            time.sleep(0.5)
        return content  # summarizer failed after retries — raw truncated content
