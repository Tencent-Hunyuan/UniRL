"""SearchTool — batched web search via Serper, SerpApi, or You.com."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests

from unirl.rollout.env.tools.base import Tool

_SERPER_URL = "https://google.serper.dev/search"
_SERPAPI_URL = "https://serpapi.com/search"
_YOU_URL = "https://ydc-index.io/v1/search"
_YOU_KEYLESS_URL = "https://api.you.com/v1/agents/search"
_YOU_KEYLESS_MAX_COUNT = 50
_YOU_USER_AGENT = "youdotcom-integration/Tencent-Hunyuan-UniRL"


class SearchTool(Tool):
    """Web search via Serper, SerpApi (Google), or You.com."""

    name = "search"

    def __init__(
        self,
        *,
        top_k: int = 10,
        timeout: float = 30.0,
        provider: str = "serper",
        max_retries: int = 3,
    ) -> None:
        self._top_k = int(top_k)
        self._timeout = float(timeout)
        self._provider = os.environ.get("SEARCH_PROVIDER", provider).lower()
        self._max_retries = max(1, int(max_retries))

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the web. Provide an array of query strings; returns the top "
                    "results for each query. Use multiple complementary queries in one call."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "One or more search query strings.",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        queries = arguments.get("query")
        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list) or not queries:
            raise ValueError("search requires a non-empty 'query' string or array")
        return "\n=======\n".join(self._search_one(str(q)) for q in queries)

    def _fetch_organic(self, query: str) -> list:
        """One provider call -> the list of organic result dicts. Raises on HTTP error."""
        if self._provider == "you":
            # keyed API when YDC_API_KEY is set, else the keyless endpoint; mapped to Serper's organic shape
            headers = {"User-Agent": _YOU_USER_AGENT}
            ydc_key = os.environ.get("YDC_API_KEY", "")
            if ydc_key:
                resp = requests.post(
                    _YOU_URL,
                    json={"query": query, "count": self._top_k},
                    headers={**headers, "X-API-Key": ydc_key},
                    timeout=self._timeout,
                )
            else:
                resp = requests.get(
                    _YOU_KEYLESS_URL,
                    params={"query": query, "count": min(self._top_k, _YOU_KEYLESS_MAX_COUNT)},
                    headers=headers,
                    timeout=self._timeout,
                )
            resp.raise_for_status()
            return [
                {
                    "title": r.get("title", ""),
                    "link": r.get("url", ""),
                    "snippet": " ".join(r.get("snippets") or []) or r.get("description", ""),
                }
                for r in resp.json()["results"].get("web") or []
            ]
        key = os.environ.get("SERPER_KEY_ID", "")
        if self._provider == "serpapi":
            resp = requests.get(
                _SERPAPI_URL,
                params={"q": query, "engine": "google", "api_key": key, "num": self._top_k},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json().get("organic_results") or []
        resp = requests.post(
            _SERPER_URL,
            json={"q": query},
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("organic") or []

    def _search_one(self, query: str) -> str:
        last = f"[search] error for {query!r}"
        for _ in range(self._max_retries):
            try:
                organic = self._fetch_organic(query)
                break
            except Exception as exc:  # noqa: BLE001 — surfaced to the model as text, not raised
                last = f"[search] error for {query!r}: {exc}"
                time.sleep(0.5)
        else:
            return last
        if not organic:
            return f"No results for {query!r}. Try a less specific query."
        lines: List[str] = []
        for i, page in enumerate(organic[: self._top_k], start=1):
            title = page.get("title", "")
            link = page.get("link", "")
            snippet = page.get("snippet", "")
            lines.append(f"{i}. [{title}]({link})\n{snippet}")
        return f"Results for {query!r}:\n" + "\n\n".join(lines)
