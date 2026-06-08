"""SGLang scheduler endpoint parsing and formatting utilities."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Tuple


def normalize_scheduler_host(value: Any) -> str:
    host = str(value or "").strip()
    if not host:
        return "127.0.0.1"
    for prefix in ("tcp://", "http://", "https://"):
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    host = host.split("/", 1)[0].strip()
    if host == "localhost":
        return "127.0.0.1"
    return host


def format_scheduler_endpoint(host: str, port: int) -> str:
    host_text = str(host).strip()
    if ":" in host_text and not host_text.startswith("["):
        host_text = f"[{host_text}]"
    return f"tcp://{host_text}:{int(port)}"


def parse_scheduler_endpoint(value: Any) -> Optional[Tuple[str, int]]:
    if value is None:
        return None

    if isinstance(value, dict):
        endpoint_value = value.get("scheduler_endpoint") or value.get("endpoint") or value.get("scheduler")
        if endpoint_value is not None:
            return parse_scheduler_endpoint(endpoint_value)

        host = value.get("host", value.get("scheduler_host"))
        port = value.get("scheduler_port", value.get("port"))
        if host is None or port is None:
            return None
        return normalize_scheduler_host(host), int(port)

    text = str(value).strip()
    if not text:
        return None
    for prefix in ("tcp://", "http://", "https://"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.split("/", 1)[0].strip()

    if text.startswith("["):
        end = text.find("]")
        if end <= 0 or end + 1 >= len(text) or text[end + 1] != ":":
            raise ValueError(f"Invalid scheduler endpoint {value!r}; expected tcp://[host]:port.")
        host = text[1:end]
        port_text = text[end + 2 :]
    else:
        if ":" not in text:
            raise ValueError(f"Invalid scheduler endpoint {value!r}; expected host:port.")
        host, port_text = text.rsplit(":", 1)

    return normalize_scheduler_host(host), int(port_text)


def parse_scheduler_endpoint_pool(value: Any) -> List[Tuple[str, int]]:
    if value is None:
        return []

    raw_items: Any = value
    if isinstance(raw_items, str):
        text = raw_items.strip()
        if not text:
            return []
        if text.startswith("["):
            raw_items = json.loads(text)
        else:
            raw_items = [part.strip() for part in text.split(",") if part.strip()]

    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, (list, tuple)):
        raise TypeError(f"Scheduler endpoint pool must be list/tuple/string/dict, got: {type(raw_items).__name__}")

    parsed: List[Tuple[str, int]] = []
    for item in raw_items:
        endpoint = parse_scheduler_endpoint(item)
        if endpoint is None:
            continue
        parsed.append(endpoint)
    return parsed


__all__ = [
    "normalize_scheduler_host",
    "format_scheduler_endpoint",
    "parse_scheduler_endpoint",
    "parse_scheduler_endpoint_pool",
]
