"""Supervised (SFT) datasets — target-carrying manifests with epoch semantics.

The RL data layer (``datasets.py`` / ``data_source.py``) is prompt-first by
design: rows carry prompts, rollout generates the targets. SFT rows carry the
target itself, so they get their own dataset shaping here — sharing the media
normalization machinery (:func:`unirl.data.datasets._normalize_media_refs`)
but NOT the RL classes (per-algorithm dataset shaping; a shared stage-keyed
dispatcher is the misrouting failure mode other frameworks hit).

Manifest row shapes (JSONL, one object per line; relative media URIs resolve
against the manifest's directory):

- AR (LLM):   ``{"prompt": str, "response": str}``
- AR (agent): ``{"messages": [..., {"role": "assistant", ...}], "tools": [...]}``
- AR (VLM):   ``{"prompt", "response", "media": [{"modality": "image",
  "role": "condition", "uri": "img/0.png"}]}``
- AR (omni):  ``{"prompt", "response", "media": [{"modality": "audio",
  "role": "prompt", "uri": "audio/0.wav"}]}`` — URI-backed prompt media
  (image / audio / video) for chat stages that render it themselves; the
  builder rejects ``role="condition"`` here rather than silently dropping it
- Diffusion:  ``{"prompt": str, "media": [{"modality": "image",
  "role": "target", "uri": "img/0.png"}]}`` (``caption`` accepted as an alias
  for ``prompt``)
- Video diffusion: ``{"prompt": str, "media": [{"modality": "video",
  "role": "target", "uri": "video/0.mp4"}]}``

Rows are OPAQUE records driver-side — media loading and tokenization happen on
the training workers (``unirl/train/sft/track_builder.py`` track builders), so nothing heavy crosses
the driver/Ray boundary.

Epoch semantics: :class:`SupervisedDataSource` walks a per-epoch reshuffled
order and exposes ``state_dict()`` / ``load_state_dict()`` with the exact
``{epoch, position}`` cursor, checkpointed by the SFT trainer — mid-epoch
resume replays neither skips nor duplicates (RL's infinite reshuffled stream
has no such notion, which is why this is a separate class).
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, Iterator, List, Optional

from unirl.data.datasets import _LEGACY_EMBEDDING_FIELDS, _normalize_media_refs

logger = logging.getLogger(__name__)

_SUPERVISED_EXCLUDED_KEYS = {
    "prompt",
    "caption",
    "response",
    "messages",
    "tools",
    "media",
    "media_refs",
    "metadata",
    "sample_id",
    "prompt_id",
    *_LEGACY_EMBEDDING_FIELDS,
}


def tokenize_agent_target(
    record: Dict[str, Any],
    *,
    tokenizer: Any,
    enable_thinking: bool,
) -> List[int]:
    """Render an agent record and return only its final assistant-turn tokens."""
    messages = record["messages"]
    history = messages[:-1]
    tools = record.get("tools")

    def apply_template(turns: List[Dict[str, Any]], *, add_generation_prompt: bool) -> List[int]:
        rendered = tokenizer.apply_chat_template(
            turns,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tokenize=True,
            return_dict=False,
            truncation=False,
        )
        if isinstance(rendered, dict):
            rendered = rendered["input_ids"]
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return [int(token_id) for token_id in rendered]

    prompt_ids = apply_template(history, add_generation_prompt=True)
    full_ids = apply_template(messages, add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Agent target is not a suffix of its rendered history. "
            "Ensure enable_thinking matches the dataset's assistant reasoning format."
        )

    target_ids = full_ids[len(prompt_ids) :]
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    if eos_id is not None and eos_id in target_ids:
        last_eos = len(target_ids) - 1 - target_ids[::-1].index(eos_id)
        target_ids = target_ids[: last_eos + 1]
    return target_ids


def normalize_supervised_example(
    item: Dict[str, Any],
    *,
    default_sample_id: str,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize one raw manifest row into the supervised record shape.

    Returns either a legacy ``{"sample_id", "prompt", ...}`` record or an
    agent ``{"sample_id", "messages", ["tools"], ...}`` record. Agent rows
    carry the target assistant turn as the final message; the worker-side
    builder renders the preceding history as the prompt and supervises only
    that final turn.
    """
    if not isinstance(item, dict):
        raise TypeError(f"Supervised example must be a dict, got {type(item).__name__}.")
    legacy = sorted(k for k in _LEGACY_EMBEDDING_FIELDS if k in item)
    if legacy:
        raise ValueError(
            f"Supervised manifests must be raw-data-first and may not include legacy embedding fields: {legacy}."
        )

    record: Dict[str, Any] = {
        "sample_id": str(item.get("sample_id", item.get("prompt_id", default_sample_id))),
    }
    messages = item.get("messages")
    if messages is not None:
        if "prompt" in item or "caption" in item or "response" in item:
            raise ValueError("Agent supervised examples use 'messages' and may not also set prompt/response fields.")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Agent supervised example 'messages' must be a list with at least two turns.")
        normalized_messages: List[Dict[str, Any]] = []
        for turn, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f"Agent message {turn} must be a dict, got {type(message).__name__}.")
            role = message.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"Agent message {turn} has unsupported role {role!r}.")
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            if content is not None and not isinstance(content, str):
                raise TypeError(
                    f"Agent message {turn} 'content' must be a string or null, got {type(content).__name__}."
                )
            if tool_calls is not None and (role != "assistant" or not isinstance(tool_calls, list)):
                raise TypeError(f"Agent message {turn} 'tool_calls' must be a list on an assistant turn.")
            if role == "assistant" and not content and not tool_calls:
                raise ValueError(f"Agent assistant message {turn} has neither content nor tool_calls.")
            normalized_messages.append(dict(message))
        if normalized_messages[-1]["role"] != "assistant":
            raise ValueError("Agent supervised example must end with the target assistant turn.")
        if not any(message["role"] == "user" for message in normalized_messages[:-1]):
            raise ValueError("Agent supervised example has no user turn before the target assistant turn.")
        record["messages"] = normalized_messages

        tools = item.get("tools")
        if tools is not None:
            if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
                raise TypeError("Agent supervised example 'tools' must be a list of tool-schema objects.")
            record["tools"] = list(tools)
    else:
        prompt = item.get("prompt", item.get("caption", ""))
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(
                "Supervised example needs either a non-empty 'prompt'/'caption' field or agent 'messages'."
            )
        record["prompt"] = prompt

        response = item.get("response")
        if response is not None:
            if not isinstance(response, str) or not response:
                raise ValueError(
                    f"Supervised example 'response' must be a non-empty string, got {type(response).__name__}."
                )
            record["response"] = response

    media_refs = _normalize_media_refs(item.get("media_refs", item.get("media")), base_dir=base_dir)
    if media_refs:
        record["media_refs"] = media_refs

    metadata = item.get("metadata")
    if metadata is None:
        metadata = {k: v for k, v in item.items() if k not in _SUPERVISED_EXCLUDED_KEYS}
    elif not isinstance(metadata, dict):
        raise TypeError(f"Supervised example metadata must be a dict, got {type(metadata).__name__}.")
    if metadata:
        record["metadata"] = dict(metadata)
    return record


class SupervisedDataset:
    """File-backed supervised dataset: parsing + per-row normalization only.

    Supports ``.jsonl`` (one object per line) and ``.json`` (list of objects).
    Epoch ordering / batching belong to :class:`SupervisedDataSource`.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self._base_dir = os.path.dirname(os.path.abspath(file_path))
        prefix = os.path.basename(file_path) or "sft_source"
        self.records: List[Dict[str, Any]] = []
        for idx, item in enumerate(self._iter_raw(file_path)):
            record = normalize_supervised_example(item, default_sample_id=f"{prefix}:{idx}", base_dir=self._base_dir)
            self.records.append(record)
        if not self.records:
            raise ValueError(f"No supervised examples found in {file_path}")
        logger.info("Loaded %d supervised examples from %s", len(self.records), file_path)

    @staticmethod
    def _iter_raw(path: str) -> Iterator[Dict[str, Any]]:
        if path.endswith(".jsonl"):
            with open(path) as fh:
                for line_num, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_num}: invalid JSON — {exc}") from exc
        elif path.endswith(".json"):
            with open(path) as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                raise ValueError(f"{path}: .json supervised manifests must be a list of objects.")
            yield from data
        else:
            raise ValueError(f"Unsupported supervised manifest format: {path} (use .jsonl or .json)")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class SupervisedDataSource:
    """Epoch-aware batch iterator over a supervised manifest (+ eval split).

    ``get_samples`` walks a per-epoch shuffled order (``seed + epoch``-seeded,
    so resume is exact) and wraps across epoch boundaries within one batch.
    The ``{epoch, position}`` cursor rides ``state_dict()`` into the trainer's
    checkpoint sidecar.
    """

    def __init__(
        self,
        manifest_path: str,
        *,
        eval_manifest_path: Optional[str] = None,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        self.dataset = SupervisedDataset(manifest_path)
        self.eval_dataset = SupervisedDataset(eval_manifest_path) if eval_manifest_path else None
        if self.eval_dataset is None:
            logger.warning(
                "SupervisedDataSource: no eval_manifest_path — eval batches fall back to the "
                "TRAIN set (eval loss then measures training data)."
            )
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = 0
        self._pos = 0
        self._order = self._make_order()

    def _make_order(self) -> List[int]:
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self._epoch).shuffle(order)
        return order

    def get_samples(self, batch_size: int) -> List[Dict[str, Any]]:
        batch: List[Dict[str, Any]] = []
        while len(batch) < batch_size:
            if self._pos >= len(self._order):
                self._epoch += 1
                self._pos = 0
                self._order = self._make_order()
            batch.append(self.dataset[self._order[self._pos]])
            self._pos += 1
        return batch

    @property
    def epoch(self) -> float:
        """Fractional epochs consumed — for logging."""
        return self._epoch + self._pos / max(1, len(self.dataset))

    def state_dict(self) -> Dict[str, int]:
        return {"epoch": self._epoch, "position": self._pos, "seed": self.seed}

    def load_state_dict(self, state: Dict[str, int]) -> None:
        if state.get("seed", self.seed) != self.seed:
            logger.warning(
                "SupervisedDataSource.load_state_dict: checkpoint seed %s != configured seed %s — "
                "the resumed shuffle order will differ from the original run.",
                state.get("seed"),
                self.seed,
            )
        self._epoch = state["epoch"]
        self._pos = state["position"]
        self._order = self._make_order()
        if self._pos > len(self._order):
            raise ValueError(
                f"SupervisedDataSource.load_state_dict: cursor position {self._pos} exceeds "
                f"dataset size {len(self._order)} — dataset changed since the checkpoint?"
            )

    def iter_eval_batches(self, batch_size: int, *, eval_num_samples: int = -1) -> Iterator[List[Dict[str, Any]]]:
        """Deterministic-order eval batches (manifest order, no shuffle).

        ``eval_num_samples``: ``-1`` = full eval set, ``0`` = nothing,
        ``N > 0`` = first N rows. The final partial batch is yielded as-is —
        the trainer pads it to the DP width with ``_eval_pad`` rows the loss
        masks out, so the full set is covered exactly.
        """
        pool = self.eval_dataset if self.eval_dataset is not None else self.dataset
        n = len(pool)
        limit = n if eval_num_samples < 0 else min(eval_num_samples, n)
        for start in range(0, limit, batch_size):
            yield [pool[i] for i in range(start, min(start + batch_size, limit))]


__all__ = [
    "SupervisedDataSource",
    "SupervisedDataset",
    "normalize_supervised_example",
    "tokenize_agent_target",
]
