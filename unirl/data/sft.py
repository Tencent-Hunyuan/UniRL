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
- AR (VLM):   ``{"prompt", "response", "media": [{"modality": "image",
  "role": "condition", "uri": "img/0.png"}]}``
- Diffusion:  ``{"prompt": str, "media": [{"modality": "image",
  "role": "target", "uri": "img/0.png"}]}`` (``caption`` accepted as an alias
  for ``prompt``)

Rows are OPAQUE records driver-side — media loading and tokenization happen on
the training workers (``unirl/train/sft/track_builder.py``), so nothing heavy
crosses the driver/Ray boundary.

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
    "media",
    "media_refs",
    "metadata",
    "sample_id",
    "prompt_id",
    *_LEGACY_EMBEDDING_FIELDS,
}


def normalize_supervised_example(
    item: Dict[str, Any],
    *,
    default_sample_id: str,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize one raw manifest row into the supervised record shape.

    Returns ``{"sample_id", "prompt", ["response"], ["media_refs"],
    ["metadata"]}``. ``response`` stays optional here — whether it is required
    is a per-domain decision the worker-side track builder enforces (AR
    requires it; diffusion requires a ``role="target"`` media ref instead).
    """
    if not isinstance(item, dict):
        raise TypeError(f"Supervised example must be a dict, got {type(item).__name__}.")
    legacy = sorted(k for k in _LEGACY_EMBEDDING_FIELDS if k in item)
    if legacy:
        raise ValueError(
            f"Supervised manifests must be raw-data-first and may not include legacy embedding fields: {legacy}."
        )

    prompt = item.get("prompt", item.get("caption", ""))
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Supervised example is missing a non-empty 'prompt' or 'caption' field.")

    record: Dict[str, Any] = {
        "sample_id": str(item.get("sample_id", item.get("prompt_id", default_sample_id))),
        "prompt": prompt,
    }

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

    # ---- train stream --------------------------------------------------

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

    # ---- resume cursor --------------------------------------------------

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

    # ---- eval ------------------------------------------------------------

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
]
