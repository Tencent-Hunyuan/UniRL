"""JSONL manifest data source for supervised training.

Model-agnostic: rows are opaque record dicts handed to the task adapter's
``load_record`` on the worker (tensors load worker-side from paths, so nothing
heavy crosses the driver/Ray boundary). ``_root`` (the manifest's directory)
is injected into every record so relative paths stay portable.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional


class JsonlSFTDataSource:
    """Epoch-cycling shuffled reader over a JSONL manifest (+ optional eval manifest)."""

    def __init__(
        self,
        manifest_path: str,
        *,
        eval_manifest_path: Optional[str] = None,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        self.records = self._load(manifest_path)
        if not self.records:
            raise ValueError(f"empty SFT manifest: {manifest_path}")
        self.eval_records_all = self._load(eval_manifest_path) if eval_manifest_path else []
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0
        self._pos = 0
        self._order = self._make_order()

    @staticmethod
    def _load(path: Optional[str]) -> List[Dict[str, Any]]:
        if path is None:
            return []
        root = os.path.dirname(os.path.abspath(path))
        records = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_root"] = root
                records.append(record)
        return records

    def _make_order(self) -> List[int]:
        order = list(range(len(self.records)))
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
            batch.append(self.records[self._order[self._pos]])
            self._pos += 1
        return batch

    def eval_samples(self, k: int) -> List[Dict[str, Any]]:
        pool = self.eval_records_all or self.records
        return pool[: max(1, int(k))]


__all__ = ["JsonlSFTDataSource"]
