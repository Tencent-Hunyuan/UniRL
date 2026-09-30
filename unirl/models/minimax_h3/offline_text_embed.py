"""Safetensors sidecar store for MiniMax-H3 layer-50 prompt embeddings."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from unirl.config.require import require
from unirl.utils.dtypes import parse_torch_dtype

from .vendor import MINIMAX_H3_TEXT_ENCODER_LAYER

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.json"
SCHEMA_VERSION = "1.0"
EXTRACTOR = "minimax_h3_text_embed"
FEATURE_DIM = 5120
_SHARD_PATTERN = "embeddings_{:05d}.safetensors"
_SHARD_GLOB = "embeddings_*.safetensors"
_STORE_FINGERPRINT: Dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "extractor": EXTRACTOR,
    "target_layer": MINIMAX_H3_TEXT_ENCODER_LAYER,
    "feature_dim": FEATURE_DIM,
}


def compute_prompt_key(prompt: str) -> str:
    """Return a 16-hex SHA-256 fingerprint of the exact prompt string."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _require_fingerprint(index_data: Dict[str, Any], expected: Dict[str, Any], message: str) -> None:
    mismatched = {key: (index_data.get(key), value) for key, value in expected.items() if index_data.get(key) != value}
    require(not mismatched, f"{message}: {{field: (stored, expected)}} = {mismatched}")


class OfflineTextEmbedStore:
    """Read-only mmap store: ``index.json`` plus sharded ``.safetensors`` files."""

    def __init__(self, cache_dir: str, entries: Dict[str, str]) -> None:
        self.cache_dir = os.path.abspath(cache_dir)
        self.entries = entries
        # mmap handles, not materialized shards: a shard holds ~2000 [L, 5120]
        # tensors, and every DP rank on a node would otherwise keep its own copy.
        self._shard_handles: Dict[str, safe_open] = {}

    @classmethod
    def from_dir(cls, cache_dir: str) -> "OfflineTextEmbedStore":
        """Load ``index.json`` from ``cache_dir``."""
        index_path = os.path.join(cache_dir, INDEX_FILENAME)
        require(
            os.path.isfile(index_path),
            f"OfflineTextEmbedStore: index file not found: {index_path}. "
            "Run `python -m unirl.tools.precompute_minimax_h3` first.",
        )
        with open(index_path, "r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        _require_fingerprint(
            index_data, _STORE_FINGERPRINT, f"OfflineTextEmbedStore: fingerprint mismatch at {index_path}"
        )
        entries = index_data["entries"]
        logger.info("Loaded MiniMax-H3 text-embed store from %s (%d prompts)", cache_dir, len(entries))
        return cls(cache_dir, entries)

    def _open_shard(self, shard_name: str) -> safe_open:
        if shard_name not in self._shard_handles:
            shard_path = os.path.join(self.cache_dir, shard_name)
            require(os.path.isfile(shard_path), f"OfflineTextEmbedStore: missing shard: {shard_path}")
            self._shard_handles[shard_name] = safe_open(shard_path, framework="pt", device="cpu")
        return self._shard_handles[shard_name]

    def get(self, prompt: str) -> torch.Tensor:
        """Fetch one prompt's unpadded ``[L, D]`` embedding."""
        key = compute_prompt_key(prompt)
        require(
            key in self.entries,
            f"OfflineTextEmbedStore: prompt not in cache: {prompt!r}. "
            "Run `python -m unirl.tools.precompute_minimax_h3` over every prompt file the run reads, eval included.",
        )
        tensor = self._open_shard(self.entries[key]).get_tensor(key)
        require(tensor.dim() == 2, f"OfflineTextEmbedStore: expected [L, D], got {tuple(tensor.shape)}")
        return tensor

    def verify_coverage_or_raise(self, prompts: Sequence[str], *, context: str) -> None:
        """Raise if any prompt is missing from the store."""
        missing = [prompt for prompt in prompts if compute_prompt_key(prompt) not in self.entries]
        require(
            not missing,
            f"Precomputed MiniMax-H3 text-embed coverage failed in {context}: "
            f"{len(missing)}/{len(prompts)} prompts missing from {self.cache_dir}. First missing: {missing[:3]!r}.",
        )


class OfflineTextEmbedWriter:
    """Write unpadded ``[L, D]`` embeddings into safetensors shards plus ``index.json``."""

    def __init__(
        self,
        output_dir: str,
        *,
        model_checkpoint: str,
        dtype: str,
        shard_size: int,
        resume: bool,
        force_overwrite: bool,
    ) -> None:
        self.output_dir = os.path.abspath(output_dir)
        self.model_checkpoint = model_checkpoint
        self.dtype = dtype
        self._torch_dtype = parse_torch_dtype(dtype, field_name="OfflineTextEmbedWriter.dtype")
        self.shard_size = shard_size
        self.entries: Dict[str, str] = {}
        self.shards: List[str] = []
        self._current_shard_tensors: Dict[str, torch.Tensor] = {}
        self._current_shard_idx = 0
        self._init_output_dir(resume=resume, force_overwrite=force_overwrite)

    @property
    def _current_shard_name(self) -> str:
        return _SHARD_PATTERN.format(self._current_shard_idx)

    def _init_output_dir(self, *, resume: bool, force_overwrite: bool) -> None:
        index_path = os.path.join(self.output_dir, INDEX_FILENAME)
        if os.path.exists(index_path):
            if force_overwrite:
                # Glob rather than read the old index: a crashed run can leave a
                # shard on disk that its index never listed.
                for shard_path in glob.glob(os.path.join(self.output_dir, _SHARD_GLOB)):
                    os.remove(shard_path)
                os.remove(index_path)
                logger.info("force_overwrite: cleared existing cache at %s", self.output_dir)
            elif resume:
                with open(index_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                _require_fingerprint(
                    data,
                    {**_STORE_FINGERPRINT, "model_checkpoint": self.model_checkpoint, "dtype": self.dtype},
                    f"OfflineTextEmbedWriter: cannot resume {index_path}; use --force-overwrite to rebuild",
                )
                self.entries = data["entries"]
                self.shards = data["shards"]
                self._current_shard_idx = len(self.shards)
                logger.info(
                    "Resumed cache at %s (%d entries, %d shards)", self.output_dir, len(self.entries), len(self.shards)
                )
            else:
                raise FileExistsError(
                    f"OfflineTextEmbedWriter: {index_path} already exists. Pass --resume or --force-overwrite."
                )
        os.makedirs(self.output_dir, exist_ok=True)

    def add(self, prompt: str, tensor: torch.Tensor) -> None:
        """Append one unpadded ``[L, D]`` embedding."""
        require(
            tensor.dim() == 2 and int(tensor.shape[1]) == FEATURE_DIM,
            f"OfflineTextEmbedWriter: expected [L, {FEATURE_DIM}], got {tuple(tensor.shape)}",
        )
        require(
            tensor.dtype == self._torch_dtype,
            f"OfflineTextEmbedWriter: expected dtype {self._torch_dtype}, got {tensor.dtype}",
        )
        key = compute_prompt_key(prompt)
        self._current_shard_tensors[key] = tensor.detach().to("cpu").contiguous()
        self.entries[key] = self._current_shard_name
        if len(self._current_shard_tensors) >= self.shard_size:
            self._flush_current_shard()

    def _flush_current_shard(self) -> None:
        shard_name = self._current_shard_name
        shard_path = os.path.join(self.output_dir, shard_name)
        logger.info("Saving %d embeddings to %s", len(self._current_shard_tensors), shard_path)
        save_file(self._current_shard_tensors, shard_path)
        self.shards.append(shard_name)
        self._current_shard_tensors.clear()
        self._current_shard_idx += 1
        self._save_index()

    def _save_index(self) -> None:
        index_data = {
            **_STORE_FINGERPRINT,
            "model_checkpoint": self.model_checkpoint,
            "dtype": self.dtype,
            "total_prompts": len(self.entries),
            "shards": self.shards,
            "entries": self.entries,
        }
        index_path = os.path.join(self.output_dir, INDEX_FILENAME)
        tmp_path = index_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(index_data, handle, indent=2, ensure_ascii=False)
        os.replace(tmp_path, index_path)

    def close(self) -> None:
        """Flush the open shard and write the final index."""
        if self._current_shard_tensors:
            self._flush_current_shard()
        else:
            self._save_index()
        logger.info("Wrote MiniMax-H3 text-embed store to %s (%d prompts)", self.output_dir, len(self.entries))


__all__ = [
    "OfflineTextEmbedStore",
    "OfflineTextEmbedWriter",
    "compute_prompt_key",
]
