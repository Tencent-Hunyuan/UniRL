"""
Dataset implementations for GRPO training.

The default user-facing data input contract is prompt-first:
- Text prompts plus optional metadata (TextPromptDataset)
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset

from unirl.types.media import MediaRef

logger = logging.getLogger(__name__)

_LEGACY_EMBEDDING_FIELDS = {
    "prompt_embed_path",
    "pooled_embed_path",
    "pooled_prompt_embeds_path",
    "text_ids_path",
    "text_ids",
    "prompt_embeds",
    "pooled_prompt_embeds",
}

_PROMPT_EXAMPLE_EXCLUDED_KEYS = {
    "prompt",
    "caption",
    "media",
    "media_refs",
    "metadata",
    "prompt_id",
    *_LEGACY_EMBEDDING_FIELDS,
}

_MEDIA_REF_FIELDS = {"modality", "role", "uri"}


def _normalize_media_refs(raw_media: Any, *, base_dir: Optional[str] = None) -> List[MediaRef]:
    """Normalize a manifest ``media`` list into typed media references.

    Relative URIs are resolved against *base_dir* when provided.
    """
    if raw_media is None:
        return []
    if not isinstance(raw_media, list):
        raise TypeError(f"Prompt example media must be a list, got {type(raw_media).__name__}.")

    media_refs: List[MediaRef] = []
    for idx, item in enumerate(raw_media):
        if isinstance(item, MediaRef):
            media_refs.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(f"Prompt example media[{idx}] must be a dict, got {type(item).__name__}.")

        extra_fields = sorted(set(item) - _MEDIA_REF_FIELDS)
        if extra_fields:
            raise ValueError(
                "Prompt example media entries may only contain 'modality', 'role', and 'uri'. "
                f"Got extra fields={extra_fields} at media[{idx}]."
            )
        missing_fields = sorted(field for field in _MEDIA_REF_FIELDS if not item.get(field))
        if missing_fields:
            raise ValueError(f"Prompt example media[{idx}] is missing required fields={missing_fields}.")

        modality = str(item["modality"]).strip().lower()
        role = str(item["role"]).strip().lower()
        uri = _resolve_media_uri(str(item["uri"]).strip(), base_dir=base_dir)
        if not modality or not role or not uri:
            raise ValueError(f"Prompt example media[{idx}] fields must be non-empty strings.")
        media_refs.append(MediaRef(modality=modality, role=role, uri=uri))
    return media_refs


def normalize_prompt_example(
    item: Dict[str, Any],
    *,
    default_prompt_id: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize one raw dataset entry into prompt/metadata form.

    Args:
        item: Raw dataset entry dict.
        default_prompt_id: Fallback prompt ID if none is present in the item.
        base_dir: Base directory for resolving relative media URIs. Typically
            the parent directory of the dataset file.
    """
    if not isinstance(item, dict):
        raise TypeError(f"Prompt example must be a dict, got {type(item).__name__}.")

    prompt = item.get("prompt", item.get("caption", ""))
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Prompt example is missing a non-empty 'prompt' or 'caption' field.")

    metadata = item.get("metadata")
    if metadata is None:
        metadata = {key: value for key, value in item.items() if key not in _PROMPT_EXAMPLE_EXCLUDED_KEYS}
    elif not isinstance(metadata, dict):
        raise TypeError(f"Prompt example metadata must be a dict when provided, got {type(metadata).__name__}.")

    raw_media = item.get("media_refs", item.get("media"))
    media_refs = _normalize_media_refs(raw_media, base_dir=base_dir)

    result: Dict[str, Any] = {"prompt": prompt}
    prompt_id = item.get("prompt_id")
    if prompt_id is None and default_prompt_id is not None:
        prompt_id = default_prompt_id
    if prompt_id is not None:
        result["prompt_id"] = str(prompt_id)
    if metadata:
        result["metadata"] = dict(metadata)
    if media_refs:
        result["media_refs"] = media_refs
    return result


def _resolve_media_uri(raw_path: str, *, base_dir: Optional[str] = None) -> str:
    """Resolve a media path to an absolute URI.

    - Absolute paths and URLs (http://, https://, s3://, gs://) are returned as-is.
    - Relative paths are joined with *base_dir* (if provided) to produce an
      absolute filesystem path.
    """
    if raw_path.startswith(("http://", "https://", "s3://", "gs://")):
        return raw_path
    if os.path.isabs(raw_path):
        return raw_path
    if base_dir:
        return os.path.join(base_dir, raw_path)
    return raw_path


class PromptExampleDataset(Dataset):
    """
    Dataset that can expose prompt/metadata examples without loading training tensors.

    This is the framework-level interface used by evaluation prompt selection.
    """

    def get_prompt_example(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement get_prompt_example().")


class TextPromptDataset(PromptExampleDataset):
    """
    File-backed dataset for text prompt examples.

    This class owns file parsing and per-row normalization only. Runtime
    concerns such as epoch ordering, batching, and drop-last policy belong in
    the data source / sampler layer.

    Supports:
    - JSON file with list of strings
    - JSON file with list of dicts containing 'prompt' or 'caption'
    - JSONL file with one JSON object per line (JSON Lines format)
    - TXT file with one prompt per line

    Example JSON formats:
        ["prompt 1", "prompt 2", ...]
        [{"prompt": "prompt 1"}, {"prompt": "prompt 2"}, ...]

    Example JSONL format:
        {"prompt": "prompt 1"}
        {"prompt": "prompt 2"}
    """

    def __init__(
        self,
        file_path: str,
        prompt_key: str = "prompt",
        seed: Optional[int] = None,
        shuffle: bool = False,
    ):
        """
        Initialize text prompt dataset.

        Args:
            file_path: Path to JSON or TXT file containing prompts
            prompt_key: Key for prompt in JSON dicts
            seed: Random seed for optional standalone load-time shuffling
            shuffle: Whether to shuffle prompts on load. Training data sources
                should normally leave this disabled and shuffle via samplers.
        """
        self.file_path = file_path
        self.prompt_key = prompt_key
        self.samples: List[Dict[str, Any]] = []
        self._base_dir = os.path.dirname(os.path.abspath(file_path))

        # Load prompts
        self._source_prefix = os.path.basename(self.file_path) or "prompt_source"
        self._load_prompts()

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.Random(seed).shuffle(self.samples)
            else:
                random.shuffle(self.samples)

        logger.info(f"Loaded {len(self.samples)} prompts from {file_path}")

    def _load_prompts(self) -> None:
        """Load prompts from file."""

        def _append_item(item: Any, *, context: str) -> None:
            sample_idx = len(self.samples)
            default_prompt_id = f"{self._source_prefix}:{sample_idx}"
            if isinstance(item, str):
                candidate: Any = {"prompt": item}
            else:
                candidate = item
            if not isinstance(item, dict):
                if not isinstance(candidate, dict):
                    logger.warning("Skipping invalid %s: %r", context, item)
                    return
            candidate = dict(candidate)
            if self.prompt_key in candidate and self.prompt_key != "prompt":
                candidate["prompt"] = candidate.pop(self.prompt_key)
            legacy_fields = sorted(field for field in _LEGACY_EMBEDDING_FIELDS if field in candidate)
            if legacy_fields:
                raise ValueError(
                    "Prompt manifests must be prompt-first and may not include legacy embedding fields. "
                    f"Got fields={legacy_fields} in {self.file_path}."
                )

            try:
                normalized = normalize_prompt_example(
                    candidate,
                    default_prompt_id=default_prompt_id,
                    base_dir=self._base_dir,
                )
                self.samples.append(normalized)
            except (TypeError, ValueError) as exc:
                has_media_contract = any(field in candidate for field in ("media", "media_refs"))
                if has_media_contract:
                    raise
                logger.warning("Skipping invalid %s: %s", context, exc)

        if self.file_path.endswith(".json"):
            with open(self.file_path, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    _append_item(item, context="item")
            elif isinstance(data, dict):
                # Handle dict format with prompts key
                if "prompts" in data:
                    prompts = data["prompts"]
                    if isinstance(prompts, list):
                        for item in prompts:
                            _append_item(item, context="prompts item")
                    elif isinstance(prompts, str):
                        _append_item(prompts, context="prompts item")
                elif self.prompt_key in data:
                    prompt_val = data[self.prompt_key]
                    if isinstance(prompt_val, list):
                        for item in prompt_val:
                            if isinstance(item, dict):
                                _append_item(item, context="prompt list item")
                            else:
                                _append_item({self.prompt_key: item}, context="prompt list item")
                    elif isinstance(prompt_val, str):
                        _append_item(data, context="top-level prompt item")
                elif "caption" in data:
                    _append_item(data, context="top-level caption item")

        elif self.file_path.endswith(".jsonl"):
            # JSON Lines format: one JSON object per line
            with open(self.file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        _append_item(item, context=f"item at line {line_num}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        elif self.file_path.endswith(".txt"):
            with open(self.file_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    prompt = line.strip()
                    if not prompt:
                        continue
                    _append_item(prompt, context=f"txt line {line_num}")

        else:
            raise ValueError(f"Unsupported file format: {self.file_path}. Supported formats: .json, .jsonl, .txt")

        if not self.samples:
            raise ValueError(f"No prompts found in {self.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

    def get_prompt_example(self, idx: int) -> Dict[str, Any]:
        return normalize_prompt_example(
            self.samples[idx],
            default_prompt_id=f"{self._source_prefix}:{idx}",
            base_dir=self._base_dir,
        )
