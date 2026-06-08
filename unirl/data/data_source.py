"""
Data source implementations for GRPO training.

The default data-source contract is prompt-first:
- prompts plus optional typed media references and metadata for rollout/eval input

Runtime prompt embeddings are produced inside rollout engines and training
pipelines, not provided by the external dataset.
"""

import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader

from unirl.types.primitives import Images, Texts
from unirl.types.prompts import RolloutInputs

from .datasets import PromptExampleDataset, TextPromptDataset, normalize_prompt_example

logger = logging.getLogger(__name__)


def _load_condition_images(media_refs: List[Any]) -> Optional[List[Any]]:
    """Load ``(modality="image", role="condition")`` media refs into ``Image``.

    Returns a per-prompt list of ``Image`` (or ``None`` for prompts that
    carry no condition image), or ``None`` when no prompt in the batch
    has a condition image — letting the caller omit the ``images`` key
    from the collated batch entirely.

    Raises ``ValueError`` if any prompt carries more than one condition
    image (WAN I2V is single-frame conditioned).
    """
    if not media_refs:
        return None
    # Local imports keep PIL / torchvision off the import path for
    # text-only training runs that never touch this code.
    import PIL.Image
    import torchvision.transforms.functional as TF

    from unirl.types.primitives import Image as PrimImage

    images_per_prompt: List[Any] = []
    any_loaded = False
    for refs in media_refs:
        selected = [
            r
            for r in (refs or [])
            if getattr(r, "modality", None) == "image" and getattr(r, "role", None) == "condition"
        ]
        if not selected:
            images_per_prompt.append(None)
            continue
        if len(selected) > 1:
            raise ValueError(f"WAN I2V expects <=1 (image, condition) MediaRef per prompt, got {len(selected)}")
        pil = PIL.Image.open(selected[0].uri).convert("RGB")
        tensor = TF.to_tensor(pil)  # [3, H, W] in [0, 1]
        images_per_prompt.append(PrimImage(pixels=tensor))
        any_loaded = True

    if not any_loaded:
        return None
    return images_per_prompt


def _validate_homogeneous_images(images: List[Any]) -> None:
    """Reject batches where some prompts have condition images and others don't."""
    populated = [img for img in images if img is not None]
    if populated and len(populated) != len(images):
        missing = [i for i, img in enumerate(images) if img is None]
        raise ValueError(
            f"Heterogeneous I2V batch — {len(missing)}/{len(images)} prompts "
            f"are missing a condition image (e.g. prompt index {missing[0]}). "
            f"Split into separate requests so each batch is either fully T2V or "
            f"fully I2V; per-sample channel-concat is not supported."
        )


_SUPPORTED_MEDIA_REF_ROLES: Set[Tuple[str, str]] = {("image", "condition")}


def _reject_unsupported_media_refs(batch: Dict[str, Any], *, context: str) -> None:
    """Fail loud when a dataset hands unsupported media_refs to the driver.

    The ``media_refs`` channel carries a ``MediaRef(uri, modality, role)``
    URI list. The driver consumes the ``(image, condition)`` (modality,
    role) pair via :func:`_load_condition_images`
    → ``RolloutInputs.primitives['image']: Images``;
    all other (modality, role) combinations are not yet typed and
    would be silently dropped (degrading I2V/V2V/text-conditioned jobs
    into a misconfigured run).

    Supported set: see :data:`_SUPPORTED_MEDIA_REF_ROLES`. Anything else
    raises ``NotImplementedError`` with a per-prompt index of the first
    offending entry so debugging is straightforward.
    """
    refs = batch.get("media_refs")
    if not refs:
        return
    if not isinstance(refs, list):
        raise TypeError(
            f"{context}: media_refs must be a list of per-prompt MediaRef lists, got {type(refs).__name__}."
        )
    bad: List[Tuple[int, Any]] = []
    for i, per_prompt in enumerate(refs or []):
        for r in per_prompt or []:
            modality = getattr(r, "modality", None)
            role = getattr(r, "role", None)
            if (modality, role) not in _SUPPORTED_MEDIA_REF_ROLES:
                bad.append((i, r))
    if not bad:
        return
    raise NotImplementedError(
        f"{context}: media_refs include {len(bad)} unsupported (modality, role) "
        f"entries; the driver currently consumes only (image, condition). "
        f"First bad entry: prompt={bad[0][0]}, ref={bad[0][1]!r}."
    )


class MultimodalRLDataSource:
    """
    Multimodal runtime data source for RL training.

    This layer owns run-time example ordering, batching, and train/eval source
    selection. Dataset implementations stay responsible for loading indexed
    examples from storage.

    Accepted user-facing formats:
    - JSON/TXT/JSONL prompt datasets
    - JSON manifests with ``prompt`` or ``caption`` plus optional ``media``
      references and extra metadata
    """

    def __init__(self, args):
        """
        Initialize data source from arguments.

        Args:
            args: Hydra ``cfg`` (DictConfig) with:
                - run.data_path: Path to data file (JSON, JSONL, or TXT)
                - run.seed: Random seed
                - algorithm.prompts_per_rollout: Batch size
        """
        self.args = args
        self.data_path = args.run.data_path
        self.eval_data_path = args.run.eval_data_path
        self.seed = args.run.seed
        self.prompts_per_rollout = int(args.algorithm.prompts_per_rollout)
        self.drop_last = True

        # Training data and eval data are treated as separate prompt sources.
        self.train_dataset = None
        self.eval_dataset = None
        self._dataloader = None
        self._iter: Optional[Iterator] = None
        self._eval_dataset_ready = False
        self._shuffle_generator = torch.Generator()
        # ``self.seed`` may be None (run.seed=null) — torch.Generator needs an
        # int, so draw from OS entropy. Per-process shuffle order then becomes
        # non-reproducible, matching the seed=null contract.
        if self.seed is None:
            _shuffle_seed = int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF
        else:
            _shuffle_seed = int(self.seed)
        self._shuffle_generator.manual_seed(_shuffle_seed)

        if not self.data_path:
            raise ValueError("MultimodalRLDataSource requires args.run.data_path.")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Training data path not found: {self.data_path}")
        self._init_dataset()

    def _init_dataset(self) -> None:
        """Initialize the training dataset and dataloader."""
        self.train_dataset = self._build_dataset(self.data_path)
        logger.info(
            "Loaded multimodal training dataset from %s (%d samples)",
            self.data_path,
            len(self.train_dataset),
        )
        self._create_dataloader()

    def _build_dataset(self, path: str) -> PromptExampleDataset:
        """Build one prompt dataset instance for either training or evaluation."""
        return TextPromptDataset(
            file_path=path,
        )

    def _resolve_eval_path(self) -> Optional[str]:
        """Resolve which path should back evaluation prompt selection."""
        if self.eval_data_path:
            if not os.path.exists(self.eval_data_path):
                raise FileNotFoundError(f"Evaluation data path not found: {self.eval_data_path}")
            return self.eval_data_path
        return self.data_path

    def _ensure_eval_dataset(self) -> None:
        """Lazily build the evaluation dataset with deterministic ordering."""
        if self._eval_dataset_ready:
            return

        eval_path = self._resolve_eval_path()
        if eval_path is None or not os.path.exists(eval_path):
            self.eval_dataset = None
            self._eval_dataset_ready = True
            return

        self.eval_dataset = self._build_dataset(eval_path)
        self._eval_dataset_ready = True

        source_label = "eval_data_path" if self.eval_data_path else "data_path"
        logger.info(
            "Loaded evaluation prompt source from %s=%s (%d examples, deterministic order)",
            source_label,
            eval_path,
            len(self.eval_dataset),
        )

    def _create_dataloader(self) -> None:
        """Create dataloader for prompt-batch sampling."""
        if self.train_dataset is None:
            return

        # prompts_per_rollout determines the DataLoader batch size; do not repeat each prompt k times here
        sampler = None
        if len(self.train_dataset) < self.prompts_per_rollout:
            raise ValueError(
                "Training dataset is smaller than prompts_per_rollout, which would produce an "
                f"empty DataLoader with drop_last=True (num_prompts={len(self.train_dataset)}, "
                f"prompts_per_rollout={self.prompts_per_rollout})."
            )

        self._dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.prompts_per_rollout,
            sampler=sampler,
            shuffle=(sampler is None),  # Only shuffle if not using custom sampler
            generator=self._shuffle_generator if sampler is None else None,
            num_workers=0,  # Keep simple for Ray
            collate_fn=self._collate_text,
            drop_last=True,
        )

        self._iter = iter(self._dataloader)

    def _collate_text(self, batch: List[Dict[str, Any]]) -> RolloutInputs:
        """Collate function for text prompt dataset."""
        prompts = [item["prompt"] for item in batch]
        prompt_ids = self._resolve_prompt_ids(batch)
        sample_ids = [f"prompt:{pid}:sample:0" for pid in prompt_ids]

        media_refs = [item.get("media_refs", []) for item in batch]
        if any(media_refs):
            _reject_unsupported_media_refs({"media_refs": media_refs}, context="MultimodalRLDataSource._collate_text")

        primitives: Dict[str, Any] = {"text": Texts(texts=prompts)}
        images = _load_condition_images(media_refs)
        if images is not None:
            _validate_homogeneous_images(images)
            primitives["image"] = Images.from_list([img for img in images if img is not None])

        metadata_list = [item.get("metadata") for item in batch]

        return RolloutInputs(
            primitives=primitives,
            sample_ids=sample_ids,
            group_ids=list(prompt_ids),
            metadata=metadata_list,
        )

    @property
    def num_prompts(self) -> int:
        """Total number of prompts in the training dataset."""
        if self.train_dataset is not None:
            return len(self.train_dataset)
        return 0

    def _prompt_examples_to_batch(self, prompt_examples: List[Dict[str, Any]]) -> RolloutInputs:
        """Convert normalized prompt examples into a RolloutInputs."""
        prompts = [item["prompt"] for item in prompt_examples]
        prompt_ids = self._resolve_prompt_ids(prompt_examples)
        sample_ids = [f"prompt:{pid}:sample:0" for pid in prompt_ids]

        media_refs = [item.get("media_refs", []) for item in prompt_examples]
        if any(media_refs):
            _reject_unsupported_media_refs(
                {"media_refs": media_refs}, context="MultimodalRLDataSource._prompt_examples_to_batch"
            )

        primitives: Dict[str, Any] = {"text": Texts(texts=prompts)}
        images = _load_condition_images(media_refs)
        if images is not None:
            _validate_homogeneous_images(images)
            primitives["image"] = Images.from_list([img for img in images if img is not None])

        metadata_list = [item.get("metadata") for item in prompt_examples]

        return RolloutInputs(
            primitives=primitives,
            sample_ids=sample_ids,
            group_ids=list(prompt_ids),
            metadata=metadata_list,
        )

    def _resolve_prompt_ids(self, prompt_examples: List[Dict[str, Any]]) -> List[str]:
        """Resolve deterministic prompt IDs even if a dataset forgot to provide them."""
        prompt_ids: List[str] = []
        for idx, item in enumerate(prompt_examples):
            prompt_id = item.get("prompt_id")
            if prompt_id is None or not str(prompt_id).strip():
                prompt_ids.append(f"prompt:{idx}")
            else:
                prompt_ids.append(str(prompt_id))
        return prompt_ids

    def get_samples(self, batch_size: int) -> RolloutInputs:
        """Get next batch of samples as a typed ``RolloutInputs``."""
        if self._iter is None:
            raise RuntimeError("MultimodalRLDataSource is not initialized. Training DataLoader is unavailable.")

        try:
            batch = next(self._iter)
        except StopIteration:
            # Reset iterator
            self._iter = iter(self._dataloader)
            batch = next(self._iter)

        return batch

    def get_eval_samples(self, batch_size: int) -> Dict[str, Any]:
        """Get a stable eval batch from the dedicated evaluation prompt source."""
        batch_size = max(0, int(batch_size))
        if batch_size == 0:
            return {"prompts": []}

        self._ensure_eval_dataset()
        if self.eval_dataset is None:
            raise RuntimeError(
                "MultimodalRLDataSource could not initialize evaluation prompt data. "
                "Provide eval_data_path or a readable training data_path."
            )

        get_prompt_example = getattr(self.eval_dataset, "get_prompt_example", None)
        if not callable(get_prompt_example):
            raise TypeError(
                f"Evaluation dataset {type(self.eval_dataset).__name__} must implement "
                "get_prompt_example(idx) -> {'prompt': ..., 'metadata': ...}."
            )

        prompt_examples = [
            normalize_prompt_example(
                get_prompt_example(idx),
                default_prompt_id=f"eval:{idx}",
            )
            for idx in range(min(batch_size, len(self.eval_dataset)))
        ]
        return self._prompt_examples_to_batch(prompt_examples)


class DefaultDataSource:
    """
    Default data source that returns simple prompts.

    Used when no data_path is specified or as fallback.
    """

    def __init__(self, args):
        """
        Initialize default data source.

        Args:
            args: Hydra ``cfg`` (DictConfig)
        """
        self.args = args
        self.drop_last = False

        # Default prompts for different scenarios
        self.prompts = [
            "A beautiful sunset over the ocean",
            "A cat playing with a ball of yarn",
            "A mountain landscape with snow",
            "A futuristic city at night",
            "A garden full of colorful flowers",
            "A cozy cabin in the woods",
            "An astronaut floating in space",
            "A tropical beach with palm trees",
        ]

        self._index = 0

    @property
    def num_prompts(self) -> int:
        return len(self.prompts)

    def get_samples(self, batch_size: int) -> RolloutInputs:
        """Get next batch of prompts."""
        prompts = []
        for _ in range(batch_size):
            prompts.append(self.prompts[self._index % len(self.prompts)])
            self._index += 1
        return RolloutInputs(
            primitives={"text": Texts(texts=prompts)},
            sample_ids=[f"prompt:{i}:sample:0" for i in range(len(prompts))],
            group_ids=[f"prompt:{i}" for i in range(len(prompts))],
        )

    def get_eval_samples(self, batch_size: int) -> Dict[str, List[str]]:
        """Get a stable eval batch."""
        batch_size = max(0, int(batch_size))
        return {"prompts": self.prompts[:batch_size]}
