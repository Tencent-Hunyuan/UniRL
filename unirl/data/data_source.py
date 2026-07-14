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

from unirl.types.primitives import Images, Texts, Videos
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


def _load_condition_videos(media_refs: List[Any]) -> Optional[List[Any]]:
    """Load ``(modality="video", role="condition")`` media refs into ``Video``.

    Returns a per-prompt list of ``Video`` (or ``None`` for prompts that carry
    no condition video), or ``None`` when no prompt in the batch has a
    condition video. WAN V2V consumes one reference video per prompt.
    """
    if not media_refs:
        return None
    # Local imports keep video IO dependencies off text/image-only runs.
    import torchvision.io

    from unirl.types.primitives import Video as PrimVideo

    videos_per_prompt: List[Any] = []
    any_loaded = False
    for refs in media_refs:
        selected = [
            r
            for r in (refs or [])
            if getattr(r, "modality", None) == "video" and getattr(r, "role", None) == "condition"
        ]
        if not selected:
            videos_per_prompt.append(None)
            continue
        if len(selected) > 1:
            raise ValueError(f"WAN V2V expects <=1 (video, condition) MediaRef per prompt, got {len(selected)}")

        uri = selected[0].uri
        if str(uri).endswith((".pt", ".pth")):
            # weights_only=True blocks arbitrary code execution from a crafted
            # manifest pointing at an untrusted .pt (condition videos are plain tensors).
            frames = torch.load(uri, map_location="cpu", weights_only=True)
        elif str(uri).endswith((".npy", ".npz")):
            import numpy as np

            loaded = np.load(uri)
            frames = loaded["frames"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
            frames = torch.as_tensor(frames)
        else:
            frames, _, _ = torchvision.io.read_video(uri, pts_unit="sec", output_format="TCHW")
        if frames.numel() == 0:
            raise ValueError(f"Condition video has no decoded frames: {uri}")
        if frames.dtype == torch.uint8:
            frames = frames.to(dtype=torch.float32).div_(255.0)
        else:
            frames = frames.to(dtype=torch.float32).clamp_(0.0, 1.0)
        if int(frames.shape[1]) != 3:
            raise ValueError(
                f"WAN V2V expects RGB condition video frames [T, 3, H, W], got {tuple(frames.shape)} from {uri}"
            )
        videos_per_prompt.append(PrimVideo(frames=frames))
        any_loaded = True

    if not any_loaded:
        return None
    return videos_per_prompt


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


def _validate_homogeneous_videos(videos: List[Any]) -> None:
    """Reject batches where some prompts have condition videos and others don't."""
    populated = [vid for vid in videos if vid is not None]
    if populated and len(populated) != len(videos):
        missing = [i for i, vid in enumerate(videos) if vid is None]
        raise ValueError(
            f"Heterogeneous V2V batch — {len(missing)}/{len(videos)} prompts "
            f"are missing a condition video (e.g. prompt index {missing[0]}). "
            f"Split into separate requests so each batch is either fully T2V/I2V or fully V2V."
        )


_SUPPORTED_MEDIA_REF_ROLES: Set[Tuple[str, str]] = {("image", "condition"), ("video", "condition")}


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
        f"entries; the driver currently consumes only (image, condition) and (video, condition). "
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
        videos = _load_condition_videos(media_refs)
        if videos is not None:
            _validate_homogeneous_videos(videos)
            primitives["video"] = Videos.from_list([vid for vid in videos if vid is not None])

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
        videos = _load_condition_videos(media_refs)
        if videos is not None:
            _validate_homogeneous_videos(videos)
            primitives["video"] = Videos.from_list([vid for vid in videos if vid is not None])

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

    def iter_eval_batches(
        self,
        batch_size: int,
        *,
        eval_num_prompts: int = -1,
    ) -> Iterator[RolloutInputs]:
        """Yield the evaluation prompt source in deterministic batches.

        Args:
            batch_size: number of prompts per yielded batch. ``batch_size <= 0``
                yields nothing (safer than clamping to 1, which would silently
                iterate the full dataset prompt-by-prompt).
            eval_num_prompts: cap on total prompts iterated across all batches.
                Sentinel encoding (matches the trainer's ``eval_num_prompts``
                config knob):
                  * ``-1`` (default, or any negative value) — full eval dataset.
                  * ``0`` — yield nothing (explicit opt-out).
                  * ``N > 0`` — first ``min(N, len(eval_dataset))`` prompts; the
                    tail batch may be shorter than ``batch_size``.
        """
        batch_size = int(batch_size)
        eval_num_prompts = int(eval_num_prompts)
        if batch_size <= 0 or eval_num_prompts == 0:
            return
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

        total = len(self.eval_dataset)
        limit = total if eval_num_prompts < 0 else min(eval_num_prompts, total)
        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
            prompt_examples = [
                normalize_prompt_example(
                    get_prompt_example(idx),
                    default_prompt_id=f"eval:{idx}",
                )
                for idx in range(start, end)
            ]
            yield self._prompt_examples_to_batch(prompt_examples)

    def get_eval_samples(self, batch_size: int) -> RolloutInputs:
        """Return the first eval batch (BC shim over :meth:`iter_eval_batches`).

        ``batch_size <= 0`` returns an empty batch. Otherwise yields the first
        deterministic batch of up to ``batch_size`` prompts.
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            return self._prompt_examples_to_batch([])
        return next(
            self.iter_eval_batches(batch_size),
            self._prompt_examples_to_batch([]),
        )


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

    def _prompts_to_inputs(self, prompts: List[str], *, offset: int = 0) -> RolloutInputs:
        return RolloutInputs(
            primitives={"text": Texts(texts=prompts)},
            sample_ids=[f"prompt:{offset + i}:sample:0" for i in range(len(prompts))],
            group_ids=[f"prompt:{offset + i}" for i in range(len(prompts))],
        )

    def iter_eval_batches(
        self,
        batch_size: int,
        *,
        eval_num_prompts: int = -1,
    ) -> Iterator[RolloutInputs]:
        """Yield the default eval prompts in deterministic batches.

        Args:
            batch_size: number of prompts per yielded batch. ``batch_size <= 0``
                yields nothing.
            eval_num_prompts: cap on total prompts iterated. Same sentinel
                encoding as :meth:`MultimodalRLDataSource.iter_eval_batches`:
                ``-1`` (default) = full list; ``0`` = empty; ``N > 0`` = first
                ``min(N, len(self.prompts))``.
        """
        batch_size = int(batch_size)
        eval_num_prompts = int(eval_num_prompts)
        if batch_size <= 0 or eval_num_prompts == 0:
            return
        total = len(self.prompts)
        limit = total if eval_num_prompts < 0 else min(eval_num_prompts, total)
        for start in range(0, limit, batch_size):
            end = min(start + batch_size, limit)
            yield self._prompts_to_inputs(self.prompts[start:end], offset=start)

    def get_eval_samples(self, batch_size: int) -> RolloutInputs:
        """Return the first eval batch (BC shim over :meth:`iter_eval_batches`)."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            return self._prompts_to_inputs([])
        return next(
            self.iter_eval_batches(batch_size),
            self._prompts_to_inputs([]),
        )
