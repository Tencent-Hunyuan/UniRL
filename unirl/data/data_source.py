"""
Data source implementations for GRPO training.

The default data-source contract is prompt-first:
- prompts plus optional typed media references and metadata for rollout/eval input

Runtime prompt embeddings are produced inside rollout engines and training
pipelines, not provided by the external dataset.
"""

import logging
import os
from collections import Counter
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, cast

import torch
from torch.utils.data import DataLoader

from unirl.types.media import MediaRef, MediaRefs
from unirl.types.primitives import Image, Images, Texts, Videos
from unirl.types.sample import Part, PrimitiveMap, Sample
from unirl.utils.video import load_video

from .datasets import PromptExampleDataset, TextPromptDataset, normalize_prompt_example

logger = logging.getLogger(__name__)


def _load_condition_images(media_refs: List[Any]) -> Optional[List[Image]]:
    """Load ``(modality="image", role="condition")`` media refs into ``Image``.

    Returns one image per prompt, or ``None`` when the batch has no condition
    images. Partially populated batches are rejected here.

    Raises ``ValueError`` if any prompt carries more than one condition image.
    """
    if not media_refs or not any(media_refs):
        return None
    import PIL.Image
    import torchvision.transforms.functional as TF

    images_per_prompt: List[Optional[Image]] = []
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
            raise ValueError(f"Expected at most one condition image per prompt, got {len(selected)}")
        pil = PIL.Image.open(selected[0].uri).convert("RGB")
        tensor = TF.to_tensor(pil)
        images_per_prompt.append(Image(pixels=tensor))
        any_loaded = True

    if not any_loaded:
        return None
    missing = [index for index, image in enumerate(images_per_prompt) if image is None]
    if missing:
        raise ValueError(
            f"Condition-image batch is incomplete: {len(missing)}/{len(images_per_prompt)} "
            f"prompts are missing an image (e.g. prompt index {missing[0]})."
        )
    return cast(List[Image], images_per_prompt)


def _load_condition_videos(media_refs: List[Any]) -> Optional[List[Any]]:
    """Load ``(modality="video", role="condition")`` media refs into ``Video``.

    Returns a per-prompt list of ``Video`` (or ``None`` for prompts that carry
    no condition video), or ``None`` when no prompt in the batch has a
    condition video. WAN V2V consumes one reference video per prompt.
    """
    if not media_refs or not any(media_refs):
        return None
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

        videos_per_prompt.append(PrimVideo(frames=load_video(selected[0].uri)))
        any_loaded = True

    if not any_loaded:
        return None
    return videos_per_prompt


def _validate_video_media_roles(media_refs: List[Any]) -> None:
    roles = {
        getattr(ref, "role", None)
        for refs in media_refs
        for ref in (refs or [])
        if getattr(ref, "modality", None) == "video"
    }
    if "condition" in roles and "prompt" in roles:
        raise ValueError("A batch cannot mix (video, condition) and (video, prompt) MediaRefs.")


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


_SUPPORTED_MEDIA_REF_ROLES: Set[Tuple[str, str]] = {
    ("image", "condition"),
    ("image", "prompt"),
    ("video", "condition"),
    ("audio", "prompt"),
    ("video", "prompt"),
}


def _dataset_metadata(items: List[Dict[str, Any]], *, context: str) -> List[Optional[Dict[str, Any]]]:
    """Keep dataset/reward metadata separate from typed model inputs."""
    values: List[Optional[Dict[str, Any]]] = []
    for row, item in enumerate(items):
        metadata = item.get("metadata")
        if isinstance(metadata, dict) and "_media_refs" in metadata:
            raise ValueError(
                f"{context}: metadata['_media_refs'] is no longer a model-input channel "
                f"(row {row}); use the dataset media/media_refs field."
            )
        values.append(metadata)
    return values


def _prompt_media_primitive(
    media_refs: List[Any],
    *,
    context: str,
) -> Optional[MediaRefs]:
    """Build a batch-aligned sparse prompt-media primitive.

    The outer row list remains rectangular while each row may carry a different
    image/video/audio combination (or no prompt media at all). Schema and URI
    normalization live here; model-specific cardinality (e.g. Qwen3-Omni's one
    input per modality) is enforced by the Omni adapter, not the data layer.
    Local-path existence is checked when the actor opens the media.
    """
    rows: List[List[MediaRef]] = []
    any_prompt_media = False
    for row, refs in enumerate(media_refs):
        selected = [ref for ref in (refs or []) if getattr(ref, "role", None) == "prompt"]
        typed: List[MediaRef] = []
        for ref in selected:
            if isinstance(ref, MediaRef):
                typed.append(ref)
                continue
            modality = getattr(ref, "modality", None)
            uri = getattr(ref, "uri", None)
            try:
                typed.append(MediaRef(modality=modality, role="prompt", uri=uri))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{context}: prompt {row} invalid media ref: {exc}") from exc
        rows.append(typed)
        any_prompt_media = any_prompt_media or bool(typed)
    return MediaRefs.from_rows(rows) if any_prompt_media else None


def _reject_unsupported_media_refs(batch: Dict[str, Any], *, context: str) -> None:
    """Fail loud when a dataset hands unsupported media_refs to the driver.

    The ``media_refs`` channel carries a ``MediaRef(uri, modality, role)``
    URI list. The driver consumes condition media into chained primitive
    Parts and preserves sparse prompt image/audio/video refs in a typed
    ``MediaRefs`` input Part. Condition images use packed ``Images`` so their
    per-sample source resolutions remain ragged until a model consumes them;
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
    _validate_video_media_roles(refs)
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
        f"entries; supported pairs are {sorted(_SUPPORTED_MEDIA_REF_ROLES)}. "
        f"First bad entry: prompt={bad[0][0]}, ref={bad[0][1]!r}."
    )


def _input_sample(
    primitives: PrimitiveMap,
    *,
    sample_ids: List[str],
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> Sample:
    """Build the data-source request as a text-rooted input Part chain."""
    if len(set(sample_ids)) != len(sample_ids):
        duplicates = sorted(sample_id for sample_id, count in Counter(sample_ids).items() if count > 1)
        raise ValueError(f"Data-source input requires unique root sample_ids; duplicates: {duplicates[:3]}")
    if metadata is not None and len(metadata) != len(sample_ids):
        raise ValueError(f"Data-source metadata count {len(metadata)} != sample_ids count {len(sample_ids)}.")
    text = primitives.get("text")
    if not isinstance(text, Texts):
        raise TypeError(f"Data-source input requires a Texts root, got {type(text).__name__}.")
    unsupported = set(primitives) - {"text", "image", "video", "media"}
    if unsupported:
        raise ValueError(f"Data-source input has unsupported primitive keys: {sorted(unsupported)}")

    root = Part.input(sample_ids, primitives={"text": text}, metadata=metadata)
    parts = [root]
    parent = root
    for key in ("image", "video", "media"):
        primitive = primitives.get(key)
        if primitive is None:
            continue
        parent = parent.input_child({key: primitive})
        parts.append(parent)
    return Sample.request(*parts)


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
                - run.shuffle: Whether to shuffle the training prompts (default: True)
                - algorithm.prompts_per_rollout: Batch size
        """
        self.args = args
        self.data_path = args.run.data_path
        self.eval_data_path = args.run.eval_data_path
        self.seed = args.run.seed
        self.shuffle = bool(getattr(args.run, "shuffle", True))
        self.prompts_per_rollout = int(args.algorithm.prompts_per_rollout)
        self.drop_last = True

        self.train_dataset = None
        self.eval_dataset = None
        self._dataloader = None
        self._iter: Optional[Iterator] = None
        self._eval_dataset_ready = False
        self._shuffle_generator = torch.Generator()
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

        should_shuffle = sampler is None and self.shuffle
        self._dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.prompts_per_rollout,
            sampler=sampler,
            shuffle=should_shuffle,
            generator=self._shuffle_generator if should_shuffle else None,
            num_workers=0,
            collate_fn=self._collate_text,
            drop_last=True,
        )

        self._iter = iter(self._dataloader)

    def _collate_text(self, batch: List[Dict[str, Any]]) -> Sample:
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
            primitives["image"] = Images.from_list(images)
        condition_videos = _load_condition_videos(media_refs)
        if condition_videos is not None:
            _validate_homogeneous_videos(condition_videos)
            primitives["video"] = Videos.from_list([video for video in condition_videos if video is not None])
        prompt_media = _prompt_media_primitive(media_refs, context="MultimodalRLDataSource._collate_text")
        if prompt_media is not None:
            primitives["media"] = prompt_media

        metadata_list = _dataset_metadata(batch, context="MultimodalRLDataSource._collate_text")

        return _input_sample(primitives, sample_ids=sample_ids, metadata=metadata_list)

    @property
    def num_prompts(self) -> int:
        """Total number of prompts in the training dataset."""
        if self.train_dataset is not None:
            return len(self.train_dataset)
        return 0

    def _prompt_examples_to_batch(self, prompt_examples: List[Dict[str, Any]]) -> Sample:
        """Convert normalized prompt examples into an input-only ``Sample``."""
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
            primitives["image"] = Images.from_list(images)
        condition_videos = _load_condition_videos(media_refs)
        if condition_videos is not None:
            _validate_homogeneous_videos(condition_videos)
            primitives["video"] = Videos.from_list([video for video in condition_videos if video is not None])
        prompt_media = _prompt_media_primitive(media_refs, context="MultimodalRLDataSource._prompt_examples_to_batch")
        if prompt_media is not None:
            primitives["media"] = prompt_media

        metadata_list = _dataset_metadata(prompt_examples, context="MultimodalRLDataSource._prompt_examples_to_batch")

        return _input_sample(primitives, sample_ids=sample_ids, metadata=metadata_list)

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

    def get_samples(self, batch_size: int) -> Sample:
        """Get the next batch as an input-only request ``Sample``."""
        if self._iter is None:
            raise RuntimeError("MultimodalRLDataSource is not initialized. Training DataLoader is unavailable.")

        try:
            batch = next(self._iter)
        except StopIteration:
            self._iter = iter(self._dataloader)
            batch = next(self._iter)

        return batch

    def iter_eval_batches(
        self,
        batch_size: int,
        *,
        eval_num_prompts: int = -1,
    ) -> Iterator[Sample]:
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

    def get_eval_samples(self, batch_size: int) -> Sample:
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


class MultiDomainRLDataSource:
    """Round-robin multi-domain prompt source — one domain per rollout batch.

    Composes one ``TextPromptDataset`` + ``DataLoader`` per named domain and
    cycles them across ``get_samples()`` calls, so every rollout batch is
    single-domain. Every row is stamped with ``metadata["domain"] = <name>``;
    downstream components route on that tag (``DiffusionOPD`` picks the frozen
    teacher adapter of the same name, ``PerDomainRewardScorer`` dispatches each
    row to its domain's scorer). Because routing is carried by the data itself,
    resume fast-forwarding or reordering can never desynchronize a batch from
    its domain.

    Config (Hydra ``args``)::

        args:
          run:
            domains:
              - name: pickscore
                data_path: datasets/pickscore/train.txt
                eval_data_path: datasets/pickscore/test.txt   # optional
              - name: ocr
                data_path: datasets/ocr/train.txt
            seed: 42
          algorithm:
            prompts_per_rollout: ${batch_size}

    Text-only prompts (``media_refs`` are rejected). Prompt ids are prefixed
    with the domain name so ids stay unique across domains — including inside
    the concatenated eval Sample.
    """

    def __init__(self, args):
        run_cfg = args.run
        entries = list(getattr(run_cfg, "domains", None) or [])
        if not entries:
            raise ValueError("MultiDomainRLDataSource requires args.run.domains (a non-empty list).")

        self.domains: List[Dict[str, Optional[str]]] = []
        for entry in entries:
            get = entry.get if hasattr(entry, "get") else lambda k, d=None: getattr(entry, k, d)
            name, data_path, eval_data_path = get("name"), get("data_path"), get("eval_data_path")
            if not name or not data_path:
                raise ValueError(f"Each domain entry needs 'name' and 'data_path'; got {entry!r}.")
            if not os.path.exists(str(data_path)):
                raise FileNotFoundError(f"Domain {name!r} data_path not found: {data_path}")
            if eval_data_path and not os.path.exists(str(eval_data_path)):
                raise FileNotFoundError(f"Domain {name!r} eval_data_path not found: {eval_data_path}")
            self.domains.append(
                {
                    "name": str(name),
                    "data_path": str(data_path),
                    "eval_data_path": str(eval_data_path) if eval_data_path else None,
                }
            )
        names = [d["name"] for d in self.domains]
        if len(set(names)) != len(names):
            raise ValueError(f"Domain names must be unique, got {names}.")

        self.seed = getattr(run_cfg, "seed", None)
        self.prompts_per_rollout = int(args.algorithm.prompts_per_rollout)
        self.drop_last = True

        self._datasets: List[TextPromptDataset] = []
        self._dataloaders: List[DataLoader] = []
        self._iters: List[Iterator] = []
        self._eval_datasets: Optional[List[TextPromptDataset]] = None
        self._iter_counter = 0

        for i, domain in enumerate(self.domains):
            ds = TextPromptDataset(file_path=domain["data_path"])
            if len(ds) < self.prompts_per_rollout:
                raise ValueError(
                    f"Domain {domain['name']!r} dataset is smaller than prompts_per_rollout, which "
                    f"would produce an empty DataLoader with drop_last=True "
                    f"(num_prompts={len(ds)}, prompts_per_rollout={self.prompts_per_rollout})."
                )
            # Per-domain generator offset by index; seed=None -> OS entropy
            # (the base class's seed=null contract).
            generator = torch.Generator()
            if self.seed is None:
                generator.manual_seed(int.from_bytes(os.urandom(8), "big") & 0x7FFFFFFF)
            else:
                generator.manual_seed(int(self.seed) + i)
            loader = DataLoader(
                ds,
                batch_size=self.prompts_per_rollout,
                shuffle=True,
                generator=generator,
                num_workers=0,  # Keep simple for Ray
                collate_fn=partial(self._collate_domain, domain["name"]),
                drop_last=True,
            )
            self._datasets.append(ds)
            self._dataloaders.append(loader)
            self._iters.append(iter(loader))
            logger.info(
                "MultiDomainRLDataSource: domain %r — %d prompts from %s",
                domain["name"],
                len(ds),
                domain["data_path"],
            )

    def _domain_examples_to_batch(self, domain: str, examples: List[Dict[str, Any]], *, tag: str) -> Sample:
        """Build a Sample from one domain's prompt examples, domain-stamped."""
        if any(item.get("media_refs") for item in examples):
            raise ValueError(
                f"MultiDomainRLDataSource is text-prompt-only; domain {domain!r} has media_refs. "
                "Extend it alongside MultimodalRLDataSource._collate_text when a recipe needs media."
            )
        prompts = [item["prompt"] for item in examples]
        prompt_ids = []
        for idx, item in enumerate(examples):
            pid = item.get("prompt_id")
            pid = f"{tag}:{idx}" if pid is None or not str(pid).strip() else str(pid)
            # Domain prefix keeps ids unique across domains (the eval Sample
            # concatenates all domains into one id namespace).
            prompt_ids.append(f"{domain}:{pid}")
        sample_ids = [f"prompt:{pid}:sample:0" for pid in prompt_ids]
        metadata_list: List[Optional[Dict[str, Any]]] = []
        for item in examples:
            md = dict(item.get("metadata") or {})
            md["domain"] = domain
            metadata_list.append(md)
        return _input_sample({"text": Texts(texts=prompts)}, sample_ids=sample_ids, metadata=metadata_list)

    def _collate_domain(self, domain: str, batch: List[Dict[str, Any]]) -> Sample:
        return self._domain_examples_to_batch(domain, batch, tag="train")

    def get_samples(self, batch_size: int) -> Sample:
        """Next single-domain batch; domains cycle in declaration order.

        ``batch_size`` is nominal — the actual size is ``prompts_per_rollout``,
        fixed at construction (mirrors :class:`MultimodalRLDataSource`).
        """
        idx = self._iter_counter % len(self.domains)
        self._iter_counter += 1
        try:
            return next(self._iters[idx])
        except StopIteration:
            self._iters[idx] = iter(self._dataloaders[idx])
            return next(self._iters[idx])

    @property
    def num_prompts(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    def _ensure_eval_datasets(self) -> List[TextPromptDataset]:
        if self._eval_datasets is None:
            self._eval_datasets = [
                TextPromptDataset(file_path=domain["eval_data_path"] or domain["data_path"]) for domain in self.domains
            ]
        return self._eval_datasets

    def get_eval_samples(self, batch_size: int) -> Sample:
        """One deterministic eval Sample of up to ``batch_size`` prompts.

        The budget is split evenly across domains (remainder to the earlier
        ones) and the per-domain slices are concatenated, so every domain is
        represented in a single eval pass — the trainer chunks the returned
        Sample itself. ``batch_size <= 0`` returns an empty batch, mirroring
        :meth:`MultimodalRLDataSource.get_eval_samples`.
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            return self._domain_examples_to_batch(self.domains[0]["name"], [], tag="eval")
        eval_datasets = self._ensure_eval_datasets()
        per_domain, remainder = divmod(batch_size, len(self.domains))
        batches: List[Sample] = []
        for i, (domain, ds) in enumerate(zip(self.domains, eval_datasets)):
            limit = min(per_domain + int(i < remainder), len(ds))
            if limit <= 0:
                continue
            examples = [
                normalize_prompt_example(ds.get_prompt_example(idx), default_prompt_id=f"eval:{idx}")
                for idx in range(limit)
            ]
            batches.append(self._domain_examples_to_batch(domain["name"], examples, tag="eval"))
        if not batches:
            return self._domain_examples_to_batch(self.domains[0]["name"], [], tag="eval")
        if len(batches) == 1:
            return batches[0]
        return Sample.concat(batches)


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

    def get_samples(self, batch_size: int) -> Sample:
        """Get next batch of prompts."""
        prompts = []
        for _ in range(batch_size):
            prompts.append(self.prompts[self._index % len(self.prompts)])
            self._index += 1
        return _input_sample(
            {"text": Texts(texts=prompts)},
            sample_ids=[f"prompt:{i}:sample:0" for i in range(len(prompts))],
        )

    def _prompts_to_inputs(self, prompts: List[str], *, offset: int = 0) -> Sample:
        return _input_sample(
            {"text": Texts(texts=prompts)},
            sample_ids=[f"prompt:{offset + i}:sample:0" for i in range(len(prompts))],
        )

    def iter_eval_batches(
        self,
        batch_size: int,
        *,
        eval_num_prompts: int = -1,
    ) -> Iterator[Sample]:
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

    def get_eval_samples(self, batch_size: int) -> Sample:
        """Return the first eval batch (BC shim over :meth:`iter_eval_batches`)."""
        batch_size = int(batch_size)
        if batch_size <= 0:
            return self._prompts_to_inputs([])
        return next(
            self.iter_eval_batches(batch_size),
            self._prompts_to_inputs([]),
        )
