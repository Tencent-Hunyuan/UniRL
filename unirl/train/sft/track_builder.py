"""Worker-side supervised Part builders for the SFT domain."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, cast

import torch

from unirl.data.sft import message_content_image_uris
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.models.types.codec import EncodeStage
from unirl.models.types.conversations import tokenize_agent_target
from unirl.types.conditions import ImageLatentCondition
from unirl.types.media import MediaRef, MediaRefs
from unirl.types.primitives import Images, Texts, Video, Videos
from unirl.types.sample import Part
from unirl.types.segments.latent import make_image_segment, make_video_segment
from unirl.types.segments.text import TextSegment
from unirl.utils.video import load_video

logger = logging.getLogger(__name__)

Record = Dict[str, Any]
_TENSOR_CACHE_SCHEMA_VERSION = 1


class _VideoSFTPipeline(Protocol):
    def build_conditions(self, texts: Texts, *, guidance_scale: float) -> Any: ...

    def build_video_encoder(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> EncodeStage[Videos, ImageLatentCondition]: ...


def _require_local_uri(uri: str, *, context: str) -> str:
    """Reject remote media URIs — builders only read local/shared paths."""
    if uri.startswith(("http://", "https://", "s3://", "gs://")):
        raise NotImplementedError(
            f"{context}: remote media URIs are not supported yet ({uri!r}); "
            "download to local/shared storage and reference the path."
        )
    return uri


def _cache_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prefetch_batch_key(records: Sequence[Record]) -> str:
    return _cache_key({"records": records})


class _TensorDiskCache:
    """Small atomic torch-object cache namespaced by an explicit model fingerprint."""

    def __init__(self, root: str, *, fingerprint: str, kind: str, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError(f"_TensorDiskCache: max_entries must be >= 1; got {max_entries!r}")
        namespace = hashlib.sha256(f"{_TENSOR_CACHE_SCHEMA_VERSION}:{fingerprint}".encode()).hexdigest()[:20]
        self.directory = Path(root).expanduser().resolve() / namespace / kind
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self._writes = 0
        self._known_entries = {path.stem for path in self.directory.glob("*.pt")}
        if len(self._known_entries) > self.max_entries:
            self._evict()

    def get(self, key: str) -> Optional[Any]:
        path = self.directory / f"{key}.pt"
        if not path.is_file():
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            return None

    def put(self, key: str, value: Any) -> None:
        path = self.directory / f"{key}.pt"
        if path.exists():
            self._known_entries.add(key)
            return
        temp = self.directory / f".{key}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            torch.save(value, temp)
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
        self._writes += 1
        self._known_entries.add(key)
        if len(self._known_entries) > self.max_entries or self._writes % 64 == 0:
            self._evict()

    def _evict(self) -> None:
        entries = []
        for path in self.directory.glob("*.pt"):
            try:
                entries.append((path.stat().st_mtime_ns, path))
            except FileNotFoundError:
                pass
        if len(entries) <= self.max_entries:
            return
        entries.sort(key=lambda item: item[0])
        for _, path in entries[: len(entries) - self.max_entries]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._known_entries = {path.stem for _, path in entries[-self.max_entries :] if path.exists()}


def _media_stat_fingerprint(uri: str) -> Dict[str, Any]:
    path = _require_local_uri(uri, context="DiffusionSupervisedTrackBuilder")
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _encoder_cache_is_safe(modules: Sequence[Any]) -> bool:
    has_parameters = False
    for module in modules:
        if (
            module is None
            or getattr(module, "training", None) is not False
            or not callable(getattr(module, "parameters", None))
        ):
            return False
        for parameter in module.parameters():
            has_parameters = True
            if parameter.requires_grad:
                return False
    return has_parameters


def _load_pil_image(uri: str):
    """Load one local image as RGB PIL (worker-side; driver never touches pixels)."""
    from PIL import Image as PILImage

    with PILImage.open(_require_local_uri(uri, context="SupervisedTrackBuilder")) as image:
        return image.convert("RGB")


def _load_pil_images(uris: Sequence[Optional[str]], *, max_workers: int) -> List[Optional[Any]]:
    """Load local images concurrently while preserving input order and ``None`` rows."""
    present = [(index, uri) for index, uri in enumerate(uris) if uri is not None]
    images: List[Optional[Any]] = [None] * len(uris)
    if not present:
        return images
    worker_count = min(max_workers, len(present))
    if worker_count == 1:
        loaded = [_load_pil_image(uri) for _, uri in present]
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="sft-image") as executor:
            loaded = list(executor.map(_load_pil_image, [uri for _, uri in present]))
    for (index, _), image in zip(present, loaded):
        images[index] = image
    return images


def _media_uris(record: Record, *, role: str, modality: Optional[str] = None) -> List[str]:
    """URIs of normalized media refs matching ``role`` and ``modality``."""
    return [
        ref.uri
        for ref in record.get("media_refs", []) or []
        if isinstance(ref, MediaRef) and ref.role == role and (modality is None or ref.modality == modality)
    ]


def _sample_ids(records: Sequence[Record]) -> List[str]:
    return [str(r.get("sample_id", f"sft:{i}")) for i, r in enumerate(records)]


def _pad_flags(records: Sequence[Record]) -> List[bool]:
    return [bool(r.get("_eval_pad", False)) for r in records]


class SupervisedTrackBuilder(Remote):
    """Worker-side interface for converting normalized records into Parts."""

    def build(self, records: List[Record]) -> Part:
        raise NotImplementedError


class ARSupervisedTrackBuilder(SupervisedTrackBuilder):
    """Dataset records → AR ``Part`` (LLM / VLM / agent); the agent chat-stage contract is in ``README.md``."""

    def __init__(
        self,
        *,
        pipeline: Any,
        chat_stage_attr: str = "chat_template",
        max_response_length: int = 4096,
        append_eos: bool = True,
        image_load_workers: int = 4,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self._chat_stage = getattr(pipeline, chat_stage_attr, None)
        if self._chat_stage is None or not callable(getattr(self._chat_stage, "embed", None)):
            raise ValueError(
                f"ARSupervisedTrackBuilder: pipeline.{chat_stage_attr} is missing or has no .embed(); "
                f"point chat_stage_attr at the pipeline's chat-template stage."
            )
        tokenizer = getattr(pipeline.bundle, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("ARSupervisedTrackBuilder: pipeline.bundle has no tokenizer.")
        self._tokenizer = tokenizer
        if max_response_length < 1:
            raise ValueError(f"ARSupervisedTrackBuilder: max_response_length must be >= 1; got {max_response_length!r}")
        self.max_response_length = max_response_length
        self.append_eos = append_eos
        if image_load_workers < 1:
            raise ValueError(f"ARSupervisedTrackBuilder: image_load_workers must be >= 1; got {image_load_workers!r}")
        self.image_load_workers = image_load_workers
        self._prefetch_executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_future = None
        self._prefetch_key = None
        # VLM chat stages take (texts, images); text-only ones take (texts).
        self._embed_takes_images = "images" in inspect.signature(self._chat_stage.embed).parameters
        self._embed_sft_prompt = getattr(self._chat_stage, "embed_sft_prompt", None)
        self._warned_truncation = False

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def build(self, records: List[Record]) -> Part:
        """Tokenize + embed one shard of supervised records into a training Part."""
        if not records:
            raise ValueError("ARSupervisedTrackBuilder.build: empty record shard.")
        images, batch_ids = self._take_prefetched(records)
        with torch.no_grad():
            conditions = self._embed_prompts(records, images=images)
            tokens, loss_masks = self._tokenize_responses(records, batch_ids=batch_ids)
        segment = TextSegment.pack(tokens=tokens, loss_mask=loss_masks)
        part = Part(
            sample_ids=_sample_ids(records),
            conditions=conditions.to_dict(),
            segment=segment,
            metadata=[dict(record.get("metadata") or {}) for record in records],
        )
        if part.batch_size != len(records):
            raise RuntimeError(
                f"ARSupervisedTrackBuilder.build: built {part.batch_size} rows from {len(records)} "
                "records — token accounting is broken."
            )
        return part

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def prefetch(self, records: List[Record]) -> None:
        if not records:
            raise ValueError("ARSupervisedTrackBuilder.prefetch: empty record shard.")
        if self._prefetch_future is not None:
            raise RuntimeError("ARSupervisedTrackBuilder.prefetch: previous prefetch was not consumed.")
        if self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sft-prefetch")
        self._prefetch_key = _prefetch_batch_key(records)
        self._prefetch_future = self._prefetch_executor.submit(self._prepare_cpu_inputs, tuple(records))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _take_prefetched(
        self,
        records: Sequence[Record],
    ) -> Tuple[Optional[List[Optional[Any]]], Optional[List[List[int]]]]:
        if self._prefetch_future is None:
            return None, None
        expected = _prefetch_batch_key(records)
        if expected != self._prefetch_key:
            # Avoid using the tokenizer concurrently when eval runs between the
            # train step that launched this prefetch and the step that consumes it.
            self._prefetch_future.result()
            return None, None
        future = self._prefetch_future
        self._prefetch_future = None
        self._prefetch_key = None
        return future.result()

    def _prepare_cpu_inputs(
        self,
        records: Sequence[Record],
    ) -> Tuple[Optional[List[Optional[Any]]], Optional[List[List[int]]]]:
        if any("messages" in record for record in records):
            return None, None
        images = self._load_prompt_images(records) if self._embed_takes_images else None
        return images, self._batch_token_ids(records)

    def _load_prompt_images(self, records: Sequence[Record]) -> List[Optional[Any]]:
        image_uris: List[Optional[str]] = []
        for record in records:
            uris = _media_uris(record, role="condition", modality="image")
            if len(uris) > 1:
                raise ValueError(
                    f"ARSupervisedTrackBuilder: at most one role='condition' image per record "
                    f"(sample {record.get('sample_id')!r} has {len(uris)})."
                )
            image_uris.append(uris[0] if uris else None)
        return _load_pil_images(image_uris, max_workers=self.image_load_workers)

    def _embed_prompts(
        self,
        records: Sequence[Record],
        *,
        images: Optional[List[Optional[Any]]] = None,
    ) -> Any:
        agent_flags = ["messages" in record for record in records]
        if any(agent_flags):
            if not all(agent_flags):
                raise ValueError("ARSupervisedTrackBuilder: a batch may not mix prompt/response and agent records.")
            if any(record.get("media_refs") for record in records):
                raise ValueError(
                    "ARSupervisedTrackBuilder: agent records carry images inside message content parts, not media_refs."
                )
            embed_messages = getattr(self._chat_stage, "embed_messages", None)
            if not callable(embed_messages):
                raise ValueError(
                    "ARSupervisedTrackBuilder: this pipeline's chat stage does not support OpenAI-style messages."
                )
            histories = [self._load_history_images(r) for r in records]
            tools = [r.get("tools") for r in records]
            return embed_messages(histories, tools=tools)

        texts = Texts(texts=[str(r["prompt"]) for r in records])
        # The data layer normalizes every manifest media entry to MediaRef, so no dict form here.
        prompt_media_rows: List[List[MediaRef]] = [
            [ref for ref in (r.get("media_refs") or []) if isinstance(ref, MediaRef) and ref.role == "prompt"]
            for r in records
        ]
        if any(prompt_media_rows):
            if not callable(self._embed_sft_prompt):
                raise ValueError(
                    "ARSupervisedTrackBuilder: this pipeline's chat stage does not support "
                    "role='prompt' media through embed_sft_prompt(texts, media_refs)."
                )
            return self._embed_sft_prompt(
                texts=texts,
                media_refs=MediaRefs.from_rows(prompt_media_rows),
            )

        if not self._embed_takes_images:
            unconsumed = [r.get("sample_id") for r in records if r.get("media_refs")]
            if unconsumed:
                raise ValueError(
                    f"ARSupervisedTrackBuilder: records {unconsumed[:3]} carry media_refs that this chat stage "
                    "cannot consume — prompt media must use role='prompt' (see unirl/data/sft.py row shapes). "
                    "Embedding them as text-only would train the model without the media, undetectably."
                )
            return self._chat_stage.embed(texts)
        if images is None:
            images = self._load_prompt_images(records)
        if all(img is None for img in images):
            return self._chat_stage.embed(texts, None)
        return self._chat_stage.embed(texts, images)

    def _load_history_images(self, record: Record) -> List[Dict[str, Any]]:
        """One record's prompt history with image-part URIs replaced by loaded PILs."""
        history = record["messages"][:-1]
        if not any(message_content_image_uris(m) for m in history):
            return history
        if not getattr(self._chat_stage, "supports_message_images", False):
            raise ValueError(
                f"ARSupervisedTrackBuilder: record {record.get('sample_id')!r} carries image parts in its "
                "message history, but this pipeline's chat stage does not declare supports_message_images."
            )
        loaded: List[Dict[str, Any]] = []
        for message in history:
            content = message.get("content")
            if not isinstance(content, list):
                loaded.append(message)
                continue
            parts = [
                {"type": "image", "image": _load_pil_image(part["image"])}
                if isinstance(part, dict) and part.get("type") == "image"
                else part
                for part in content
            ]
            loaded.append({**message, "content": parts})
        return loaded

    def _batch_token_ids(self, records: Sequence[Record]) -> List[List[int]]:
        responses: List[str] = []
        for record in records:
            response = record.get("response")
            if not isinstance(response, str) or not response:
                raise ValueError(
                    f"ARSupervisedTrackBuilder: record {record.get('sample_id')!r} has no non-empty 'response' — "
                    "AR SFT manifests must carry the target text."
                )
            responses.append(response)
        encoded = self._tokenizer(
            responses,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        batch_ids = encoded["input_ids"]
        if len(batch_ids) != len(records):
            raise RuntimeError(
                f"ARSupervisedTrackBuilder: tokenizer returned {len(batch_ids)} rows for {len(records)} responses."
            )
        return [list(ids) for ids in batch_ids]

    def _tokenize_responses(
        self,
        records: Sequence[Record],
        *,
        batch_ids: Optional[List[List[int]]] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        device = getattr(self.pipeline.bundle, "device", torch.device("cpu"))
        eos_id = self._tokenizer.eos_token_id
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0] if eos_id else None
        if self.append_eos and eos_id is None:
            raise ValueError("ARSupervisedTrackBuilder: append_eos=True but the tokenizer has no eos_token_id.")
        if batch_ids is None:
            batch_ids = (
                [self._tokenize_agent_target(record) for record in records]
                if any("messages" in record for record in records)
                else self._batch_token_ids(records)
            )

        tokens: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        truncated = 0
        for r, is_pad, ids in zip(records, _pad_flags(records), batch_ids):
            if not ids:
                raise ValueError(
                    f"ARSupervisedTrackBuilder: target of record {r.get('sample_id')!r} tokenized to zero "
                    "tokens — a sample with no supervision would poison the loss denominator."
                )
            needs_eos = self.append_eos and ids[-1] != eos_id
            budget = self.max_response_length - (1 if needs_eos else 0)
            if len(ids) > budget:
                if "messages" in r:
                    raise ValueError(
                        f"ARSupervisedTrackBuilder: agent target of record {r.get('sample_id')!r} exceeds "
                        f"max_response_length={self.max_response_length}; filter overlong agent targets "
                        "during manifest preparation instead of truncating a structured assistant turn."
                    )
                ids = ids[:budget]
                truncated += 1
                if self.append_eos and not needs_eos:
                    ids[-1] = eos_id
            if needs_eos:
                ids = ids + [eos_id]
            tokens.append(torch.tensor(ids, dtype=torch.long, device=device))
            fill = 0.0 if is_pad else 1.0
            masks.append(torch.full((len(ids),), fill, dtype=torch.float32, device=device))
        if truncated and not self._warned_truncation:
            self._warned_truncation = True
            logger.warning(
                "ARSupervisedTrackBuilder: %d/%d responses truncated to max_response_length=%d "
                "(append_eos=%s). This warning is emitted once.",
                truncated,
                len(records),
                self.max_response_length,
                self.append_eos,
            )
        return tokens, masks

    def _tokenize_agent_target(self, record: Record) -> List[int]:
        """Render one final assistant turn and return only its supervised suffix."""
        stage_tokenize = getattr(self._chat_stage, "tokenize_agent_target", None)
        if callable(stage_tokenize):
            return [int(t) for t in stage_tokenize(record)]
        return tokenize_agent_target(
            record,
            tokenizer=self._tokenizer,
            enable_thinking=bool(getattr(self._chat_stage, "enable_thinking", False)),
        )


class DiffusionSupervisedTrackBuilder(SupervisedTrackBuilder):
    """Dataset records → diffusion ``Part`` with an x0-only segment."""

    def __init__(
        self,
        *,
        pipeline: Any,
        height: int = 512,
        width: int = 512,
        encode_stage_attr: str = "vae_encode",
        guidance_scale: float = 1.0,
        resolution_align: int = 16,
        image_load_workers: int = 4,
        cache_dir: Optional[str] = None,
        cache_fingerprint: Optional[str] = None,
        cache_text_conditions: bool = False,
        cache_vae_latents: bool = False,
        cache_max_entries: int = 4096,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        if image_load_workers < 1:
            raise ValueError(
                f"DiffusionSupervisedTrackBuilder: image_load_workers must be >= 1; got {image_load_workers!r}"
            )
        self.image_load_workers = image_load_workers
        cache_requested = cache_text_conditions or cache_vae_latents
        if cache_requested and (not cache_dir or not cache_fingerprint):
            raise ValueError(
                "DiffusionSupervisedTrackBuilder: cache_dir and cache_fingerprint are required "
                "when an encoder cache is enabled."
            )
        self._validate_encoder_cache_targets(cache_text_conditions, cache_vae_latents)
        self._text_cache = (
            _TensorDiskCache(
                cache_dir,
                fingerprint=cache_fingerprint,
                kind="text-conditions",
                max_entries=cache_max_entries,
            )
            if cache_text_conditions
            else None
        )
        self._vae_cache = (
            _TensorDiskCache(
                cache_dir,
                fingerprint=cache_fingerprint,
                kind="vae-latents",
                max_entries=cache_max_entries,
            )
            if cache_vae_latents
            else None
        )
        align = resolution_align
        if self.height % align or self.width % align:
            raise ValueError(
                f"DiffusionSupervisedTrackBuilder: height/width ({self.height}x{self.width}) must be "
                f"divisible by {align} (VAE downsample × transformer patch size)."
            )
        self._encode = getattr(pipeline, encode_stage_attr, None)
        if self._encode is None or not callable(getattr(self._encode, "encode", None)):
            raise ValueError(
                f"DiffusionSupervisedTrackBuilder: pipeline.{encode_stage_attr} is missing or has no "
                f".encode() — this model family needs a VAE encode stage (see the add-model-bundle "
                f"skill, checklist item 10; WAN21ImageLatentEncodeStage is the template)."
            )
        build_conditions = getattr(pipeline, "build_conditions", None)
        if not callable(build_conditions):
            raise ValueError(
                "DiffusionSupervisedTrackBuilder: pipeline has no build_conditions(texts, ...) — "
                "add one (every diffusion pipeline exposes it) so SFT encodes prompts exactly "
                "like rollout does."
            )
        self._conditions_kwargs: Dict[str, Any] = {"guidance_scale": self.guidance_scale}
        if "image_shape" in inspect.signature(build_conditions).parameters:
            self._conditions_kwargs["image_shape"] = (self.height, self.width)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def build(self, records: List[Record]) -> Part:
        """Encode one shard of (prompt, target image) records into a training Part."""
        if not records:
            raise ValueError("DiffusionSupervisedTrackBuilder.build: empty record shard.")
        with torch.no_grad():
            conditions = self._build_conditions(records)
            latents = self._encode_latents(records)
        if latents.shape[0] != len(records):
            raise RuntimeError(
                f"DiffusionSupervisedTrackBuilder.build: encoded {latents.shape[0]} latents "
                f"from {len(records)} records."
            )
        pad = torch.tensor([0.0 if p else 1.0 for p in _pad_flags(records)], dtype=torch.float32)
        segment = make_image_segment(
            latents=latents.unsqueeze(1),  # [B, 1, ...] — clean x0 at the last (only) position
            loss_mask=pad.to(latents.device),
        )
        return Part(
            sample_ids=_sample_ids(records),
            conditions=conditions.to_dict(),
            segment=segment,
            metadata=[dict(record.get("metadata") or {}) for record in records],
        )

    def _validate_encoder_cache_targets(
        self,
        cache_text_conditions: bool,
        cache_vae_latents: bool,
    ) -> None:
        bundle = self.pipeline.bundle
        if cache_text_conditions:
            text_modules = []
            for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                module = getattr(bundle, name, None)
                if module is not None:
                    text_modules.append(module)
            if not _encoder_cache_is_safe(text_modules):
                raise ValueError(
                    "DiffusionSupervisedTrackBuilder: cache_text_conditions requires non-empty, "
                    "frozen text encoders in eval mode."
                )
        if cache_vae_latents:
            vae = getattr(bundle, "vae", None)
            if not _encoder_cache_is_safe([vae]):
                raise ValueError(
                    "DiffusionSupervisedTrackBuilder: cache_vae_latents requires a frozen VAE in eval mode."
                )

    def _text_cache_key(self, record: Record) -> str:
        return _cache_key(
            {
                "prompt": str(record["prompt"]),
                "conditions": self._conditions_kwargs,
            }
        )

    def _vae_cache_key(self, uri: str) -> str:
        return _cache_key(
            {
                "media": _media_stat_fingerprint(uri),
                "height": self.height,
                "width": self.width,
                "resize": "pil-bicubic-stretch-v1",
            }
        )

    def _build_conditions(self, records: Sequence[Record]) -> Any:
        if self._text_cache is None:
            texts = Texts(texts=[str(record["prompt"]) for record in records])
            return self.pipeline.build_conditions(texts, **self._conditions_kwargs)

        keys = [self._text_cache_key(record) for record in records]
        items: List[Optional[Any]] = [self._text_cache.get(key) for key in keys]
        missing = [index for index, item in enumerate(items) if item is None]
        if missing:
            texts = Texts(texts=[str(records[index]["prompt"]) for index in missing])
            encoded = self.pipeline.build_conditions(texts, **self._conditions_kwargs)
            for encoded_index, record_index in enumerate(missing):
                item = encoded.slice(encoded_index, encoded_index + 1)
                self._text_cache.put(keys[record_index], item.to_device("cpu"))
                items[record_index] = item
        concrete = cast(List[Any], items)
        device = self.pipeline.bundle.device
        on_device = [item.to_device(device) for item in concrete]
        return type(on_device[0]).concat(on_device)

    def _encode_latents(self, records: Sequence[Record]) -> torch.Tensor:
        target_uris = self._target_image_uris(records)
        if self._vae_cache is None:
            pixels = self._load_target_pixels(target_uris)
            return self._encode.encode(Images.from_dense(pixels)).latents

        keys = [self._vae_cache_key(uri) for uri in target_uris]
        items: List[Optional[torch.Tensor]] = [self._vae_cache.get(key) for key in keys]
        missing = [index for index, item in enumerate(items) if item is None]
        if missing:
            pixels = self._load_target_pixels([target_uris[index] for index in missing])
            encoded = self._encode.encode(Images.from_dense(pixels)).latents
            for encoded_index, record_index in enumerate(missing):
                item = encoded[encoded_index].detach()
                self._vae_cache.put(keys[record_index], item.cpu())
                items[record_index] = item
        concrete = cast(List[torch.Tensor], items)
        device = self.pipeline.bundle.device
        return torch.stack([item.to(device) for item in concrete], dim=0)

    def _target_image_uris(self, records: Sequence[Record]) -> List[str]:
        target_uris: List[str] = []
        for r in records:
            uris = _media_uris(r, role="target", modality="image")
            if len(uris) != 1:
                raise ValueError(
                    f"DiffusionSupervisedTrackBuilder: record {r.get('sample_id')!r} must carry exactly one "
                    f"role='target' image media ref (got {len(uris)}) — diffusion SFT manifests are "
                    "(prompt, target image) pairs."
                )
            target_uris.append(uris[0])
        return target_uris

    def _load_target_pixels(self, target_uris: Sequence[str]) -> torch.Tensor:
        """Load + resize target images → ``[B, 3, H, W]`` fp32 in ``[0, 1]``."""
        import numpy as np
        from PIL import Image as PILImage

        images = _load_pil_images(target_uris, max_workers=self.image_load_workers)
        rows: List[torch.Tensor] = []
        for img in images:
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height), PILImage.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
            rows.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        return torch.stack(rows, dim=0)


class VideoDiffusionSupervisedTrackBuilder(SupervisedTrackBuilder):
    """Dataset records → video-diffusion ``Part`` with a clean x0 latent."""

    def __init__(
        self,
        *,
        pipeline: _VideoSFTPipeline,
        num_frames: int,
        height: int,
        width: int,
        guidance_scale: float = 1.0,
        max_decode_frames: int = 256,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        num_frames = int(num_frames)
        self._encode = pipeline.build_video_encoder(
            num_frames=num_frames,
            height=height,
            width=width,
        )
        self._conditions_kwargs: Dict[str, Any] = {"guidance_scale": float(guidance_scale)}
        self._max_decode_frames = int(max_decode_frames)
        if self._max_decode_frames < num_frames:
            raise ValueError(
                "VideoDiffusionSupervisedTrackBuilder: max_decode_frames must be >= num_frames "
                f"({num_frames}), got {self._max_decode_frames}."
            )

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def build(self, records: List[Record]) -> Part:
        if not records:
            raise ValueError("VideoDiffusionSupervisedTrackBuilder.build: empty record shard.")
        with torch.no_grad():
            texts = Texts(texts=[str(r["prompt"]) for r in records])
            conditions = self.pipeline.build_conditions(texts, **self._conditions_kwargs)
            videos = self._load_target_videos(records)
            latents = self._encode.encode(videos).latents
        if latents.shape[0] != len(records):
            raise RuntimeError(
                f"VideoDiffusionSupervisedTrackBuilder.build: encoded {latents.shape[0]} latents "
                f"from {len(records)} records."
            )
        loss_mask = torch.tensor([not is_pad for is_pad in _pad_flags(records)], dtype=torch.float32)
        segment = make_video_segment(
            latents=latents.unsqueeze(1),
            loss_mask=loss_mask.to(latents.device),
        )
        return Part(
            sample_ids=_sample_ids(records),
            conditions=conditions.to_dict(),
            segment=segment,
            metadata=[dict(record.get("metadata") or {}) for record in records],
        )

    def _load_target_videos(self, records: Sequence[Record]) -> Videos:
        rows: List[Video] = []
        for r in records:
            uris = _media_uris(r, role="target", modality="video")
            if len(uris) != 1:
                raise ValueError(
                    f"VideoDiffusionSupervisedTrackBuilder: record {r.get('sample_id')!r} must carry exactly "
                    f"one role='target' video media ref (got {len(uris)})."
                )
            rows.append(Video(frames=load_video(uris[0], max_frames=self._max_decode_frames)))
        return Videos.from_list(rows)


__all__ = [
    "ARSupervisedTrackBuilder",
    "DiffusionSupervisedTrackBuilder",
    "SupervisedTrackBuilder",
    "VideoDiffusionSupervisedTrackBuilder",
]
