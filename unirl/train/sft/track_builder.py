"""Worker-side supervised track builders for the SFT domain.

In the RL loop the rollout engine is the data producer: it turns a request
into a ``RolloutTrack`` (conditions + segment) that ``TrainStack.train_track``
consumes. SFT swaps that producer for a dataset-backed one and keeps the whole
consumer side (stack / algorithm / backend) unchanged — these classes are the
swap, mirroring :class:`~unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine`'s
shape (a ``Remote`` sibling holding the trainer-injected ``pipeline``, one
``DP_SCATTER`` method, ``torch.no_grad()`` inside).

Per-model logic stays in the model packages: prompts go through the bundle's
own chat-template / text-embed stages, targets through the bundle's VAE encode
stage — a supervised track is indistinguishable from a rollout-built one to
``ARStage.replay`` / ``predict_noise_at_step``. A new modality plugs in as
(bundle stages) + (a track builder here) + (a loss in ``unirl/algorithms``)
only when its record→(conditions, segment) mapping or loss math is genuinely
new — never as a per-model SFT file.

Eval padding contract: the SFT trainer pads the final eval batch up to the DP
width with ``{"_eval_pad": True}`` copies so ``DP_SCATTER`` divisibility holds
without dropping tail samples; builders zero those rows' ``loss_mask`` and the
losses count them as weight 0 — full-set eval stays exact.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from unirl.data.sft import tokenize_agent_target
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.segments.latent import make_image_segment
from unirl.types.segments.text import TextSegment

logger = logging.getLogger(__name__)

Record = Dict[str, Any]


def _cache_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _prefetch_batch_key(records: Sequence[Record]) -> str:
    return _cache_key({"records": list(records)})


class _TensorDiskCache:
    """Small atomic torch-object cache namespaced by an explicit model fingerprint."""

    def __init__(self, root: str, *, fingerprint: str, kind: str, max_entries: int) -> None:
        namespace = hashlib.sha256(fingerprint.encode()).hexdigest()[:20]
        self.directory = Path(root).expanduser().resolve() / namespace / kind
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_entries = max(1, int(max_entries))
        self._writes = 0
        self._known_entries = {path.stem for path in self.directory.glob("*.pt")}

    def get(self, key: str) -> Optional[Any]:
        path = self.directory / f"{key}.pt"
        if not path.is_file():
            return None
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning("Ignoring unreadable SFT cache entry %s: %s", path, exc)
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
    stat = os.stat(uri)
    return {
        "path": os.path.abspath(uri),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _all_parameters_frozen(modules: Sequence[Any]) -> bool:
    parameters = []
    for module in modules:
        if module is None or not callable(getattr(module, "parameters", None)):
            return False
        parameters.extend(module.parameters())
    return bool(parameters) and not any(parameter.requires_grad for parameter in parameters)


def _load_pil_image(uri: str):
    """Load one local image as RGB PIL (worker-side; driver never touches pixels)."""
    from PIL import Image as PILImage

    if uri.startswith(("http://", "https://", "s3://", "gs://")):
        raise NotImplementedError(
            f"SupervisedTrackBuilder: remote media URIs are not supported yet ({uri!r}); "
            "download to local/shared storage and reference the path."
        )
    with PILImage.open(uri) as image:
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


def _media_uris(record: Record, *, role: str) -> List[str]:
    """URIs of the record's media refs with the given role (dataclass or dict form)."""
    uris: List[str] = []
    for ref in record.get("media_refs", []) or []:
        ref_role = getattr(ref, "role", None) if not isinstance(ref, dict) else ref.get("role")
        ref_uri = getattr(ref, "uri", None) if not isinstance(ref, dict) else ref.get("uri")
        if ref_role == role and ref_uri:
            uris.append(str(ref_uri))
    return uris


def _sample_ids(records: Sequence[Record]) -> List[str]:
    return [str(r.get("sample_id", f"sft:{i}")) for i, r in enumerate(records)]


def _pad_flags(records: Sequence[Record]) -> List[bool]:
    return [bool(r.get("_eval_pad", False)) for r in records]


class SupervisedTrackBuilder(Remote):
    """Worker-side interface for converting normalized records into tracks."""

    def build(self, records: List[Record]) -> RolloutTrack:
        raise NotImplementedError

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def prefetch(self, records: List[Record]) -> None:
        """Optionally start worker-local CPU preparation for a future batch."""
        del records


class ARSupervisedTrackBuilder(SupervisedTrackBuilder):
    """Dataset records → AR ``RolloutTrack`` (LLM + VLM), via the bundle's stages.

    Prompt side: the pipeline's chat-template stage (``add_generation_prompt``
    baked in, byte-identical to what rollout engines render — the SFT model is
    trained on exactly the token sequence inference will see). Target side:
    ``bundle.tokenizer`` on the raw response + EOS, matching the rollout
    convention that the stop token is the last supervised token.

    Args:
        pipeline: trainer-injected sibling (``Qwen3Pipeline`` / ``QwenVLPipeline`` /
            any pipeline exposing a chat stage + tokenizer-carrying bundle).
        chat_stage_attr: chat/template stage attribute on the pipeline.
        max_response_length: hard token cap per response (uncapped targets OOM'd
            other frameworks); legacy responses are truncated with EOS kept,
            while agent targets must be filtered before training.
        append_eos: append ``tokenizer.eos_token_id`` to every response —
            disable only for models whose template ends turns with a non-EOS
            token that the dataset already includes.
        image_load_workers: bounded worker-side image I/O concurrency for VLM
            records. Set to 1 for serial loading.
        prefetch_cpu: overlap next-batch tokenization/image I/O with the current
            GPU train step. The trainer must also set ``prefetch_next_batch``.
    """

    def __init__(
        self,
        *,
        pipeline: Any,
        chat_stage_attr: str = "chat_template",
        max_response_length: int = 4096,
        append_eos: bool = True,
        image_load_workers: int = 4,
        prefetch_cpu: bool = False,
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
        if int(image_load_workers) < 1:
            raise ValueError(f"ARSupervisedTrackBuilder: image_load_workers must be >= 1; got {image_load_workers!r}")
        self.image_load_workers = int(image_load_workers)
        self.prefetch_cpu = bool(prefetch_cpu)
        self._prefetch_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="sft-prefetch") if self.prefetch_cpu else None
        )
        self._prefetch_future = None
        self._prefetch_key = None
        # VLM chat stages take (texts, images); text-only ones take (texts).
        self._embed_takes_images = "images" in inspect.signature(self._chat_stage.embed).parameters
        self._warned_truncation = False

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def build(self, records: List[Record]) -> RolloutTrack:
        """Tokenize + embed one shard of supervised records into a root track."""
        if not records:
            raise ValueError("ARSupervisedTrackBuilder.build: empty record shard.")
        prepared = self._take_prefetched(records)
        images, batch_ids = prepared if prepared is not None else (None, None)
        with torch.no_grad():
            conditions = self._embed_prompts(records, images=images)
            tokens, loss_masks = self._tokenize_responses(records, batch_ids=batch_ids)
        segment = TextSegment.pack(tokens=tokens, loss_mask=loss_masks)
        track = RolloutTrack(
            sample_ids=_sample_ids(records),
            parent_ids=None,
            parent_track=None,
            conditions=conditions.to_dict(),
            segment=segment,
        )
        if track.batch_size != len(records):
            raise RuntimeError(
                f"ARSupervisedTrackBuilder.build: built {track.batch_size} rows from {len(records)} "
                "records — token accounting is broken."
            )
        return track

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def prefetch(self, records: List[Record]) -> None:
        if not self.prefetch_cpu:
            return
        if not records:
            raise ValueError("ARSupervisedTrackBuilder.prefetch: empty record shard.")
        if self._prefetch_future is not None:
            raise RuntimeError("ARSupervisedTrackBuilder.prefetch: previous prefetch was not consumed.")
        self._prefetch_key = _prefetch_batch_key(records)
        assert self._prefetch_executor is not None
        self._prefetch_future = self._prefetch_executor.submit(self._prepare_cpu_inputs, tuple(records))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _take_prefetched(
        self,
        records: Sequence[Record],
    ) -> Optional[Tuple[List[Optional[Any]], List[List[int]]]]:
        if self._prefetch_future is None:
            return None
        expected = _prefetch_batch_key(records)
        if expected != self._prefetch_key:
            # Eval may run between the train step that started this prefetch and
            # the next train step that consumes it. Build eval normally while
            # leaving the pending train batch intact.
            return None
        future = self._prefetch_future
        self._prefetch_future = None
        self._prefetch_key = None
        return future.result()

    def _prepare_cpu_inputs(
        self,
        records: Sequence[Record],
    ) -> Tuple[List[Optional[Any]], List[List[int]]]:
        agent_flags = ["messages" in record for record in records]
        if any(agent_flags) and not all(agent_flags):
            raise ValueError("ARSupervisedTrackBuilder: a batch may not mix prompt/response and agent records.")
        images = (
            self._load_prompt_images(records)
            if self._embed_takes_images and not any(agent_flags)
            else [None] * len(records)
        )
        return images, self._batch_token_ids(records)

    def _load_prompt_images(self, records: Sequence[Record]) -> List[Optional[Any]]:
        image_uris: List[Optional[str]] = []
        for record in records:
            uris = _media_uris(record, role="condition")
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
        agent_flags = ["messages" in r for r in records]
        if any(agent_flags):
            if not all(agent_flags):
                raise ValueError("ARSupervisedTrackBuilder: a batch may not mix prompt/response and agent records.")
            if any(r.get("media_refs") for r in records):
                raise ValueError("ARSupervisedTrackBuilder: agent messages currently support text-only records.")
            embed_messages = getattr(self._chat_stage, "embed_messages", None)
            if not callable(embed_messages):
                raise ValueError(
                    "ARSupervisedTrackBuilder: this pipeline's chat stage does not support OpenAI-style messages."
                )
            histories = [r["messages"][:-1] for r in records]
            tools = [r.get("tools") for r in records]
            return embed_messages(histories, tools=tools)

        texts = Texts(texts=[str(r["prompt"]) for r in records])
        if not self._embed_takes_images:
            return self._chat_stage.embed(texts)
        if images is None:
            images = self._load_prompt_images(records)
        if all(img is None for img in images):
            return self._chat_stage.embed(texts, None)
        return self._chat_stage.embed(texts, images)

    def _batch_token_ids(self, records: Sequence[Record]) -> List[List[int]]:
        agent_flags = ["messages" in record for record in records]
        if any(agent_flags):
            if not all(agent_flags):
                raise ValueError("ARSupervisedTrackBuilder: a batch may not mix prompt/response and agent records.")
            return [self._tokenize_agent_target(record) for record in records]

        responses: List[str] = []
        for record in records:
            response = record.get("response")
            if not isinstance(response, str) or not response:
                raise ValueError(
                    f"ARSupervisedTrackBuilder: record {record.get('sample_id')!r} has no non-empty 'response' — "
                    "AR SFT manifests must carry the target text."
                )
            responses.append(response)
        tokenizer_kwargs: Dict[str, Any] = {"add_special_tokens": False}
        try:
            tokenizer_parameters = inspect.signature(self._tokenizer).parameters
        except (TypeError, ValueError):
            tokenizer_parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in tokenizer_parameters.values()
        )
        if accepts_kwargs or "padding" in tokenizer_parameters:
            tokenizer_kwargs["padding"] = False
        if accepts_kwargs or "truncation" in tokenizer_parameters:
            tokenizer_kwargs["truncation"] = False
        encoded = self._tokenizer(responses, **tokenizer_kwargs)
        batch_ids = encoded["input_ids"]
        if len(responses) == 1 and batch_ids and isinstance(batch_ids[0], int):
            batch_ids = [batch_ids]
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
            batch_ids = self._batch_token_ids(records)
        elif len(batch_ids) != len(records):
            raise RuntimeError(
                f"ARSupervisedTrackBuilder: received {len(batch_ids)} prefetched token rows for {len(records)} records."
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
            ids = list(ids)
            needs_eos = self.append_eos and (not ids or ids[-1] != eos_id)
            budget = self.max_response_length - (1 if needs_eos else 0)
            if len(ids) > budget:
                if "messages" in r:
                    raise ValueError(
                        f"ARSupervisedTrackBuilder: agent target of record {r.get('sample_id')!r} exceeds "
                        f"max_response_length={self.max_response_length}; filter overlong agent targets "
                        "during manifest preparation instead of truncating a structured assistant turn."
                    )
                ids = list(ids[:budget])
                truncated += 1
                if self.append_eos and not needs_eos and eos_id is not None:
                    ids[-1] = eos_id
            if needs_eos:
                ids = list(ids) + [eos_id]
            tokens.append(torch.tensor(ids, dtype=torch.long, device=device))
            # _eval_pad rows ride the forward but carry zero loss weight — the
            # trainer pads eval batches to the DP width with duplicates.
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
        return tokenize_agent_target(
            record,
            tokenizer=self._tokenizer,
            enable_thinking=bool(getattr(self._chat_stage, "enable_thinking", False)),
        )


class DiffusionSupervisedTrackBuilder(SupervisedTrackBuilder):
    """Dataset records → diffusion ``RolloutTrack`` with an x0-only segment.

    Prompt side: the pipeline's own ``build_conditions`` (the exact conditions
    ``diffuse``/``replay`` consume — CFG defaults included). Target side: the
    bundle's VAE encode stage (``pipeline.<encode_stage_attr>``), whose
    normalization is the strict inverse of the decode stage by construction.
    The clean latent lands at ``segment.latents[:, -1]`` — the slot
    :class:`~unirl.algorithms.FlowMatchSFT` (and DiffusionNFT) read.

    Args:
        height / width: target resolution; images are bicubic-resized. Must be
            divisible by ``resolution_align`` (latent patching constraint).
        encode_stage_attr: VAE encode stage attribute on the pipeline
            (``vae_encode``; add one per the add-model-bundle skill if the
            family lacks it).
        guidance_scale: forwarded to ``build_conditions``; keep 1.0 — SFT runs
            the pure conditional branch.
        image_load_workers: bounded target-image I/O concurrency. Set to 1 for
            serial loading.
        cache_dir / cache_fingerprint: persistent cache root and an explicit
            model/config revision. Both are required when either cache is on.
        cache_text_conditions: cache per-sample frozen text-encoder outputs.
        cache_vae_latents: cache deterministic frozen-VAE target latents.
        cache_max_entries: per-kind on-disk entry bound.
    """

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
        if int(image_load_workers) < 1:
            raise ValueError(
                f"DiffusionSupervisedTrackBuilder: image_load_workers must be >= 1; got {image_load_workers!r}"
            )
        self.image_load_workers = int(image_load_workers)
        cache_requested = bool(cache_text_conditions or cache_vae_latents)
        if cache_requested and (not cache_dir or not cache_fingerprint):
            raise ValueError(
                "DiffusionSupervisedTrackBuilder: cache_dir and cache_fingerprint are required "
                "when an encoder cache is enabled."
            )
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
        self._cache_guards_checked = False
        self._cache_stats = {
            "text_hits": 0,
            "text_misses": 0,
            "vae_hits": 0,
            "vae_misses": 0,
        }
        self._cache_builds = 0
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
    def build(self, records: List[Record]) -> RolloutTrack:
        """Encode one shard of (prompt, target image) records into a root track."""
        if not records:
            raise ValueError("DiffusionSupervisedTrackBuilder.build: empty record shard.")
        self._ensure_cache_guards()
        with torch.no_grad():
            conditions = self._build_conditions(records)
            latents = self._encode_latents(records)
        self._cache_builds += 1
        if (self._text_cache is not None or self._vae_cache is not None) and (
            self._cache_builds <= 3 or self._cache_builds % 50 == 0
        ):
            logger.info("SFT encoder cache stats after build %d: %s", self._cache_builds, self._cache_stats)
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
        return RolloutTrack(
            sample_ids=_sample_ids(records),
            parent_ids=None,
            parent_track=None,
            conditions=conditions.to_dict(),
            segment=segment,
        )

    def _ensure_cache_guards(self) -> None:
        if self._cache_guards_checked:
            return
        self._cache_guards_checked = True
        bundle = getattr(self.pipeline, "bundle", None)
        if self._text_cache is not None:
            text_modules = [
                getattr(bundle, name, None)
                for name in ("text_encoder", "text_encoder_2", "text_encoder_3")
                if getattr(bundle, name, None) is not None
            ]
            if not _all_parameters_frozen(text_modules):
                logger.warning(
                    "DiffusionSupervisedTrackBuilder: text-condition cache disabled because "
                    "text encoders are trainable or cannot be introspected."
                )
                self._text_cache = None
        if self._vae_cache is not None:
            vae = getattr(bundle, "vae", None)
            if not _all_parameters_frozen([vae]):
                logger.warning(
                    "DiffusionSupervisedTrackBuilder: VAE cache disabled because the VAE "
                    "is trainable or cannot be introspected."
                )
                self._vae_cache = None

    def _text_cache_key(self, record: Record) -> str:
        return _cache_key(
            {
                "manifest": record.get("_manifest_fingerprint"),
                "sample_id": record.get("sample_id"),
                "prompt": str(record["prompt"]),
                "conditions": self._conditions_kwargs,
            }
        )

    def _vae_cache_key(self, record: Record) -> str:
        uris = _media_uris(record, role="target")
        if len(uris) != 1:
            raise ValueError(
                f"DiffusionSupervisedTrackBuilder: record {record.get('sample_id')!r} must carry exactly one "
                f"role='target' image media ref (got {len(uris)})."
            )
        return _cache_key(
            {
                "manifest": record.get("_manifest_fingerprint"),
                "sample_id": record.get("sample_id"),
                "media": _media_stat_fingerprint(uris[0]),
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
        self._cache_stats["text_hits"] += len(records) - len(missing)
        self._cache_stats["text_misses"] += len(missing)
        if missing:
            texts = Texts(texts=[str(records[index]["prompt"]) for index in missing])
            encoded = self.pipeline.build_conditions(texts, **self._conditions_kwargs)
            if not all(callable(getattr(encoded, name, None)) for name in ("slice", "to_device", "concat")):
                raise TypeError(
                    "DiffusionSupervisedTrackBuilder: cached text conditions must be a Batch-like "
                    "object exposing slice/to_device/concat."
                )
            for encoded_index, record_index in enumerate(missing):
                item = encoded.slice(encoded_index, encoded_index + 1).to_device("cpu")
                self._text_cache.put(keys[record_index], item)
                items[record_index] = item
        concrete = [item for item in items if item is not None]
        if len(concrete) != len(records):
            raise RuntimeError("DiffusionSupervisedTrackBuilder: text cache did not resolve every record.")
        conditions = type(concrete[0]).concat(concrete)
        device = getattr(self.pipeline.bundle, "device", torch.device("cpu"))
        return conditions.to_device(device)

    def _encode_latents(self, records: Sequence[Record]) -> torch.Tensor:
        if self._vae_cache is None:
            pixels = self._load_target_pixels(records)
            return self._encode.encode(Images(pixels=pixels)).latents

        keys = [self._vae_cache_key(record) for record in records]
        items: List[Optional[torch.Tensor]] = [self._vae_cache.get(key) for key in keys]
        missing = [index for index, item in enumerate(items) if item is None]
        self._cache_stats["vae_hits"] += len(records) - len(missing)
        self._cache_stats["vae_misses"] += len(missing)
        if missing:
            pixels = self._load_target_pixels([records[index] for index in missing])
            encoded = self._encode.encode(Images(pixels=pixels)).latents
            for encoded_index, record_index in enumerate(missing):
                item = encoded[encoded_index].detach().cpu()
                self._vae_cache.put(keys[record_index], item)
                items[record_index] = item
        concrete = [item for item in items if isinstance(item, torch.Tensor)]
        if len(concrete) != len(records):
            raise RuntimeError("DiffusionSupervisedTrackBuilder: VAE cache did not resolve every record.")
        device = getattr(self.pipeline.bundle, "device", torch.device("cpu"))
        return torch.stack(concrete, dim=0).to(device)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_target_pixels(self, records: Sequence[Record]) -> torch.Tensor:
        """Load + resize target images → ``[B, 3, H, W]`` fp32 in ``[0, 1]``."""
        import numpy as np
        from PIL import Image as PILImage

        target_uris: List[str] = []
        for r in records:
            uris = _media_uris(r, role="target")
            if len(uris) != 1:
                raise ValueError(
                    f"DiffusionSupervisedTrackBuilder: record {r.get('sample_id')!r} must carry exactly one "
                    f"role='target' image media ref (got {len(uris)}) — diffusion SFT manifests are "
                    "(prompt, target image) pairs."
                )
            target_uris.append(uris[0])
        images = _load_pil_images(target_uris, max_workers=self.image_load_workers)
        rows: List[torch.Tensor] = []
        for img in images:
            assert img is not None
            if img.size != (self.width, self.height):
                img = img.resize((self.width, self.height), PILImage.BICUBIC)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
            rows.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        return torch.stack(rows, dim=0)


__all__ = [
    "ARSupervisedTrackBuilder",
    "DiffusionSupervisedTrackBuilder",
    "SupervisedTrackBuilder",
]
