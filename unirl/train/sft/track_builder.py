"""Worker-side supervised Part builders for the SFT domain."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import torch

from unirl.data.sft import tokenize_agent_target
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.models.types.codec import EncodeStage
from unirl.types.conditions import ImageLatentCondition
from unirl.types.media import MediaRef, MediaRefs
from unirl.types.primitives import Images, Texts, Video, Videos
from unirl.types.sample import Part
from unirl.types.segments.latent import make_image_segment, make_video_segment
from unirl.types.segments.text import TextSegment
from unirl.utils.video import load_video

logger = logging.getLogger(__name__)

Record = Dict[str, Any]


class _VideoSFTPipeline(Protocol):
    def build_conditions(self, texts: Texts, *, guidance_scale: float) -> Any: ...

    def build_video_encoder(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> EncodeStage[Videos, ImageLatentCondition]: ...


def _load_pil_image(uri: str):
    """Load one local image as RGB PIL (worker-side; driver never touches pixels)."""
    from PIL import Image as PILImage

    if uri.startswith(("http://", "https://", "s3://", "gs://")):
        raise NotImplementedError(
            f"SupervisedTrackBuilder: remote media URIs are not supported yet ({uri!r}); "
            "download to local/shared storage and reference the path."
        )
    return PILImage.open(uri).convert("RGB")


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
    """Dataset records → AR ``Part`` (LLM + VLM), via the bundle's stages."""

    def __init__(
        self,
        *,
        pipeline: Any,
        chat_stage_attr: str = "chat_template",
        max_response_length: int = 4096,
        append_eos: bool = True,
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
        self._embed_takes_images = "images" in inspect.signature(self._chat_stage.embed).parameters
        self._embed_sft_prompt = getattr(self._chat_stage, "embed_sft_prompt", None)
        self._warned_truncation = False

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def build(self, records: List[Record]) -> Part:
        """Tokenize + embed one shard of supervised records into a training Part."""
        if not records:
            raise ValueError("ARSupervisedTrackBuilder.build: empty record shard.")
        with torch.no_grad():
            conditions = self._embed_prompts(records)
            tokens, loss_masks = self._tokenize_responses(records)
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

    def _embed_prompts(self, records: Sequence[Record]) -> Any:
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
        images: List[Optional[Any]] = []
        for r in records:
            uris = _media_uris(r, role="condition", modality="image")
            if len(uris) > 1:
                raise ValueError(
                    f"ARSupervisedTrackBuilder: at most one role='condition' image per record "
                    f"(sample {r.get('sample_id')!r} has {len(uris)})."
                )
            images.append(_load_pil_image(uris[0]) if uris else None)
        if all(img is None for img in images):
            images = None  # type: ignore[assignment]
        return self._chat_stage.embed(texts, images)

    def _tokenize_responses(self, records: Sequence[Record]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        device = getattr(self.pipeline.bundle, "device", torch.device("cpu"))
        eos_id = self._tokenizer.eos_token_id
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0] if eos_id else None
        if self.append_eos and eos_id is None:
            raise ValueError("ARSupervisedTrackBuilder: append_eos=True but the tokenizer has no eos_token_id.")

        tokens: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        truncated = 0
        for r, is_pad in zip(records, _pad_flags(records)):
            if "messages" in r:
                ids = self._tokenize_agent_target(r)
            else:
                response = r.get("response")
                if not isinstance(response, str) or not response:
                    raise ValueError(
                        f"ARSupervisedTrackBuilder: record {r.get('sample_id')!r} has no non-empty 'response' — "
                        "AR SFT manifests must carry the target text."
                    )
                ids = self._tokenizer(response, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(
                    f"ARSupervisedTrackBuilder: target of record {r.get('sample_id')!r} tokenized to zero "
                    "tokens — a sample with no supervision would poison the loss denominator."
                )
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
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
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
            texts = Texts(texts=[str(r["prompt"]) for r in records])
            conditions = self.pipeline.build_conditions(texts, **self._conditions_kwargs)
            pixels = self._load_target_pixels(records)
            latents = self._encode.encode(Images.from_dense(pixels)).latents
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

    def _load_target_pixels(self, records: Sequence[Record]) -> torch.Tensor:
        """Load + resize target images → ``[B, 3, H, W]`` fp32 in ``[0, 1]``."""
        import numpy as np
        from PIL import Image as PILImage

        rows: List[torch.Tensor] = []
        for r in records:
            uris = _media_uris(r, role="target", modality="image")
            if len(uris) != 1:
                raise ValueError(
                    f"DiffusionSupervisedTrackBuilder: record {r.get('sample_id')!r} must carry exactly one "
                    f"role='target' image media ref (got {len(uris)}) — diffusion SFT manifests are "
                    "(prompt, target image) pairs."
                )
            img = _load_pil_image(uris[0])
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
