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

import inspect
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.segments.latent import make_image_segment
from unirl.types.segments.text import TextSegment

logger = logging.getLogger(__name__)

Record = Dict[str, Any]


def _load_pil_image(uri: str):
    """Load one local image as RGB PIL (worker-side; driver never touches pixels)."""
    from PIL import Image as PILImage

    if uri.startswith(("http://", "https://", "s3://", "gs://")):
        raise NotImplementedError(
            f"SupervisedTrackBuilder: remote media URIs are not supported yet ({uri!r}); "
            "download to local/shared storage and reference the path."
        )
    return PILImage.open(uri).convert("RGB")


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
            other frameworks); truncated responses keep their EOS and log once.
        append_eos: append ``tokenizer.eos_token_id`` to every response —
            disable only for models whose template ends turns with a non-EOS
            token that the dataset already includes.
    """

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
        # VLM chat stages take (texts, images); text-only ones take (texts).
        self._embed_takes_images = "images" in inspect.signature(self._chat_stage.embed).parameters
        self._warned_truncation = False

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def build(self, records: List[Record]) -> RolloutTrack:
        """Tokenize + embed one shard of supervised records into a root track."""
        if not records:
            raise ValueError("ARSupervisedTrackBuilder.build: empty record shard.")
        with torch.no_grad():
            conditions = self._embed_prompts(records)
            tokens, loss_masks = self._tokenize_responses(records)
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed_prompts(self, records: Sequence[Record]) -> Any:
        for r in records:
            if "messages" in r:
                raise NotImplementedError(
                    "ARSupervisedTrackBuilder: multi-turn 'messages' records are not supported yet — "
                    "use single-turn {'prompt', 'response'} rows (multi-turn interleaved masking "
                    "is a follow-up with its own template-consistency tests)."
                )
        texts = Texts(texts=[str(r["prompt"]) for r in records])
        if not self._embed_takes_images:
            return self._chat_stage.embed(texts)
        images: List[Optional[Any]] = []
        for r in records:
            uris = _media_uris(r, role="condition")
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
            response = r.get("response")
            if not isinstance(response, str) or not response:
                raise ValueError(
                    f"ARSupervisedTrackBuilder: record {r.get('sample_id')!r} has no non-empty 'response' — "
                    "AR SFT manifests must carry the target text."
                )
            ids = self._tokenizer(response, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(
                    f"ARSupervisedTrackBuilder: response of record {r.get('sample_id')!r} tokenized to zero "
                    "tokens — a sample with no supervision would poison the loss denominator."
                )
            budget = self.max_response_length - (1 if self.append_eos else 0)
            if len(ids) > budget:
                ids = ids[:budget]
                truncated += 1
            if self.append_eos:
                ids = list(ids) + [eos_id]
            tokens.append(torch.tensor(ids, dtype=torch.long, device=device))
            # _eval_pad rows ride the forward but carry zero loss weight — the
            # trainer pads eval batches to the DP width with duplicates.
            fill = 0.0 if is_pad else 1.0
            masks.append(torch.full((len(ids),), fill, dtype=torch.float32, device=device))
        if truncated and not self._warned_truncation:
            self._warned_truncation = True
            logger.warning(
                "ARSupervisedTrackBuilder: %d/%d responses truncated to max_response_length=%d (EOS kept). "
                "This warning is emitted once.",
                truncated,
                len(records),
                self.max_response_length,
            )
        return tokens, masks


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
    def build(self, records: List[Record]) -> RolloutTrack:
        """Encode one shard of (prompt, target image) records into a root track."""
        if not records:
            raise ValueError("DiffusionSupervisedTrackBuilder.build: empty record shard.")
        with torch.no_grad():
            texts = Texts(texts=[str(r["prompt"]) for r in records])
            conditions = self.pipeline.build_conditions(texts, **self._conditions_kwargs)
            pixels = self._load_target_pixels(records)
            latents = self._encode.encode(Images(pixels=pixels)).latents
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_target_pixels(self, records: Sequence[Record]) -> torch.Tensor:
        """Load + resize target images → ``[B, 3, H, W]`` fp32 in ``[0, 1]``."""
        import numpy as np
        from PIL import Image as PILImage

        rows: List[torch.Tensor] = []
        for r in records:
            uris = _media_uris(r, role="target")
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


__all__ = [
    "ARSupervisedTrackBuilder",
    "DiffusionSupervisedTrackBuilder",
    "SupervisedTrackBuilder",
]
