"""Face similarity reward scorer (REFL-compatible / BPTT-differentiable)."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.base import LocalRewardBackend
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .face_tools import Face, FaceAnalysis

logger = logging.getLogger(__name__)

# Reference-embedding LRU bound (see FaceRewardScorer._ref_cache).
_REF_CACHE_MAX = 64


# ---------------------------------------------------------------------------
# Reference-video loader (imageio + resize-to-cover + center_crop)
# ---------------------------------------------------------------------------


def _load_ref_video_frames(
    video_path: str,
    *,
    max_frames: int = 81,
    max_pixels: int = 480 * 480,
    height_div: int = 16,
    width_div: int = 16,
) -> torch.Tensor:
    """Load a reference video as ``(C, T, H, W)`` float32 in ``[-1, 1]``.

    Frame-for-frame preprocessing so the reference-side stays aligned with the
    reference-loader used to train the REFL data pipeline.
    """
    import imageio
    import PIL.Image
    import torchvision.transforms.functional as TF

    reader = imageio.get_reader(video_path)
    total = int(reader.count_frames())  # type: ignore[attr-defined]

    # time_division_factor=4, remainder=1
    nf = min(max_frames, total)
    while nf > 1 and nf % 4 != 1:
        nf -= 1

    first = PIL.Image.fromarray(reader.get_data(0))
    fw, fh = first.size
    if fw * fh > max_pixels:
        scale = (fw * fh / max_pixels) ** 0.5
        fh, fw = int(fh / scale), int(fw / scale)
    target_h = fh // height_div * height_div
    target_w = fw // width_div * width_div

    frames = []
    for i in range(nf):
        frame = PIL.Image.fromarray(reader.get_data(i))
        iw, ih = frame.size
        scale = max(target_w / iw, target_h / ih)
        frame = TF.resize(
            frame,
            (round(ih * scale), round(iw * scale)),
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        frame = TF.center_crop(frame, (target_h, target_w))
        t = TF.to_tensor(frame)  # (C, H, W) [0, 1]
        frame = TF.to_pil_image(t)  # round-trip PIL to match training-time preprocessing
        frames.append(TF.to_tensor(frame) * 2.0 - 1.0)
    reader.close()

    # (T, C, H, W) -> (C, T, H, W)
    return torch.stack(frames).permute(1, 0, 2, 3).contiguous()


# ---------------------------------------------------------------------------
# Reward scorer
# ---------------------------------------------------------------------------


class FaceRewardScorer(LocalRewardBackend):
    """SCRFD detection + ArcFace embedding + pool cosine similarity for REFL."""

    canonical_model_name = "face"
    input_kind = "video"

    def __init__(self, *, config: "FaceRewardSpec", base_device: str) -> None:
        super().__init__(
            model_name=self.canonical_model_name,
            device=resolve_device(config.device, base_device),
            batch_size=int(config.batch_size),
            model_path=config.model_path,
            image_size=config.image_size,
            ref_max_frames=config.ref_max_frames,
            ref_max_pixels=config.ref_max_pixels,
            differentiable=config.differentiable,
        )

    def _load_model(self) -> None:
        model_path = self.model_kwargs["model_path"]
        self.model = FaceAnalysis(root=model_path, device=self.device)
        self._image_size = int(self.model_kwargs.get("image_size", 112))
        self._ref_max_frames = int(self.model_kwargs.get("ref_max_frames", 81))
        self._ref_max_pixels = int(self.model_kwargs.get("ref_max_pixels", 480 * 480))
        self._differentiable = bool(self.model_kwargs.get("differentiable", True))
        # Bounded LRU: reference embeddings are small, but a many-identity
        # dataset must not grow GPU residency without bound.
        self._ref_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        raise NotImplementedError("FaceRewardScorer is REFL-only; use compute_rewards_differentiable().")

    # ------------------------------------------------------------------
    # Per-video face embedding (recipe-local REFL implementation)
    # ------------------------------------------------------------------

    def _extract_face_embeddings(
        self,
        video: torch.Tensor,
        *,
        with_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract per-frame face embeddings.

        Args:
            video: ``(C, T, H, W)`` float in ``[-1, 1]``.
            with_grad: When True, the ArcFace forward is kept inside autograd
                (used for the GENERATED video on the BPTT path); when False,
                the whole computation runs under ``torch.no_grad`` (used for
                the REFERENCE video, which never receives gradient).

        Returns:
            embeddings: ``(1, T, 512)`` on ``self.device``. Differentiable in
                ``video`` when ``with_grad=True``.
            mask:       ``(1, T)`` on ``self.device`` (1 = face found).
        """
        fa = self.model
        fa.detection_model.torch_model.to(self.device)
        fa.arcface_model.torch_model.to(self.device)

        T = int(video.shape[1])
        embeddings: List[torch.Tensor] = []
        mask: List[int] = []
        zero_emb = None

        autograd_ctx = torch.enable_grad if with_grad else torch.no_grad

        for t in range(T):
            frame = video[:, t].float().to(self.device)
            bboxes, kpss = fa.detection_model.detect(frame)
            if bboxes.shape[0] > 0:
                indexed = [(i, x) for i, x in enumerate(bboxes)]
                sorted_bboxes = sorted(
                    indexed,
                    key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
                )
                max_index, _ = sorted_bboxes[-1]
                face = Face(
                    bbox=bboxes[max_index][0:4],
                    kps=kpss[max_index],
                    det_score=bboxes[max_index][4],
                )
                with autograd_ctx():
                    emb = fa.arcface_model.get(frame, face)
                embeddings.append(emb)
                mask.append(1)
            else:
                if zero_emb is None:
                    zero_emb = torch.zeros(512, device=self.device)
                embeddings.append(zero_emb)
                mask.append(0)

        emb_stack = torch.stack(embeddings).unsqueeze(0)  # (1, T, 512)
        mask_tensor = torch.tensor(mask, device=self.device).unsqueeze(0).float()  # (1, T)
        return emb_stack, mask_tensor

    # ------------------------------------------------------------------
    # Reference-video caching
    # ------------------------------------------------------------------

    def _get_ref_embeddings(self, ref_video_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._ref_cache.get(ref_video_path)
        if cached is not None:
            self._ref_cache.move_to_end(ref_video_path)
            return cached
        with torch.no_grad():
            video = _load_ref_video_frames(
                ref_video_path,
                max_frames=self._ref_max_frames,
                max_pixels=self._ref_max_pixels,
            ).to(self.device)
            ref_emb, ref_mask = self._extract_face_embeddings(video, with_grad=False)
        # Detach (defence-in-depth) before caching.
        cached = (ref_emb.detach(), ref_mask.detach())
        self._ref_cache[ref_video_path] = cached
        while len(self._ref_cache) > _REF_CACHE_MAX:
            self._ref_cache.popitem(last=False)
        return cached

    # ------------------------------------------------------------------
    # Differentiable REFL reward entry point
    # ------------------------------------------------------------------

    def compute_rewards_differentiable(
        self,
        media_tensor: torch.Tensor,
        prompts: List[str],
        records: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Score generated videos ``[B,C,T,H,W]`` against reference-video metadata."""
        if media_tensor.ndim != 5:
            raise ValueError(f"FaceRewardScorer expects [B,C,T,H,W], got {tuple(media_tensor.shape)}")
        B = int(media_tensor.shape[0])
        if len(prompts) != B:
            raise ValueError(f"FaceRewardScorer: prompts length {len(prompts)} != batch size {B}.")
        metadata = records or [None] * B
        if len(metadata) != B:
            raise ValueError(f"FaceRewardScorer: records length {len(metadata)} != batch size {B}.")

        rewards: List[torch.Tensor] = []
        for i in range(B):
            md = metadata[i] or {}
            ref_path: Optional[str] = md.get("ref_video_path") or md.get("ref_video")
            if not ref_path:
                raise ValueError(
                    f"FaceRewardScorer: sample {i} is missing metadata['ref_video_path']. "
                    "Make sure the JSONL row was written with absolute ref video paths."
                )

            ref_emb, ref_mask = self._get_ref_embeddings(str(ref_path))

            gen_video = torch.clamp(media_tensor[i], -1, 1).to(self.device)
            gen_emb, gen_mask = self._extract_face_embeddings(gen_video, with_grad=self._differentiable)

            if int(gen_mask.sum().item()) == 0:
                rewards.append(gen_video.sum() * 0.0)
                continue

            score = self.model.pool_embedding_loss(
                id_embedding=gen_emb,
                gt_embedding=ref_emb,
                id_mask=gen_mask,
            )
            rewards.append(score)

        return torch.stack(rewards).float()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def offload(self) -> None:
        self._ref_cache.clear()
        fa = self.model
        if fa is not None:
            for sub in (fa.detection_model, fa.landmark_model, fa.arcface_model):
                if sub is not None and hasattr(sub.torch_model, "cpu"):
                    sub.torch_model.cpu()
            torch.cuda.empty_cache()

    def onload(self) -> None:
        fa = self.model
        if fa is not None:
            for sub in (fa.detection_model, fa.landmark_model, fa.arcface_model):
                if sub is not None and hasattr(sub.torch_model, "to"):
                    sub.torch_model.to(self.device)

    def is_available(self) -> bool:
        return bool(self._is_loaded)

    def dispose(self) -> None:
        self.offload()


@dataclass
class FaceRewardSpec(BaseRewardComponentSpec):
    """Typed config for :class:`FaceRewardScorer`.

    Args:
        model_path: Directory containing ONNX model files
            (scrfd_10g_bnkps.onnx, 2d106det.onnx, glintr100.onnx). Passed
            straight to :class:`FaceAnalysis` as its model root.
        device: "auto" / "cuda" / "cuda:N" — resolved against ``base_device``.
        batch_size: kept for parity with sibling specs; the scorer iterates
            per-sample internally because each sample needs its own ref video.
        image_size: ArcFace alignment size, must be a multiple of 112 or 128.
        ref_max_frames: Frame cap for ref-video decoding (default 81, matches
            the REFL training-time cap).
        ref_max_pixels: Pixel cap for ref-video decoding (default 480*480 —
            keeps SCRFD's short-side input near its trained resolution).
        differentiable: Keep autograd on the generated ArcFace forward (default
            True — required for REFL). Set False to score under no_grad for
            the historical GRPO/replay path.
    """

    model_path: str = ""
    device: str = "auto"
    batch_size: int = 1
    image_size: int = 112
    ref_max_frames: int = 81
    ref_max_pixels: int = 480 * 480
    differentiable: bool = True


__all__ = [
    "FaceRewardScorer",
    "FaceRewardSpec",
]
