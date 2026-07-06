"""Audio-video / audio-text semantic alignment reward using Meta ImageBind.

Mirrors Flow-Factory's ImageBind reward. Used for LTX-2.3 T2AV where the reward
service injects the jointly-generated audio into ``request.generated["audio"]``
alongside the video in ``request.generated["video"]``.

IMPORTANT: ImageBind is licensed under CC-BY-NC-SA 4.0 (NonCommercial). The
package is NOT a base dependency — it is imported lazily inside ``_load_model``
so the scorer only pulls it in when a recipe explicitly selects ``imagebind``.
Install with::

    pip install git+https://github.com/facebookresearch/ImageBind.git
    pip install git+https://github.com/facebookresearch/pytorchvideo.git
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

_IMAGEBIND_LICENSE_WARNING = (
    "ImageBind is licensed under CC-BY-NC-SA 4.0 (NonCommercial). Using it in "
    "commercial applications may violate the license. "
    "See https://github.com/facebookresearch/ImageBind/blob/main/LICENSE"
)
_IMAGEBIND_INSTALL_MSG = (
    "ImageBind is not installed. Install with:\n"
    "  pip install git+https://github.com/facebookresearch/ImageBind.git\n"
    "  pip install git+https://github.com/facebookresearch/pytorchvideo.git\n"
    "Note: ImageBind is CC-BY-NC-SA 4.0 (NonCommercial only)."
)

_IB_AUDIO_SAMPLE_RATE = 16_000
_IB_AUDIO_NUM_MEL_BINS = 128
_IB_AUDIO_TARGET_LENGTH = 204
_IB_AUDIO_CLIP_DURATION = 2
_IB_AUDIO_CLIPS_PER_SAMPLE = 3
_IB_AUDIO_MEAN = -4.268
_IB_AUDIO_STD = 9.138

_IB_VISION_SIZE = 224
_IB_VISION_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IB_VISION_STD = (0.26862954, 0.26130258, 0.27577711)


class ImageBindRewardScorer(LocalRewardBackend):
    """Audio-video / audio-text alignment reward using Meta ImageBind.

    Modes (``mode`` on the Spec):
        - "audio_video" (default): cos_sim(audio, video)
        - "text_audio":            cos_sim(text, audio)
        - "text_video":            cos_sim(text, video)
        - "all":                   weighted sum of all three

    ``input_kind = "video"``: video is the primary decoded media; audio arrives
    as the parallel side-channel (``request.generated["audio"]``).

    IMPORTANT: ImageBind is CC-BY-NC-SA 4.0 (NonCommercial).
    """

    canonical_model_name = "imagebind"
    input_kind = "video"
    DEFAULT_MODE = "audio_video"

    def __init__(self, *, config: "ImageBindSpec", base_device: str) -> None:
        self._mode = str(config.mode or self.DEFAULT_MODE)
        self._weights = dict(config.weights or {"audio_video": 0.5, "text_audio": 0.25, "text_video": 0.25})
        super().__init__(
            device=resolve_device(config.device, base_device),
            batch_size=config.batch_size,
        )

    def _load_model(self) -> None:
        warnings.warn(_IMAGEBIND_LICENSE_WARNING, stacklevel=2)
        try:
            from imagebind.models import imagebind_model
        except ImportError as e:
            raise ImportError(_IMAGEBIND_INSTALL_MSG) from e

        self.model = imagebind_model.imagebind_huge(pretrained=True).to(self.device).eval()

    # ---- audio preprocessing -------------------------------------------------

    def _preprocess_audio_to_melspec(self, audio_list: List[torch.Tensor], src_sample_rate: int) -> torch.Tensor:
        import torch.nn.functional as Fn
        import torchaudio.functional as AF

        batch_clips = []
        samples_per_clip = _IB_AUDIO_CLIP_DURATION * _IB_AUDIO_SAMPLE_RATE
        for waveform in audio_list:
            wf = waveform.detach().float()
            if wf.ndim == 2:
                ch_axis = 0 if wf.shape[0] <= wf.shape[1] else 1
                wf = wf.mean(dim=ch_axis)
            wf = wf.reshape(1, -1)  # (1, T)
            if src_sample_rate != _IB_AUDIO_SAMPLE_RATE:
                wf = AF.resample(wf, src_sample_rate, _IB_AUDIO_SAMPLE_RATE)

            duration_s = wf.shape[1] / _IB_AUDIO_SAMPLE_RATE
            clip_starts = self._compute_clip_starts(duration_s, _IB_AUDIO_CLIP_DURATION, _IB_AUDIO_CLIPS_PER_SAMPLE)
            mel_clips = []
            for start_s in clip_starts:
                start_idx = int(start_s * _IB_AUDIO_SAMPLE_RATE)
                clip = wf[:, start_idx : start_idx + samples_per_clip]
                if clip.shape[1] < samples_per_clip:
                    clip = Fn.pad(clip, (0, samples_per_clip - clip.shape[1]))
                mel = self._waveform_to_melspec(clip)
                mel = (mel - _IB_AUDIO_MEAN) / _IB_AUDIO_STD
                mel_clips.append(mel)
            batch_clips.append(torch.stack(mel_clips, dim=0))
        return torch.stack(batch_clips, dim=0).to(self.device)

    @staticmethod
    def _waveform_to_melspec(waveform: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as Fn
        import torchaudio.compliance.kaldi as kaldi

        waveform = waveform.float()
        waveform = waveform - waveform.mean()
        fbank = kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=_IB_AUDIO_SAMPLE_RATE,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=_IB_AUDIO_NUM_MEL_BINS,
            dither=0.0,
            frame_length=25,
            frame_shift=10,
        )
        fbank = fbank.transpose(0, 1)
        n_frames = fbank.shape[1]
        if n_frames < _IB_AUDIO_TARGET_LENGTH:
            fbank = Fn.pad(fbank, (0, _IB_AUDIO_TARGET_LENGTH - n_frames))
        else:
            fbank = fbank[:, :_IB_AUDIO_TARGET_LENGTH]
        return fbank.unsqueeze(0)

    @staticmethod
    def _compute_clip_starts(duration_s: float, clip_duration: float, num_clips: int) -> List[float]:
        if duration_s <= clip_duration:
            return [0.0] * num_clips
        spacing = (duration_s - clip_duration) / max(num_clips - 1, 1)
        return [i * spacing for i in range(num_clips)]

    # ---- video preprocessing -------------------------------------------------

    def _preprocess_video(self, video_list: List[torch.Tensor]) -> torch.Tensor:
        batch_result = []
        for video in video_list:
            video_f = video.float() / 255.0 if video.dtype == torch.uint8 else video.float()
            clips = self._temporal_subsample_clips(video_f, num_clips=5, frames_per_clip=2)
            all_crops = []
            for clip in clips:
                clip = self._resize_short_side(clip, _IB_VISION_SIZE)
                clip = self._normalize_video_clip(clip)
                all_crops.extend(self._spatial_crop(clip, _IB_VISION_SIZE))
            batch_result.append(torch.stack(all_crops, dim=0))
        return torch.stack(batch_result, dim=0).to(self.device)

    @staticmethod
    def _temporal_subsample_clips(video: torch.Tensor, num_clips: int, frames_per_clip: int) -> List[torch.Tensor]:
        T = video.shape[0]
        clips = []
        for i in range(num_clips):
            center = int((i + 0.5) * T / num_clips)
            indices = torch.linspace(
                max(0, center - frames_per_clip // 2),
                min(T - 1, center + frames_per_clip // 2 - 1),
                frames_per_clip,
            ).long()
            clips.append(video[indices].permute(1, 0, 2, 3))
        return clips

    @staticmethod
    def _resize_short_side(clip: torch.Tensor, size: int) -> torch.Tensor:
        C, T, H, W = clip.shape
        if W <= H:
            new_w, new_h = size, int(H / W * size)
        else:
            new_w, new_h = int(W / H * size), size
        clip_flat = clip.reshape(C * T, 1, H, W)
        clip_resized = F.interpolate(clip_flat, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return clip_resized.reshape(C, T, new_h, new_w)

    @staticmethod
    def _normalize_video_clip(clip: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(_IB_VISION_MEAN, device=clip.device).view(3, 1, 1, 1)
        std = torch.tensor(_IB_VISION_STD, device=clip.device).view(3, 1, 1, 1)
        return (clip - mean) / std

    @staticmethod
    def _spatial_crop(clip: torch.Tensor, crop_size: int) -> List[torch.Tensor]:
        C, T, H, W = clip.shape
        crops = []
        if H > W:
            offsets = [0, (H - crop_size) // 2, H - crop_size]
            for y in offsets:
                crops.append(clip[:, :, y : y + crop_size, :])
        else:
            offsets = [0, (W - crop_size) // 2, W - crop_size]
            for x in offsets:
                crops.append(clip[:, :, :, x : x + crop_size])
        return crops

    # ---- scoring -------------------------------------------------------------

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        from imagebind.data import load_and_transform_text
        from imagebind.models.imagebind_model import ModalityType

        audio = request.audio
        videos = request.videos
        prompts = request.prompts

        need_text = self._mode in ("text_audio", "text_video", "all")
        need_audio = self._mode in ("audio_video", "text_audio", "all")
        need_video = self._mode in ("audio_video", "text_video", "all")

        if need_audio and audio is None:
            raise ValueError("ImageBindRewardScorer: mode needs audio but request.generated['audio'] is None.")
        if need_video and videos is None:
            raise ValueError("ImageBindRewardScorer: mode needs video but request.generated['video'] is None.")

        src_rate = int(request.audio_sample_rate) if request.audio_sample_rate is not None else _IB_AUDIO_SAMPLE_RATE

        inputs: Dict[str, torch.Tensor] = {}
        if need_text:
            inputs[ModalityType.TEXT] = load_and_transform_text(prompts, self.device)
        if need_audio:
            inputs[ModalityType.AUDIO] = self._preprocess_audio_to_melspec(audio, src_rate)
        if need_video:
            inputs[ModalityType.VISION] = self._preprocess_video(videos)

        with torch.no_grad():
            embeddings = self.model(inputs)
        rewards = self._compute_similarity(embeddings, ModalityType)
        return rewards.float().cpu().tolist()

    def _compute_similarity(self, embeddings: dict, ModalityType) -> torch.Tensor:
        def cos(a, b):
            return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)

        if self._mode == "audio_video":
            return cos(embeddings[ModalityType.AUDIO], embeddings[ModalityType.VISION])
        if self._mode == "text_audio":
            return cos(embeddings[ModalityType.TEXT], embeddings[ModalityType.AUDIO])
        if self._mode == "text_video":
            return cos(embeddings[ModalityType.TEXT], embeddings[ModalityType.VISION])
        if self._mode == "all":
            w = self._weights
            av = cos(embeddings[ModalityType.AUDIO], embeddings[ModalityType.VISION])
            ta = cos(embeddings[ModalityType.TEXT], embeddings[ModalityType.AUDIO])
            tv = cos(embeddings[ModalityType.TEXT], embeddings[ModalityType.VISION])
            return w["audio_video"] * av + w["text_audio"] * ta + w["text_video"] * tv
        raise ValueError(f"Unknown ImageBind mode {self._mode!r}; expected audio_video|text_audio|text_video|all.")


@dataclass
class ImageBindSpec(BaseRewardComponentSpec):
    """Typed config for the ImageBind audio-video reward component (NonCommercial)."""

    batch_size: int = 8
    device: str = "auto"
    mode: str = "audio_video"
    weights: Optional[Dict[str, float]] = field(default=None)
