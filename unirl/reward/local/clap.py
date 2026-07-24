"""CLAP audio-text alignment reward — LAION CLAP via HuggingFace transformers.

Scores cosine similarity between generated audio and the text prompt. Used for
LTX-2.3 T2AV where the reward service injects the jointly-generated audio
waveform into ``request.generated["audio"]`` (a side-channel alongside the
video in ``request.generated["video"]``). Mirrors Flow-Factory's CLAP reward.

Zero extra dependencies — ``transformers.ClapModel`` is already in the dep tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend


class CLAPRewardScorer(LocalRewardBackend):
    """Audio-text alignment reward using LAION CLAP.

    ``input_kind = "video"``: the primary decoded media is the video (so the
    track routes through the video path), and the audio arrives as a parallel
    side-channel (``request.generated["audio"]`` + ``request.audio_sample_rate``)
    that the reward service injects when the track has ``decoded_audio``.
    """

    canonical_model_name = "clap"
    input_kind = "video"
    CLAP_SAMPLE_RATE = 48_000

    def __init__(self, *, config: "CLAPSpec", base_device: str) -> None:
        super().__init__(
            device=resolve_device(config.device, base_device),
            batch_size=config.batch_size,
            model_id=config.model_id,
        )

    def _load_model(self) -> None:
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as e:
            raise ImportError("transformers with ClapModel/ClapProcessor is required for the CLAP reward") from e

        model_id = self.model_kwargs.get("model_id", "laion/larger_clap_general")
        # float32 required: CLAP audio encoder uses BatchNorm, which is unstable / unsupported in fp16/bf16.
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.model = self.model.to(dtype=torch.float32)
        self.processor = ClapProcessor.from_pretrained(model_id)

    def _preprocess_audio(self, audio_list: List[torch.Tensor], src_sample_rate: int) -> List["torch.Tensor"]:
        """Downmix to mono and resample each waveform to CLAP's 48 kHz.

        Accepts per-sample tensors shaped ``[L]``, ``[C, L]``, or ``[L, C]``
        (the ``Audios`` primitive packs along the leading L axis, so ``to_list``
        yields ``[L]`` / ``[L, C]``). Returns mono numpy float32 arrays ``[L']``.
        """
        import numpy as np
        import torchaudio.functional as AF

        processed: List[np.ndarray] = []
        for waveform in audio_list:
            wf = waveform.detach().float()
            if wf.isnan().any() or wf.isinf().any():
                wf = torch.zeros_like(wf)
            if wf.ndim == 2:
                # Reduce the channel axis to mono regardless of [C, L] vs [L, C]:
                # the channel axis is the smaller of the two.
                ch_axis = 0 if wf.shape[0] <= wf.shape[1] else 1
                wf = wf.mean(dim=ch_axis)
            wf = wf.reshape(-1)  # (L,)

            if src_sample_rate != self.CLAP_SAMPLE_RATE:
                wf = AF.resample(
                    wf.unsqueeze(0),
                    orig_freq=int(src_sample_rate),
                    new_freq=self.CLAP_SAMPLE_RATE,
                ).squeeze(0)

            processed.append(wf.cpu().numpy())
        return processed

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        audio = request.audio
        prompts = request.prompts
        if audio is None:
            raise ValueError(
                "CLAPRewardScorer requires audio in the reward request "
                "(request.generated['audio']); got none. Ensure the pipeline "
                "decodes audio into track.decoded_audio for LTX-2.3 T2AV."
            )
        if request.audio_sample_rate is None:
            raise ValueError("CLAPRewardScorer requires request.audio_sample_rate (source Hz); got None.")
        src_rate = int(request.audio_sample_rate)

        all_rewards: List[float] = []
        for i in range(0, len(audio), self.batch_size):
            batch_audio = audio[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]
            waveforms_np = self._preprocess_audio(batch_audio, src_rate)

            inputs = self.processor(
                text=batch_prompts,
                audios=waveforms_np,
                sampling_rate=self.CLAP_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                audio_embeds = F.normalize(outputs.audio_embeds, p=2, dim=-1)
                text_embeds = F.normalize(outputs.text_embeds, p=2, dim=-1)
                scores = (audio_embeds * text_embeds).sum(dim=-1)
            all_rewards.extend(scores.float().cpu().tolist())

        return all_rewards


@dataclass
class CLAPSpec(BaseRewardComponentSpec):
    """Typed config for the CLAP audio-text reward component."""

    batch_size: int = 8
    device: str = "auto"
    model_id: str = "laion/larger_clap_general"
