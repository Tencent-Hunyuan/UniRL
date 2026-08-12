"""Hard fail-closed gates for TTS rollout and waveform stability."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.reward.local.tts_metrics import mono_waveform
from unirl.types.reward import RewardRequest, RewardResponse


class TTSStabilityRewardScorer(RewardBackend):
    """Return 1 only when every configured rollout/audio gate passes."""

    input_kind = "audio"

    def __init__(self, *, config: "TTSStabilitySpec", base_device: str) -> None:
        super().__init__(model_name="tts_stability", batch_size=config.batch_size)
        self._spec = config
        self.input_kind = "audio"

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        audios = request.audio
        if audios is None:
            raise ValueError("TTSStabilityRewardScorer requires generated audio.")
        sr = int(request.audio_sample_rate or self._spec.default_sample_rate)
        rewards: List[float] = []
        details: List[Dict[str, Any]] = []
        gate_names = (
            "empty",
            "decode_failure",
            "missing_eos",
            "extreme_duration",
            "silence",
            "clipping",
            "codec_repetition",
            "audio_repetition",
            "non_finite",
        )
        components: Dict[str, List[float]] = {f"gate_{name}": [] for name in gate_names}
        components.update({"duration_s": [], "rms": [], "clip_fraction": [], "audio_repetition_fraction": []})
        for i, wav in enumerate(audios):
            meta = request.metadata[i] if request.metadata and i < len(request.metadata) else {}
            meta = meta if isinstance(meta, dict) else {}
            rollout = meta.get("rollout", {})
            rollout = rollout if isinstance(rollout, dict) else {}
            merged = {**meta, **rollout}

            gates = {name: False for name in gate_names}
            duration_s = rms = clip_fraction = repetition_fraction = 0.0
            if wav is None or wav.numel() == 0:
                gates["empty"] = True
            else:
                w = mono_waveform(wav)
                gates["non_finite"] = not bool(torch.isfinite(w).all())
                if not gates["non_finite"]:
                    duration_s = float(w.numel()) / float(sr)
                    rms = float(w.square().mean().sqrt().item())
                    clip_fraction = float((w.abs() >= self._spec.clipping_amplitude).float().mean().item())
                    repetition_fraction = self._audio_repetition_fraction(w, sr)
                    gates["silence"] = rms < self._spec.silence_rms
                    gates["clipping"] = clip_fraction > self._spec.max_clipping_fraction
                    gates["audio_repetition"] = repetition_fraction > self._spec.max_audio_repetition_fraction

                    expected = merged.get("expected_duration_s")
                    extreme = duration_s < self._spec.min_seconds or duration_s > self._spec.max_seconds
                    if expected is not None:
                        expected = float(expected)
                        if expected <= 0:
                            extreme = True
                        else:
                            ratio = duration_s / expected
                            extreme = extreme or ratio < self._spec.min_expected_duration_ratio
                            extreme = extreme or ratio > self._spec.max_expected_duration_ratio
                    gates["extreme_duration"] = extreme

            gates["decode_failure"] = bool(merged.get("decode_failure", False))
            has_eos = merged.get("has_eos")
            if has_eos is None:
                status = merged.get("segment_status")
                if isinstance(status, str):
                    has_eos = status.strip().lower() in {"completed", "stop"}
                elif status is not None:
                    has_eos = int(status) == 1
            gates["missing_eos"] = has_eos is not True if self._spec.require_eos else False

            codec_max_run = float(merged.get("codec_max_run_fraction", 0.0) or 0.0)
            codec_repetition = float(merged.get("codec_repetition_fraction", 0.0) or 0.0)
            codec_unique = float(merged.get("codec_unique_ratio", 1.0) or 0.0)
            gates["codec_repetition"] = (
                codec_max_run > self._spec.max_codec_run_fraction
                or codec_repetition > self._spec.max_codec_repetition_fraction
                or codec_unique < self._spec.min_codec_unique_ratio
            )

            failed = [name for name, failed_gate in gates.items() if failed_gate]
            score = 0.0 if failed else 1.0
            rewards.append(score)
            for name in gate_names:
                components[f"gate_{name}"].append(float(gates[name]))
            components["duration_s"].append(duration_s)
            components["rms"].append(rms)
            components["clip_fraction"].append(clip_fraction)
            components["audio_repetition_fraction"].append(repetition_fraction)
            details.append(
                {
                    "status": "passed" if not failed else "gated",
                    "failed_gates": failed,
                    "gates": gates,
                    "duration_s": duration_s,
                    "rms": rms,
                    "clip_fraction": clip_fraction,
                    "audio_repetition_fraction": repetition_fraction,
                    "rollout": rollout,
                }
            )
        return RewardResponse(
            rewards=rewards,
            component_rewards=components,
            details=details,
            successes=[True] * len(rewards),
            errors=[None] * len(rewards),
            compute_time=time.time() - start,
        )

    def _audio_repetition_fraction(self, waveform: torch.Tensor, sample_rate: int) -> float:
        frame = max(8, round(self._spec.repetition_frame_seconds * sample_rate))
        if waveform.numel() < frame * 2:
            return 0.0
        frames = waveform.unfold(0, frame, frame)
        if frames.shape[0] < 2:
            return 0.0
        frames = frames - frames.mean(dim=1, keepdim=True)
        norms = frames.norm(dim=1).clamp_min(1e-8)
        best = 0.0
        max_lag = min(self._spec.repetition_max_lag_frames, frames.shape[0] - 1)
        for lag in range(1, max_lag + 1):
            similarities = (frames[lag:] * frames[:-lag]).sum(dim=1) / (norms[lag:] * norms[:-lag])
            fraction = float((similarities > self._spec.repetition_cosine_threshold).float().mean().item())
            best = max(best, fraction)
        return best

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind

    def is_available(self) -> bool:
        return True


@dataclass
class TTSStabilitySpec(BaseRewardComponentSpec):
    batch_size: int = 8
    device: str = "auto"
    default_sample_rate: int = 24000
    min_seconds: float = 0.2
    max_seconds: float = 60.0
    silence_rms: float = 1e-4
    clipping_amplitude: float = 0.999
    max_clipping_fraction: float = 0.01
    min_expected_duration_ratio: float = 0.25
    max_expected_duration_ratio: float = 4.0
    max_codec_run_fraction: float = 0.20
    max_codec_repetition_fraction: float = 0.85
    min_codec_unique_ratio: float = 0.02
    repetition_frame_seconds: float = 0.20
    repetition_max_lag_frames: int = 8
    repetition_cosine_threshold: float = 0.995
    max_audio_repetition_fraction: float = 0.80
    require_eos: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_seconds) or self.min_seconds < 0:
            raise ValueError("min_seconds must be finite and non-negative.")
        if self.max_seconds <= self.min_seconds:
            raise ValueError("max_seconds must be greater than min_seconds.")
        for name in (
            "max_clipping_fraction",
            "max_codec_run_fraction",
            "max_codec_repetition_fraction",
            "min_codec_unique_ratio",
            "max_audio_repetition_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
