"""TTS intelligibility reward via fail-closed multilingual ASR."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.base import LocalRewardBackend
from unirl.reward.local.tts_metrics import (
    RewardUnavailableError,
    metric_for_language,
    score_edit_metric,
    unavailable_response,
)
from unirl.types.reward import RewardRequest, RewardResponse

logger = logging.getLogger(__name__)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Backward-compatible WER helper using the shared normalization."""
    return float(score_edit_metric(reference, hypothesis, metric="wer")["rate"])


def char_error_rate(reference: str, hypothesis: str, *, language: str = "") -> float:
    return float(score_edit_metric(reference, hypothesis, metric="cer", language=language)["rate"])


def wer_to_reward(wer: float, *, wer_cap: float = 1.0) -> float:
    cap = max(float(wer_cap), 1e-6)
    return float(max(0.0, min(1.0, 1.0 - float(wer) / cap)))


class TTSWerRewardScorer(LocalRewardBackend):
    """Reward = clipped ``1 - selected_error_rate / error_rate_cap``.

    Expected request fields:
    - ``generated["audio"]`` + ``audio_sample_rate``
    - reference transcript in metadata or conditioning text
    - optional metadata ``language``; configured languages use CER, others WER
    """

    canonical_model_name = "tts_wer"
    input_kind = "audio"

    def __init__(self, *, config: "TTSWerSpec", base_device: str) -> None:
        self._spec = config
        self.training_eligible = not config.dry_run
        device = config.device if config.device and config.device != "auto" else base_device
        super().__init__(model_name="tts_wer", device=device, batch_size=config.batch_size)
        self.input_kind = "audio"

    def _load_model(self) -> None:
        self._asr = None
        self._unavailable_reason = None
        if self._spec.dry_run:
            self._unavailable_reason = "dry_run=True; ASR was intentionally not loaded"
            return
        model_id = str(self._spec.asr_model_id or "").strip()
        if not model_id:
            raise RewardUnavailableError("TTSWerRewardScorer requires a non-empty asr_model_id in training mode.")
        try:
            import torch
            from transformers import pipeline

            device: Any = -1
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                suffix = str(self.device).partition(":")[2]
                device = int(suffix) if suffix else 0
            self._asr = pipeline(
                "automatic-speech-recognition",
                model=model_id,
                device=device,
            )
            logger.info("TTSWerRewardScorer: loaded ASR %s", model_id)
        except Exception as exc:  # pragma: no cover - optional dep
            raise RewardUnavailableError(f"TTSWerRewardScorer failed to load ASR {model_id!r}: {exc}") from exc

    def _transcribe(self, waveform, sample_rate: int, language: str) -> str:
        if self._asr is None:
            raise RewardUnavailableError(self._unavailable_reason or "ASR model is not loaded")

        wav = waveform.detach().float().cpu().numpy()
        if wav.ndim > 1:
            wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[-1] else -1)
        kwargs: Dict[str, Any] = {}
        language_key = language.strip().lower().replace("_", "-")
        language_root = language_key.split("-", 1)[0]
        mapped_language = self._spec.asr_language_map.get(
            language_key,
            self._spec.asr_language_map.get(language_root, language_root),
        )
        if self._spec.pass_language_to_asr and mapped_language:
            kwargs["generate_kwargs"] = {"language": mapped_language}
        out = self._asr(
            {"array": wav.astype("float32"), "sampling_rate": int(sample_rate)},
            **kwargs,
        )
        if isinstance(out, dict):
            return str(out.get("text", ""))
        return str(out)

    def _sample_context(self, request: RewardRequest, idx: int) -> Tuple[str, str]:
        meta = None
        if request.metadata is not None and idx < len(request.metadata):
            meta = request.metadata[idx]
        language = str(meta.get("language", "") if isinstance(meta, dict) else "")
        if isinstance(meta, dict):
            for key in ("normalized_transcript", "transcript", "text", "prompt"):
                if meta.get(key):
                    return str(meta[key]), language
        prompts = request.prompts
        if idx < len(prompts):
            return str(prompts[idx]), language
        return "", language

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        if self._asr is None:
            return unavailable_response(
                request.batch_size,
                component=self.canonical_model_name,
                reason=self._unavailable_reason or "ASR model is not loaded",
            )
        start = time.time()
        audios = request.audio
        if audios is None:
            raise ValueError("TTSWerRewardScorer requires generated audio.")
        sr = int(request.audio_sample_rate or self._spec.default_sample_rate)
        rewards: List[float] = []
        successes: List[bool] = []
        errors: List[str | None] = []
        details: List[Dict[str, Any]] = []
        components: Dict[str, List[float]] = {
            key: []
            for key in (
                "selected_error_rate",
                "wer",
                "cer",
                "substitutions",
                "deletions",
                "insertions",
                "reference_units",
            )
        }
        for i, wav in enumerate(audios):
            try:
                reference, language = self._sample_context(request, i)
                if not reference.strip():
                    raise ValueError("reference transcript is missing")
                if wav is None or getattr(wav, "numel", lambda: 0)() == 0:
                    raise ValueError("generated audio is empty")
                transcript = self._transcribe(wav, sr, language)
                selected_metric = metric_for_language(language, self._spec.cer_languages)
                selected = score_edit_metric(
                    reference,
                    transcript,
                    metric=selected_metric,
                    language=language,
                )
                wer = score_edit_metric(reference, transcript, metric="wer", language=language)
                cer = score_edit_metric(reference, transcript, metric="cer", language=language)
                reward = wer_to_reward(float(selected["rate"]), wer_cap=self._spec.error_rate_cap)
                rewards.append(reward)
                successes.append(True)
                errors.append(None)
                details.append(
                    {
                        "status": "available",
                        "language": language,
                        "transcript": transcript,
                        "selected": selected,
                        "wer": wer,
                        "cer": cer,
                    }
                )
                components["selected_error_rate"].append(float(selected["rate"]))
                components["wer"].append(float(wer["rate"]))
                components["cer"].append(float(cer["rate"]))
                for key in ("substitutions", "deletions", "insertions", "reference_units"):
                    components[key].append(float(selected[key]))
            except Exception as exc:
                rewards.append(0.0)
                successes.append(False)
                errors.append(f"{type(exc).__name__}: {exc}")
                details.append({"status": "failed", "error": errors[-1]})
                for values in components.values():
                    values.append(0.0)
        return RewardResponse(
            rewards=rewards,
            component_rewards=components,
            details=details,
            successes=successes,
            errors=errors,
            compute_time=time.time() - start,
        )

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        # ``compute_rewards`` is overridden to preserve transcript/edit details.
        return list(self.compute_rewards(request).rewards)

    def is_available(self) -> bool:
        return self._asr is not None and not self._spec.dry_run

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind


@dataclass
class TTSWerSpec(BaseRewardComponentSpec):
    batch_size: int = 4
    device: str = "auto"
    asr_model_id: str = "openai/whisper-small"
    error_rate_cap: float = 1.0
    # Backward-compatible alias. If set, callers should keep it equal to
    # ``error_rate_cap``; retained so old Hydra recipes do not fail parsing.
    wer_cap: float | None = None
    default_sample_rate: int = 24000
    cer_languages: Tuple[str, ...] = ("zh", "yue", "ja", "ko")
    pass_language_to_asr: bool = True
    asr_language_map: Dict[str, str] = field(default_factory=dict)
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.wer_cap is not None:
            self.error_rate_cap = float(self.wer_cap)
        if self.error_rate_cap <= 0:
            raise ValueError("TTSWerSpec.error_rate_cap must be positive.")
