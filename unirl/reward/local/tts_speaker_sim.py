"""Fail-closed TTS speaker similarity using speaker-verification embeddings."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.base import LocalRewardBackend
from unirl.reward.local.tts_metrics import (
    RewardUnavailableError,
    mono_waveform,
    resample_waveform,
    unavailable_response,
)
from unirl.types.reward import RewardRequest, RewardResponse

logger = logging.getLogger(__name__)


class TTSSpeakerSimRewardScorer(LocalRewardBackend):
    """Cosine similarity between generated speech and a reference wav / speaker embedding.

    Every sample must provide ``speaker_embedding`` or ``ref_audio``, or resolve
    ``speaker`` / ``speaker_id`` in a fixed reference bank. Missing references
    are failures, never an energy proxy.
    """

    canonical_model_name = "tts_speaker_sim"
    input_kind = "audio"

    def __init__(self, *, config: "TTSSpeakerSimSpec", base_device: str) -> None:
        self._spec = config
        self.training_eligible = not config.dry_run
        device = config.device if config.device and config.device != "auto" else base_device
        super().__init__(model_name="tts_speaker_sim", device=device, batch_size=config.batch_size)
        self.input_kind = "audio"

    def _load_model(self) -> None:
        self._encoder = None
        self._processor = None
        self._reference_bank: Dict[str, Any] = {}
        self._unavailable_reason = None
        if self._spec.reference_bank_path:
            bank_path = Path(self._spec.reference_bank_path)
            try:
                raw = json.loads(bank_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise TypeError("reference bank root must be a JSON object")
                self._reference_bank = raw
                self._reference_bank_dir = bank_path.parent
            except Exception as exc:
                raise RewardUnavailableError(f"Failed to load speaker reference bank {bank_path}: {exc}") from exc
        else:
            self._reference_bank_dir = Path(".")
        if self._spec.dry_run:
            self._unavailable_reason = "dry_run=True; speaker encoder was intentionally not loaded"
            return
        model_id = str(self._spec.speaker_model_id or "").strip()
        if not model_id:
            raise RewardUnavailableError(
                "TTSSpeakerSimRewardScorer requires speaker_model_id in training mode."
            )
        try:
            from transformers import AutoFeatureExtractor

            try:
                from transformers import AutoModelForAudioXVector

                model_cls = AutoModelForAudioXVector
            except ImportError:  # pragma: no cover - older transformers
                from transformers import AutoModel

                model_cls = AutoModel

            self._processor = AutoFeatureExtractor.from_pretrained(model_id)
            self._encoder = model_cls.from_pretrained(model_id).to(self.device).eval()
            logger.info("TTSSpeakerSimRewardScorer: loaded %s", model_id)
        except Exception as exc:  # pragma: no cover
            raise RewardUnavailableError(
                f"TTSSpeakerSimRewardScorer failed to load {model_id!r}: {exc}"
            ) from exc

    def _embed(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if self._encoder is None or self._processor is None:
            raise RewardUnavailableError(self._unavailable_reason or "speaker encoder is not loaded")
        target_rate = int(getattr(self._processor, "sampling_rate", sample_rate) or sample_rate)
        wav = resample_waveform(waveform, sample_rate, target_rate)
        inputs = self._processor(
            wav.numpy(),
            sampling_rate=target_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._encoder(**inputs)
        embedding = getattr(out, "embeddings", None)
        if embedding is not None:
            emb = embedding.squeeze(0)
        else:
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None or hidden.ndim != 3:
                raise RuntimeError("Speaker encoder returned neither embeddings nor last_hidden_state.")
            # Statistics pooling is the standard utterance-level fallback for
            # speaker verification; a plain hidden-state mean discards useful
            # speaker-discriminative variance.
            mask = inputs.get("attention_mask")
            if mask is None or mask.shape[-1] != hidden.shape[1]:
                weights = torch.ones(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
            else:
                weights = mask.to(hidden.dtype)
            weights = weights.unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            mean = (hidden * weights).sum(dim=1) / denom
            variance = ((hidden - mean.unsqueeze(1)).square() * weights).sum(dim=1) / denom
            emb = torch.cat([mean, variance.clamp_min(1e-9).sqrt()], dim=-1).squeeze(0)
        return torch.nn.functional.normalize(emb.float(), dim=-1)

    def _load_ref_wav(self, path: str) -> Tuple[torch.Tensor, int]:
        try:
            import soundfile as sf

            data, sr = sf.read(path, always_2d=False)
            return mono_waveform(torch.tensor(data, dtype=torch.float32)), int(sr)
        except Exception as exc:
            raise ValueError(f"Failed to read ref_audio {path!r}: {exc}") from exc

    def _reference(self, meta: Dict[str, Any], generated_embedding: torch.Tensor) -> Tuple[torch.Tensor, str]:
        source: Any = None
        source_name = ""
        if meta.get("speaker_embedding") is not None:
            source = {"speaker_embedding": meta["speaker_embedding"]}
            source_name = "sample_embedding"
        elif meta.get("ref_audio") is not None:
            source = {
                "ref_audio": meta["ref_audio"],
                "sample_rate": meta.get("ref_audio_sample_rate"),
            }
            source_name = "sample_audio"
        else:
            speaker = str(meta.get("speaker_id") or meta.get("speaker") or "")
            if speaker and speaker in self._reference_bank:
                source = self._reference_bank[speaker]
                source_name = f"reference_bank:{speaker}"
        if source is None:
            raise ValueError(
                "speaker reference missing; provide per-sample ref_audio/speaker_embedding "
                "or a speaker key present in reference_bank_path"
            )

        if isinstance(source, list):
            source = {"speaker_embedding": source}
        elif isinstance(source, str):
            source = {"ref_audio": source}
        if not isinstance(source, dict):
            raise TypeError(f"Invalid speaker reference entry: {type(source).__name__}")

        if source.get("speaker_embedding") is not None:
            ref = torch.as_tensor(
                source["speaker_embedding"],
                dtype=torch.float32,
                device=generated_embedding.device,
            ).view(-1)
            ref = torch.nn.functional.normalize(ref, dim=-1)
        elif source.get("ref_audio") is not None:
            ref_audio = source["ref_audio"]
            if torch.is_tensor(ref_audio):
                ref_rate = int(source.get("sample_rate") or self._spec.default_sample_rate)
                ref_wav = ref_audio
            else:
                ref_path = Path(str(ref_audio))
                if not ref_path.is_absolute():
                    ref_path = self._reference_bank_dir / ref_path
                ref_wav, ref_rate = self._load_ref_wav(str(ref_path))
            ref = self._embed(ref_wav, ref_rate)
        else:
            raise ValueError("Speaker reference entry has neither speaker_embedding nor ref_audio.")
        if ref.numel() != generated_embedding.numel():
            raise ValueError(
                f"Speaker embedding dimension mismatch: generated={generated_embedding.numel()}, "
                f"reference={ref.numel()}."
            )
        return ref, source_name

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        if self._encoder is None:
            return unavailable_response(
                request.batch_size,
                component=self.canonical_model_name,
                reason=self._unavailable_reason or "speaker encoder is not loaded",
            )
        start = time.time()
        audios = request.audio
        if audios is None:
            raise ValueError("TTSSpeakerSimRewardScorer requires generated audio.")
        sr = int(request.audio_sample_rate or self._spec.default_sample_rate)
        rewards: List[float] = []
        raw_similarities: List[float] = []
        successes: List[bool] = []
        errors: List[str | None] = []
        details: List[Dict[str, Any]] = []
        for i, wav in enumerate(audios):
            try:
                if wav is None or wav.numel() == 0:
                    raise ValueError("generated audio is empty")
                meta = request.metadata[i] if request.metadata and i < len(request.metadata) else None
                meta = meta if isinstance(meta, dict) else {}
                gen_emb = self._embed(wav, sr)
                ref_emb, reference_source = self._reference(meta, gen_emb)
                similarity = float(torch.dot(gen_emb.view(-1), ref_emb.view(-1)).clamp(-1.0, 1.0).item())
                scale = self._spec.similarity_ceiling - self._spec.similarity_floor
                reward = (similarity - self._spec.similarity_floor) / scale
                rewards.append(float(max(0.0, min(1.0, reward))))
                raw_similarities.append(similarity)
                successes.append(True)
                errors.append(None)
                details.append(
                    {
                        "status": "available",
                        "raw_similarity": similarity,
                        "reference_source": reference_source,
                    }
                )
            except Exception as exc:
                rewards.append(0.0)
                raw_similarities.append(-1.0)
                successes.append(False)
                errors.append(f"{type(exc).__name__}: {exc}")
                details.append({"status": "failed", "error": errors[-1]})
        return RewardResponse(
            rewards=rewards,
            component_rewards={"raw_similarity": raw_similarities},
            details=details,
            successes=successes,
            errors=errors,
            compute_time=time.time() - start,
        )

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        return list(self.compute_rewards(request).rewards)

    def is_available(self) -> bool:
        return self._encoder is not None and not self._spec.dry_run

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind


@dataclass
class TTSSpeakerSimSpec(BaseRewardComponentSpec):
    batch_size: int = 4
    device: str = "auto"
    speaker_model_id: str = "microsoft/wavlm-base-plus-sv"
    reference_bank_path: str = ""
    default_sample_rate: int = 24000
    similarity_floor: float = 0.0
    similarity_ceiling: float = 1.0
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.similarity_ceiling <= self.similarity_floor:
            raise ValueError("similarity_ceiling must be greater than similarity_floor.")
