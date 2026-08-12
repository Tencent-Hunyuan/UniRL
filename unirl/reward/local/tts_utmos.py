"""Fail-closed UTMOS/DNSMOS perceptual quality reward."""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

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


class TTSUTMOSRewardScorer(LocalRewardBackend):
    """Map configured UTMOS or DNSMOS predictions into one common [0, 1] scale."""

    canonical_model_name = "tts_utmos"
    input_kind = "audio"

    def __init__(self, *, config: "TTSUTMOSSpec", base_device: str) -> None:
        self._spec = config
        self.training_eligible = not config.dry_run
        device = config.device if config.device and config.device != "auto" else base_device
        super().__init__(model_name="tts_utmos", device=device, batch_size=config.batch_size)
        self.input_kind = "audio"

    def _load_model(self) -> None:
        self._predictor = None
        self._unavailable_reason = None
        if self._spec.dry_run:
            self._unavailable_reason = "dry_run=True; MOS model was intentionally not loaded"
            return
        backend = self._spec.backend.strip().lower()
        try:
            if backend == "utmos":
                import utmos  # type: ignore

                parameters = inspect.signature(utmos.Score).parameters
                kwargs: Dict[str, Any] = {}
                if self._spec.model_path and "model_path" in parameters:
                    kwargs["model_path"] = self._spec.model_path
                if "device" in parameters:
                    kwargs["device"] = self.device
                self._predictor = utmos.Score(**kwargs)
                if self._spec.model_path and "model_path" not in parameters:
                    raise TypeError("Installed utmos.Score does not support configured model_path.")
            elif backend == "dnsmos":
                if not self._spec.model_path:
                    raise ValueError("DNSMOS requires model_path to an ONNX model.")
                import onnxruntime as ort

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if str(self.device).startswith("cuda")
                    else ["CPUExecutionProvider"]
                )
                self._predictor = ort.InferenceSession(self._spec.model_path, providers=providers)
            else:
                raise ValueError(f"Unsupported MOS backend {self._spec.backend!r}; expected utmos or dnsmos.")
            logger.info("TTSUTMOSRewardScorer: loaded %s backend", backend)
        except Exception as exc:
            raise RewardUnavailableError(
                f"TTSUTMOSRewardScorer failed to load {backend or '<empty>'} model: {exc}"
            ) from exc

    def _predict_mos(self, waveform: torch.Tensor, sample_rate: int) -> float:
        if self._predictor is None:
            raise RewardUnavailableError(self._unavailable_reason or "MOS model is not loaded")
        wav = resample_waveform(waveform, sample_rate, self._spec.model_sample_rate)
        if self._spec.backend == "utmos":
            return float(self._predictor.score(wav.numpy(), self._spec.model_sample_rate))

        import numpy as np

        session = self._predictor
        input_name = self._spec.dnsmos_input_name or session.get_inputs()[0].name
        outputs = session.run(None, {input_name: wav.numpy().astype(np.float32)[None, :]})
        if not outputs:
            raise RuntimeError("DNSMOS returned no outputs.")
        values = np.asarray(outputs[self._spec.dnsmos_output_tensor]).reshape(-1)
        if self._spec.dnsmos_output_index >= values.size:
            raise IndexError(
                f"DNSMOS output index {self._spec.dnsmos_output_index} exceeds output size {values.size}."
            )
        return float(values[self._spec.dnsmos_output_index])

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        if self._predictor is None:
            return unavailable_response(
                request.batch_size,
                component=self.canonical_model_name,
                reason=self._unavailable_reason or "MOS model is not loaded",
            )
        start = time.time()
        audios = request.audio
        if audios is None:
            raise ValueError("TTSUTMOSRewardScorer requires generated audio.")
        sr = int(request.audio_sample_rate or self._spec.default_sample_rate)
        rewards: List[float] = []
        raw_mos: List[float] = []
        quality_pass: List[float] = []
        successes: List[bool] = []
        errors: List[str | None] = []
        details: List[Dict[str, Any]] = []
        for wav in audios:
            try:
                if wav is None or wav.numel() == 0:
                    raise ValueError("generated audio is empty")
                if not torch.isfinite(wav).all():
                    raise ValueError("generated audio contains non-finite samples")
                mos = self._predict_mos(mono_waveform(wav), sr)
                if not torch.isfinite(torch.tensor(mos)):
                    raise ValueError(f"MOS model returned non-finite value {mos!r}")
                mapped = (mos - self._spec.mos_floor) / (self._spec.mos_ceiling - self._spec.mos_floor)
                reward = float(max(0.0, min(1.0, mapped)))
                passed = mos >= self._spec.mos_threshold
                rewards.append(reward)
                raw_mos.append(mos)
                quality_pass.append(float(passed))
                successes.append(True)
                errors.append(None)
                details.append(
                    {
                        "status": "available",
                        "backend": self._spec.backend,
                        "raw_mos": mos,
                        "mapped_reward": reward,
                        "threshold": self._spec.mos_threshold,
                        "passed": passed,
                    }
                )
            except Exception as exc:
                rewards.append(0.0)
                raw_mos.append(0.0)
                quality_pass.append(0.0)
                successes.append(False)
                errors.append(f"{type(exc).__name__}: {exc}")
                details.append({"status": "failed", "error": errors[-1]})
        return RewardResponse(
            rewards=rewards,
            component_rewards={"raw_mos": raw_mos, "quality_pass": quality_pass},
            details=details,
            successes=successes,
            errors=errors,
            compute_time=time.time() - start,
        )

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        return list(self.compute_rewards(request).rewards)

    def is_available(self) -> bool:
        return self._predictor is not None and not self._spec.dry_run

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind


@dataclass
class TTSUTMOSSpec(BaseRewardComponentSpec):
    batch_size: int = 4
    device: str = "auto"
    backend: str = "utmos"
    model_path: str = ""
    default_sample_rate: int = 24000
    model_sample_rate: int = 16000
    mos_floor: float = 1.0
    mos_ceiling: float = 5.0
    mos_threshold: float = 2.5
    dnsmos_input_name: str = ""
    dnsmos_output_tensor: int = 0
    # DNSMOS primary output is commonly [SIG, BAK, OVRL], so OVRL is index 2.
    dnsmos_output_index: int = 2
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.backend = self.backend.strip().lower()
        if self.backend not in {"utmos", "dnsmos"}:
            raise ValueError("TTSUTMOSSpec.backend must be 'utmos' or 'dnsmos'.")
        if self.mos_ceiling <= self.mos_floor:
            raise ValueError("mos_ceiling must be greater than mos_floor.")
        if not self.mos_floor <= self.mos_threshold <= self.mos_ceiling:
            raise ValueError("mos_threshold must be within [mos_floor, mos_ceiling].")
        if self.model_sample_rate <= 0:
            raise ValueError("model_sample_rate must be positive.")


__all__ = ["TTSUTMOSRewardScorer", "TTSUTMOSSpec"]
