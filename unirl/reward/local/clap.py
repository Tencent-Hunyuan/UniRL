"""CLAP audio-text alignment reward — LAION CLAP via HuggingFace transformers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from unirl.reward.base import PromptRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest, RewardResponse

from .base import LocalRewardBackend


class CLAPRewardScorer(LocalRewardBackend):
    """Audio-text alignment reward using LAION CLAP."""

    canonical_model_name = "clap"
    input_kind = "video"
    CLAP_SAMPLE_RATE = 48_000

    def __init__(self, *, config: "CLAPSpec", base_device: str) -> None:
        self.negative_prompts_metadata_key = config.negative_prompts_metadata_key
        self.event_prompts_metadata_key = config.event_prompts_metadata_key
        self.matched_cosine_weight = float(config.matched_cosine_weight)
        self.retrieval_margin_weight = float(config.retrieval_margin_weight)
        self.event_coverage_weight = float(config.event_coverage_weight)
        self.ast_event_weight = float(config.ast_event_weight)
        self.ast_model_id = config.ast_model_id
        reward_weights = (
            self.matched_cosine_weight,
            self.retrieval_margin_weight,
            self.event_coverage_weight,
            self.ast_event_weight,
        )
        if not all(math.isfinite(weight) for weight in reward_weights):
            raise ValueError("CLAP reward weights must be finite.")
        if any(weight < 0.0 for weight in reward_weights):
            raise ValueError("CLAP reward weights must be non-negative.")
        if not any(weight > 0.0 for weight in reward_weights):
            raise ValueError("At least one CLAP reward weight must be positive.")
        if self.retrieval_margin_weight > 0.0 and not self.negative_prompts_metadata_key:
            raise ValueError("CLAPSpec.retrieval_margin_weight requires negative_prompts_metadata_key.")
        if self.event_coverage_weight > 0.0 and not self.event_prompts_metadata_key:
            raise ValueError("CLAPSpec.event_coverage_weight requires event_prompts_metadata_key.")
        if self.ast_event_weight > 0.0 and not self.event_prompts_metadata_key:
            raise ValueError("CLAPSpec.ast_event_weight requires event_prompts_metadata_key.")
        self.audio_normalization = config.audio_normalization
        self.target_rms_dbfs = float(config.target_rms_dbfs)
        self.peak_limit = float(config.peak_limit)
        if self.audio_normalization not in {"none", "rms"}:
            raise ValueError("CLAPSpec.audio_normalization must be 'none' or 'rms'.")
        if not math.isfinite(self.target_rms_dbfs) or self.target_rms_dbfs > 0.0:
            raise ValueError("CLAPSpec.target_rms_dbfs must be finite and at most 0 dBFS.")
        if not math.isfinite(self.peak_limit) or not 0.0 < self.peak_limit <= 1.0:
            raise ValueError("CLAPSpec.peak_limit must be in (0, 1].")
        self.ast_model = None
        self.ast_processor = None
        self._ast_label_to_index: Dict[str, int] = {}
        super().__init__(
            device=resolve_device(config.device, base_device),
            batch_size=config.batch_size,
            prompt_source=config.prompt_source,
            model_id=config.model_id,
        )

    def _load_model(self) -> None:
        try:
            from transformers import ClapModel, ClapProcessor
        except ImportError as e:
            raise ImportError("transformers with ClapModel/ClapProcessor is required for the CLAP reward") from e

        model_id = self.model_kwargs.get("model_id", "laion/larger_clap_general")
        self.model = ClapModel.from_pretrained(model_id).to(self.device).eval()
        self.model = self.model.to(dtype=torch.float32)
        self.processor = ClapProcessor.from_pretrained(model_id)
        if self.ast_event_weight > 0.0:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            self.ast_processor = AutoFeatureExtractor.from_pretrained(self.ast_model_id)
            self.ast_model = (
                AutoModelForAudioClassification.from_pretrained(self.ast_model_id)
                .to(self.device)
                .eval()
                .to(dtype=torch.float32)
            )
            self._ast_label_to_index = {
                str(label).lower(): int(index) for index, label in self.ast_model.config.id2label.items()
            }

    def _prepare_waveforms(
        self, audio_list: List[torch.Tensor], src_sample_rate: int, target_sample_rate: int
    ) -> List[np.ndarray]:
        """Downmix, resample, and optionally RMS-normalize each ``[L]`` / ``[C, L]`` / ``[L, C]`` waveform."""
        processed: List[np.ndarray] = []
        for waveform in audio_list:
            wf = waveform.detach().float()
            if not torch.isfinite(wf).all():
                wf = torch.zeros_like(wf)
            if wf.ndim == 2:
                ch_axis = 0 if wf.shape[0] <= wf.shape[1] else 1
                wf = wf.mean(dim=ch_axis)
            wf = wf.reshape(-1)

            if src_sample_rate != target_sample_rate:
                import torchaudio.functional as AF

                wf = AF.resample(
                    wf.unsqueeze(0),
                    orig_freq=int(src_sample_rate),
                    new_freq=int(target_sample_rate),
                ).squeeze(0)

            if self.audio_normalization == "rms":
                rms = wf.square().mean().sqrt()
                peak = wf.abs().max()
                if rms > torch.finfo(wf.dtype).eps:
                    gain = (10.0 ** (self.target_rms_dbfs / 20.0)) / rms
                    if peak * gain > self.peak_limit:
                        gain = self.peak_limit / peak
                    wf = wf * gain

            processed.append(wf.cpu().numpy())
        return processed

    def _encode_texts(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            return self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            ).pooler_output

    def _encode_audio(self, waveforms: List[torch.Tensor], src_sample_rate: int) -> torch.Tensor:
        inputs = self.processor(
            audio=self._prepare_waveforms(waveforms, src_sample_rate, self.CLAP_SAMPLE_RATE),
            sampling_rate=self.CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            return self.model.get_audio_features(
                input_features=inputs["input_features"],
                is_longer=inputs.get("is_longer"),
                attention_mask=inputs.get("attention_mask"),
            ).pooler_output

    def _encode_ast_events(
        self,
        waveforms: List[torch.Tensor],
        src_sample_rate: int,
        event_prompts: List[List[str]],
    ) -> List[float]:
        """Return the minimum AudioSet event probability requested by each sample."""
        ast_sample_rate = int(self.ast_processor.sampling_rate)
        inputs = self.ast_processor(
            self._prepare_waveforms(waveforms, src_sample_rate, ast_sample_rate),
            sampling_rate=ast_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            probabilities = self.ast_model(**inputs).logits.float().sigmoid()

        scores: List[float] = []
        for row, labels in zip(probabilities, event_prompts):
            missing = [label for label in labels if label.lower() not in self._ast_label_to_index]
            if missing:
                raise ValueError(f"AST model does not define AudioSet event labels: {missing!r}.")
            indices = torch.tensor([self._ast_label_to_index[label.lower()] for label in labels], device=row.device)
            scores.append(float(row.index_select(0, indices).min()))
        return scores

    @staticmethod
    def _metadata_prompt_sets(request: RewardRequest, key: str, exclude: Optional[List[str]] = None) -> List[List[str]]:
        """Read one non-empty, de-duplicated caption list per sample from ``request.metadata[i][key]``."""
        if request.metadata is None:
            raise ValueError(f"CLAPRewardScorer needs per-sample metadata[{key!r}]; the request has no metadata.")
        resolved: List[List[str]] = []
        for index, row in enumerate(request.metadata):
            candidates = row.get(key) if isinstance(row, dict) else None
            if not isinstance(candidates, (list, tuple)):
                raise ValueError(f"CLAP metadata[{key!r}] must be a list of strings; sample_index={index}.")
            excluded = exclude[index] if exclude is not None else None
            prompts = list(
                dict.fromkeys(
                    candidate.strip()
                    for candidate in candidates
                    if isinstance(candidate, str) and candidate.strip() and candidate.strip() != excluded
                )
            )
            if not prompts:
                raise ValueError(f"CLAP metadata[{key!r}] has no usable caption; sample_index={index}.")
            resolved.append(prompts)
        return resolved

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Score matched captions and expose the retrieval and event diagnostics as components."""
        if not self._is_loaded:
            raise RuntimeError(
                f"{type(self).__name__}.compute_rewards called before _load_model "
                f"completed (model_name={self.model_name!r}, batch_size={request.batch_size})."
            )
        start = time.time()
        rewards, components = self._compute_rewards_and_components(request)
        return RewardResponse(
            rewards=rewards,
            component_rewards=components,
            successes=[True] * len(rewards),
            errors=[None] * len(rewards),
            compute_time=time.time() - start,
        )

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        rewards, _ = self._compute_rewards_and_components(request)
        return rewards

    def _compute_rewards_and_components(self, request: RewardRequest) -> tuple[List[float], Dict[str, List[float]]]:
        audio = request.audio
        prompts = self.prompts(request)
        if audio is None:
            raise ValueError(
                "CLAPRewardScorer requires audio in the reward request "
                "(request.generated['audio']); got none. Ensure the pipeline "
                "decodes audio into Part.primitives['audio'] for LTX-2.3 T2AV."
            )
        if request.audio_sample_rate is None:
            raise ValueError("CLAPRewardScorer requires request.audio_sample_rate (source Hz); got None.")
        src_rate = int(request.audio_sample_rate)
        negative_sets = (
            self._metadata_prompt_sets(request, self.negative_prompts_metadata_key, exclude=prompts)
            if self.negative_prompts_metadata_key
            else None
        )
        event_sets = (
            self._metadata_prompt_sets(request, self.event_prompts_metadata_key)
            if self.event_prompts_metadata_key
            else None
        )

        texts = list(prompts)
        for prompt_sets in (negative_sets, event_sets):
            if prompt_sets is not None:
                texts.extend(prompt for row in prompt_sets for prompt in row)
        unique_texts = list(dict.fromkeys(texts))
        text_embeds = self._encode_texts(unique_texts)
        text_index = {text: index for index, text in enumerate(unique_texts)}

        def cosines(row_texts: List[str], audio_embed: torch.Tensor) -> torch.Tensor:
            rows = torch.tensor([text_index[text] for text in row_texts], device=text_embeds.device)
            return text_embeds.index_select(0, rows) @ audio_embed

        components: Dict[str, List[float]] = {"matched_cosine": []}
        if negative_sets is not None:
            components.update(mismatched_cosine=[], retrieval_margin=[], retrieval_top1=[])
        if event_sets is not None:
            components["event_coverage_min_cosine"] = []
        if self.ast_event_weight > 0.0:
            components["ast_event_min_probability"] = []

        rewards: List[float] = []
        for batch_start in range(0, len(audio), self.batch_size):
            batch_audio = audio[batch_start : batch_start + self.batch_size]
            audio_embeds = self._encode_audio(batch_audio, src_rate)
            ast_scores = (
                self._encode_ast_events(batch_audio, src_rate, event_sets[batch_start : batch_start + len(batch_audio)])
                if self.ast_event_weight > 0.0
                else None
            )
            for offset, audio_embed in enumerate(audio_embeds):
                index = batch_start + offset
                matched = float(cosines([prompts[index]], audio_embed)[0])
                reward = self.matched_cosine_weight * matched
                components["matched_cosine"].append(matched)
                if negative_sets is not None:
                    negative_scores = cosines(negative_sets[index], audio_embed)
                    hardest_negative = float(negative_scores.max())
                    margin = matched - hardest_negative
                    reward += self.retrieval_margin_weight * margin
                    components["mismatched_cosine"].append(float(negative_scores.mean()))
                    components["retrieval_margin"].append(margin)
                    components["retrieval_top1"].append(float(matched > hardest_negative))
                if event_sets is not None:
                    coverage = float(cosines(event_sets[index], audio_embed).min())
                    reward += self.event_coverage_weight * coverage
                    components["event_coverage_min_cosine"].append(coverage)
                if ast_scores is not None:
                    reward += self.ast_event_weight * ast_scores[offset]
                    components["ast_event_min_probability"].append(ast_scores[offset])
                rewards.append(reward)

        return rewards, components

    def offload(self) -> None:
        super().offload()
        if self.ast_model is not None:
            self.ast_model = self.ast_model.cpu()

    def onload(self) -> None:
        super().onload()
        if self.ast_model is not None:
            self.ast_model = self.ast_model.to(self.device)


@dataclass
class CLAPSpec(PromptRewardComponentSpec):
    """Typed config for the CLAP audio-text reward component."""

    batch_size: int = 8
    device: str = "auto"
    model_id: str = "laion/larger_clap_general"
    negative_prompts_metadata_key: Optional[str] = None
    event_prompts_metadata_key: Optional[str] = None
    matched_cosine_weight: float = 1.0
    retrieval_margin_weight: float = 0.0
    event_coverage_weight: float = 0.0
    ast_event_weight: float = 0.0
    ast_model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    audio_normalization: str = "none"
    target_rms_dbfs: float = -20.0
    peak_limit: float = 0.95
