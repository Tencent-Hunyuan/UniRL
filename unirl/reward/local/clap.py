"""CLAP audio-text alignment reward — LAION CLAP via HuggingFace transformers."""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

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
        self.prompt_metadata_key = str(config.prompt_metadata_key or "").strip() or None
        self.negative_prompts_metadata_key = str(config.negative_prompts_metadata_key or "").strip() or None
        self.event_prompts_metadata_key = str(config.event_prompts_metadata_key or "").strip() or None
        self.matched_cosine_weight = float(config.matched_cosine_weight)
        self.retrieval_margin_weight = float(config.retrieval_margin_weight)
        self.event_coverage_weight = float(config.event_coverage_weight)
        self.ast_event_weight = float(config.ast_event_weight)
        self.ast_model_id = str(config.ast_model_id).strip()
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
        self.audio_normalization = str(config.audio_normalization).strip().lower()
        self.target_rms_dbfs = float(config.target_rms_dbfs)
        self.peak_limit = float(config.peak_limit)
        if self.audio_normalization not in {"none", "rms"}:
            raise ValueError("CLAPSpec.audio_normalization must be 'none' or 'rms'.")
        if not math.isfinite(self.target_rms_dbfs) or self.target_rms_dbfs > 0.0:
            raise ValueError("CLAPSpec.target_rms_dbfs must be finite and at most 0 dBFS.")
        if not math.isfinite(self.peak_limit) or not 0.0 < self.peak_limit <= 1.0:
            raise ValueError("CLAPSpec.peak_limit must be in (0, 1].")
        self.retrieval_prompts = [str(prompt).strip() for prompt in config.retrieval_prompts]
        if any(not prompt for prompt in self.retrieval_prompts):
            raise ValueError("CLAPSpec.retrieval_prompts must contain only non-empty strings.")
        if len(set(self.retrieval_prompts)) != len(self.retrieval_prompts):
            raise ValueError("CLAPSpec.retrieval_prompts must not contain duplicates.")
        if self.retrieval_prompts and len(self.retrieval_prompts) < 2:
            raise ValueError("CLAPSpec.retrieval_prompts needs at least two prompts for retrieval diagnostics.")
        if self.retrieval_prompts and self.negative_prompts_metadata_key:
            raise ValueError("CLAPSpec.retrieval_prompts and negative_prompts_metadata_key are mutually exclusive.")
        if self.retrieval_margin_weight > 0.0 and not (self.retrieval_prompts or self.negative_prompts_metadata_key):
            raise ValueError(
                "CLAPSpec.retrieval_margin_weight requires retrieval_prompts or negative_prompts_metadata_key."
            )
        if self.event_coverage_weight > 0.0 and not self.event_prompts_metadata_key:
            raise ValueError("CLAPSpec.event_coverage_weight requires event_prompts_metadata_key.")
        if self.ast_event_weight > 0.0 and not self.event_prompts_metadata_key:
            raise ValueError("CLAPSpec.ast_event_weight requires event_prompts_metadata_key.")
        if self.ast_event_weight > 0.0 and not self.ast_model_id:
            raise ValueError("CLAPSpec.ast_model_id must be non-empty when ast_event_weight is positive.")
        self._retrieval_text_embeds: Optional[torch.Tensor] = None
        self.ast_model = None
        self.ast_processor = None
        self._ast_sample_rate: Optional[int] = None
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
        self._processor_audio_keyword = (
            "audio" if "audio" in inspect.signature(self.processor.__call__).parameters else "audios"
        )
        if self.ast_event_weight > 0.0:
            try:
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            except ImportError as e:
                raise ImportError(
                    "transformers with AutoModelForAudioClassification is required for the AST event reward"
                ) from e

            self.ast_processor = AutoFeatureExtractor.from_pretrained(self.ast_model_id)
            self.ast_model = (
                AutoModelForAudioClassification.from_pretrained(self.ast_model_id)
                .to(self.device)
                .eval()
                .to(dtype=torch.float32)
            )
            self._ast_sample_rate = int(self.ast_processor.sampling_rate)
            self._ast_label_to_index = {
                str(label).strip().lower(): int(index)
                for index, label in self.ast_model.config.id2label.items()
            }

    def _preprocess_audio(self, audio_list: List[torch.Tensor], src_sample_rate: int) -> List["torch.Tensor"]:
        """Downmix and resample each ``[L]`` / ``[C, L]`` / ``[L, C]`` waveform to CLAP's 48 kHz mono ``[L']``."""
        import numpy as np

        processed: List[np.ndarray] = []
        for waveform in audio_list:
            wf = waveform.detach().float()
            if wf.isnan().any() or wf.isinf().any():
                wf = torch.zeros_like(wf)
            if wf.ndim == 2:
                ch_axis = 0 if wf.shape[0] <= wf.shape[1] else 1
                wf = wf.mean(dim=ch_axis)
            wf = wf.reshape(-1)

            if src_sample_rate != self.CLAP_SAMPLE_RATE:
                import torchaudio.functional as AF

                wf = AF.resample(
                    wf.unsqueeze(0),
                    orig_freq=int(src_sample_rate),
                    new_freq=self.CLAP_SAMPLE_RATE,
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

    @staticmethod
    def _feature_tensor(value: Any) -> torch.Tensor:
        """Normalize the tensor/wrapper return types used across transformers releases."""
        if torch.is_tensor(value):
            return value
        pooled = getattr(value, "pooler_output", None)
        if torch.is_tensor(pooled):
            return pooled
        if isinstance(value, tuple) and value and torch.is_tensor(value[0]):
            return value[0]
        raise TypeError(f"Cannot extract CLAP feature tensor from {type(value).__name__}.")

    def _encode_texts(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.model.get_text_features(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
            )
        return F.normalize(self._feature_tensor(features).float(), p=2, dim=-1)

    def _encode_audio(self, waveforms: List[torch.Tensor], src_sample_rate: int) -> torch.Tensor:
        waveforms_np = self._preprocess_audio(waveforms, src_sample_rate)
        inputs = self.processor(
            **{self._processor_audio_keyword: waveforms_np},
            sampling_rate=self.CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            features = self.model.get_audio_features(
                input_features=inputs.get("input_features"),
                is_longer=inputs.get("is_longer"),
                attention_mask=inputs.get("attention_mask"),
            )
        return F.normalize(self._feature_tensor(features).float(), p=2, dim=-1)

    def _encode_ast_events(
        self,
        waveforms: List[torch.Tensor],
        src_sample_rate: int,
        event_prompts: List[List[str]],
    ) -> torch.Tensor:
        """Return the minimum AudioSet event probability requested by each sample."""
        if self.ast_model is None or self.ast_processor is None or self._ast_sample_rate is None:
            raise RuntimeError("AST event reward is enabled but the AST model is not loaded.")

        import numpy as np
        processed: List[np.ndarray] = []
        for waveform in waveforms:
            wf = waveform.detach().float()
            if wf.isnan().any() or wf.isinf().any():
                wf = torch.zeros_like(wf)
            if wf.ndim == 2:
                channel_axis = 0 if wf.shape[0] <= wf.shape[1] else 1
                wf = wf.mean(dim=channel_axis)
            wf = wf.reshape(-1)
            if src_sample_rate != self._ast_sample_rate:
                if src_sample_rate % self._ast_sample_rate == 0:
                    factor = src_sample_rate // self._ast_sample_rate
                    wf = F.avg_pool1d(
                        wf.reshape(1, 1, -1),
                        kernel_size=factor,
                        stride=factor,
                    ).reshape(-1)
                else:
                    target_length = max(1, round(wf.numel() * self._ast_sample_rate / src_sample_rate))
                    wf = F.interpolate(
                        wf.reshape(1, 1, -1),
                        size=target_length,
                        mode="linear",
                        align_corners=False,
                    ).reshape(-1)
            if self.audio_normalization == "rms":
                rms = wf.square().mean().sqrt()
                peak = wf.abs().max()
                if rms > torch.finfo(wf.dtype).eps:
                    gain = (10.0 ** (self.target_rms_dbfs / 20.0)) / rms
                    if peak * gain > self.peak_limit:
                        gain = self.peak_limit / peak
                    wf = wf * gain
            processed.append(wf.cpu().numpy())

        inputs = self.ast_processor(
            processed,
            sampling_rate=self._ast_sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            probabilities = self.ast_model(**inputs).logits.float().sigmoid()

        scores = []
        for row_index, labels in enumerate(event_prompts):
            missing = [label for label in labels if label.strip().lower() not in self._ast_label_to_index]
            if missing:
                raise ValueError(f"AST model does not define AudioSet event labels: {missing!r}.")
            indices = torch.tensor(
                [self._ast_label_to_index[label.strip().lower()] for label in labels],
                device=probabilities.device,
                dtype=torch.long,
            )
            scores.append(probabilities[row_index].index_select(0, indices).min())
        return torch.stack(scores)

    def _reward_prompts(self, request: RewardRequest) -> List[str]:
        prompts = self.prompts(request)
        if self.prompt_metadata_key is None:
            return prompts

        metadata = request.metadata or []
        resolved: List[str] = []
        for index, prompt in enumerate(prompts):
            row = metadata[index] if index < len(metadata) else None
            candidate = row.get(self.prompt_metadata_key) if isinstance(row, dict) else None
            resolved.append(candidate.strip() if isinstance(candidate, str) and candidate.strip() else prompt)
        return resolved

    def _negative_prompts(self, request: RewardRequest, reward_prompts: List[str]) -> Optional[List[List[str]]]:
        if self.negative_prompts_metadata_key is None:
            return None

        metadata = request.metadata or []
        resolved: List[List[str]] = []
        for index, target_prompt in enumerate(reward_prompts):
            row = metadata[index] if index < len(metadata) else None
            candidates = row.get(self.negative_prompts_metadata_key) if isinstance(row, dict) else None
            if not isinstance(candidates, (list, tuple)):
                raise ValueError(
                    "CLAP negative prompt metadata must be a list of strings; "
                    f"key={self.negative_prompts_metadata_key!r}, sample_index={index}."
                )
            negatives = list(
                dict.fromkeys(
                    candidate.strip()
                    for candidate in candidates
                    if isinstance(candidate, str) and candidate.strip() and candidate.strip() != target_prompt
                )
            )
            if not negatives:
                raise ValueError(
                    "CLAP negative prompt metadata must contain at least one non-target caption; "
                    f"key={self.negative_prompts_metadata_key!r}, sample_index={index}."
                )
            resolved.append(negatives)
        return resolved

    def _event_prompts(self, request: RewardRequest) -> Optional[List[List[str]]]:
        """Resolve per-sample sound-event prompts used for minimum-coverage scoring."""
        if self.event_prompts_metadata_key is None:
            return None

        metadata = request.metadata or []
        resolved: List[List[str]] = []
        for index in range(len(self.prompts(request))):
            row = metadata[index] if index < len(metadata) else None
            candidates = row.get(self.event_prompts_metadata_key) if isinstance(row, dict) else None
            if not isinstance(candidates, (list, tuple)):
                raise ValueError(
                    "CLAP event prompt metadata must be a list of strings; "
                    f"key={self.event_prompts_metadata_key!r}, sample_index={index}."
                )
            event_prompts = list(
                dict.fromkeys(
                    candidate.strip() for candidate in candidates if isinstance(candidate, str) and candidate.strip()
                )
            )
            if not event_prompts:
                raise ValueError(
                    "CLAP event prompt metadata must contain at least one event; "
                    f"key={self.event_prompts_metadata_key!r}, sample_index={index}."
                )
            resolved.append(event_prompts)
        return resolved

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Score matched captions and expose optional retrieval diagnostics."""
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
        prompts = self._reward_prompts(request)
        if audio is None:
            raise ValueError(
                "CLAPRewardScorer requires audio in the reward request "
                "(request.generated['audio']); got none. Ensure the pipeline "
                "decodes audio into Part.primitives['audio'] for LTX-2.3 T2AV."
            )
        if request.audio_sample_rate is None:
            raise ValueError("CLAPRewardScorer requires request.audio_sample_rate (source Hz); got None.")
        if len(audio) != len(prompts):
            raise ValueError(f"CLAPRewardScorer got {len(audio)} audio samples but {len(prompts)} reward prompts.")
        src_rate = int(request.audio_sample_rate)
        negative_prompt_sets = self._negative_prompts(request, prompts)
        event_prompt_sets = self._event_prompts(request)
        dynamic_text_embeds: Optional[torch.Tensor] = None
        dynamic_prompt_to_index: Dict[str, int] = {}
        if not self.retrieval_prompts or event_prompt_sets is not None:
            dynamic_prompts = [] if self.retrieval_prompts else list(prompts)
            if negative_prompt_sets is not None:
                dynamic_prompts.extend(prompt for row in negative_prompt_sets for prompt in row)
            if event_prompt_sets is not None:
                dynamic_prompts.extend(prompt for row in event_prompt_sets for prompt in row)
            unique_dynamic_prompts = list(dict.fromkeys(dynamic_prompts))
            dynamic_text_embeds = self._encode_texts(unique_dynamic_prompts)
            dynamic_prompt_to_index = {prompt: index for index, prompt in enumerate(unique_dynamic_prompts)}

        all_rewards: List[float] = []
        component_rewards: Dict[str, List[float]] = {"matched_cosine": []}
        retrieval_prompt_to_index = {prompt: index for index, prompt in enumerate(self.retrieval_prompts)}
        if self.retrieval_prompts or negative_prompt_sets is not None:
            component_rewards.update(
                {
                    "mismatched_cosine": [],
                    "retrieval_margin": [],
                    "retrieval_top1": [],
                }
            )
        if event_prompt_sets is not None:
            component_rewards["event_coverage_min_cosine"] = []
        if self.ast_event_weight > 0.0:
            component_rewards["ast_event_min_probability"] = []
        if self.retrieval_prompts:
            unknown = sorted(set(prompts) - set(retrieval_prompt_to_index))
            if unknown:
                raise ValueError(
                    "CLAP reward prompts must appear in CLAPSpec.retrieval_prompts when retrieval diagnostics "
                    f"are enabled; missing={unknown[:3]!r}."
                )

        for i in range(0, len(audio), self.batch_size):
            batch_audio = audio[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]
            audio_embeds = self._encode_audio(batch_audio, src_rate)

            if self.retrieval_prompts:
                score_matrix = audio_embeds @ self._get_retrieval_text_embeds().T
                target_indices = torch.tensor(
                    [retrieval_prompt_to_index[prompt] for prompt in batch_prompts],
                    device=score_matrix.device,
                    dtype=torch.long,
                )
                rows = torch.arange(score_matrix.shape[0], device=score_matrix.device)
                matched = score_matrix[rows, target_indices]
                negative_mask = torch.ones_like(score_matrix, dtype=torch.bool)
                negative_mask[rows, target_indices] = False
                negative_scores = score_matrix[negative_mask].view(score_matrix.shape[0], -1)
                mismatched = negative_scores.mean(dim=-1)
                margin = matched - negative_scores.max(dim=-1).values
                top1 = (score_matrix.argmax(dim=-1) == target_indices).float()
            elif negative_prompt_sets is not None:
                assert dynamic_text_embeds is not None
                target_indices = torch.tensor(
                    [dynamic_prompt_to_index[prompt] for prompt in batch_prompts],
                    device=dynamic_text_embeds.device,
                    dtype=torch.long,
                )
                text_embeds = dynamic_text_embeds.index_select(0, target_indices)
                matched = (audio_embeds * text_embeds).sum(dim=-1)
                batch_negative_prompts = negative_prompt_sets[i : i + self.batch_size]
                mismatched_values = []
                hardest_negative_values = []
                for row_index, row_prompts in enumerate(batch_negative_prompts):
                    indices = torch.tensor(
                        [dynamic_prompt_to_index[prompt] for prompt in row_prompts],
                        device=dynamic_text_embeds.device,
                        dtype=torch.long,
                    )
                    row_negative_embeds = dynamic_text_embeds.index_select(0, indices)
                    row_negative_scores = row_negative_embeds @ audio_embeds[row_index]
                    mismatched_values.append(row_negative_scores.mean())
                    hardest_negative_values.append(row_negative_scores.max())
                mismatched = torch.stack(mismatched_values)
                hardest_negative = torch.stack(hardest_negative_values)
                margin = matched - hardest_negative
                top1 = (matched > hardest_negative).float()
            else:
                assert dynamic_text_embeds is not None
                target_indices = torch.tensor(
                    [dynamic_prompt_to_index[prompt] for prompt in batch_prompts],
                    device=dynamic_text_embeds.device,
                    dtype=torch.long,
                )
                text_embeds = dynamic_text_embeds.index_select(0, target_indices)
                matched = (audio_embeds * text_embeds).sum(dim=-1)

            event_coverage: Optional[torch.Tensor] = None
            if event_prompt_sets is not None:
                assert dynamic_text_embeds is not None
                event_values = []
                for row_index, row_prompts in enumerate(event_prompt_sets[i : i + self.batch_size]):
                    indices = torch.tensor(
                        [dynamic_prompt_to_index[prompt] for prompt in row_prompts],
                        device=dynamic_text_embeds.device,
                        dtype=torch.long,
                    )
                    row_event_embeds = dynamic_text_embeds.index_select(0, indices)
                    event_values.append((row_event_embeds @ audio_embeds[row_index]).min())
                event_coverage = torch.stack(event_values)
                component_rewards["event_coverage_min_cosine"].extend(event_coverage.float().cpu().tolist())

            ast_event_score: Optional[torch.Tensor] = None
            if self.ast_event_weight > 0.0:
                assert event_prompt_sets is not None
                ast_event_score = self._encode_ast_events(
                    batch_audio,
                    src_rate,
                    event_prompt_sets[i : i + self.batch_size],
                )
                component_rewards["ast_event_min_probability"].extend(ast_event_score.float().cpu().tolist())

            if self.retrieval_prompts or negative_prompt_sets is not None:
                component_rewards["mismatched_cosine"].extend(mismatched.float().cpu().tolist())
                component_rewards["retrieval_margin"].extend(margin.float().cpu().tolist())
                component_rewards["retrieval_top1"].extend(top1.cpu().tolist())

            reward = matched * self.matched_cosine_weight
            if self.retrieval_margin_weight > 0.0:
                reward = reward + margin * self.retrieval_margin_weight
            if self.event_coverage_weight > 0.0:
                assert event_coverage is not None
                reward = reward + event_coverage * self.event_coverage_weight
            if self.ast_event_weight > 0.0:
                assert ast_event_score is not None
                reward = reward + ast_event_score * self.ast_event_weight

            matched_values = matched.float().cpu().tolist()
            all_rewards.extend(reward.float().cpu().tolist())
            component_rewards["matched_cosine"].extend(matched_values)

        return all_rewards, component_rewards

    def _get_retrieval_text_embeds(self) -> torch.Tensor:
        if self._retrieval_text_embeds is None:
            self._retrieval_text_embeds = self._encode_texts(self.retrieval_prompts)
        return self._retrieval_text_embeds

    def offload(self) -> None:
        super().offload()
        if self.ast_model is not None:
            self.ast_model = self.ast_model.cpu()
        if self._retrieval_text_embeds is not None:
            self._retrieval_text_embeds = self._retrieval_text_embeds.cpu()

    def onload(self) -> None:
        super().onload()
        if self.ast_model is not None:
            self.ast_model = self.ast_model.to(self.device)
        if self._retrieval_text_embeds is not None:
            self._retrieval_text_embeds = self._retrieval_text_embeds.to(self.device)


@dataclass
class CLAPSpec(PromptRewardComponentSpec):
    """Typed config for the CLAP audio-text reward component."""

    batch_size: int = 8
    device: str = "auto"
    model_id: str = "laion/larger_clap_general"
    prompt_metadata_key: Optional[str] = None
    negative_prompts_metadata_key: Optional[str] = None
    event_prompts_metadata_key: Optional[str] = None
    retrieval_prompts: List[str] = field(default_factory=list)
    matched_cosine_weight: float = 1.0
    retrieval_margin_weight: float = 0.0
    event_coverage_weight: float = 0.0
    ast_event_weight: float = 0.0
    ast_model_id: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    audio_normalization: str = "none"
    target_rms_dbfs: float = -20.0
    peak_limit: float = 0.95
