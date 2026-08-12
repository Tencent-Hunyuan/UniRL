"""Fail-closed Qwen3-Omni direct-TTS vLLM-Omni adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from unirl.config.require import require
from unirl.models.qwen3_omni.talker_conditions import Qwen3OmniTalkerConditions
from unirl.models.qwen3_omni.talker_contract import AUDIO_SAMPLE_RATE, NUM_CODE_GROUPS
from unirl.models.qwen3_omni.talker_sampling import (
    TalkerSamplingConfig,
    suppress_special_codec_ids,
)
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.capabilities import (
    DIRECT_TTS_CONTRACT_VERSION,
    require_direct_tts_runtime,
)
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Audio, Audios, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import SegmentStatus, TextSegment

_TALKER_STAGE_ID = 0
_CODE2WAV_STAGE_ID = 1
_RAW_CONTRACT_KEY = "unirl_qwen3_omni_direct_tts"


def build_direct_tts_prefix_ids(prompt_ids: Sequence[int], config: Any) -> List[int]:
    """Mirror direct-TTS prefix ID construction without running Thinker."""

    ids = [int(token_id) for token_id in prompt_ids]
    starts = [index for index, token_id in enumerate(ids) if token_id == int(config.im_start_token_id)]
    starts.append(len(ids))
    if len(starts) < 3:
        raise ValueError("direct-TTS prompt must contain ChatML user and assistant spans")

    prefix: List[int] = []
    saw_final_assistant = False
    for span_index, (start, end) in enumerate(zip(starts, starts[1:])):
        if start + 1 >= end:
            raise ValueError("direct-TTS prompt contains an empty ChatML role span")
        role = ids[start + 1]
        if role == int(config.system_token_id):
            continue
        if role == int(config.user_token_id):
            prefix.extend(ids[start:end])
            continue
        if role == int(config.assistant_token_id) and span_index == len(starts) - 2:
            if end - start < 4:
                raise ValueError("direct-TTS final assistant span is too short for the <audio> bootstrap")
            # Official prefix: 3 text slots + 4 codec control slots + TTS BOS
            # + first assistant content slot. talker_input_ids uses TTS PAD for
            # all nine because the actual codec controls ride in embeddings.
            prefix.extend([int(config.tts_pad_token_id)] * 9)
            saw_final_assistant = True
            continue
        if role == int(config.assistant_token_id):
            continue
        raise ValueError(f"direct-TTS prompt has unsupported ChatML role token {role}")
    if not saw_final_assistant or not prefix:
        raise ValueError("direct-TTS prompt is missing the final assistant <audio> span")
    return prefix


@dataclass(frozen=True)
class DirectTTSPreparedRequest:
    prompt_ids: List[int]
    speaker_id: int
    prefix_ids: List[int]
    behavior_sampling: Dict[str, Any]


class Qwen3OmniTalkerInputAdapter:
    """Build the patched runtime's direct prefix→Talker→Code2Wav request."""

    def __init__(
        self,
        modality: str,
        *,
        model_path: str,
        max_prompt_length: int = 2048,
        system_instruction: Optional[str] = None,
        default_speaker: str = "Ethan",
        repetition_penalty: float = 1.05,
        suppress_special_tokens: bool = True,
    ) -> None:
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer

        self.modality = modality
        self.model_path = str(model_path)
        self.max_prompt_length = int(max_prompt_length)
        self.system_instruction = system_instruction or (
            "You are a high-quality TTS model. Convert the user's text into natural speech audio."
        )
        self.default_speaker = str(default_speaker)
        self.repetition_penalty = float(repetition_penalty)
        self.suppress_special_tokens = bool(suppress_special_tokens)
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self._last_requests: List[DirectTTSPreparedRequest] = []

    def _speaker_id(self, metadata: Any) -> int:
        mapping = getattr(self._config.talker_config, "speaker_id", None) or {}
        if isinstance(metadata, dict) and metadata.get("speaker_id") is not None:
            speaker_id = int(metadata["speaker_id"])
            if speaker_id not in {int(value) for value in mapping.values()}:
                raise ValueError(
                    f"Unknown Talker speaker_id {speaker_id}; known={sorted(mapping.values())}"
                )
            return speaker_id
        speaker = (
            str(metadata["speaker"])
            if isinstance(metadata, dict) and metadata.get("speaker")
            else self.default_speaker
        )
        speaker_id = mapping.get(speaker.lower())
        if speaker_id is None:
            raise ValueError(f"Unknown Talker speaker {speaker!r}; known={sorted(mapping)}")
        return int(speaker_id)

    def _prompt_ids(self, text: str) -> List[int]:
        from unirl.models.qwen3_omni.talker_pipeline import build_tts_messages

        messages = build_tts_messages(
            text=str(text),
            system_instruction=self.system_instruction,
        )
        ids = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors=None,
        )
        if isinstance(ids, Mapping):
            ids = ids.get("input_ids")
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        elif (
            isinstance(ids, (list, tuple))
            and len(ids) == 1
            and isinstance(ids[0], (list, tuple))
        ):
            ids = ids[0]
        if not isinstance(ids, (list, tuple)):
            raise TypeError(f"direct-TTS chat template returned {type(ids).__name__}, expected token ids")
        row = [int(token_id) for token_id in ids]
        if not row:
            raise ValueError("direct-TTS chat template returned no token ids")
        if len(row) > self.max_prompt_length:
            raise ValueError(
                f"direct-TTS prompt produced {len(row)} tokens, exceeding "
                f"max_prompt_length={self.max_prompt_length}; truncation would corrupt the prefix"
            )
        return row

    def _behavior(self, ar: ARSamplingParams) -> TalkerSamplingConfig:
        codec_cfg = self._config.talker_config
        eos = int(codec_cfg.codec_eos_token_id)
        if ar.stop_token_id is not None and int(ar.stop_token_id) != eos:
            raise ValueError(
                f"direct-TTS stop_token_id must be codec EOS {eos}, got {ar.stop_token_id}"
            )
        suppress = (
            suppress_special_codec_ids(
                vocab_size=int(codec_cfg.text_config.vocab_size),
                codec_eos_token_id=eos,
            )
            if self.suppress_special_tokens
            else ()
        )
        return TalkerSamplingConfig(
            temperature=float(ar.temperature),
            top_k=max(0, int(ar.top_k)),
            top_p=float(ar.top_p),
            repetition_penalty=self.repetition_penalty,
            suppress_token_ids=tuple(suppress),
            eos_token_id=eos,
            do_sample=float(ar.temperature) > 0.0,
        )

    def build(self, sample: Sample) -> List[GenerateCall]:
        frontier = sample.frontier_gen_part(ARSamplingParams)
        ar = frontier.sampling_params
        assert isinstance(ar, ARSamplingParams)
        texts_prim = next((primitive for primitive in sample.conditioning() if isinstance(primitive, Texts)), None)
        require(texts_prim is not None, "qwen3_omni_talker requires Texts conditioning")
        texts = list(texts_prim.texts)
        require(
            len(texts) == len(frontier.sample_ids),
            f"qwen3_omni_talker text count {len(texts)} != frontier {len(frontier.sample_ids)}",
        )

        behavior = self._behavior(ar)
        behavior_dict = behavior.to_dict()
        codec_vocab = int(self._config.talker_config.text_config.vocab_size)
        suppressed = set(behavior.suppress_token_ids)
        allowed = [
            token_id
            for token_id in range(codec_vocab)
            if token_id not in suppressed
        ]
        metadata = sample.root_metadata(-1)
        prompts: List[Dict[str, Any]] = []
        self._last_requests = []
        for index, text in enumerate(texts):
            row_metadata = metadata[index] if metadata is not None and index < len(metadata) else None
            prompt_ids = self._prompt_ids(text)
            speaker_id = self._speaker_id(row_metadata)
            prefix_ids = build_direct_tts_prefix_ids(prompt_ids, self._config)
            prepared = DirectTTSPreparedRequest(
                prompt_ids=prompt_ids,
                speaker_id=speaker_id,
                prefix_ids=prefix_ids,
                behavior_sampling=dict(behavior_dict),
            )
            self._last_requests.append(prepared)
            prompts.append(
                {
                    # The patched Talker stage replaces these placeholders with
                    # the exact direct-TTS prefix carried below.
                    "prompt_token_ids": [0] * len(prefix_ids),
                    "additional_information": {
                        _RAW_CONTRACT_KEY: {
                            "contract_version": DIRECT_TTS_CONTRACT_VERSION,
                            "prompt_input_ids": prompt_ids,
                            "speaker_id": speaker_id,
                            "prefix_ids": prefix_ids,
                            "behavior_sampling": dict(behavior_dict),
                        }
                    },
                }
            )

        talker_sampling = StageSampling(
            kind=STAGE_KIND_AR,
            kwargs={
                "temperature": float(behavior.temperature),
                "top_p": float(behavior.top_p),
                "top_k": int(behavior.top_k) if behavior.top_k > 0 else -1,
                "repetition_penalty": float(behavior.repetition_penalty),
                "max_tokens": int(ar.max_new_tokens),
                "stop_token_ids": [int(behavior.eos_token_id)],
                "allowed_token_ids": allowed,
                "detokenize": False,
                "logprobs": 1,
            },
        )
        code2wav_sampling = StageSampling(
            kind=STAGE_KIND_AR,
            kwargs={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 65536,
            },
        )
        return [GenerateCall(prompts=prompts, sampling=[talker_sampling, code2wav_sampling])]


def _contract_payload(output: OmniRawResult, *, label: str) -> Mapping[str, Any]:
    custom = getattr(output, "custom_output", None)
    if not isinstance(custom, Mapping):
        raise RuntimeError(f"direct-TTS {label} output has no custom_output mapping")
    payload = custom.get(_RAW_CONTRACT_KEY)
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"direct-TTS {label} output lacks custom_output[{_RAW_CONTRACT_KEY!r}]"
        )
    if int(payload.get("contract_version", -1)) != DIRECT_TTS_CONTRACT_VERSION:
        raise RuntimeError(
            f"direct-TTS {label} output contract version "
            f"{payload.get('contract_version')!r} != {DIRECT_TTS_CONTRACT_VERSION}"
        )
    return payload


def _required(payload: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in payload:
        raise RuntimeError(f"direct-TTS {label} contract is missing required field {key!r}")
    return payload[key]


def _sampled_processed_logprobs(completion: Any, token_ids: List[int]) -> List[float]:
    raw = getattr(completion, "logprobs", None)
    if not isinstance(raw, (list, tuple)) or len(raw) != len(token_ids):
        raise RuntimeError(
            "direct-TTS Talker requires one processed logprob map per layer0 token"
        )
    values: List[float] = []
    for position, (token_id, token_map) in enumerate(zip(token_ids, raw)):
        if not isinstance(token_map, Mapping):
            raise RuntimeError(f"direct-TTS processed logprobs[{position}] is not a mapping")
        entry = token_map.get(token_id)
        if entry is None:
            entry = token_map.get(str(token_id))
        if entry is None:
            raise RuntimeError(
                f"direct-TTS processed logprobs[{position}] lacks sampled token {token_id}"
            )
        value = float(getattr(entry, "logprob", entry))
        if not math.isfinite(value):
            raise RuntimeError(f"direct-TTS processed logprob at position {position} is non-finite")
        values.append(value)
    return values


class Qwen3OmniTalkerOutputAdapter:
    """Strictly decode the patched runtime's two-stage raw-result contract."""

    def __init__(self, modality: str, input_adapter: Qwen3OmniTalkerInputAdapter) -> None:
        self.modality = modality
        self.input_adapter = input_adapter

    @staticmethod
    def _stage(group: List[OmniRawResult], stage_id: int, label: str) -> OmniRawResult:
        matches = [output for output in group if getattr(output, "stage_id", None) == stage_id]
        if len(matches) != 1:
            raise RuntimeError(
                f"direct-TTS result requires exactly one {label} stage-{stage_id} output; "
                f"got {len(matches)}"
            )
        return matches[0]

    def _parse_one(
        self,
        group: List[OmniRawResult],
        prepared: DirectTTSPreparedRequest,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        stage_ids = [getattr(output, "stage_id", None) for output in group]
        if len(stage_ids) != 2 or set(stage_ids) != {_TALKER_STAGE_ID, _CODE2WAV_STAGE_ID}:
            raise RuntimeError(
                "direct-TTS output must contain only Talker stage 0 and "
                f"Code2Wav stage 1; got stage_ids={stage_ids}"
            )
        talker_output = self._stage(group, _TALKER_STAGE_ID, "Talker")
        audio_output = self._stage(group, _CODE2WAV_STAGE_ID, "Code2Wav")
        if getattr(talker_output, "final_output_type", None) != "text":
            raise RuntimeError("direct-TTS Talker output must be exposed as final_output_type='text'")
        if getattr(audio_output, "final_output_type", None) != "audio":
            raise RuntimeError("direct-TTS Code2Wav output must be final_output_type='audio'")
        talker_payload = _contract_payload(talker_output, label="Talker")
        audio_payload = _contract_payload(audio_output, label="Code2Wav")

        request_output = getattr(talker_output, "request_output", None)
        completions = getattr(request_output, "outputs", None)
        if not isinstance(completions, (list, tuple)) or len(completions) != 1:
            raise RuntimeError("direct-TTS Talker output requires exactly one vLLM completion")
        completion = completions[0]
        token_ids = [int(token_id) for token_id in (getattr(completion, "token_ids", None) or [])]
        if not token_ids:
            raise RuntimeError("direct-TTS Talker output returned no layer0 tokens")
        processed_logps = _sampled_processed_logprobs(completion, token_ids)

        echoed_tokens = [
            int(token_id)
            for token_id in _required(talker_payload, "layer0_token_ids", label="Talker")
        ]
        echoed_logps = [
            float(value)
            for value in _required(talker_payload, "processed_logprobs", label="Talker")
        ]
        if echoed_tokens != token_ids:
            raise RuntimeError("direct-TTS custom layer0 tokens differ from vLLM completion tokens")
        if len(echoed_logps) != len(processed_logps) or any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)
            for left, right in zip(echoed_logps, processed_logps)
        ):
            raise RuntimeError("direct-TTS custom processed logprobs differ from vLLM completion logprobs")

        residual = torch.as_tensor(
            _required(talker_payload, "residual_codes", label="Talker"),
            dtype=torch.long,
        )
        expected_shape = (NUM_CODE_GROUPS - 1, len(token_ids))
        if residual.dim() != 2 or tuple(residual.shape) != expected_shape:
            raise RuntimeError(
                f"direct-TTS residual_codes expected {expected_shape}, got {tuple(residual.shape)}"
            )
        prefix_ids = [
            int(token_id)
            for token_id in _required(talker_payload, "prefix_ids", label="Talker")
        ]
        if prefix_ids != prepared.prefix_ids:
            raise RuntimeError("direct-TTS engine prefix_ids differ from the driver-authored prefix")
        if int(_required(talker_payload, "speaker_id", label="Talker")) != prepared.speaker_id:
            raise RuntimeError("direct-TTS engine speaker_id differs from the request")
        raw_behavior = _required(talker_payload, "behavior_sampling", label="Talker")
        behavior = TalkerSamplingConfig.from_dict(dict(raw_behavior) if isinstance(raw_behavior, Mapping) else {})
        if behavior.to_dict() != prepared.behavior_sampling:
            raise RuntimeError("direct-TTS engine behavior_sampling differs from the request")

        codec_eos = int(
            _required(talker_payload, "codec_eos_token_id", label="Talker")
        )
        if codec_eos != behavior.eos_token_id:
            raise RuntimeError("direct-TTS engine codec EOS differs from behavior_sampling")
        eos_reached = _required(talker_payload, "eos_reached", label="Talker")
        if not isinstance(eos_reached, bool):
            raise RuntimeError("direct-TTS Talker output requires boolean eos_reached")
        finish_reason = str(getattr(completion, "finish_reason", ""))
        if eos_reached:
            if codec_eos in token_ids[:-1] or token_ids[-1] != codec_eos or finish_reason != "stop":
                raise RuntimeError("direct-TTS EOS/status contract is inconsistent")
            status = int(SegmentStatus.COMPLETED)
        elif finish_reason == "length":
            if codec_eos in token_ids:
                raise RuntimeError("direct-TTS length-truncated output contains codec EOS")
            status = int(SegmentStatus.TRUNCATED)
        elif finish_reason == "abort":
            status = int(SegmentStatus.ABORTED)
        else:
            raise RuntimeError(f"direct-TTS unsupported finish_reason {finish_reason!r}")

        if [
            int(token_id)
            for token_id in _required(audio_payload, "layer0_token_ids", label="Code2Wav")
        ] != token_ids:
            raise RuntimeError("Code2Wav source layer0 tokens differ from Talker output")
        expected_frames = len(token_ids) - (1 if eos_reached else 0)
        if int(
            _required(audio_payload, "source_num_codec_frames", label="Code2Wav")
        ) != expected_frames:
            raise RuntimeError("Code2Wav source frame count is inconsistent with Talker EOS")
        if int(_required(audio_payload, "sample_rate", label="Code2Wav")) != AUDIO_SAMPLE_RATE:
            raise RuntimeError(
                f"direct-TTS audio sample rate must be {AUDIO_SAMPLE_RATE}"
            )
        waveform = torch.as_tensor(
            _required(audio_payload, "audio", label="Code2Wav"),
            dtype=torch.float32,
        ).reshape(-1)
        if not bool(torch.isfinite(waveform).all()):
            raise RuntimeError("direct-TTS Code2Wav output contains non-finite audio")
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(processed_logps, dtype=torch.float32),
            residual.cpu(),
            waveform.cpu(),
            status,
        )

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        frontier = sample.frontier_gen_part(ARSamplingParams)
        prepared = self.input_adapter._last_requests
        require(
            len(per_request) == len(prepared) == len(frontier.sample_ids),
            "direct-TTS result/prepared/frontier batch sizes differ",
        )

        tokens: List[torch.Tensor] = []
        logps: List[torch.Tensor] = []
        residuals: List[torch.Tensor] = []
        waveforms: List[torch.Tensor] = []
        statuses: List[int] = []
        for group, request in zip(per_request, prepared):
            token, logp, residual, waveform, status = self._parse_one(group, request)
            tokens.append(token)
            logps.append(logp)
            residuals.append(residual)
            waveforms.append(waveform)
            statuses.append(status)

        max_prompt = max(len(request.prompt_ids) for request in prepared)
        pad_id = int(getattr(self.input_adapter._tokenizer, "pad_token_id", None) or 0)
        prompt_ids = []
        prompt_masks = []
        for request in prepared:
            padding = max_prompt - len(request.prompt_ids)
            prompt_ids.append([pad_id] * padding + request.prompt_ids)
            prompt_masks.append([0] * padding + [1] * len(request.prompt_ids))
        conditions = Qwen3OmniTalkerConditions(
            prompt=TextTokenCondition(
                input_ids=torch.tensor(prompt_ids, dtype=torch.long),
                attention_mask=torch.tensor(prompt_masks, dtype=torch.long),
            ),
            speaker_ids=[request.speaker_id for request in prepared],
            prefix_ids=[torch.tensor(request.prefix_ids, dtype=torch.long) for request in prepared],
            residual_codes=residuals,
            behavior_sampling=dict(prepared[0].behavior_sampling),
        )
        segment = TextSegment.pack(
            tokens=tokens,
            log_probs=logps,
            status=torch.tensor(statuses, dtype=torch.long),
        )
        primitives: Dict[str, Any] = {
            "audio": Audios.from_list([Audio(waveform=waveform) for waveform in waveforms])
        }
        texts_prim = next((primitive for primitive in sample.conditioning() if isinstance(primitive, Texts)), None)
        if texts_prim is not None:
            primitives["text"] = texts_prim
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives=primitives,
                conditions=conditions.to_dict(),
                primitive_metadata={"audio": {"sample_rate": AUDIO_SAMPLE_RATE}},
                status=segment.status,
            )
        )


@register_adapter("qwen3_omni_talker")
class Qwen3OmniTalkerAdapter(ModelAdapter):
    """Direct TTS only; refuses unpatched spoken-response runtimes."""

    stage_yaml = "qwen3_omni_talker_tts_rl_1x4.yaml"
    stage_yaml_source = "local"
    omni_mode = None
    needs_sigmas = False
    needs_driver_tokenizer = False
    ar_lora_passthrough = True
    clear_cuda_visible = False
    lora_copy_transport = True
    lora_stage_ids = (_TALKER_STAGE_ID,)

    def __init__(
        self,
        config: Any,
        model_config: Any,
        *,
        strategy: Any = None,
        tokenize_fn: Any = None,
    ) -> None:
        # This happens before backend spawn. The currently pinned upstream
        # vLLM-Omni 0.20.0 intentionally fails here.
        self.runtime_capability = require_direct_tts_runtime()
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = Qwen3OmniTalkerInputAdapter(
            self.modality,
            model_path=str(config.model_path),
            max_prompt_length=int(getattr(config, "max_prompt_length", 2048) or 2048),
            system_instruction=(
                getattr(model_config, "system_instruction", None)
                if model_config is not None
                else None
            ),
            default_speaker=(
                str(getattr(model_config, "default_speaker", "Ethan"))
                if model_config is not None
                else "Ethan"
            ),
            repetition_penalty=float(
                getattr(model_config, "repetition_penalty", 1.05)
                if model_config is not None
                else 1.05
            ),
            suppress_special_tokens=bool(
                getattr(model_config, "suppress_special_tokens", True)
                if model_config is not None
                else True
            ),
        )
        self.output_adapter = Qwen3OmniTalkerOutputAdapter(self.modality, self.input_adapter)

    def schedule_policy(self) -> Any:
        return None

    def validate_request(self, sample: Sample) -> None:
        sample.frontier_gen_part(ARSamplingParams)
        require(
            any(isinstance(primitive, Texts) for primitive in sample.conditioning()),
            "qwen3_omni_talker requires Texts prompts",
        )

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = [
    "DirectTTSPreparedRequest",
    "Qwen3OmniTalkerAdapter",
    "Qwen3OmniTalkerInputAdapter",
    "Qwen3OmniTalkerOutputAdapter",
    "build_direct_tts_prefix_ids",
]
