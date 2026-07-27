"""Qwen3-Omni Thinker rollout adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.models.qwen3_omni.media import extract_audio_from_video_pyav
from unirl.models.qwen3_omni.video import limit_video_frames, sample_video_frames_pyav
from unirl.models.types.conversations import build_video_messages
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.hi3 import Hi3TextOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment
from unirl.types.segments.base import SegmentStatus

logger = logging.getLogger(__name__)


def _compress_qwen3_omni_prompt_ids(
    token_ids: List[int],
    *,
    audio_token_id: int,
    image_token_id: int,
    video_token_id: int,
    vision_bos_token_id: int,
    vision_eos_token_id: int,
    audio_bos_token_id: int,
    audio_eos_token_id: int,
    use_audio_in_video: bool,
) -> List[int]:
    """Undo HF multimodal expansion before vLLM processes the raw media.

    This mirrors vLLM's
    ``Qwen3OmniMoeThinkerMultiModalProcessor._get_raw_input_ids``. Replay
    keeps the original expanded IDs; only the IDs sent to vLLM are compressed.
    """
    result = list(token_ids)
    if use_audio_in_video:
        while True:
            start = next(
                (i for i in range(len(result) - 1) if result[i : i + 2] == [vision_bos_token_id, audio_bos_token_id]),
                None,
            )
            if start is None:
                break
            end = next(
                (
                    i
                    for i in range(start + 2, len(result) - 1)
                    if result[i : i + 2] == [audio_eos_token_id, vision_eos_token_id]
                ),
                None,
            )
            if end is None:
                raise ValueError(
                    "Qwen3OmniThinkerInputAdapter: expanded audio-in-video span "
                    "has no matching audio/vision end tokens."
                )
            result = result[:start] + [vision_bos_token_id, video_token_id, vision_eos_token_id] + result[end + 2 :]

    for mm_token_id in (audio_token_id, image_token_id, video_token_id):
        compressed: List[int] = []
        for token_id in result:
            if token_id != mm_token_id or not compressed or compressed[-1] != mm_token_id:
                compressed.append(token_id)
        result = compressed
    return result


class Qwen3OmniThinkerInputAdapter:
    """Build one batched AR generate call from a role-aware request Sample."""

    def __init__(
        self,
        modality: str,
        *,
        model_path: str,
        video_fps: float = 1.0,
        video_max_frames: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
        max_prompt_length: int = 12288,
        system_instruction: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.modality = modality
        self.model_path = str(model_path)
        self.video_fps = float(video_fps)
        self.video_max_frames = int(video_max_frames) if video_max_frames is not None else None
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        self.use_audio_in_video = bool(use_audio_in_video)
        self.max_prompt_length = int(max_prompt_length)
        self.system_instruction = system_instruction
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

        # Reuse the processor and tokenizer across requests.
        from transformers import AutoProcessor, AutoTokenizer

        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # Some checkpoints store the chat template separately.
        if getattr(self._tokenizer, "chat_template", None) is None:
            import json
            import os

            path = os.path.join(self.model_path, "chat_template.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                        if data.get("chat_template"):
                            self._tokenizer.chat_template = data["chat_template"]
                except (OSError, json.JSONDecodeError):
                    pass
        processor_tokenizer = getattr(self._processor, "tokenizer", None)
        if (
            processor_tokenizer is not None
            and getattr(processor_tokenizer, "chat_template", None) is None
            and getattr(self._tokenizer, "chat_template", None) is not None
        ):
            processor_tokenizer.chat_template = self._tokenizer.chat_template

        # Cached for replay-condition construction by the output adapter. The
        # engine's _generate_lock encloses build -> generate -> response, so no
        # second caller can replace this request-local batch in between.
        self._last_encodings: List[Dict[str, Any]] = []

    def _multimodal_processor_kwargs(self, *, video_fps: float) -> Dict[str, Any]:
        """Processor kwargs shared by driver encoding and vLLM."""
        kwargs: Dict[str, Any] = {
            "fps": video_fps,
            "do_sample_frames": False,
            "truncation": True,
        }
        if self.video_max_pixels is not None:
            kwargs["size"] = {
                "shortest_edge": int(self._processor.video_processor.size["shortest_edge"]),
                "longest_edge": self.video_max_pixels,
            }
        return kwargs

    def _token_id(self, token: str) -> int:
        token_id = self._tokenizer.convert_tokens_to_ids(token)
        if token_id is None or int(token_id) < 0:
            encoded = self._tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Qwen3OmniThinkerInputAdapter: cannot resolve token id for {token!r}")
            token_id = encoded[0]
        return int(token_id)

    def _compress_prompt_ids(self, token_ids: List[int], *, use_audio_in_video: bool) -> List[int]:
        """Compress processor-expanded placeholders for vLLM re-expansion."""
        return _compress_qwen3_omni_prompt_ids(
            token_ids,
            audio_token_id=self._token_id("<|audio_pad|>"),
            image_token_id=self._token_id("<|image_pad|>"),
            video_token_id=self._token_id("<|video_pad|>"),
            vision_bos_token_id=self._token_id("<|vision_start|>"),
            vision_eos_token_id=self._token_id("<|vision_end|>"),
            audio_bos_token_id=self._token_id("<|audio_start|>"),
            audio_eos_token_id=self._token_id("<|audio_end|>"),
            use_audio_in_video=use_audio_in_video,
        )

    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Optional[Any], float, Optional[Any]]:
        """Decode/sample the one supported video block without mutating turns."""
        prepared: List[Dict[str, Any]] = []
        video_frames: Optional[Any] = None
        effective_fps = self.video_fps
        audio_wave: Optional[Any] = None
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                prepared.append(dict(message))
                continue
            blocks: List[Dict[str, Any]] = []
            for raw_block in content:
                block = dict(raw_block)
                if block.get("type") == "video":
                    if video_frames is not None:
                        raise ValueError(
                            "Qwen3OmniThinkerInputAdapter supports at most one persistent "
                            "source video per conversation."
                        )
                    raw_video = block.get("video")
                    if isinstance(raw_video, str):
                        video_frames, effective_fps = sample_video_frames_pyav(
                            raw_video,
                            target_fps=self.video_fps,
                            max_frames=self.video_max_frames,
                        )
                        if self.use_audio_in_video:
                            sample_rate = int(
                                getattr(
                                    getattr(self._processor, "feature_extractor", None),
                                    "sampling_rate",
                                    16000,
                                )
                            )
                            audio_wave = extract_audio_from_video_pyav(raw_video, sample_rate)
                    else:
                        video_frames, effective_fps = limit_video_frames(
                            raw_video,
                            fps=self.video_fps,
                            max_frames=self.video_max_frames,
                        )
                    block["video"] = video_frames
                blocks.append(block)
            copied = dict(message)
            copied["content"] = blocks
            prepared.append(copied)
        return prepared, video_frames, effective_fps, audio_wave

    def _encode_one(
        self,
        messages: List[Dict[str, Any]],
        template_overrides: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Optional[Any], float, Optional[Any]]:
        prepared, video_frames, effective_fps, audio_wave = self._prepare_messages(messages)
        template_kwargs = dict(self.chat_template_kwargs)
        template_kwargs.update(template_overrides)

        if audio_wave is not None:
            template_kwargs.update(add_generation_prompt=True, tokenize=False)
            prompt_text = self._processor.apply_chat_template(prepared, **template_kwargs)
            processor_kwargs = self._multimodal_processor_kwargs(video_fps=effective_fps)
            processor_kwargs.update(
                text=[prompt_text],
                videos=[video_frames],
                audio=[audio_wave],
                use_audio_in_video=True,
                return_tensors="pt",
            )
            encoding = self._processor(**processor_kwargs)
            return encoding, video_frames, effective_fps, audio_wave

        if video_frames is not None:
            template_kwargs.update(self._multimodal_processor_kwargs(video_fps=effective_fps))
        # These fields define the replay-condition wire shape.
        template_kwargs.update(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoding = self._processor.apply_chat_template(prepared, **template_kwargs)
        return encoding, video_frames, effective_fps, audio_wave

    def build(self, sample: Sample) -> List[GenerateCall]:
        frontier = sample.frontier_gen_part(ARSamplingParams)
        ar = frontier.sampling_params
        assert isinstance(ar, ARSamplingParams)

        chat_overrides = dict((sample.parts[0].control or {}).get("chat") or {})
        system_instruction = chat_overrides.get("system_instruction", self.system_instruction)
        template_overrides = dict(chat_overrides.get("template_kwargs") or {})
        conversations = build_video_messages(sample.turns(), system_instruction)
        require(
            bool(conversations),
            "Qwen3OmniThinkerInputAdapter: Sample carries no text/video conditioning turns.",
        )
        require(
            len(conversations) == len(frontier.sample_ids),
            f"Qwen3OmniThinkerInputAdapter: conversation count {len(conversations)} "
            f"!= frontier id count {len(frontier.sample_ids)}",
        )

        prompts: List[Dict[str, Any]] = []
        # The output adapter consumes this cache after generation.
        self._last_encodings = []
        for messages in conversations:
            enc, video_frames, effective_fps, audio_wave = self._encode_one(messages, template_overrides)
            expanded_ids = enc["input_ids"].squeeze(0).tolist()
            if len(expanded_ids) > self.max_prompt_length:
                # Multimodal token truncation would break feature alignment.
                if video_frames is not None or audio_wave is not None:
                    raise ValueError(
                        f"Qwen3OmniThinkerInputAdapter: multimodal prompt produced {len(expanded_ids)} tokens, "
                        f"exceeding max_prompt_length={self.max_prompt_length}. Reduce video_max_frames, "
                        "video_max_pixels, or video_fps, or raise max_prompt_length."
                    )
                enc = dict(enc)
                enc["input_ids"] = enc["input_ids"][..., -self.max_prompt_length :]
                enc["attention_mask"] = enc["attention_mask"][..., -self.max_prompt_length :]
                expanded_ids = enc["input_ids"].squeeze(0).tolist()
            rollout_ids = (
                self._compress_prompt_ids(expanded_ids, use_audio_in_video=audio_wave is not None)
                if video_frames is not None or audio_wave is not None
                else expanded_ids
            )
            entry: Dict[str, Any] = {"prompt_token_ids": rollout_ids}
            media: Dict[str, Any] = {}
            if video_frames is not None:
                media["video"] = [video_frames]
            if audio_wave is not None:
                media["audio"] = [audio_wave]
            if media:
                entry["multi_modal_data"] = media
            if video_frames is not None or audio_wave is not None:
                # Keep this request-local: globally enabling audio-in-video
                # breaks vLLM's video-only dummy profiling during engine boot.
                mm_processor_kwargs = self._multimodal_processor_kwargs(video_fps=effective_fps)
                if audio_wave is not None:
                    mm_processor_kwargs["use_audio_in_video"] = True
                    temporal_patch_size = int(getattr(self._processor.video_processor, "temporal_patch_size", 2))
                    mm_processor_kwargs["second_per_grid_ts"] = [temporal_patch_size / effective_fps]
                entry["mm_processor_kwargs"] = mm_processor_kwargs
                logger.debug(
                    "Qwen3-Omni rollout prompt: expanded_tokens=%d compressed_tokens=%d "
                    "modalities=%s mm_processor_kwargs=%s",
                    len(expanded_ids),
                    len(rollout_ids),
                    sorted(media),
                    mm_processor_kwargs,
                )
            prompts.append(entry)
            self._last_encodings.append(enc)

        max_new_tokens = int(ar.max_new_tokens)
        temperature = float(ar.temperature)
        top_p = float(ar.top_p)
        top_k_val = int(ar.top_k)
        top_k = top_k_val if top_k_val > 0 else -1  # vLLM: -1 disables top_k
        stop_token_id = ar.stop_token_id

        base_sampling_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_new_tokens,
            "logprobs": 1,
        }
        if stop_token_id is not None:
            base_sampling_kwargs["stop_token_ids"] = [int(stop_token_id)]

        # Keep seed unset: AsyncOmniEngine.add_request receives each prompt as an
        # independent request, and patch_per_request_ar_seed clones the shared
        # SamplingParams with a fresh seed for every request.
        return [
            GenerateCall(
                prompts=prompts,
                sampling=[StageSampling(kind=STAGE_KIND_AR, kwargs=base_sampling_kwargs)],
            )
        ]


class Qwen3OmniThinkerOutputAdapter(Hi3TextOutputAdapter):
    """Build AR responses and replay conditions from cached encodings."""

    def __init__(self, modality: str, input_adapter: "Qwen3OmniThinkerInputAdapter") -> None:
        super().__init__(modality)
        self._input_adapter = input_adapter

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Assemble replay conditions from cached processor outputs."""
        del per_request

        from unirl.models.qwen3_omni.conditions import Qwen3OmniARConditions
        from unirl.types.conditions import TextTokenCondition

        encs = self._input_adapter._last_encodings
        if not encs:
            raise RuntimeError(
                "Qwen3OmniThinkerOutputAdapter.build_conditions: input adapter "
                "cache is empty — ``build_inputs`` must run before ``build_response``."
            )
        frontier = sample.frontier_gen_part(ARSamplingParams)
        require(
            len(encs) == len(frontier.sample_ids),
            f"Qwen3OmniThinkerOutputAdapter: encoding count {len(encs)} "
            f"!= frontier id count {len(frontier.sample_ids)}",
        )

        pad_id = self._input_adapter._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._input_adapter._tokenizer.eos_token_id or 0

        # Right-pad token tensors to the batch maximum.
        raw_ids = [e["input_ids"].squeeze(0) for e in encs]
        raw_masks = [e["attention_mask"].squeeze(0) for e in encs]
        max_len = int(max(t.shape[-1] for t in raw_ids))
        batch = len(raw_ids)
        input_ids = torch.full((batch, max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=raw_masks[0].dtype if raw_masks else torch.long)
        for i, (ids, mask) in enumerate(zip(raw_ids, raw_masks)):
            L = int(ids.shape[-1])
            input_ids[i, :L] = ids[:L].to(torch.long)
            attention_mask[i, :L] = mask[:L]

        pixel_values_videos: List[Any] = []
        video_grid_thw: List[Any] = []
        video_second_per_grid: List[Any] = []
        input_features: List[Any] = []
        feature_attention_mask: List[Any] = []
        for e in encs:
            pvv = e.get("pixel_values_videos")
            vgt = e.get("video_grid_thw")
            vspg = e.get("video_second_per_grid")
            pixel_values_videos.append(pvv if pvv is not None else None)
            video_grid_thw.append(vgt if vgt is not None else None)
            if vspg is not None and not isinstance(vspg, torch.Tensor):
                vspg = torch.as_tensor(vspg)
            video_second_per_grid.append(vspg if vspg is not None else None)
            input_features.append(e.get("input_features"))
            feature_attention_mask.append(e.get("feature_attention_mask"))

        has_video = any(p is not None for p in pixel_values_videos)
        has_audio = any(a is not None for a in input_features)
        cond = Qwen3OmniARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values_videos=pixel_values_videos if has_video else None,
            video_grid_thw=video_grid_thw if has_video else None,
            video_second_per_grid=video_second_per_grid if has_video else None,
            input_features=input_features if has_audio else None,
            feature_attention_mask=feature_attention_mask if has_audio else None,
        )
        return cond.to_dict()

    @staticmethod
    def _stage0(group: List[OmniRawResult]) -> OmniRawResult:
        for output in group:
            if getattr(output, "stage_id", None) == 0:
                return output
        raise RuntimeError("Qwen3OmniThinkerOutputAdapter: backend result group has no stage-0 AR output.")

    @classmethod
    def _status(cls, per_request: List[List[OmniRawResult]]) -> torch.Tensor:
        mapping = {
            "stop": SegmentStatus.COMPLETED,
            "length": SegmentStatus.TRUNCATED,
            "abort": SegmentStatus.ABORTED,
        }
        values: List[int] = []
        for group in per_request:
            output = cls._stage0(group)
            request_output = getattr(output, "request_output", None)
            completions = getattr(request_output, "outputs", None) or []
            finish_reason = getattr(completions[0], "finish_reason", None) if completions else None
            values.append(int(mapping.get(str(finish_reason), SegmentStatus.PENDING)))
        return torch.tensor(values, dtype=torch.long)

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        """Fill exactly the existing AR frontier from one result per row."""
        frontier = sample.frontier_gen_part(ARSamplingParams)
        require(
            len(per_request) == len(frontier.sample_ids),
            f"Qwen3OmniThinkerOutputAdapter: result count {len(per_request)} "
            f"!= frontier id count {len(frontier.sample_ids)}",
        )
        # Missing stage outputs are infrastructure failures, not valid empty
        # assistant turns. Validate before any best-effort helper runs.
        for group in per_request:
            self._stage0(group)

        segment = self.build_segment(sample, per_request)
        if segment is None:
            # A present stage may legitimately finish without an emitted token.
            empty = [torch.zeros(0, dtype=torch.long) for _ in per_request]
            empty_logp = [torch.zeros(0, dtype=torch.float32) for _ in per_request]
            segment = TextSegment.pack(tokens=empty, log_probs=empty_logp)
        decoded = self.build_decoded(sample, per_request)
        conditions = self.build_conditions(sample, per_request)
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"text": decoded},
                conditions=conditions,
                status=self._status(per_request),
            )
        )


@register_adapter("qwen3_omni_thinker")
class Qwen3OmniThinkerAdapter(ModelAdapter):
    """Qwen3-Omni Thinker — text/video → AR text (single stage, TP>1, LoRA)."""

    # TODO: This anchored AR topology is temporarily migrated from unified
    # models; replace these knobs with formal TP/DP/PP support.
    stage_yaml = "qwen3_omni_thinker_only_rl_1x4.yaml"
    stage_yaml_source = "local"
    omni_mode = None
    needs_sigmas = False
    needs_driver_tokenizer = False
    ar_lora_passthrough = True
    clear_cuda_visible = True
    lora_copy_transport = True

    def __init__(
        self,
        config: Any,
        model_config: Any,
        *,
        strategy: Any = None,
        tokenize_fn: Any = None,
    ) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)

        mc = model_config
        model_path = str(config.model_path)
        video_fps = float(getattr(mc, "video_fps", 1.0)) if mc is not None else 1.0
        video_max_frames = getattr(mc, "video_max_frames", None) if mc is not None else None
        video_max_pixels = getattr(mc, "video_max_pixels", None) if mc is not None else None
        use_audio_in_video = bool(getattr(mc, "use_audio_in_video", False)) if mc is not None else False
        max_prompt_length = int(getattr(mc, "max_prompt_length", 12288)) if mc is not None else 12288
        config_video_fps = getattr(config, "video_fps", None)
        config_video_max_pixels = getattr(config, "video_max_pixels", None)
        config_use_audio_in_video = getattr(config, "use_audio_in_video", None)
        config_max_prompt_length = getattr(config, "max_prompt_length", None)
        if config_video_fps is not None:
            video_fps = float(config_video_fps)
        if config_video_max_pixels is not None:
            video_max_pixels = int(config_video_max_pixels)
        if config_use_audio_in_video is not None:
            use_audio_in_video = bool(config_use_audio_in_video)
        if config_max_prompt_length is not None:
            max_prompt_length = int(config_max_prompt_length)
        system_instruction = getattr(mc, "system_instruction", None) if mc is not None else None
        chat_template_kwargs = dict(getattr(config, "chat_template_kwargs", {}) or {})

        logger.info(
            "Resolved Qwen3-Omni rollout adapter config: model_path=%s video_fps=%s "
            "video_max_frames=%s video_max_pixels=%s use_audio_in_video=%s max_prompt_length=%s "
            "system_instruction_set=%s model_config_available=%s",
            model_path,
            video_fps,
            video_max_frames,
            video_max_pixels,
            use_audio_in_video,
            max_prompt_length,
            system_instruction is not None,
            model_config is not None,
        )

        self.input_adapter = Qwen3OmniThinkerInputAdapter(
            self.modality,
            model_path=model_path,
            video_fps=video_fps,
            video_max_frames=video_max_frames,
            video_max_pixels=video_max_pixels,
            use_audio_in_video=use_audio_in_video,
            max_prompt_length=max_prompt_length,
            system_instruction=system_instruction,
            chat_template_kwargs=chat_template_kwargs,
        )
        self.output_adapter = Qwen3OmniThinkerOutputAdapter(self.modality, self.input_adapter)

    def schedule_policy(self) -> Any:
        """Return no diffusion schedule for the AR-only stage."""
        return None

    def validate_request(self, sample: Sample) -> None:
        sample.frontier_gen_part(ARSamplingParams)
        # Rendering is the single validation source for supported modalities,
        # frontier alignment, and the one-video contract.
        conversations = build_video_messages(sample.turns())
        require(bool(conversations), f"modality={self.modality!r} requires text/video conditioning turns.")

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = ["Qwen3OmniThinkerAdapter", "Qwen3OmniThinkerInputAdapter"]
