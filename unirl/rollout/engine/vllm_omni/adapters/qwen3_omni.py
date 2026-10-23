"""Qwen3-Omni Thinker rollout adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch

from unirl.config.require import require
from unirl.models.qwen3_omni.conditions import Qwen3OmniARConditions
from unirl.models.qwen3_omni.media import build_omni_messages, omni_processor_media_kwargs, prepare_omni_media
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.utils import build_ar_segment, decoded_text_from_ar
from unirl.types.conditions import TextTokenCondition
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment
from unirl.types.segments.base import SegmentStatus

logger = logging.getLogger(__name__)


@register_adapter("qwen3_omni_thinker")
class Qwen3OmniThinkerAdapter(ModelAdapter):
    """Qwen3-Omni Thinker — text/video → AR text (single stage, TP>1, LoRA)."""

    deploy_config = "qwen3_omni_thinker_only_rl_1x4.yaml"
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
        require(
            model_config is not None,
            "Qwen3OmniThinkerAdapter requires model_config (Qwen3OmniPipelineConfig).",
        )

        mc = model_config
        self.model_path = config.model_path
        self.image_max_pixels = config.image_max_pixels if config.image_max_pixels is not None else mc.image_max_pixels
        self.video_fps = config.video_fps if config.video_fps is not None else mc.video_fps
        self.video_max_frames = mc.video_max_frames
        self.video_max_pixels = config.video_max_pixels if config.video_max_pixels is not None else mc.video_max_pixels
        self.use_audio_in_video = (
            config.use_audio_in_video if config.use_audio_in_video is not None else mc.use_audio_in_video
        )
        self.max_prompt_length = (
            config.max_prompt_length if config.max_prompt_length is not None else mc.max_prompt_length
        )
        self.system_instruction = mc.system_instruction
        self.chat_template_kwargs = {} if config.chat_template_kwargs is None else dict(config.chat_template_kwargs)

        logger.info(
            "Resolved Qwen3-Omni rollout adapter config: model_path=%s image_max_pixels=%s video_fps=%s "
            "video_max_frames=%s video_max_pixels=%s use_audio_in_video=%s max_prompt_length=%s "
            "system_instruction_set=%s",
            self.model_path,
            self.image_max_pixels,
            self.video_fps,
            self.video_max_frames,
            self.video_max_pixels,
            self.use_audio_in_video,
            self.max_prompt_length,
            self.system_instruction is not None,
        )

        from transformers import AutoProcessor, AutoTokenizer

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.chat_template is None:
            import json
            import os

            path = os.path.join(self.model_path, "chat_template.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                    if data.get("chat_template"):
                        self.tokenizer.chat_template = data["chat_template"]
        processor_tokenizer = self.processor.tokenizer
        if processor_tokenizer.chat_template is None and self.tokenizer.chat_template is not None:
            processor_tokenizer.chat_template = self.tokenizer.chat_template

        self.last_encodings: List[Dict[str, Any]] = []

    def token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None or int(token_id) < 0:
            encoded = self.tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Qwen3OmniThinkerAdapter: cannot resolve token id for {token!r}")
            token_id = encoded[0]
        return int(token_id)

    def schedule_policy(self) -> Any:
        """Return no diffusion schedule for the AR-only stage."""
        return None

    def validate_request(self, sample: Sample) -> None:
        sample.frontier_gen_part(ARSamplingParams)
        conversations = build_omni_messages(sample.turns())
        require(bool(conversations), f"modality={self.modality!r} requires conditioning turns.")

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        frontier = sample.frontier_gen_part(ARSamplingParams)
        ar = frontier.sampling_params
        chat_overrides = dict(sample.parts[0].control.get("chat") or {})
        system_instruction = chat_overrides.get("system_instruction", self.system_instruction)
        template_overrides = dict(chat_overrides.get("template_kwargs") or {})
        conversations = build_omni_messages(
            sample.turns(),
            system_instruction,
        )
        require(
            len(conversations) == len(frontier.sample_ids),
            f"Qwen3OmniThinkerAdapter: conversation count {len(conversations)} "
            f"!= frontier id count {len(frontier.sample_ids)}",
        )

        prompts: List[Dict[str, Any]] = []
        self.last_encodings = []
        for messages in conversations:
            media_bundle = prepare_omni_media(
                messages,
                sample_rate=self.processor.feature_extractor.sampling_rate,
                video_fps=self.video_fps,
                video_max_frames=self.video_max_frames,
                use_audio_in_video=self.use_audio_in_video,
            )
            prepared = media_bundle.messages
            image = media_bundle.image
            video_frames = media_bundle.video_frames
            effective_fps = media_bundle.effective_fps
            audio_wave = media_bundle.audio_waveform
            audio_sample_rate = media_bundle.audio_sample_rate
            audio_in_video = media_bundle.audio_in_video
            has_mm = image is not None or video_frames is not None or audio_wave is not None
            template_kwargs = dict(self.chat_template_kwargs)
            template_kwargs.update(template_overrides)

            if has_mm:
                template_kwargs.update(add_generation_prompt=True, tokenize=False)
                prompt_text = self.processor.apply_chat_template(prepared, **template_kwargs)
                processor_kwargs: Dict[str, Any] = {"text": [prompt_text], "truncation": True, "return_tensors": "pt"}
                processor_kwargs.update(
                    omni_processor_media_kwargs(
                        self.processor,
                        has_image=image is not None,
                        has_video=video_frames is not None,
                        image_max_pixels=self.image_max_pixels,
                        video_fps=effective_fps,
                        video_max_pixels=self.video_max_pixels,
                    )
                )
                if image is not None:
                    processor_kwargs["images"] = [image]
                if video_frames is not None:
                    processor_kwargs["videos"] = [video_frames]
                if audio_wave is not None:
                    processor_kwargs["audio"] = [audio_wave]
                if audio_in_video:
                    processor_kwargs["use_audio_in_video"] = True
                enc = self.processor(**processor_kwargs)
            else:
                template_kwargs.update(
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                enc = self.processor.apply_chat_template(prepared, **template_kwargs)

            enc["_unirl_audio_in_video"] = audio_in_video
            expanded_ids = enc["input_ids"].squeeze(0).tolist()
            if len(expanded_ids) > self.max_prompt_length:
                if has_mm:
                    raise ValueError(
                        f"Qwen3OmniThinkerAdapter: multimodal prompt produced {len(expanded_ids)} tokens, "
                        f"exceeding max_prompt_length={self.max_prompt_length}. Reduce image_max_pixels, "
                        "video_max_frames, video_max_pixels, or video_fps, or raise max_prompt_length."
                    )
                enc = dict(enc)
                enc["input_ids"] = enc["input_ids"][..., -self.max_prompt_length :]
                enc["attention_mask"] = enc["attention_mask"][..., -self.max_prompt_length :]
                expanded_ids = enc["input_ids"].squeeze(0).tolist()
            if has_mm:
                audio_token_id = self.token_id("<|audio_pad|>")
                image_token_id = self.token_id("<|image_pad|>")
                video_token_id = self.token_id("<|video_pad|>")
                vision_bos_token_id = self.token_id("<|vision_start|>")
                vision_eos_token_id = self.token_id("<|vision_end|>")
                audio_bos_token_id = self.token_id("<|audio_start|>")
                audio_eos_token_id = self.token_id("<|audio_end|>")
                rollout_ids = list(expanded_ids)
                if audio_in_video:
                    while True:
                        start = next(
                            (
                                i
                                for i in range(len(rollout_ids) - 1)
                                if rollout_ids[i : i + 2] == [vision_bos_token_id, audio_bos_token_id]
                            ),
                            None,
                        )
                        if start is None:
                            break
                        end = next(
                            (
                                i
                                for i in range(start + 2, len(rollout_ids) - 1)
                                if rollout_ids[i : i + 2] == [audio_eos_token_id, vision_eos_token_id]
                            ),
                            None,
                        )
                        if end is None:
                            raise ValueError(
                                "Qwen3OmniThinkerAdapter: expanded audio-in-video span has no matching audio/vision end tokens."
                            )
                        rollout_ids = (
                            rollout_ids[:start]
                            + [vision_bos_token_id, video_token_id, vision_eos_token_id]
                            + rollout_ids[end + 2 :]
                        )
                for mm_token_id in (audio_token_id, image_token_id, video_token_id):
                    compressed: List[int] = []
                    for token_id in rollout_ids:
                        if token_id != mm_token_id or not compressed or compressed[-1] != mm_token_id:
                            compressed.append(token_id)
                    rollout_ids = compressed
            else:
                rollout_ids = expanded_ids
            entry: Dict[str, Any] = {"prompt_token_ids": rollout_ids}
            media: Dict[str, Any] = {}
            if image is not None:
                media["image"] = [image]
            if video_frames is not None:
                media["video"] = [video_frames]
            if audio_wave is not None:
                if audio_sample_rate is None:
                    raise RuntimeError("Qwen3-Omni canonical audio is missing its sampling rate.")
                media["audio"] = [(audio_wave, audio_sample_rate)]
            if media:
                entry["multi_modal_data"] = media
            if has_mm:
                mm_processor_kwargs: Dict[str, Any] = {"truncation": True}
                mm_processor_kwargs.update(
                    omni_processor_media_kwargs(
                        self.processor,
                        has_image=image is not None,
                        has_video=video_frames is not None,
                        image_max_pixels=self.image_max_pixels,
                        video_fps=effective_fps,
                        video_max_pixels=self.video_max_pixels,
                    )
                )
                if audio_in_video:
                    mm_processor_kwargs["use_audio_in_video"] = True
                    temporal_patch_size = self.processor.video_processor.temporal_patch_size
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
            self.last_encodings.append(enc)

        top_k = ar.top_k if ar.top_k > 0 else -1  # vLLM: -1 disables top_k

        base_sampling_kwargs: Dict[str, Any] = {
            "temperature": ar.temperature,
            "top_p": ar.top_p,
            "top_k": top_k,
            "max_tokens": ar.max_new_tokens,
            "logprobs": 1,
        }
        if ar.stop_token_id is not None:
            base_sampling_kwargs["stop_token_ids"] = [ar.stop_token_id]

        return [
            GenerateCall(
                prompts=prompts,
                sampling=[StageSampling(kind=STAGE_KIND_AR, kwargs=base_sampling_kwargs)],
            )
        ]

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        """Fill the existing AR frontier and assemble replay conditions from cached processor outputs."""
        frontier = sample.frontier_gen_part(ARSamplingParams)
        require(
            len(per_request) == len(frontier.sample_ids),
            f"Qwen3OmniThinkerAdapter: result count {len(per_request)} != frontier id count {len(frontier.sample_ids)}",
        )
        mapping = {
            "stop": SegmentStatus.COMPLETED,
            "length": SegmentStatus.TRUNCATED,
            "abort": SegmentStatus.ABORTED,
        }
        status_values: List[int] = []
        for group in per_request:
            stage0 = next(
                (output for output in group if output.stage_id == 0),
                None,
            )
            if stage0 is None:
                raise RuntimeError("Qwen3OmniThinkerAdapter: backend result group has no stage-0 AR output.")
            completions = stage0.outputs
            finish_reason = completions[0].finish_reason if completions else None
            status_values.append(int(mapping.get(finish_reason, SegmentStatus.PENDING)))

        segment = build_ar_segment(per_request)
        if segment is None:
            empty = [torch.zeros(0, dtype=torch.long) for _ in per_request]
            empty_logp = [torch.zeros(0, dtype=torch.float32) for _ in per_request]
            segment = TextSegment.pack(tokens=empty, log_probs=empty_logp)
        decoded = decoded_text_from_ar(per_request)

        encs = self.last_encodings
        require(
            len(encs) == len(frontier.sample_ids),
            f"Qwen3OmniThinkerAdapter: encoding count {len(encs)} != frontier id count {len(frontier.sample_ids)}",
        )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            eos_id = self.tokenizer.eos_token_id
            pad_id = 0 if eos_id is None else eos_id

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

        pixel_values: List[Any] = []
        image_grid_thw: List[Any] = []
        pixel_values_videos: List[Any] = []
        video_grid_thw: List[Any] = []
        video_second_per_grid: List[Any] = []
        input_features: List[Any] = []
        feature_attention_mask: List[Any] = []
        use_audio_in_video: List[bool] = []
        for e in encs:
            pixel_values.append(e.get("pixel_values"))
            image_grid_thw.append(e.get("image_grid_thw"))
            pixel_values_videos.append(e.get("pixel_values_videos"))
            video_grid_thw.append(e.get("video_grid_thw"))
            vspg = e.get("video_second_per_grid")
            if vspg is not None and not isinstance(vspg, torch.Tensor):
                vspg = torch.as_tensor(vspg)
            video_second_per_grid.append(vspg)
            input_features.append(e.get("input_features"))
            feature_attention_mask.append(e.get("feature_attention_mask"))
            use_audio_in_video.append(e["_unirl_audio_in_video"])

        cond = Qwen3OmniARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
        )
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"text": decoded},
                conditions=cond.to_dict(),
                status=torch.tensor(status_values, dtype=torch.long),
            )
        )


__all__ = ["Qwen3OmniThinkerAdapter"]
