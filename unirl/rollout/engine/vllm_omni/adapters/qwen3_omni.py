"""Qwen3-Omni Thinker rollout adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from unirl.config.require import require
from unirl.models.qwen3_omni.media import extract_audio_from_video_pyav
from unirl.models.qwen3_omni.video import limit_video_frames, sample_video_frames_pyav
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.hi3 import Hi3TextOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni.utils import texts_from_req
from unirl.types.primitives import Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


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
                (
                    i
                    for i in range(len(result) - 1)
                    if result[i : i + 2] == [vision_bos_token_id, audio_bos_token_id]
                ),
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
            result = (
                result[:start]
                + [vision_bos_token_id, video_token_id, vision_eos_token_id]
                + result[end + 2 :]
            )

    for mm_token_id in (audio_token_id, image_token_id, video_token_id):
        compressed: List[int] = []
        for token_id in result:
            if token_id != mm_token_id or not compressed or compressed[-1] != mm_token_id:
                compressed.append(token_id)
        result = compressed
    return result


class Qwen3OmniThinkerInputAdapter:
    """Build one batched AR generate call from a rollout request."""

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
    ) -> None:
        self.modality = modality
        self.model_path = str(model_path)
        self.video_fps = float(video_fps)
        self.video_max_frames = int(video_max_frames) if video_max_frames is not None else None
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        self.use_audio_in_video = bool(use_audio_in_video)
        self.max_prompt_length = int(max_prompt_length)
        self.system_instruction = system_instruction

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

        # Cached for replay-condition construction by the output adapter.
        self._last_encodings: List[Dict[str, Any]] = []

    def _multimodal_processor_kwargs(self, *, video_fps: float) -> Dict[str, Any]:
        """Processor kwargs shared by driver encoding and vLLM."""
        kwargs: Dict[str, Any] = {
            "fps": video_fps,
            "do_sample_frames": False,
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

    def _extract_videos(self, req: RolloutReq, n: int) -> List[Optional[tuple[Any, float]]]:
        """Return one ``(frames, effective_fps)`` entry per prompt, or ``None``."""
        prim = req.primitives.get("video")
        if prim is None:
            return [None] * n
        if not isinstance(prim, Videos):
            raise TypeError(
                f"Qwen3OmniThinkerInputAdapter: req.primitives['video'] must be Videos, got {type(prim).__name__}"
            )
        # Prefer raw paths and sample them at the configured rate.
        uris = getattr(prim, "uris", None)
        if uris:
            require(
                len(uris) == n,
                f"Qwen3OmniThinkerInputAdapter: uris count {len(uris)} != prompt count {n}",
            )
            return [
                sample_video_frames_pyav(
                    uri,
                    target_fps=self.video_fps,
                    max_frames=self.video_max_frames,
                )
                for uri in uris
            ]
        # Unpack pre-decoded frames using cumulative boundaries.
        frames = prim.frames
        cu = prim.cu_frames
        if frames is None or cu is None:
            raise ValueError("Qwen3OmniThinkerInputAdapter: Videos primitive carries neither uris nor packed frames.")
        cu_list = [int(x) for x in cu.tolist()]
        require(
            len(cu_list) - 1 == n,
            f"Qwen3OmniThinkerInputAdapter: video batch {len(cu_list) - 1} != prompt count {n}",
        )
        return [
            limit_video_frames(
                frames[cu_list[i] : cu_list[i + 1]],
                fps=self.video_fps,
                max_frames=self.video_max_frames,
            )
            for i in range(n)
        ]

    def _extract_audios(self, req: RolloutReq, n: int) -> List[Optional[Any]]:
        """Extract one waveform from each source video when audio fusion is enabled."""
        if not self.use_audio_in_video:
            return [None] * n
        prim = req.primitives.get("video")
        uris = getattr(prim, "uris", None) if prim is not None else None
        if not uris:
            return [None] * n
        require(
            len(uris) == n,
            f"Qwen3OmniThinkerInputAdapter: audio uris count {len(uris)} != prompt count {n}",
        )
        sample_rate = int(
            getattr(getattr(self._processor, "feature_extractor", None), "sampling_rate", 16000)
        )
        return [extract_audio_from_video_pyav(uri, sample_rate) for uri in uris]

    def _encode_one(
        self,
        text: str,
        video_frames: Optional[Any],
        video_fps: float,
        audio_wave: Optional[Any],
        system_instruction: Optional[str],
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        if video_frames is not None:
            content.append({"type": "video", "video": video_frames})
        content.append({"type": "text", "text": text})

        messages: List[Dict[str, Any]] = []
        if system_instruction is not None:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": content})

        if audio_wave is not None:
            prompt_text = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            processor_kwargs: Dict[str, Any] = {
                "text": [prompt_text],
                "videos": [video_frames],
                "audio": [audio_wave],
                "use_audio_in_video": True,
                "fps": video_fps,
                "do_sample_frames": False,
                "return_tensors": "pt",
            }
            if self.video_max_pixels is not None:
                processor_kwargs["size"] = {
                    "shortest_edge": int(self._processor.video_processor.size["shortest_edge"]),
                    "longest_edge": self.video_max_pixels,
                }
            return self._processor(**processor_kwargs)

        template_kwargs: Dict[str, Any] = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if video_frames is not None:
            template_kwargs.update(self._multimodal_processor_kwargs(video_fps=video_fps))
        return self._processor.apply_chat_template(messages, **template_kwargs)

    def build(self, req: RolloutReq) -> List[GenerateCall]:
        texts = texts_from_req(req)
        n = len(texts.texts)
        require(
            len(req.sample_ids) == n,
            f"Qwen3OmniThinkerInputAdapter: sample id count {len(req.sample_ids)} != prompt count {n}",
        )
        video_frames = self._extract_videos(req, n)
        audio_waves = self._extract_audios(req, n)

        # Allow a per-request system instruction.
        chat_overrides = dict(req.task_config.get("chat") or {})
        sys_instr = chat_overrides.get("system_instruction", self.system_instruction)

        prompts: List[Dict[str, Any]] = []
        # The output adapter consumes this cache after generation.
        self._last_encodings = []
        for text, video_sample, aw in zip(texts.texts, video_frames, audio_waves):
            vf, effective_fps = video_sample if video_sample is not None else (None, self.video_fps)
            enc = self._encode_one(text, vf, effective_fps, aw, sys_instr)
            expanded_ids = enc["input_ids"].squeeze(0).tolist()
            if len(expanded_ids) > self.max_prompt_length:
                # Multimodal token truncation would break feature alignment.
                if vf is not None or aw is not None:
                    raise ValueError(
                        f"Qwen3OmniThinkerInputAdapter: multimodal prompt produced {len(expanded_ids)} tokens, "
                        f"exceeding max_prompt_length={self.max_prompt_length}. Reduce video_max_pixels "
                        "or video_fps, or raise max_prompt_length."
                    )
                enc = dict(enc)
                enc["input_ids"] = enc["input_ids"][..., -self.max_prompt_length :]
                enc["attention_mask"] = enc["attention_mask"][..., -self.max_prompt_length :]
                expanded_ids = enc["input_ids"].squeeze(0).tolist()
            rollout_ids = (
                self._compress_prompt_ids(expanded_ids, use_audio_in_video=aw is not None)
                if vf is not None or aw is not None
                else expanded_ids
            )
            entry: Dict[str, Any] = {"prompt_token_ids": rollout_ids}
            media: Dict[str, Any] = {}
            if vf is not None:
                media["video"] = [vf]
            if aw is not None:
                media["audio"] = [aw]
            if media:
                entry["multi_modal_data"] = media
            if vf is not None or aw is not None:
                # Keep this request-local: globally enabling audio-in-video
                # breaks vLLM's video-only dummy profiling during engine boot.
                mm_processor_kwargs = self._multimodal_processor_kwargs(video_fps=effective_fps)
                if aw is not None:
                    mm_processor_kwargs["use_audio_in_video"] = True
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

        ar = req.sampling_params.get("ar")
        max_new_tokens = int(getattr(ar, "max_new_tokens", 512))
        temperature = float(getattr(ar, "temperature", 1.0))
        top_p = float(getattr(ar, "top_p", 1.0))
        top_k_val = int(getattr(ar, "top_k", 0))
        top_k = top_k_val if top_k_val > 0 else -1  # vLLM: -1 disables top_k
        stop_token_id = getattr(ar, "stop_token_id", None)

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

    def build_conditions(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Assemble replay conditions from cached processor outputs."""
        del req, per_request
        import torch

        from unirl.models.qwen3_omni.conditions import Qwen3OmniARConditions
        from unirl.types.conditions import TextTokenCondition

        encs = self._input_adapter._last_encodings
        if not encs:
            raise RuntimeError(
                "Qwen3OmniThinkerOutputAdapter.build_conditions: input adapter "
                "cache is empty — ``build_inputs`` must run before ``build_response``."
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
        )
        self.output_adapter = Qwen3OmniThinkerOutputAdapter(self.modality, self.input_adapter)

    def schedule_policy(self) -> Any:
        """Return no diffusion schedule for the AR-only stage."""
        return None

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; "
                "use req.primitives['video'] for multimodal input."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


__all__ = ["Qwen3OmniThinkerAdapter", "Qwen3OmniThinkerInputAdapter"]
