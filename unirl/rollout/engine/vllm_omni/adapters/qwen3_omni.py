"""Qwen3-Omni Thinker rollout adapters."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.config.require import require
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


def _sample_video_frames_pyav(path: str, target_fps: float) -> Any:
    """Decode and sample video frames as ``[T, C, H, W]`` uint8."""
    import av
    import numpy as np
    import torch

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else target_fps
        step = max(1, round(src_fps / float(target_fps)))
        frames = [
            frame.to_ndarray(format="rgb24") for i, frame in enumerate(container.decode(video=0)) if i % step == 0
        ]
    finally:
        container.close()
    if not frames:
        raise ValueError(f"pyav decoded no frames from video: {path}")
    arr = np.stack(frames)  # [T, H, W, C]
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class Qwen3OmniThinkerInputAdapter:
    """Build one batched AR generate call from a rollout request."""

    def __init__(
        self,
        modality: str,
        *,
        model_path: str,
        video_fps: float = 1.0,
        video_max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
        max_prompt_length: int = 12288,
        system_instruction: Optional[str] = None,
    ) -> None:
        self.modality = modality
        self.model_path = str(model_path)
        self.video_fps = float(video_fps)
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

    def _multimodal_processor_kwargs(self) -> Dict[str, Any]:
        """Processor kwargs shared by driver encoding and vLLM."""
        kwargs: Dict[str, Any] = {
            "fps": self.video_fps,
            "do_sample_frames": False,
        }
        if self.video_max_pixels is not None:
            kwargs["size"] = {
                "shortest_edge": int(self._processor.video_processor.size["shortest_edge"]),
                "longest_edge": self.video_max_pixels,
            }
        if self.use_audio_in_video:
            kwargs["use_audio_in_video"] = True
        return kwargs

    def _extract_videos(self, req: RolloutReq, n: int) -> List[Optional[Any]]:
        """Return one entry per prompt: pyav-decoded frames tensor, or ``None``."""
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
            return [_sample_video_frames_pyav(u, self.video_fps) for u in uris]
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
        return [frames[cu_list[i] : cu_list[i + 1]] for i in range(n)]

    def _encode_one(
        self,
        text: str,
        video_frames: Optional[Any],
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

        template_kwargs: Dict[str, Any] = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if video_frames is not None:
            template_kwargs.update(self._multimodal_processor_kwargs())
        return self._processor.apply_chat_template(messages, **template_kwargs)

    def build(self, req: RolloutReq) -> List[GenerateCall]:
        texts = texts_from_req(req)
        n = len(texts.texts)
        require(
            len(req.sample_ids) == n,
            f"Qwen3OmniThinkerInputAdapter: sample id count {len(req.sample_ids)} != prompt count {n}",
        )
        video_frames = self._extract_videos(req, n)

        # Allow a per-request system instruction.
        chat_overrides = dict(req.stage_config.get("chat") or {})
        sys_instr = chat_overrides.get("system_instruction", self.system_instruction)

        prompts: List[Dict[str, Any]] = []
        # The output adapter consumes this cache after generation.
        self._last_encodings = []
        for text, vf in zip(texts.texts, video_frames):
            enc = self._encode_one(text, vf, sys_instr)
            ids = enc["input_ids"].squeeze(0).tolist()
            if len(ids) > self.max_prompt_length:
                # Multimodal token truncation would break feature alignment.
                if vf is not None:
                    raise ValueError(
                        f"Qwen3OmniThinkerInputAdapter: multimodal prompt produced {len(ids)} tokens, "
                        f"exceeding max_prompt_length={self.max_prompt_length}. Reduce video_max_pixels "
                        "or video_fps, or raise max_prompt_length."
                    )
                enc = dict(enc)
                enc["input_ids"] = enc["input_ids"][..., -self.max_prompt_length :]
                enc["attention_mask"] = enc["attention_mask"][..., -self.max_prompt_length :]
                ids = enc["input_ids"].squeeze(0).tolist()
            entry: Dict[str, Any] = {"prompt_token_ids": ids}
            if vf is not None:
                entry["multi_modal_data"] = {"video": [vf]}
                entry["mm_processor_kwargs"] = self._multimodal_processor_kwargs()
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
        for e in encs:
            pvv = e.get("pixel_values_videos")
            vgt = e.get("video_grid_thw")
            vspg = e.get("video_second_per_grid")
            pixel_values_videos.append(pvv if pvv is not None else None)
            video_grid_thw.append(vgt if vgt is not None else None)
            if vspg is not None and not isinstance(vspg, torch.Tensor):
                vspg = torch.as_tensor(vspg)
            video_second_per_grid.append(vspg if vspg is not None else None)

        has_video = any(p is not None for p in pixel_values_videos)
        cond = Qwen3OmniARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values_videos=pixel_values_videos if has_video else None,
            video_grid_thw=video_grid_thw if has_video else None,
            video_second_per_grid=video_second_per_grid if has_video else None,
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
        video_max_pixels = getattr(mc, "video_max_pixels", None) if mc is not None else None
        use_audio_in_video = bool(getattr(mc, "use_audio_in_video", False)) if mc is not None else False
        max_prompt_length = int(getattr(mc, "max_prompt_length", 12288)) if mc is not None else 12288
        system_instruction = getattr(mc, "system_instruction", None) if mc is not None else None

        self.input_adapter = Qwen3OmniThinkerInputAdapter(
            self.modality,
            model_path=model_path,
            video_fps=video_fps,
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
