"""Build token and TMRoPE video conditions for Qwen3-Omni."""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts

from .bundle import Qwen3OmniBundle
from .conditions import Qwen3OmniARConditions
from .media import extract_audio_from_video_pyav
from .video import limit_video_frames, sample_video_frames_pyav


class Qwen3OmniChatTemplateStage:
    def __init__(
        self,
        bundle: Qwen3OmniBundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        pad_to_max_length: bool = False,
        video_fps: float = 1.0,
        video_max_frames: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        # Cross-worker CONCAT requires a common sequence length when enabled.
        self.pad_to_max_length = bool(pad_to_max_length)
        self.video_fps = float(video_fps)
        self.video_max_frames = int(video_max_frames) if video_max_frames is not None else None
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        self.use_audio_in_video = bool(use_audio_in_video)

    def embed(
        self,
        texts: Texts,
        videos: Optional[List[Optional[Any]]] = None,
    ) -> Qwen3OmniARConditions:
        """Build conditions from prompts and optional per-sample videos.

        ``videos[i]`` is a decoded video for sample ``i`` (or ``None``); pass
        ``videos=None`` for a text-only batch.
        """
        processor = self.bundle.processor
        device = self.bundle.device
        dtype = self.bundle.dtype
        batch_size = len(texts.texts)
        audio_sr = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000))

        per_sample_inputs = []
        for i, text in enumerate(texts.texts):
            content: list = []
            sample_video = None
            sample_video_fps = self.video_fps
            sample_audio = None
            if videos is not None and i < len(videos) and videos[i] is not None:
                raw_video = videos[i]
                # Decode paths here; decoded tensors/arrays pass through.
                if isinstance(raw_video, str):
                    sample_video, sample_video_fps = sample_video_frames_pyav(
                        raw_video,
                        target_fps=self.video_fps,
                        max_frames=self.video_max_frames,
                    )
                    if self.use_audio_in_video:
                        sample_audio = extract_audio_from_video_pyav(raw_video, audio_sr)
                else:
                    sample_video, sample_video_fps = limit_video_frames(
                        raw_video,
                        fps=self.video_fps,
                        max_frames=self.video_max_frames,
                    )
                # The processor materializes the video placeholder.
                content.append({"type": "video", "video": sample_video})
            content.append({"type": "text", "text": text})

            messages: list = []
            if self.system_instruction is not None:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": content})

            per_sample_inputs.append(
                self._encode(processor, messages, sample_video, sample_video_fps, sample_audio)
            )

        for inp in per_sample_inputs:
            prompt_len = int(inp["input_ids"].shape[-1])
            if prompt_len > self.max_prompt_length and inp.get("pixel_values_videos") is not None:
                raise ValueError(
                    "Qwen3OmniChatTemplateStage: multimodal prompt produced "
                    f"{prompt_len} tokens, exceeding max_prompt_length={self.max_prompt_length}. "
                    "Reduce video_max_frames, video_max_pixels, or video_fps, or raise max_prompt_length."
                )

        if self.pad_to_max_length:
            max_len = self.max_prompt_length
        else:
            max_len = min(
                max(inp["input_ids"].shape[-1] for inp in per_sample_inputs),
                self.max_prompt_length,
            )
        pad_id = self.bundle.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError(
                "Qwen3OmniChatTemplateStage.embed: tokenizer has no pad_token_id; "
                "Qwen3OmniBundle.from_config sets pad_token=eos_token when absent."
            )

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        for i, inp in enumerate(per_sample_inputs):
            ids = inp["input_ids"].squeeze(0)
            L = min(int(ids.shape[0]), max_len)
            input_ids[i, :L] = ids[:L].to(device)
            mask = inp["attention_mask"].squeeze(0)
            attention_mask[i, :L] = mask[:L].to(device)

        # Keep media as per-sample CONCAT lists.
        pixel_values_videos: List[Optional[torch.Tensor]] = []
        video_grid_thw: List[Optional[torch.Tensor]] = []
        video_second_per_grid: List[Optional[torch.Tensor]] = []
        input_features: List[Optional[torch.Tensor]] = []
        feature_attention_mask: List[Optional[torch.Tensor]] = []
        for inp in per_sample_inputs:
            pvv = inp.get("pixel_values_videos")
            vgt = inp.get("video_grid_thw")
            vspg = inp.get("video_second_per_grid")
            pixel_values_videos.append(pvv.to(device=device, dtype=dtype) if pvv is not None else None)
            video_grid_thw.append(vgt.to(device=device) if vgt is not None else None)
            if vspg is not None:
                vspg_t = vspg if isinstance(vspg, torch.Tensor) else torch.as_tensor(vspg)
                video_second_per_grid.append(vspg_t.to(device=device))
            else:
                video_second_per_grid.append(None)
            ivf = inp.get("input_features")
            fam = inp.get("feature_attention_mask")
            input_features.append(ivf.to(device=device, dtype=dtype) if ivf is not None else None)
            feature_attention_mask.append(fam.to(device=device) if fam is not None else None)

        has_video = any(p is not None for p in pixel_values_videos)
        has_audio = any(a is not None for a in input_features)
        return Qwen3OmniARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values_videos=pixel_values_videos if has_video else None,
            video_grid_thw=video_grid_thw if has_video else None,
            video_second_per_grid=video_second_per_grid if has_video else None,
            input_features=input_features if has_audio else None,
            feature_attention_mask=feature_attention_mask if has_audio else None,
        )

    def _encode(self, processor, messages, sample_video, sample_video_fps, sample_audio):
        """Encode video-only prompts or interleaved audio/video prompts."""
        size = None
        if sample_video is not None and self.video_max_pixels is not None:
            size = {
                "shortest_edge": int(processor.video_processor.size["shortest_edge"]),
                "longest_edge": self.video_max_pixels,
            }

        if sample_audio is not None:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            kwargs: dict = {
                "text": [text],
                "videos": [sample_video],
                "audio": [sample_audio],
                "use_audio_in_video": True,
                "fps": sample_video_fps,
                "do_sample_frames": False,
                "truncation": True,
                "return_tensors": "pt",
            }
            if size is not None:
                kwargs["size"] = size
            return processor(**kwargs)

        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if sample_video is not None:
            kwargs["fps"] = sample_video_fps
            kwargs["do_sample_frames"] = False
            if size is not None:
                kwargs["size"] = size
        return processor.apply_chat_template(messages, **kwargs)


__all__ = ["Qwen3OmniChatTemplateStage"]
