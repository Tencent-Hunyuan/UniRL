"""Build role-aware token and TMRoPE video conditions for Qwen3-Omni."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch

from unirl.models.types.conversations import build_video_messages
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts
from unirl.types.sample import Turn

from .bundle import Qwen3OmniBundle
from .conditions import Qwen3OmniARConditions
from .video import limit_video_frames, sample_video_frames_pyav

Qwen3OmniChatInput = Union[List[Turn], Texts]


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
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        # Cross-worker CONCAT requires a common sequence length when enabled.
        self.pad_to_max_length = bool(pad_to_max_length)
        self.video_fps = float(video_fps)
        self.video_max_frames = int(video_max_frames) if video_max_frames is not None else None
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    def embed(
        self,
        value: Qwen3OmniChatInput,
        videos: Optional[List[Optional[Any]]] = None,
    ) -> Qwen3OmniARConditions:
        """Render Sample-native turns or supervised text/video rows.

        ``List[Turn]`` is the rollout path and retains the complete role-aware
        trajectory. ``Texts`` plus optional videos is the supervised path. Both
        normalize to the same processor-message representation so rollout and
        replay share the exact encoding stored on the generated Part.
        """
        if isinstance(value, Texts):
            batch_size = len(value)
            if batch_size == 0:
                raise ValueError("Qwen3OmniChatTemplateStage.embed: expected at least one text row.")
            video_rows = [None] * batch_size if videos is None else list(videos)
            if len(video_rows) != batch_size:
                raise ValueError(
                    f"Qwen3OmniChatTemplateStage.embed: videos length {len(video_rows)} != text batch {batch_size}."
                )
            conversations = []
            for text, video in zip(value.texts, video_rows):
                messages: List[Dict[str, Any]] = []
                if self.system_instruction is not None:
                    messages.append({"role": "system", "content": self.system_instruction})
                content: List[Dict[str, Any]] = []
                if video is not None:
                    content.append({"type": "video", "video": video})
                content.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": content})
                conversations.append(messages)
        else:
            if videos is not None:
                raise ValueError(
                    "Qwen3OmniChatTemplateStage.embed: videos must be carried by Turn content; "
                    "the separate videos argument is only valid with Texts input."
                )
            if not value:
                raise ValueError("Qwen3OmniChatTemplateStage.embed: expected at least one conversation turn.")
            conversations = build_video_messages(value, self.system_instruction)

        return self.embed_messages(conversations)

    def _multimodal_processor_kwargs(self, *, video_fps: float) -> Dict[str, Any]:
        processor = self.bundle.processor
        kwargs: Dict[str, Any] = {
            "fps": video_fps,
            "do_sample_frames": False,
        }
        if self.video_max_pixels is not None:
            kwargs["size"] = {
                "shortest_edge": int(processor.video_processor.size["shortest_edge"]),
                "longest_edge": self.video_max_pixels,
            }
        return kwargs

    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
        """Decode/sample the single supported video block without mutating input."""
        prepared: List[Dict[str, Any]] = []
        processor_kwargs: Dict[str, Any] = {}
        video_count = 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                prepared.append(dict(message))
                continue
            blocks: List[Dict[str, Any]] = []
            for raw_block in content:
                block = dict(raw_block)
                if block.get("type") == "video":
                    video_count += 1
                    if video_count > 1:
                        raise ValueError(
                            "Qwen3OmniChatTemplateStage supports at most one persistent source video per conversation."
                        )
                    raw_video = block.get("video")
                    if isinstance(raw_video, str):
                        frames, effective_fps = sample_video_frames_pyav(
                            raw_video,
                            target_fps=self.video_fps,
                            max_frames=self.video_max_frames,
                        )
                    else:
                        frames, effective_fps = limit_video_frames(
                            raw_video,
                            fps=self.video_fps,
                            max_frames=self.video_max_frames,
                        )
                    block["video"] = frames
                    processor_kwargs = self._multimodal_processor_kwargs(video_fps=effective_fps)
                blocks.append(block)
            copied = dict(message)
            copied["content"] = blocks
            prepared.append(copied)
        return prepared, processor_kwargs, video_count > 0

    def embed_messages(
        self,
        conversations: List[List[Dict[str, Any]]],
    ) -> Qwen3OmniARConditions:
        """Processor-encode one role-aware conversation per frontier row."""
        if not conversations:
            raise ValueError("Qwen3OmniChatTemplateStage.embed_messages: empty conversation batch.")

        processor = self.bundle.processor
        device = self.bundle.device
        dtype = self.bundle.dtype
        batch_size = len(conversations)

        per_sample_inputs: List[Dict[str, Any]] = []
        has_video_by_row: List[bool] = []
        for messages in conversations:
            prepared, mm_kwargs, has_video = self._prepare_messages(messages)
            template_kwargs = dict(self.chat_template_kwargs)
            template_kwargs.update(mm_kwargs)
            # These define the replay-condition wire shape and cannot be
            # overridden by recipe-level chat kwargs.
            template_kwargs.update(
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = processor.apply_chat_template(prepared, **template_kwargs)
            prompt_len = int(inputs["input_ids"].shape[-1])
            if prompt_len > self.max_prompt_length:
                if has_video:
                    raise ValueError(
                        "Qwen3OmniChatTemplateStage: multimodal prompt produced "
                        f"{prompt_len} tokens, exceeding max_prompt_length={self.max_prompt_length}. "
                        "Reduce video_max_frames, video_max_pixels, or video_fps, or raise max_prompt_length."
                    )
                inputs = dict(inputs)
                inputs["input_ids"] = inputs["input_ids"][..., -self.max_prompt_length :]
                inputs["attention_mask"] = inputs["attention_mask"][..., -self.max_prompt_length :]
            per_sample_inputs.append(inputs)
            has_video_by_row.append(has_video)

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

        has_video = any(has_video_by_row)
        return Qwen3OmniARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values_videos=pixel_values_videos if has_video else None,
            video_grid_thw=video_grid_thw if has_video else None,
            video_second_per_grid=video_second_per_grid if has_video else None,
        )


__all__ = ["Qwen3OmniChatInput", "Qwen3OmniChatTemplateStage"]
