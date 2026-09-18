"""Qwen3_5ChatTemplateStage — text/image conversations → AR conditions."""

from __future__ import annotations

from typing import Any, List, Optional, Union

import torch

from unirl.config.require import require
from unirl.models.types.conversations import build_text_messages, build_vision_messages
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Turn

from .bundle import Qwen3_5Bundle
from .conditions import Qwen3_5ARConditions

Qwen3_5ChatInput = Union[List[Turn], Texts]


class Qwen3_5ChatTemplateStage:
    """Apply the Qwen3.5 chat template, right-pad in batch, return AR conditions."""

    def __init__(
        self,
        bundle: Qwen3_5Bundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        enable_thinking: bool = False,
        pad_to_max_length: bool = False,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        self.enable_thinking = bool(enable_thinking)
        self.pad_to_max_length = bool(pad_to_max_length)

    def embed(
        self,
        value: Qwen3_5ChatInput,
        images: Optional[List[Optional[Any]]] = None,
    ) -> Qwen3_5ARConditions:
        """Render Sample-native turns or supervised text/image rows."""
        if isinstance(value, Texts):
            batch_size = len(value)
            if batch_size == 0:
                raise ValueError("Qwen3_5ChatTemplateStage.embed: expected at least one text row.")
            image_rows = [None] * batch_size if images is None else list(images)
            if len(image_rows) != batch_size:
                raise ValueError(
                    f"Qwen3_5ChatTemplateStage.embed: images length {len(image_rows)} != text batch {batch_size}."
                )
            conversations = []
            for text, image in zip(value.texts, image_rows):
                messages = []
                if self.system_instruction:
                    messages.append({"role": "system", "content": self.system_instruction})
                if image is not None:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": text},
                            ],
                        }
                    )
                else:
                    messages.append({"role": "user", "content": text})
                conversations.append(messages)
        else:
            turns = value
            if images is not None:
                raise ValueError(
                    "Qwen3_5ChatTemplateStage.embed: images must be carried by Turn "
                    "content; the separate images argument is only valid with Texts input."
                )
            if not turns:
                raise ValueError("Qwen3_5ChatTemplateStage.embed: expected at least one conversation turn.")
            unsupported = [
                type(turn.content).__name__ for turn in turns if not isinstance(turn.content, (Texts, Images))
            ]
            if unsupported:
                raise ValueError(
                    f"Qwen3_5ChatTemplateStage.embed: only text and image turns are supported; got {unsupported}."
                )
            image_turn_count = sum(isinstance(turn.content, Images) for turn in turns)
            require(
                image_turn_count <= 1,
                "Qwen3_5ChatTemplateStage.embed: at most one image turn per request "
                "is supported (multi-image trajectories are out of scope).",
            )
            if image_turn_count:
                conversations = build_vision_messages(turns, self.system_instruction)
            else:
                conversations = build_text_messages(turns, self.system_instruction)

        processor = self.bundle.processor
        device = self.bundle.device
        dtype = self.bundle.dtype

        def apply_template(
            messages: List[dict],
            *,
            add_generation_prompt: bool,
        ) -> dict:
            has_media = any(
                isinstance(message.get("content"), list)
                and any(isinstance(part, dict) and part.get("type") == "image" for part in message["content"])
                for message in messages
            )
            if has_media:
                processor_messages = [
                    {
                        **message,
                        "content": (
                            [{"type": "text", "text": message["content"]}]
                            if isinstance(message.get("content"), str)
                            else message.get("content")
                        ),
                    }
                    for message in messages
                ]
                return processor.apply_chat_template(
                    processor_messages,
                    add_generation_prompt=add_generation_prompt,
                    enable_thinking=self.enable_thinking,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

            ids = processor.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=self.enable_thinking,
                tokenize=True,
                return_dict=False,
                return_tensors="pt",
                truncation=False,
            )
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
            }

        per_sample_inputs = []
        for messages in conversations:
            inputs = apply_template(messages, add_generation_prompt=True)
            has_media = any(
                isinstance(message.get("content"), list)
                and any(isinstance(part, dict) and part.get("type") == "image" for part in message["content"])
                for message in messages
            )
            prompt_len = int(inputs["input_ids"].shape[-1])
            if has_media and prompt_len > self.max_prompt_length:
                raise ValueError(
                    "Qwen3_5ChatTemplateStage.embed: multimodal prompt length "
                    f"{prompt_len} exceeds max_prompt_length={self.max_prompt_length}. "
                    "Refusing to truncate after vision preprocessing because that can "
                    "desynchronize image placeholder tokens from pixel_values or grids. "
                    "Increase pipeline.max_prompt_length or shorten the prompt."
                )
            if not has_media and prompt_len > self.max_prompt_length:
                base = apply_template(messages, add_generation_prompt=False)
                suffix_len = prompt_len - int(base["input_ids"].shape[-1])
                if suffix_len < 0 or suffix_len >= self.max_prompt_length:
                    raise ValueError(
                        "Qwen3_5ChatTemplateStage.embed: cannot truncate prompt of "
                        f"length {prompt_len} to max_prompt_length={self.max_prompt_length} "
                        f"while preserving the generation-prompt suffix (suffix_len={suffix_len})."
                    )
                head = self.max_prompt_length - suffix_len
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"][:, :head], inputs["input_ids"][:, prompt_len - suffix_len :]], dim=-1
                )
                inputs["attention_mask"] = torch.cat(
                    [inputs["attention_mask"][:, :head], inputs["attention_mask"][:, prompt_len - suffix_len :]],
                    dim=-1,
                )
            per_sample_inputs.append(inputs)
        batch_size = len(per_sample_inputs)

        if self.pad_to_max_length:
            max_len = self.max_prompt_length
        else:
            max_len = min(
                max(inp["input_ids"].shape[-1] for inp in per_sample_inputs),
                self.max_prompt_length,
            )

        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError(
                "Qwen3_5ChatTemplateStage.embed: tokenizer has no pad_token_id; "
                "Qwen3_5Bundle.from_config sets pad_token=eos_token when absent."
            )

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for i, inp in enumerate(per_sample_inputs):
            ids = inp["input_ids"].squeeze(0)
            L = min(int(ids.shape[0]), max_len)
            input_ids[i, :L] = ids[:L].to(device)
            mask = inp["attention_mask"].squeeze(0)
            attention_mask[i, :L] = mask[:L].to(device)

        pixel_values: List[Optional[torch.Tensor]] = []
        image_grid_thw: List[Optional[torch.Tensor]] = []
        for inp in per_sample_inputs:
            pv = inp.get("pixel_values")
            igt = inp.get("image_grid_thw")
            pixel_values.append(pv.to(device=device, dtype=dtype) if pv is not None else None)
            image_grid_thw.append(igt.to(device=device) if igt is not None else None)

        has_img = any(p is not None for p in pixel_values)

        return Qwen3_5ARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values=pixel_values if has_img else None,
            image_grid_thw=image_grid_thw if has_img else None,
        )


__all__ = ["Qwen3_5ChatTemplateStage"]
