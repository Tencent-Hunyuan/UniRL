from __future__ import annotations

from typing import Any, List, Optional, Union

import torch

from unirl.config.require import require
from unirl.models.types.conversations import build_vision_messages
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Turn

from .bundle import QwenVLBundle
from .conditions import QwenVLARConditions

QwenVLChatInput = Union[List[Turn], Texts]


class QwenVLChatTemplateStage:
    def __init__(
        self,
        bundle: QwenVLBundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        pad_to_max_length: bool = False,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        self.pad_to_max_length = bool(pad_to_max_length)

    def embed(
        self,
        value: QwenVLChatInput,
        images: Optional[List[Optional[Any]]] = None,
    ) -> QwenVLARConditions:
        """Render role-aware turns or supervised single-turn rows.

        ``List[Turn]`` is the Sample-native rollout path; image content is already
        carried by image Turns. ``Texts`` plus an optional PIL per row is the SFT
        path. Both normalize to the same processor-message representation and
        share all tokenization, truncation, padding, and condition packing below.
        """
        if isinstance(value, Texts):
            batch_size = len(value)
            if batch_size == 0:
                raise ValueError("QwenVLChatTemplateStage.embed: expected at least one text row.")
            image_rows = [None] * batch_size if images is None else list(images)
            if len(image_rows) != batch_size:
                raise ValueError(
                    f"QwenVLChatTemplateStage.embed: images length {len(image_rows)} != text batch {batch_size}."
                )
            conversations = []
            for text, image in zip(value.texts, image_rows):
                messages = []
                if self.system_instruction:
                    messages.append({"role": "system", "content": self.system_instruction})
                content = []
                if image is not None:
                    content.append({"type": "image", "image": image})
                content.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": content})
                conversations.append(messages)
        else:
            turns = value
            if images is not None:
                raise ValueError(
                    "QwenVLChatTemplateStage.embed: images must be carried by Turn content; "
                    "the separate images argument is only valid with Texts input."
                )
            if not turns:
                raise ValueError("QwenVLChatTemplateStage.embed: expected at least one conversation turn.")
            require(
                sum(isinstance(t.content, Images) for t in turns) <= 1,
                "QwenVLChatTemplateStage.embed: at most one image turn per request is "
                "supported (multi-image trajectories are out of scope).",
            )
            conversations = build_vision_messages(turns, self.system_instruction)

        processor = self.bundle.processor
        device = self.bundle.device
        dtype = self.bundle.dtype

        per_sample_inputs = []
        for messages in conversations:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
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
                "QwenVLChatTemplateStage.embed: tokenizer has no pad_token_id; "
                "QwenVLBundle.from_config sets pad_token=eos_token when absent."
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

        return QwenVLARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values=pixel_values if any(p is not None for p in pixel_values) else None,
            image_grid_thw=image_grid_thw if any(g is not None for g in image_grid_thw) else None,
        )


__all__ = ["QwenVLChatTemplateStage"]
