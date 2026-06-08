from __future__ import annotations

from typing import List, Optional

import PIL.Image
import torch

from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts

from .bundle import QwenVLBundle
from .conditions import QwenVLARConditions


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
        # When True, pad every prompt to a fixed `max_prompt_length` instead of
        # the per-batch dynamic max. Required by the v2 DP trainer: shards from
        # different rollout workers are concatenated (dim 0) at merge time, so
        # input_ids/attention_mask must share one sequence length across shards.
        # Default False preserves the v1 dynamic-pad behavior.
        self.pad_to_max_length = bool(pad_to_max_length)

    def embed(
        self,
        texts: Texts,
        images: Optional[List[Optional[PIL.Image.Image]]] = None,
    ) -> QwenVLARConditions:
        processor = self.bundle.processor
        device = self.bundle.device
        dtype = self.bundle.dtype
        batch_size = len(texts.texts)

        per_sample_inputs = []
        for i, text in enumerate(texts.texts):
            content: list = []
            sample_images: list = []
            if images is not None and i < len(images) and images[i] is not None:
                content.append({"type": "image", "image": images[i]})
                sample_images.append(images[i])
            content.append({"type": "text", "text": text})

            messages: list = []
            if self.system_instruction is not None:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": content})

            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            per_sample_inputs.append(inputs)

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

        # Per-sample lists for pixel_values and image_grid_thw.
        # Each list has batch_size elements (one per sample, possibly None).
        # Using per-sample lists with FieldKind.CONCAT ensures correct
        # concatenation when multiple rollout workers' conditions are merged.
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
