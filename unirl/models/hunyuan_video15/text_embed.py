"""HunyuanVideo15TextEmbedStage — mllm (skip_layers+1 from last) + ByT5; null ByT5 is a zero placeholder."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch

from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import HunyuanVideo15Bundle

PROMPT_TEMPLATE_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Describe the video by detailing the following aspects: "
    "1. The main content and theme of the video. "
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
    "4. background environment, light, style and atmosphere. "
    "5. camera angles, movements, and transitions used in the video."
)

_GLYPH_PATTERN = re.compile(r"\"(.*?)\"|“(.*?)”")


def _extract_glyph_texts(prompt: str) -> Optional[str]:
    """Extract quoted glyph snippets and reformat to ``Text "...". `` form."""
    matches = _GLYPH_PATTERN.findall(prompt)
    result = [m[0] or m[1] for m in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result
    if not result:
        return None
    return ". ".join([f'Text "{text}"' for text in result]) + ". "


def _format_chat_template(prompts: List[str], system_message: str) -> List[List[Dict[str, str]]]:
    """Build the (system, user) chat conversation list expected by Qwen2.5-VL."""
    return [
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": p if p else " "},
        ]
        for p in prompts
    ]


class HunyuanVideo15TextEmbedStage:
    """Dual-encoder text → two ``TextEmbedCondition`` instances."""

    def __init__(
        self,
        bundle: HunyuanVideo15Bundle,
        *,
        mllm_max_length: int = 1000,
        mllm_crop_start: int = 108,
        mllm_skip_layers: int = 2,
        byt5_max_length: int = 256,
    ) -> None:
        self.bundle = bundle
        self.mllm_max_length = int(mllm_max_length)
        self.mllm_crop_start = int(mllm_crop_start)
        self.mllm_skip_layers = int(mllm_skip_layers)
        self.byt5_max_length = int(byt5_max_length)

    def embed_mllm(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts via the Qwen2.5-VL MLLM into a TextEmbedCondition."""
        embeds, mask = self._encode_mllm(list(p.texts))
        return TextEmbedCondition(embeds=embeds, attn_mask=mask, pooled=None)

    def _encode_mllm(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        tokenizer = bundle.tokenizer
        text_encoder = bundle.text_encoder
        device = bundle.device
        dtype = next(text_encoder.parameters()).dtype
        crop_start = self.mllm_crop_start

        chat = _format_chat_template(prompts, PROMPT_TEMPLATE_SYSTEM_MESSAGE)
        text_inputs = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding="max_length",
            max_length=self.mllm_max_length + crop_start,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device=device)
        attention_mask = text_inputs.attention_mask.to(device=device)

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        prompt_embeds = outputs.hidden_states[-(self.mllm_skip_layers + 1)]

        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            attention_mask = attention_mask[:, crop_start:]

        return prompt_embeds.to(dtype=dtype), attention_mask

    def embed_glyph(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts via the ByT5 glyph encoder into a TextEmbedCondition."""
        embeds, mask = self._encode_byt5(list(p.texts))
        return TextEmbedCondition(embeds=embeds, attn_mask=mask, pooled=None)

    def _encode_byt5(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        tokenizer = bundle.tokenizer_2
        text_encoder = bundle.text_encoder_2
        device = bundle.device
        max_length = self.byt5_max_length
        d_model = int(getattr(text_encoder.config, "d_model", 1472))
        enc_dtype = next(text_encoder.parameters()).dtype

        embeds_list: List[torch.Tensor] = []
        masks_list: List[torch.Tensor] = []

        for raw in prompts:
            glyph = _extract_glyph_texts(raw or "")
            if glyph is None:
                emb = torch.zeros(1, max_length, d_model, device=device, dtype=enc_dtype)
                mask = torch.zeros(1, max_length, device=device, dtype=torch.int64)
            else:
                tokens = tokenizer(
                    glyph,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                input_ids = tokens.input_ids.to(device=device)
                attn = tokens.attention_mask.to(device=device)
                with torch.no_grad():
                    out = text_encoder(input_ids=input_ids, attention_mask=attn.float())[0]
                emb = out.to(device=device)
                mask = attn.to(device=device)
            embeds_list.append(emb)
            masks_list.append(mask)

        prompt_embeds_2 = torch.cat(embeds_list, dim=0).to(dtype=enc_dtype)
        prompt_embeds_mask_2 = torch.cat(masks_list, dim=0)
        return prompt_embeds_2, prompt_embeds_mask_2


__all__ = [
    "HunyuanVideo15TextEmbedStage",
    "PROMPT_TEMPLATE_SYSTEM_MESSAGE",
    "_extract_glyph_texts",
]
