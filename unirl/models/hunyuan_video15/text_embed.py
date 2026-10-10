"""HunyuanVideo15TextEmbedStage — mllm (skip_layers+1 from last) + ByT5; null ByT5 is a zero placeholder."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch

from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import HunyuanVideo15Bundle

# fmt: off
PROMPT_TEMPLATE_SYSTEM_MESSAGE = "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
# fmt: on

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

    DEFAULT_CACHE_SIZE: int = 8

    def __init__(
        self,
        bundle: HunyuanVideo15Bundle,
        *,
        mllm_max_length: int = 1000,
        mllm_crop_start: int = 108,
        mllm_skip_layers: int = 2,
        byt5_max_length: int = 256,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self.bundle = bundle
        self.mllm_max_length = int(mllm_max_length)
        self.mllm_crop_start = int(mllm_crop_start)
        self.mllm_skip_layers = int(mllm_skip_layers)
        self.byt5_max_length = int(byt5_max_length)
        self.cache_size = int(cache_size)
        self._mllm_cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._glyph_cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    def clear_cache(self) -> None:
        """Clear both MLLM and ByT5 prompt-embedding caches."""
        self._mllm_cache.clear()
        self._glyph_cache.clear()

    def _embed_with_cache(
        self,
        prompts: List[str],
        cache: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]],
        encode_fn: Callable[[List[str]], Tuple[torch.Tensor, torch.Tensor]],
    ) -> TextEmbedCondition:
        """Encode prompts with in-batch deduplication and cross-call bounded caching."""
        if not prompts:
            return TextEmbedCondition(
                embeds=torch.empty(0, device=self.bundle.device),
                attn_mask=torch.empty(0, device=self.bundle.device, dtype=torch.int64),
                pooled=None,
            )

        index_of: Dict[str, int] = {}
        inverse: List[int] = []
        uniq: List[str] = []
        for s in prompts:
            i = index_of.get(s)
            if i is None:
                i = len(uniq)
                index_of[s] = i
                uniq.append(s)
            inverse.append(i)

        resolved: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        missing: List[str] = []
        for p in uniq:
            if p in cache:
                resolved[p] = cache[p]
                cache.move_to_end(p)
            else:
                missing.append(p)

        if missing:
            enc_embeds, enc_masks = encode_fn(missing)
            for p, emb, mask in zip(
                missing,
                enc_embeds.detach().to("cpu").contiguous().unbind(0),
                enc_masks.detach().to("cpu").contiguous().unbind(0),
            ):
                entry = (emb.unsqueeze(0), mask.unsqueeze(0))
                resolved[p] = entry
                if self.cache_size > 0:
                    cache[p] = entry
                    if len(cache) > self.cache_size:
                        cache.popitem(last=False)

        device = self.bundle.device
        uniq_embeds = torch.cat([resolved[p][0].to(device=device) for p in uniq], dim=0)
        uniq_masks = torch.cat([resolved[p][1].to(device=device) for p in uniq], dim=0)

        if len(uniq) == len(prompts):
            return TextEmbedCondition(embeds=uniq_embeds, attn_mask=uniq_masks, pooled=None)

        inv = torch.tensor(inverse, device=device)
        return TextEmbedCondition(
            embeds=uniq_embeds.index_select(0, inv),
            attn_mask=uniq_masks.index_select(0, inv),
            pooled=None,
        )

    def embed_mllm(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts via the Qwen2.5-VL MLLM into a TextEmbedCondition."""
        return self._embed_with_cache(list(p.texts), self._mllm_cache, self._encode_mllm)

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
        return self._embed_with_cache(list(p.texts), self._glyph_cache, self._encode_byt5)

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
