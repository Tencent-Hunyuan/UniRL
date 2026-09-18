"""HunyuanVideo10TextEmbedStage — LLaMA 3D ``[B, seq, 4096]`` and CLIP pooled 2D ``[B, 768]`` streams."""

from __future__ import annotations

from typing import List, Tuple

import torch

from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import HunyuanVideo10Bundle

PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


class HunyuanVideo10TextEmbedStage:
    """Dual-encoder text -> two ``TextEmbedCondition`` instances."""

    def __init__(
        self,
        bundle: HunyuanVideo10Bundle,
        *,
        llama_max_length: int = 256,
        clip_max_length: int = 77,
        crop_start: int = 95,
        hidden_state_skip_layer: int = 2,
    ) -> None:
        self.bundle = bundle
        self.llama_max_length = int(llama_max_length)
        self.clip_max_length = int(clip_max_length)
        self.crop_start = int(crop_start)
        self.hidden_state_skip_layer = int(hidden_state_skip_layer)
        if self.hidden_state_skip_layer < 0:
            raise ValueError(
                f"HunyuanVideo10TextEmbedStage.hidden_state_skip_layer must be >= 0, got {self.hidden_state_skip_layer}"
            )

    def embed_llama(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts via LLaMA — embeds ``[B, llama_max_length, 4096]``, mask ``[B, llama_max_length]``."""
        embeds, mask = self._encode_llama(list(p.texts))
        return TextEmbedCondition(embeds=embeds, attn_mask=mask, pooled=None)

    def _encode_llama(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        tokenizer = bundle.tokenizer
        text_encoder = bundle.text_encoder
        device = bundle.device
        dtype = next(text_encoder.parameters()).dtype
        crop_start = self.crop_start

        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)

        template = PROMPT_TEMPLATE["template"]
        formatted = [template.format(p if p else "") for p in prompts]

        text_inputs = tokenizer(
            formatted,
            padding="max_length",
            max_length=self.llama_max_length + crop_start,
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
        hidden_states = getattr(outputs, "hidden_states", None)
        selected_from_end = self.hidden_state_skip_layer + 1
        if hidden_states is None or selected_from_end > len(hidden_states):
            available = 0 if hidden_states is None else len(hidden_states)
            raise ValueError(
                "HunyuanVideo10TextEmbedStage.hidden_state_skip_layer selects a "
                f"nonexistent state: skip={self.hidden_state_skip_layer}, "
                f"encoder returned {available} hidden states"
            )
        prompt_embeds = hidden_states[-selected_from_end]

        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            attention_mask = attention_mask[:, crop_start:]

        return prompt_embeds.to(dtype=dtype), attention_mask

    def embed_clip(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts via CLIP into a TextEmbedCondition — embeds are the pooled ``[B, 768]``."""
        embeds = self._encode_clip(list(p.texts))
        return TextEmbedCondition(embeds=embeds, attn_mask=None, pooled=None)

    def _encode_clip(self, prompts: List[str]) -> torch.Tensor:
        bundle = self.bundle
        tokenizer = bundle.tokenizer_2
        text_encoder = bundle.text_encoder_2
        device = bundle.device
        dtype = next(text_encoder.parameters()).dtype

        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=self.clip_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device=device)
        attention_mask = text_inputs.attention_mask.to(device=device)

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        pooled_output = outputs.pooler_output

        return pooled_output.to(dtype=dtype)


__all__ = [
    "HunyuanVideo10TextEmbedStage",
    "PROMPT_TEMPLATE",
]
