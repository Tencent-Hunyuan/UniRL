"""BooguImageTextEmbedStage — Qwen3-VL chat-template text → TextEmbedCondition."""

from __future__ import annotations

import torch

from unirl.models.types.embedding import EmbedStage, dedup_prompts
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import BooguImageBundle

SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images based on "
    "user instructions. The instructions are as follows."
)
SYSTEM_PROMPT_DROP = (
    "Describe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text "
    "instruction should alter or modify the image. Generate a new image that "
    "meets the user's requirements while maintaining consistency with the "
    "original input where appropriate."
)


class BooguImageTextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """Qwen3-VL chat-template text → ``TextEmbedCondition`` stage."""

    def __init__(
        self,
        bundle: BooguImageBundle,
        *,
        max_sequence_length: int = 1280,
    ) -> None:
        self.bundle = bundle
        self.max_sequence_length = int(max_sequence_length)

    def embed(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts into a ``TextEmbedCondition``."""
        bundle = self.bundle
        if bundle.text_encoder is None or bundle.processor is None:
            raise ValueError(
                "BooguImageTextEmbedStage.embed: bundle has no text encoder/processor (load_text_encoder=False?)"
            )
        prompts, inverse = dedup_prompts(p.texts)
        device = bundle.device
        dtype = next(bundle.text_encoder.parameters()).dtype

        message_lists = [
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT_T2I if prompt and prompt.strip() else SYSTEM_PROMPT_DROP,
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in prompts
        ]
        vlm_inputs = bundle.processor.apply_chat_template(
            message_lists,
            padding="longest",
            max_length=self.max_sequence_length,
            truncation=False,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        input_ids = vlm_inputs["input_ids"].to(device)
        attention_mask = vlm_inputs["attention_mask"].to(device)

        with torch.no_grad():
            encoder_out = bundle.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        hidden_states = encoder_out.last_hidden_state

        cond = TextEmbedCondition(
            embeds=hidden_states.to(device=device, dtype=dtype),
            attn_mask=attention_mask,
            pooled=None,
        )
        return cond if inverse is None else cond.select(inverse)


__all__ = ["BooguImageTextEmbedStage", "SYSTEM_PROMPT_T2I", "SYSTEM_PROMPT_DROP"]
