"""QwenImage21TextEmbedStage - Qwen3-VL raw-template prompt -> pre-final-norm ``TextEmbedCondition`` (see README)."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from unirl.models.qwen_image.text_embed import extract_masked_hidden
from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import QwenImage21Bundle

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
)


class QwenImage21TextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """Qwen3-VL text-to-image prompt -> right-padded ``TextEmbedCondition`` ``[B, L, 4096]``."""

    def __init__(self, bundle: QwenImage21Bundle) -> None:
        self.bundle = bundle
        sys_message = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
        sys_tokens = bundle.processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)
        self.drop_idx = len(sys_tokens[0])

    def embed(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts; the system-prefix tokens are dropped and the rest right-padded."""
        bundle = self.bundle
        prompts = [PROMPT_TEMPLATE.format(text or " ") for text in p.texts]
        inputs = bundle.processor(text=prompts, padding=True, padding_side="left", return_tensors="pt").to(
            bundle.device
        )
        encoder = bundle.text_encoder.model
        handle = encoder.language_model.norm.register_forward_hook(lambda module, args, output: args[0])
        try:
            with torch.no_grad():
                hidden = encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask).last_hidden_state
        finally:
            handle.remove()

        split = [h[self.drop_idx :] for h in extract_masked_hidden(hidden, inputs.attention_mask)]
        embeds = pad_sequence(split, batch_first=True)
        lengths = torch.tensor([h.shape[0] for h in split], device=embeds.device)
        attn_mask = (torch.arange(embeds.shape[1], device=embeds.device) < lengths[:, None]).long()
        return TextEmbedCondition(embeds=embeds, attn_mask=attn_mask, pooled=None)


__all__ = ["QwenImage21TextEmbedStage"]
