"""QwenImageTextEmbedStage — Qwen-VL chat-template text → TextEmbedCondition.

Implements ``EmbedStage[Texts, TextEmbedCondition]``. Diverges from
:class:`unirl.models.sd3.SD3TextEmbedStage` in three ways:

- **Single multimodal LLM encoder** (vs SD3's CLIP1 + CLIP2 + T5 stack).
  Qwen-Image wraps prompts in a chat template before tokenizing and
  forwards through a ``Qwen2_5_VLForConditionalGeneration`` model.
- **System prefix stripping**. The chat template prefix
  (``PROMPT_TEMPLATE_START_IDX = 34`` tokens) is identical across every
  prompt and is discarded after the encoder forward — only the
  per-prompt portion of the last hidden state participates in
  conditioning.
- **Variable-length token output**. After prefix stripping each prompt
  has a different residual length; the stage pads to the batch-max with
  zero embeddings and emits a parallel ``attn_mask`` so the diffusion
  transformer (and any CFG concatenation downstream) sees real-vs-pad
  positions explicitly.

No ``pooled`` vector is produced — Qwen-Image's transformer accepts
token-level hidden states only. ``TextEmbedCondition.pooled`` is left
as ``None``.

Math mirrors PR #104's ``_get_qwen_prompt_embeds`` / ``encode_prompt``
byte-for-byte at the spec level.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import QwenImageBundle

# Chat-template constants (upstream Qwen-Image convention).
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_START_IDX = 34
TOKENIZER_MAX_LENGTH = 1024


class QwenImageTextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """Qwen-VL chat-template text → ``TextEmbedCondition`` stage."""

    def __init__(
        self,
        bundle: QwenImageBundle,
        *,
        max_sequence_length: int = 512,
    ) -> None:
        if max_sequence_length > TOKENIZER_MAX_LENGTH:
            raise ValueError(
                f"QwenImageTextEmbedStage.max_sequence_length cannot exceed "
                f"{TOKENIZER_MAX_LENGTH} (tokenizer cap) but got {max_sequence_length}"
            )
        self.bundle = bundle
        self.max_sequence_length = int(max_sequence_length)

    def embed(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts into a ``TextEmbedCondition``."""
        prompt_embeds, prompt_embeds_mask = self._encode(list(p.texts))
        return TextEmbedCondition(
            embeds=prompt_embeds,
            attn_mask=prompt_embeds_mask,
            pooled=None,
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        """Split a padded ``[B, T, D]`` tensor into ``B`` variable-length
        ``[t_i, D]`` slices using a ``[B, T]`` 0/1 mask."""
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return list(torch.split(selected, valid_lengths.tolist(), dim=0))

    def _encode(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        device = bundle.device
        dtype = next(bundle.text_encoder.parameters()).dtype

        texts = [PROMPT_TEMPLATE.format(item) for item in prompts]
        # Tokenizer cap includes the chat-template prefix.
        max_length = TOKENIZER_MAX_LENGTH + PROMPT_TEMPLATE_START_IDX
        text_inputs = bundle.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            encoder_out = bundle.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                output_hidden_states=True,
            )
        hidden_states = encoder_out.hidden_states[-1]

        split_hidden_states = self._extract_masked_hidden(hidden_states, text_inputs.attention_mask)
        # Strip the chat-template prefix from every prompt.
        split_hidden_states = [item[PROMPT_TEMPLATE_START_IDX:] for item in split_hidden_states]
        attn_mask_list = [
            torch.ones(item.size(0), dtype=torch.long, device=item.device) for item in split_hidden_states
        ]
        max_seq_len = max(item.size(0) for item in split_hidden_states)

        prompt_embeds = torch.stack(
            [
                torch.cat([item, item.new_zeros(max_seq_len - item.size(0), item.size(1))])
                for item in split_hidden_states
            ]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([item, item.new_zeros(max_seq_len - item.size(0))]) for item in attn_mask_list]
        )

        # Final slice to the configured budget.
        prompt_embeds = prompt_embeds[:, : self.max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, : self.max_sequence_length]
        return prompt_embeds.to(device=device, dtype=dtype), prompt_embeds_mask


__all__ = ["QwenImageTextEmbedStage"]
