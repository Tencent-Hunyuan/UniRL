"""ZImageTextEmbedStage — Qwen3 chat-template text → TextEmbedCondition.

Implements ``EmbedStage[Texts, TextEmbedCondition]``. Mirrors the
diffusers ``ZImagePipeline._encode_prompt`` byte-for-byte at the spec
level:

- **Single causal-LM encoder** (``Qwen3Model``). Each prompt is wrapped in
  a chat template (``add_generation_prompt=True``, ``enable_thinking=True``)
  before tokenizing.
- **Second-to-last hidden layer**. Z-Image conditions on
  ``encoder_out.hidden_states[-2]`` (not the final layer), per the
  reference pipeline.
- **No fixed-prefix strip** (unlike Qwen-Image). Every non-pad token of
  the chat-templated prompt participates in conditioning.
- **Variable-length token output**. After dropping pad positions each
  prompt has a different residual length; the stage pads to the batch-max
  with zero embeddings and emits a parallel ``attn_mask`` so the single-
  stream transformer can rebuild the per-prompt caption list (the
  transformer itself runs variable-length attention over the list).

No ``pooled`` vector is produced — Z-Image's transformer accepts
token-level hidden states only. ``TextEmbedCondition.pooled`` is left as
``None``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import ZImageBundle


class ZImageTextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """Qwen3 chat-template text → ``TextEmbedCondition`` stage."""

    def __init__(
        self,
        bundle: ZImageBundle,
        *,
        max_sequence_length: int = 512,
    ) -> None:
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

    def _encode(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        device = bundle.device
        dtype = next(bundle.text_encoder.parameters()).dtype
        tokenizer = bundle.tokenizer

        templated = []
        for prompt_item in prompts:
            messages = [{"role": "user", "content": prompt_item}]
            templated.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )

        text_inputs = tokenizer(
            templated,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        with torch.no_grad():
            encoder_out = bundle.text_encoder(
                input_ids=input_ids,
                attention_mask=prompt_masks,
                output_hidden_states=True,
            )
        # Z-Image conditions on the second-to-last hidden layer.
        hidden_states = encoder_out.hidden_states[-2]

        # Split into per-prompt variable-length slices (drop pad positions).
        split_hidden_states = [hidden_states[i][prompt_masks[i]] for i in range(len(prompts))]
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

        return prompt_embeds.to(device=device, dtype=dtype), prompt_embeds_mask


__all__ = ["ZImageTextEmbedStage"]
