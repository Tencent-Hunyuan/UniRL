"""Flux2KleinTextEmbedStage — Qwen3 chat-template text → TextEmbedCondition."""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch

from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import Flux2KleinBundle

logger = logging.getLogger(__name__)

POOLED_DIM = 768
DEFAULT_KLEIN_EXTRACTION_LAYERS: Tuple[int, ...] = (9, 18, 27)


class Flux2KleinTextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """Qwen3 chat-template text → ``TextEmbedCondition`` stage."""

    def __init__(
        self,
        bundle: Flux2KleinBundle,
        *,
        max_sequence_length: int = 512,
        extraction_layers: Tuple[int, ...] = DEFAULT_KLEIN_EXTRACTION_LAYERS,
    ) -> None:
        self.bundle = bundle
        self.max_sequence_length = int(max_sequence_length)
        self.extraction_layers = tuple(extraction_layers)

    def embed(self, p: Texts) -> TextEmbedCondition:
        embeds, pooled = self._encode(list(p.texts))
        return TextEmbedCondition(
            embeds=embeds,
            pooled=pooled,
        )

    def _encode(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        device = bundle.device
        dtype = next(bundle.text_encoder.parameters()).dtype

        tokenizer = bundle.tokenizer

        all_input_ids: List[torch.Tensor] = []
        all_attention_masks: List[torch.Tensor] = []
        if hasattr(tokenizer, "apply_chat_template"):
            for text in prompts:
                messages = [{"role": "user", "content": text}]
                templated = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(
                    templated,
                    padding="max_length",
                    max_length=self.max_sequence_length,
                    truncation=True,
                    return_tensors="pt",
                )
                all_input_ids.append(inputs.input_ids)
                all_attention_masks.append(inputs.attention_mask)
            input_ids = torch.cat(all_input_ids, dim=0).to(device)
            attention_mask = torch.cat(all_attention_masks, dim=0).to(device)
        else:
            inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=self.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            outputs = bundle.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        extracted = []
        for layer_idx in self.extraction_layers:
            if layer_idx < len(hidden_states):
                extracted.append(hidden_states[layer_idx])
            else:
                logger.warning(
                    "Flux2KleinTextEmbedStage: layer %d requested but model only has %d layers; "
                    "using last layer as fallback",
                    layer_idx,
                    len(hidden_states),
                )
                extracted.append(hidden_states[-1])

        prompt_embeds = torch.cat(extracted, dim=-1).to(dtype=dtype)
        pooled = self._pool(outputs, attention_mask, dtype=dtype)
        return prompt_embeds, pooled

    @staticmethod
    def _pool(outputs, attention_mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
            if pooled.shape[-1] != POOLED_DIM:
                pooled = pooled[..., :POOLED_DIM]
            return pooled.to(dtype=dtype)

        last_hidden = outputs.hidden_states[-1]
        seq_lens = attention_mask.sum(dim=-1, keepdim=True) - 1
        seq_lens = seq_lens.clamp(min=0)
        eos_hidden = torch.gather(
            last_hidden,
            dim=1,
            index=seq_lens.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1]),
        ).squeeze(1)
        return eos_hidden[..., :POOLED_DIM].to(dtype=dtype)


__all__ = ["Flux2KleinTextEmbedStage"]
