"""LTX2 text embedding stage — Gemma3 encoding + connector projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

if TYPE_CHECKING:
    from .bundle import LTX2Bundle


class LTX2TextEmbedStage:
    """Encode text prompts via Gemma3 + optional connector projection.

    For LTX-2.3 with connectors: Gemma3 → connector → (video_embeds, audio_embeds).
    For LTX-2.0 without connectors: Gemma3 → caption_projection on transformer.
    """

    def __init__(self, bundle: "LTX2Bundle") -> None:
        self.text_encoder = bundle.text_encoder
        self.tokenizer = bundle.tokenizer
        self.connectors = bundle.connectors
        self.max_sequence_length = bundle.max_sequence_length
        self.dtype = bundle.dtype
        self.device = bundle.device

    @torch.no_grad()
    def encode(
        self,
        texts: Texts,
        negative_texts: Optional[Texts] = None,
    ) -> dict:
        """Encode prompts → TextEmbedCondition for video (and audio).

        LTX-2.0 ALWAYS routes Gemma hidden states through the text connectors
        (the DiT was trained on connector outputs, not raw Gemma). Returns a
        dict with keys: 'text', 'audio_text', optionally 'negative_text',
        'negative_audio_text'.
        """
        if self.connectors is None:
            raise RuntimeError(
                "LTX2TextEmbedStage: bundle.connectors is None. LTX-2.0 requires "
                "the LTX2TextConnectors; the DiT cannot consume raw Gemma hidden "
                "states. Ensure the checkpoint's 'connectors' subfolder loaded."
            )

        packed_hidden, attention_mask = self._encode_prompts(texts.texts)
        video_embeds, audio_embeds, conn_mask = self._apply_connectors(packed_hidden, attention_mask)

        result = {
            "text": TextEmbedCondition(embeds=video_embeds, attn_mask=conn_mask),
            "audio_text": TextEmbedCondition(embeds=audio_embeds, attn_mask=conn_mask),
        }

        # Negative prompts for CFG
        if negative_texts is not None:
            neg_packed, neg_mask = self._encode_prompts(negative_texts.texts)
            neg_video, neg_audio, neg_conn_mask = self._apply_connectors(neg_packed, neg_mask)
            result["negative_text"] = TextEmbedCondition(embeds=neg_video, attn_mask=neg_conn_mask)
            result["negative_audio_text"] = TextEmbedCondition(embeds=neg_audio, attn_mask=neg_conn_mask)

        return result

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and encode through Gemma3, returning the STACKED all-layers
        packed hidden states ``(B, seq, caption_channels * num_layers)`` that
        the text connector expects (mirrors diffusers ``_get_gemma_prompt_embeds``).
        """
        # Gemma expects LEFT padding for chat-style prompts (diffusers sets this).
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        inputs = self.tokenizer(
            [p.strip() for p in prompts],
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )
        # Stack ALL hidden layers (embedding + each block) on a new trailing
        # axis, then flatten into the channel dim → (B, seq, C * num_layers).
        # This is the connector's expected ``text_encoder_dim`` input.
        stacked = torch.stack(outputs.hidden_states, dim=-1)  # (B, seq, C, L)
        packed = stacked.flatten(2, 3).to(self.dtype)  # (B, seq, C*L)
        return packed, inputs.attention_mask

    def _apply_connectors(
        self,
        packed_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route packed Gemma hidden states through LTX2TextConnectors.

        ``LTX2TextConnectors.forward(text_encoder_hidden_states, attention_mask,
        padding_side="left")`` returns a 3-tuple
        ``(video_text_embedding, audio_text_embedding, binary_attn_mask)``.
        """
        padding_side = getattr(self.tokenizer, "padding_side", "left")
        video_embeds, audio_embeds, conn_mask = self.connectors(
            packed_hidden,
            attention_mask,
            padding_side=padding_side,
        )
        return video_embeds, audio_embeds, conn_mask


__all__ = ["LTX2TextEmbedStage"]
