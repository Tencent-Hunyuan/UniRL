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

        hidden_states, attention_mask = self._encode_prompts(texts.texts)
        video_embeds, audio_embeds, conn_mask = self._apply_connectors(hidden_states, attention_mask)

        result = {
            "text": TextEmbedCondition(embeds=video_embeds, attn_mask=conn_mask),
            "audio_text": TextEmbedCondition(embeds=audio_embeds, attn_mask=conn_mask),
        }

        # Negative prompts for CFG
        if negative_texts is not None:
            neg_hidden_states, neg_mask = self._encode_prompts(negative_texts.texts)
            neg_video, neg_audio, neg_conn_mask = self._apply_connectors(neg_hidden_states, neg_mask)
            result["negative_text"] = TextEmbedCondition(embeds=neg_video, attn_mask=neg_conn_mask)
            result["negative_audio_text"] = TextEmbedCondition(embeds=neg_audio, attn_mask=neg_conn_mask)

        return result

    def _encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and return Gemma3's stacked all-layer hidden states."""
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
        return torch.stack(outputs.hidden_states, dim=-1), inputs.attention_mask

    @staticmethod
    def _pack_text_embeds(
        text_hidden_states: torch.Tensor,
        *,
        sequence_lengths: torch.Tensor,
        padding_side: str = "left",
        scale_factor: int = 8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Normalize and pack Gemma all-layer states exactly like diffusers."""
        batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
        original_dtype = text_hidden_states.dtype
        token_indices = torch.arange(seq_len, device=text_hidden_states.device).unsqueeze(0)
        if padding_side == "right":
            mask = token_indices < sequence_lengths[:, None]
        elif padding_side == "left":
            mask = token_indices >= (seq_len - sequence_lengths[:, None])
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
        mask = mask[:, :, None, None]

        masked = text_hidden_states.masked_fill(~mask, 0.0)
        valid_count = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
        masked_mean = masked.sum(dim=(1, 2), keepdim=True) / (valid_count + eps)
        x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
        x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

        normalized = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
        normalized = normalized * scale_factor
        normalized = normalized.flatten(2)
        mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
        normalized = normalized.masked_fill(~mask_flat, 0.0)
        return normalized.to(dtype=original_dtype)

    def _apply_connectors(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route packed Gemma hidden states through LTX2TextConnectors.

        ``LTX2TextConnectors.forward(text_encoder_hidden_states, attention_mask,
        padding_side="left")`` returns a 3-tuple
        ``(video_text_embedding, audio_text_embedding, binary_attn_mask)``.
        """
        # diffusers >=0.38 accepts stacked 4-D Gemma states and performs the
        # masked normalization itself. Version 0.37 expects the already
        # normalized/flattened 3-D tensor and an additive mask. Support both
        # without normalizing twice on the newer path.
        import inspect

        params = inspect.signature(self.connectors.forward).parameters
        conn_kwargs = {}
        connector_mask = attention_mask
        if "padding_side" in params:
            connector_hidden_states = hidden_states.to(self.dtype)
            conn_kwargs["padding_side"] = getattr(self.tokenizer, "padding_side", "left")
        else:
            connector_hidden_states = self._pack_text_embeds(
                hidden_states,
                sequence_lengths=attention_mask.sum(dim=-1),
                padding_side=getattr(self.tokenizer, "padding_side", "left"),
            ).to(self.dtype)
            if "additive_mask" in params:
                connector_mask = (1 - attention_mask.to(self.dtype)) * -1_000_000.0
                conn_kwargs["additive_mask"] = True
        video_embeds, audio_embeds, conn_mask = self.connectors(
            connector_hidden_states,
            connector_mask,
            **conn_kwargs,
        )
        return video_embeds, audio_embeds, conn_mask


__all__ = ["LTX2TextEmbedStage"]
