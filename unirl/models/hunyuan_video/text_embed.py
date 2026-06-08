"""HunyuanVideoTextEmbedStage -- dual-encoder text -> two TextEmbedConditions.

HunyuanVideo-1.0 cross-attends to **two text streams**:

- ``llama``  -- LlamaModel (transformers). The prompt is wrapped in a
  LLaMA-style prompt template with a system header, tokenized with padding
  to ``llama_max_length + crop_start``, encoded, and the template prefix
  is stripped by slicing ``[:, crop_start:]``. Output is the last hidden
  state (3D: ``[B, seq, 4096]``).
- ``clip`` -- CLIPTextModel (transformers). Standard CLIP text encoding,
  output is the pooled vector (2D: ``[B, 768]``).

Implements two unary embed methods (``embed_llama`` / ``embed_clip``)
rather than the strict :class:`EmbedStage` protocol -- the dual output
doesn't fit ``embed(p) -> C``. The pipeline calls them in sequence.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts

from .bundle import HunyuanVideoBundle

# --------------------------------------------------------------------------
# LLaMA prompt template. The system header instructs the model to describe
# video content; ``crop_start`` (default 95) is the prefix token count to
# strip after encoding.
# --------------------------------------------------------------------------

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


class HunyuanVideoTextEmbedStage:
    """Dual-encoder text -> two ``TextEmbedCondition`` instances.

    Not a strict ``EmbedStage[Texts, TextEmbedCondition]`` because the
    dual-stream output doesn't fit a unary ``embed(p) -> C`` shape.
    Provides ``embed_llama`` / ``embed_clip`` instead; the pipeline
    calls them sequentially.
    """

    def __init__(
        self,
        bundle: HunyuanVideoBundle,
        *,
        llama_max_length: int = 256,
        clip_max_length: int = 77,
        crop_start: int = 95,
    ) -> None:
        self.bundle = bundle
        self.llama_max_length = int(llama_max_length)
        self.clip_max_length = int(clip_max_length)
        self.crop_start = int(crop_start)

    # ------------------------------------------------------------------
    # LLaMA stream
    # ------------------------------------------------------------------

    def embed_llama(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts via the LLaMA encoder into a TextEmbedCondition.

        Returns embeds of shape ``[B, llama_max_length, 4096]`` and
        attention_mask of shape ``[B, llama_max_length]`` (after cropping
        the prompt template prefix).
        """
        embeds, mask = self._encode_llama(list(p.texts))
        return TextEmbedCondition(embeds=embeds, attn_mask=mask, pooled=None)

    def _encode_llama(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        tokenizer = bundle.tokenizer
        text_encoder = bundle.text_encoder
        device = bundle.device
        dtype = next(text_encoder.parameters()).dtype
        crop_start = self.crop_start

        # Apply the prompt template to each prompt.
        template = PROMPT_TEMPLATE["template"]
        formatted = [template.format(p if p else "") for p in prompts]

        # Tokenize with padding to (llama_max_length + crop_start) so that
        # after cropping we have exactly llama_max_length tokens.
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
        # Use the last hidden state (unlike HV15 which uses skip_layers).
        prompt_embeds = outputs.last_hidden_state

        # Strip the prompt template prefix tokens.
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            attention_mask = attention_mask[:, crop_start:]

        return prompt_embeds.to(dtype=dtype), attention_mask

    # ------------------------------------------------------------------
    # CLIP stream
    # ------------------------------------------------------------------

    def embed_clip(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts via CLIP into a TextEmbedCondition.

        Returns embeds of shape ``[B, 768]`` (the pooled output).
        ``attn_mask`` is ``None`` because the transformer reads the CLIP
        output as ``pooled_projections`` (no sequence-level masking).
        """
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
        # CLIP pooled output: [B, 768].
        pooled_output = outputs.pooler_output

        return pooled_output.to(dtype=dtype)


__all__ = [
    "HunyuanVideoTextEmbedStage",
    "PROMPT_TEMPLATE",
]
