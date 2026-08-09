"""BooguImageTextEmbedStage — Qwen3-VL chat-template text → TextEmbedCondition.

Implements ``EmbedStage[Texts, TextEmbedCondition]``. Mirrors the reference
``BooguImagePipeline._get_instruction_feature_embeds`` /
``_apply_chat_template`` (T2I path, no images) at the spec level:

- **Single multimodal LLM encoder** (the lm_head-stripped ``Qwen3VLModel``
  from the checkpoint's ``mllm/`` subfolder). Each prompt becomes a chat
  message list ``[system, user(text)]`` and the paired ``Qwen3VLProcessor``
  templates + tokenizes it in one ``apply_chat_template`` call
  (``padding="longest"``, ``padding_side="right"``, truncation off — the
  reference passes ``truncate_instruction_sequence=False``).
- **Adaptive system prompt** (reference ``_apply_chat_template`` with its
  ``system_prompt_follows_task_type=False`` default): a non-empty prompt
  gets the fixed T2I system prompt; an **empty/whitespace prompt — which is
  exactly what the CFG negative ``""`` is — gets the DROP system prompt**
  ("dataset logic"). Reproducing this switch is required for CFG parity
  with the reference pipeline.
- **Last hidden layer, full sequence, no repacking** (unlike z_image's
  pad-drop repack and qwen_image's fixed-prefix strip): the checkpoint's
  ``instruction_feature_configs`` selects 1 layer with mean-reduce ==
  identity, and the transformer consumes the right-padded embeds plus the
  processor's attention mask directly (its ``rope_embedder`` /
  context-refiner run variable-length attention off that mask).

No ``pooled`` vector is produced — the DiT accepts token-level hidden
states only. ``TextEmbedCondition.pooled`` is left as ``None``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from unirl.models.types.embedding import EmbedStage
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
        prompt_embeds, prompt_embeds_mask = self._encode(list(p.texts))
        return TextEmbedCondition(
            embeds=prompt_embeds,
            attn_mask=prompt_embeds_mask,
            pooled=None,
        )

    @staticmethod
    def _messages(prompt: str) -> List[dict]:
        """Build the reference chat-message list for one T2I prompt.

        Empty/whitespace prompts (the CFG negative ``""``) switch to the
        DROP system prompt, mirroring the reference adaptive branch
        (``_apply_chat_template``, pipeline_boogu.py:1594-1612).
        """
        system_prompt = SYSTEM_PROMPT_T2I if prompt and prompt.strip() else SYSTEM_PROMPT_DROP
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]

    def _encode(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        if bundle.text_encoder is None or bundle.processor is None:
            raise ValueError(
                "BooguImageTextEmbedStage._encode: bundle has no text encoder/processor (load_text_encoder=False?)"
            )
        device = bundle.device
        dtype = next(bundle.text_encoder.parameters()).dtype

        message_lists = [self._messages(prompt) for prompt in prompts]
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

        return hidden_states.to(device=device, dtype=dtype), attention_mask


__all__ = ["BooguImageTextEmbedStage", "SYSTEM_PROMPT_T2I", "SYSTEM_PROMPT_DROP"]
