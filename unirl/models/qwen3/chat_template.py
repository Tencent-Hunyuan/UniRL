"""Qwen3ChatTemplateStage — ``Texts → Qwen3ARConditions``.

Implements ``EmbedStage[Texts, Qwen3ARConditions]`` from
:mod:`unirl.models.types.embedding`. Applies the bundle
tokenizer's chat template (with ``add_generation_prompt=True``) so the
AR stage starts from the canonical assistant-turn prefix the model was
trained against.

An optional ``system_instruction`` string is prepended as a ``system``
message — callers that need byte-for-byte parity with an SFT template
(e.g. Qwen3's ``qwen3_nothink`` recipe) pass the exact string here. The
stage does not interpret it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts

from .bundle import Qwen3Bundle
from .conditions import Qwen3ARConditions


class Qwen3ChatTemplateStage(EmbedStage[Texts, Qwen3ARConditions]):
    """Apply the Qwen3 chat template, right-pad in batch, return AR conditions."""

    def __init__(
        self,
        bundle: Qwen3Bundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        enable_thinking: bool = False,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        # MUST agree with the rollout engine's chat template
        # (rollout.config.chat_template_kwargs.enable_thinking) or train/rollout
        # prompts diverge and the importance ratio breaks.
        self.enable_thinking = bool(enable_thinking)

    def embed(self, p: Texts) -> Qwen3ARConditions:
        """Tokenize ``p.texts`` via the chat template and pack into AR conditions."""
        conversations: List[List[Dict[str, Any]]] = []
        for text in p.texts:
            messages: List[Dict[str, Any]] = []
            if self.system_instruction is not None:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": text})
            conversations.append(messages)
        return self.embed_messages(conversations)

    def embed_messages(
        self,
        conversations: Sequence[Sequence[Dict[str, Any]]],
        *,
        tools: Optional[Sequence[Optional[Sequence[Dict[str, Any]]]]] = None,
    ) -> Qwen3ARConditions:
        """Render OpenAI-style histories and pack them as generation conditions.

        Agent SFT passes each conversation without its final target assistant
        turn. Long histories are cropped from the left after templating so the
        latest user/tool interaction and assistant generation prefix survive.
        """
        if not conversations:
            raise ValueError("Qwen3ChatTemplateStage.embed_messages: empty conversation batch.")
        if tools is None:
            tools = [None] * len(conversations)
        if len(tools) != len(conversations):
            raise ValueError(
                "Qwen3ChatTemplateStage.embed_messages: tools and conversations batch sizes differ "
                f"({len(tools)} != {len(conversations)})."
            )

        tokenizer = self.bundle.tokenizer
        device = self.bundle.device
        per_sample_ids: List[torch.Tensor] = []
        for messages, sample_tools in zip(conversations, tools):
            ids = tokenizer.apply_chat_template(
                messages,
                tools=sample_tools,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tokenize=True,
                return_tensors="pt",
                return_dict=False,
                truncation=False,
            )
            # transformers v5 flipped apply_chat_template's `return_dict` default
            # from False to True: it now returns a BatchEncoding, and integer-
            # indexing a fast-tokenizer BatchEncoding yields a tokenizers.Encoding
            # (no `.to`), not a tensor. Pin return_dict=False so we keep the bare
            # input_ids [1, L] tensor identically across v4/v5; squeeze leading dim.
            ids = ids[0]
            if ids.shape[0] > self.max_prompt_length:
                ids = ids[-self.max_prompt_length :]
            per_sample_ids.append(ids.to(device=device, dtype=torch.long))

        # Right-pad to the in-batch max so the AR loop can use a single tensor.
        max_len = max(int(t.shape[0]) for t in per_sample_ids)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError(
                "Qwen3ChatTemplateStage.embed: tokenizer has no pad_token_id; "
                "Qwen3Bundle.from_config sets pad_token=eos_token when absent — "
                "check the bundle was constructed via from_config."
            )

        batch = len(per_sample_ids)
        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
        for i, t in enumerate(per_sample_ids):
            n = int(t.shape[0])
            input_ids[i, :n] = t
            attention_mask[i, :n] = 1

        return Qwen3ARConditions(prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask))


__all__ = ["Qwen3ChatTemplateStage"]
