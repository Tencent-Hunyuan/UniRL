from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from unirl.config.require import require
from unirl.data.sft import tokenize_agent_target
from unirl.models.types.conversations import build_vision_messages
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Turn

from .bundle import QwenVLBundle
from .conditions import QwenVLARConditions

QwenVLChatInput = Union[List[Turn], Texts]


class QwenVLChatTemplateStage:
    supports_message_images = True  # embed_messages accepts image parts in message content

    def __init__(
        self,
        bundle: QwenVLBundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        pad_to_max_length: bool = False,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        self.pad_to_max_length = bool(pad_to_max_length)

    def embed(
        self,
        value: QwenVLChatInput,
        images: Optional[List[Optional[Any]]] = None,
    ) -> QwenVLARConditions:
        """Render role-aware turns or supervised single-turn rows."""
        if isinstance(value, Texts):
            batch_size = len(value)
            if batch_size == 0:
                raise ValueError("QwenVLChatTemplateStage.embed: expected at least one text row.")
            image_rows = [None] * batch_size if images is None else list(images)
            if len(image_rows) != batch_size:
                raise ValueError(
                    f"QwenVLChatTemplateStage.embed: images length {len(image_rows)} != text batch {batch_size}."
                )
            conversations = []
            for text, image in zip(value.texts, image_rows):
                messages = []
                if self.system_instruction:
                    messages.append({"role": "system", "content": self.system_instruction})
                content = []
                if image is not None:
                    content.append({"type": "image", "image": image})
                content.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": content})
                conversations.append(messages)
        else:
            turns = value
            if images is not None:
                raise ValueError(
                    "QwenVLChatTemplateStage.embed: images must be carried by Turn content; "
                    "the separate images argument is only valid with Texts input."
                )
            if not turns:
                raise ValueError("QwenVLChatTemplateStage.embed: expected at least one conversation turn.")
            require(
                sum(isinstance(t.content, Images) for t in turns) <= 1,
                "QwenVLChatTemplateStage.embed: at most one image turn per request is "
                "supported (multi-image trajectories are out of scope).",
            )
            conversations = build_vision_messages(turns, self.system_instruction)

        processor = self.bundle.processor
        per_sample_inputs = []
        for messages in conversations:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            per_sample_inputs.append(inputs)
        return self._pack_conditions(per_sample_inputs, truncate=True)

    def embed_messages(
        self,
        conversations: Sequence[Sequence[Dict[str, Any]]],
        *,
        tools: Optional[Sequence[Optional[Sequence[Dict[str, Any]]]]] = None,
    ) -> QwenVLARConditions:
        """Render OpenAI-style histories (interleaved text/image parts) as AR conditions."""
        if not conversations:
            raise ValueError("QwenVLChatTemplateStage.embed_messages: empty conversation batch.")
        if tools is not None and any(t for t in tools):
            raise ValueError(
                "QwenVLChatTemplateStage.embed_messages: the Qwen2.5-VL chat template has no tool "
                "rendering, so tool schemas would silently vanish from the prompt; agent records "
                "with tools are unsupported on this backbone."
            )
        processor = self.bundle.processor
        per_sample_inputs = []
        for row, messages in enumerate(conversations):
            for turn, message in enumerate(messages):
                if message.get("role") == "tool":
                    raise ValueError(
                        f"QwenVLChatTemplateStage.embed_messages: conversation {row} message {turn} has "
                        "role='tool' — the Qwen2.5-VL chat template has no tool template and would render "
                        "it as a bare ChatML block the model never saw in training."
                    )
                if message.get("content") is None:
                    raise ValueError(
                        f"QwenVLChatTemplateStage.embed_messages: conversation {row} message {turn} has "
                        "content=null — the Qwen2.5-VL chat template iterates content parts and cannot render it."
                    )
                if message.get("tool_calls"):
                    raise ValueError(
                        f"QwenVLChatTemplateStage.embed_messages: conversation {row} message {turn} carries "
                        "tool_calls — the Qwen2.5-VL chat template has no tool-call rendering, so they "
                        "would silently vanish from the prompt."
                    )
            # transformers 4.5x ProcessorMixin iterates every message's content as a
            # part list (string content TypeErrors); 5.x normalizes internally. Wrap
            # strings as single text parts — byte-identical under the chat template.
            normalized = [
                {**m, "content": [{"type": "text", "text": m["content"]}]} if isinstance(m.get("content"), str) else m
                for m in messages
            ]
            inputs = processor.apply_chat_template(
                normalized,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            per_sample_inputs.append(inputs)
        return self._pack_conditions(per_sample_inputs, truncate=False)

    def tokenize_agent_target(self, record: Dict[str, Any]) -> List[int]:
        """Reject tool-call targets the template cannot render, then HF-suffix-tokenize the final turn."""
        target = record["messages"][-1]
        if target.get("tool_calls"):
            raise ValueError(
                f"QwenVLChatTemplateStage.tokenize_agent_target: record {record.get('sample_id')!r} target "
                "turn carries tool_calls — the Qwen2.5-VL chat template has no tool-call rendering, so only "
                "its text would be supervised. Filter tool-call targets out of Qwen-VL manifests."
            )
        return tokenize_agent_target(record, tokenizer=self.bundle.tokenizer, enable_thinking=False)

    def _pack_conditions(self, per_sample_inputs: List[Any], *, truncate: bool) -> QwenVLARConditions:
        """Right-pad per-sample processor outputs into batched AR conditions."""
        device = self.bundle.device
        dtype = self.bundle.dtype
        batch_size = len(per_sample_inputs)

        if not truncate:
            for i, inp in enumerate(per_sample_inputs):
                length = int(inp["input_ids"].shape[-1])
                if length > self.max_prompt_length:
                    raise ValueError(
                        f"QwenVLChatTemplateStage.embed_messages: conversation {i} renders to {length} tokens "
                        f"> max_prompt_length={self.max_prompt_length}; truncation would desync vision spans, "
                        "so filter or shorten overlong histories during manifest preparation."
                    )

        if self.pad_to_max_length:
            max_len = self.max_prompt_length
        else:
            max_len = min(
                max(inp["input_ids"].shape[-1] for inp in per_sample_inputs),
                self.max_prompt_length,
            )
        pad_id = self.bundle.processor.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError(
                "QwenVLChatTemplateStage.embed: tokenizer has no pad_token_id; "
                "QwenVLBundle.from_config sets pad_token=eos_token when absent."
            )

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for i, inp in enumerate(per_sample_inputs):
            ids = inp["input_ids"].squeeze(0)
            L = min(int(ids.shape[0]), max_len)
            input_ids[i, :L] = ids[:L].to(device)
            mask = inp["attention_mask"].squeeze(0)
            attention_mask[i, :L] = mask[:L].to(device)

        pixel_values: List[Optional[torch.Tensor]] = []
        image_grid_thw: List[Optional[torch.Tensor]] = []
        for inp in per_sample_inputs:
            pv = inp.get("pixel_values")
            igt = inp.get("image_grid_thw")
            pixel_values.append(pv.to(device=device, dtype=dtype) if pv is not None else None)
            image_grid_thw.append(igt.to(device=device) if igt is not None else None)

        # Always batch-aligned lists (None per image-free row), never a collapsed
        # None: concat drops None contributors, so a collapsed value from one DP
        # shard would desync the merged list from the batch rows (and a later
        # slice would broadcast it whole). _merge_pv/_merge_igt filter row Nones.
        return QwenVLARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )


__all__ = ["QwenVLChatTemplateStage"]
