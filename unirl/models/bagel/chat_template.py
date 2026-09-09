"""BagelChatTemplateStage — supervised prompts/conversations → :class:`BagelARConditions`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from unirl.types.primitives import Texts

from .conditions import BagelARConditions


class BagelChatTemplateStage:
    supports_message_images = True  # embed_messages accepts image parts in message content

    def __init__(
        self,
        bundle: Any,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 8192,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)

    def _text_split(self, text: str) -> Dict[str, Any]:
        """One ``[bos] + encode + [eos]`` wrapped text split (prepare_prompts parity)."""
        ntk = self.bundle.new_token_ids
        ids = [ntk["bos_token_id"]] + self.bundle.tokenizer.encode(text) + [ntk["eos_token_id"]]
        return {"kind": "text", "ids": torch.tensor(ids, dtype=torch.long)}

    def _vit_split(self, image: Any) -> Dict[str, Any]:
        """One ViT-only image split (inferencer.py:249 preproc: rgb → vae resize → vit_transform)."""
        from .vendor.data.data_utils import pil_img2rgb

        resized = self.bundle.vae_transform.resize_transform(pil_img2rgb(image))
        return {"kind": "vit", "image": self.bundle.vit_transform(resized)}

    def _split_tokens(self, split: Dict[str, Any]) -> int:
        """Context length of one split (image block = patches + start/end markers)."""
        if split["kind"] == "text":
            return int(split["ids"].numel())
        patch = int(getattr(self.bundle.model, "vit_patch_size", 14))
        _, h, w = split["image"].shape
        return (h // patch) * (w // patch) + 2

    def _check_length(self, splits: List[Dict[str, Any]], *, row: int) -> None:
        total = sum(self._split_tokens(sp) for sp in splits)
        if total > self.max_prompt_length:
            raise ValueError(
                f"BagelChatTemplateStage: conversation {row} renders to {total} context tokens "
                f"> max_prompt_length={self.max_prompt_length}; filter or shorten overlong "
                "histories during manifest preparation (fewer/smaller images, shorter turns)."
            )

    def embed(self, value: Texts, images: Optional[List[Optional[Any]]] = None) -> BagelARConditions:
        """Supervised single-turn rows: per row ``[vit?, text]`` ordered splits."""
        if not isinstance(value, Texts) or len(value) == 0:
            raise ValueError("BagelChatTemplateStage.embed: expected a non-empty Texts batch.")
        image_rows = [None] * len(value) if images is None else list(images)
        if len(image_rows) != len(value):
            raise ValueError(
                f"BagelChatTemplateStage.embed: images length {len(image_rows)} != text batch {len(value)}."
            )
        splits_per_sample: List[List[Dict[str, Any]]] = []
        for row, (text, image) in enumerate(zip(value.texts, image_rows)):
            splits: List[Dict[str, Any]] = []
            if self.system_instruction:
                splits.append(self._text_split(self.system_instruction))
            if image is not None:
                splits.append(self._vit_split(image))
            splits.append(self._text_split(str(text)))
            self._check_length(splits, row=row)
            splits_per_sample.append(splits)
        return BagelARConditions(prompt_splits=splits_per_sample)

    def embed_messages(
        self,
        conversations: Sequence[Sequence[Dict[str, Any]]],
        *,
        tools: Optional[Sequence[Optional[Sequence[Dict[str, Any]]]]] = None,
    ) -> BagelARConditions:
        """Agent histories → ordered interleaved splits (text chunks + ViT images)."""
        if not conversations:
            raise ValueError("BagelChatTemplateStage.embed_messages: empty conversation batch.")
        if tools is not None and any(t for t in tools):
            raise ValueError(
                "BagelChatTemplateStage.embed_messages: Bagel has no tool-call template; "
                "agent records with tools are unsupported on this backbone."
            )
        splits_per_sample: List[List[Dict[str, Any]]] = []
        for row, messages in enumerate(conversations):
            splits: List[Dict[str, Any]] = []
            for turn, message in enumerate(messages):
                if message.get("role") == "tool":
                    raise ValueError(
                        f"BagelChatTemplateStage.embed_messages: conversation {row} message {turn} has "
                        "role='tool' — Bagel has no tool template, and rendering a tool result as a bare "
                        "text chunk would present it as the model's own utterance."
                    )
                if message.get("tool_calls"):
                    raise ValueError(
                        f"BagelChatTemplateStage.embed_messages: conversation {row} message {turn} carries "
                        "tool_calls — Bagel has no tool-call rendering."
                    )
                content = message.get("content")
                if content is None:
                    continue
                if isinstance(content, str):
                    splits.append(self._text_split(content))
                    continue
                for part in content:
                    kind = part.get("type") if isinstance(part, dict) else None
                    if kind == "text":
                        splits.append(self._text_split(part["text"]))
                    elif kind == "image":
                        splits.append(self._vit_split(part["image"]))
                    else:
                        raise ValueError(
                            f"BagelChatTemplateStage.embed_messages: conversation {row} message {turn} "
                            f"has unsupported content part {kind!r}."
                        )
            if not splits:
                raise ValueError(f"BagelChatTemplateStage.embed_messages: conversation {row} rendered no splits.")
            self._check_length(splits, row=row)
            splits_per_sample.append(splits)
        return BagelARConditions(prompt_splits=splits_per_sample)

    def tokenize_agent_target(self, record: Dict[str, Any]) -> List[int]:
        """Final assistant turn → ``encode(text) + [<|im_end|>]`` (the AR segment layout)."""
        target = record["messages"][-1]
        # Manifests may carry tool-call targets (prepare_sft_agent supervises them on Qwen);
        # tokenizing only this turn's text would drop the tool call out of the CE target silently.
        if target.get("tool_calls"):
            raise ValueError(
                f"BagelChatTemplateStage.tokenize_agent_target: record {record.get('sample_id')!r} target "
                "turn carries tool_calls — Bagel has no tool-call rendering, so only its text would be "
                "supervised. Filter tool-call targets out of Bagel manifests during preparation."
            )
        content = target.get("content")
        if not isinstance(content, str) or not content:
            raise ValueError(
                f"BagelChatTemplateStage.tokenize_agent_target: record {record.get('sample_id')!r} target "
                "turn must be non-empty text (tool_calls / image targets are unsupported on Bagel)."
            )
        return [int(t) for t in self.bundle.tokenizer.encode(content)] + [
            int(self.bundle.new_token_ids["eos_token_id"])
        ]


__all__ = ["BagelChatTemplateStage"]
