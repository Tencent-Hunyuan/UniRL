from __future__ import annotations

from typing import List

import torch

from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts

from .bundle import JanusProBundle
from .conditions import JanusProImageARConditions


def _right_pad_rows(rows: List[List[int]], *, pad_id: int, device: torch.device) -> TextTokenCondition:
    if not rows:
        raise ValueError("JanusProImagePromptStage.embed: empty text batch.")

    max_len = max(len(row) for row in rows)
    if max_len <= 0:
        raise ValueError("JanusProImagePromptStage.embed: tokenizer produced an empty prompt.")

    input_ids = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        n = len(row)
        input_ids[i, :n] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[i, :n] = 1
    return TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask)


class JanusProImagePromptStage:
    """Build Janus-Pro Text -> Image prompt tokens, including CFG prompts."""

    def __init__(
        self,
        bundle: JanusProBundle,
        *,
        user_role: str = "<|User|>",
        assistant_role: str = "<|Assistant|>",
        system_prompt: str = "",
        max_prompt_length: int = 4096,
    ) -> None:
        self.bundle = bundle
        self.user_role = str(user_role)
        self.assistant_role = str(assistant_role)
        self.system_prompt = str(system_prompt)
        self.max_prompt_length = int(max_prompt_length)

    def embed(self, texts: Texts, *, cfg_weight: float = 5.0) -> JanusProImageARConditions:
        processor = self.bundle.processor
        tokenizer = self.bundle.tokenizer
        prompt_rows: List[List[int]] = []
        cfg_rows: List[List[int]] = []

        for text in texts.texts:
            conversation = [
                {"role": self.user_role, "content": text},
                {"role": self.assistant_role, "content": ""},
            ]
            sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=processor.sft_format,
                system_prompt=self.system_prompt,
            )
            prompt = sft_format + processor.image_start_tag
            token_ids = [int(token_id) for token_id in tokenizer.encode(prompt)]
            if len(token_ids) > self.max_prompt_length:
                raise ValueError(
                    "JanusProImagePromptStage.embed: prompt length "
                    f"{len(token_ids)} exceeds max_prompt_length={self.max_prompt_length}."
                )

            cfg_ids = list(token_ids)
            if len(cfg_ids) > 2:
                cfg_ids[1:-1] = [int(self.bundle.pad_token_id)] * (len(cfg_ids) - 2)
            prompt_rows.append(token_ids)
            cfg_rows.append(cfg_ids)

        return JanusProImageARConditions(
            prompt=_right_pad_rows(prompt_rows, pad_id=self.bundle.pad_token_id, device=self.bundle.device),
            cfg_prompt=_right_pad_rows(cfg_rows, pad_id=self.bundle.pad_token_id, device=self.bundle.device),
            cfg_weight=float(cfg_weight),
        )


__all__ = ["JanusProImagePromptStage"]
