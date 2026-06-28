from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional

import PIL.Image

from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts

from .bundle import JanusProBundle
from .conditions import JanusProARConditions


@contextmanager
def _temporary_system_prompt(processor, system_prompt: Optional[str]):
    if system_prompt is None:
        yield
        return
    old = getattr(processor, "system_prompt", None)
    processor.system_prompt = system_prompt
    try:
        yield
    finally:
        processor.system_prompt = old


class JanusProChatTemplateStage:
    def __init__(
        self,
        bundle: JanusProBundle,
        *,
        user_role: str = "<|User|>",
        assistant_role: str = "<|Assistant|>",
        image_placeholder: str = "<image_placeholder>",
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
    ) -> None:
        self.bundle = bundle
        self.user_role = str(user_role)
        self.assistant_role = str(assistant_role)
        self.image_placeholder = str(image_placeholder)
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)

    def embed(
        self,
        texts: Texts,
        images: Optional[List[Optional[PIL.Image.Image]]] = None,
    ) -> JanusProARConditions:
        if images is None:
            raise TypeError("JanusProChatTemplateStage.embed requires one conditioning image per text prompt.")
        if len(images) != len(texts.texts) or any(img is None for img in images):
            raise ValueError(
                "JanusProChatTemplateStage.embed expects a homogeneous Text+Image batch "
                f"(texts={len(texts.texts)}, images={len(images)})."
            )

        processor = self.bundle.processor
        prepares = []
        with _temporary_system_prompt(processor, self.system_instruction):
            for text, image in zip(texts.texts, images):
                conversation = [
                    {
                        "role": self.user_role,
                        "content": f"{self.image_placeholder}\n{text}",
                        "images": [image],
                    },
                    {"role": self.assistant_role, "content": ""},
                ]
                prepares.append(
                    processor.process_one(
                        conversations=conversation,
                        images=[image],
                    )
                )

        batched = processor.batchify(prepares).to(self.bundle.device, dtype=self.bundle.dtype)
        if batched.input_ids.shape[1] > self.max_prompt_length:
            raise ValueError(
                "JanusProChatTemplateStage.embed: prompt length "
                f"{batched.input_ids.shape[1]} exceeds max_prompt_length={self.max_prompt_length}."
            )

        return JanusProARConditions(
            prompt=TextTokenCondition(input_ids=batched.input_ids, attention_mask=batched.attention_mask),
            pixel_values=batched.pixel_values,
            images_seq_mask=batched.images_seq_mask,
            images_emb_mask=batched.images_emb_mask,
        )


__all__ = ["JanusProChatTemplateStage"]
