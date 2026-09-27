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
    old = processor.system_prompt
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
        max_prompt_length: int,
    ) -> None:
        self.bundle = bundle
        self.user_role, self.assistant_role = bundle.processor.new_chat_template().roles
        self.max_prompt_length = max_prompt_length

    def embed(
        self,
        texts: Texts,
        images: List[PIL.Image.Image],
        *,
        system_instruction: Optional[str],
    ) -> JanusProARConditions:
        if len(images) != len(texts.texts):
            raise ValueError(
                "JanusProChatTemplateStage.embed expects a homogeneous Text+Image batch "
                f"(texts={len(texts.texts)}, images={len(images)})."
            )

        processor = self.bundle.processor
        prepares = []
        with _temporary_system_prompt(processor, system_instruction):
            for text, image in zip(texts.texts, images):
                conversation = [
                    {
                        "role": self.user_role,
                        "content": f"{processor.image_tag}\n{text}",
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
        expected_image_tokens = len(images) * processor.num_image_tokens
        seq_tokens = batched.images_seq_mask.sum().item()
        emb_tokens = batched.images_emb_mask.sum().item()
        if seq_tokens != expected_image_tokens or emb_tokens != expected_image_tokens:
            raise RuntimeError(
                "JanusProChatTemplateStage failed to encode every conditioning image: "
                f"expected {expected_image_tokens} image tokens, got sequence_mask={seq_tokens}, "
                f"embedding_mask={emb_tokens}."
            )
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
