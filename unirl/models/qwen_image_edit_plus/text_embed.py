"""QwenImageEditPlusTextEmbedStage — edit chat-template text (+ optional image) → TextEmbedCondition."""

from __future__ import annotations

from typing import Optional

import torch

from unirl.models.qwen_image.text_embed import extract_masked_hidden
from unirl.models.types.embedding import ImageConditionedEmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Images, Texts

from .bundle import QwenImageEditPlusBundle
from .vae import aligned_size_for_aspect

PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, "
    "background), then explain how the user's text instruction should alter or modify the image. Generate a new "
    "image that meets the user's requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_START_IDX = 64
IMG_PROMPT_TEMPLATE = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
TOKENIZER_MAX_LENGTH = 1024

CONDITION_IMAGE_AREA = 384 * 384


class QwenImageEditPlusTextEmbedStage(ImageConditionedEmbedStage[Texts, Images, TextEmbedCondition]):
    """Edit-template text (+ optional source images) → ``TextEmbedCondition``."""

    def __init__(
        self,
        bundle: QwenImageEditPlusBundle,
        *,
        max_sequence_length: int = 512,
        processor_path: Optional[str] = None,
    ) -> None:
        if max_sequence_length > TOKENIZER_MAX_LENGTH:
            raise ValueError(
                f"QwenImageEditPlusTextEmbedStage.max_sequence_length cannot exceed "
                f"{TOKENIZER_MAX_LENGTH} (tokenizer cap) but got {max_sequence_length}"
            )
        from transformers import Qwen2VLProcessor

        self.bundle = bundle
        self.max_sequence_length = int(max_sequence_length)
        self.processor = Qwen2VLProcessor.from_pretrained(
            processor_path or bundle.pretrained_path,
            subfolder="processor",
        )

    def embed(self, p: Texts, images: Optional[Images] = None) -> TextEmbedCondition:
        """Encode prompts; optionally condition on source images."""
        prompts = list(p.texts)
        bundle = self.bundle
        device = bundle.device
        dtype = next(bundle.text_encoder.parameters()).dtype

        condition_pils = None
        if images is not None:
            import PIL.Image

            condition_pils = []
            for pil in images.to_pils():
                cond_w, cond_h = aligned_size_for_aspect(pil.width, pil.height, target_area=CONDITION_IMAGE_AREA)
                if pil.width != cond_w or pil.height != cond_h:
                    pil = pil.resize((cond_w, cond_h), PIL.Image.LANCZOS)
                condition_pils.append(pil)
            if len(condition_pils) != len(prompts):
                raise ValueError(
                    f"QwenImageEditPlusTextEmbedStage.embed: image count {len(condition_pils)} "
                    f"!= prompt count {len(prompts)}"
                )
            base_img_prompt = IMG_PROMPT_TEMPLATE.format(1)
        else:
            base_img_prompt = ""

        txt = [PROMPT_TEMPLATE.format(base_img_prompt + e) for e in prompts]

        model_inputs = self.processor(
            text=txt,
            images=condition_pils,
            padding=True,
            return_tensors="pt",
        ).to(device)

        forward_kwargs = {
            "input_ids": model_inputs.input_ids,
            "attention_mask": model_inputs.attention_mask,
            "output_hidden_states": True,
        }
        pixel_values = model_inputs["pixel_values"] if "pixel_values" in model_inputs else None
        image_grid_thw = model_inputs["image_grid_thw"] if "image_grid_thw" in model_inputs else None
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values.to(dtype=dtype)
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw

        with torch.no_grad():
            encoder_out = bundle.text_encoder(**forward_kwargs)
        hidden_states = encoder_out.hidden_states[-1]

        split_hidden_states = extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [item[PROMPT_TEMPLATE_START_IDX:] for item in split_hidden_states]
        attn_mask_list = [
            torch.ones(item.size(0), dtype=torch.long, device=item.device) for item in split_hidden_states
        ]
        max_seq_len = max(item.size(0) for item in split_hidden_states)

        prompt_embeds = torch.stack(
            [
                torch.cat([item, item.new_zeros(max_seq_len - item.size(0), item.size(1))])
                for item in split_hidden_states
            ]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([item, item.new_zeros(max_seq_len - item.size(0))]) for item in attn_mask_list]
        )

        prompt_embeds = prompt_embeds[:, : self.max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, : self.max_sequence_length]
        return TextEmbedCondition(
            embeds=prompt_embeds.to(device=device, dtype=dtype),
            attn_mask=prompt_embeds_mask,
            pooled=None,
        )


__all__ = ["QwenImageEditPlusTextEmbedStage"]
