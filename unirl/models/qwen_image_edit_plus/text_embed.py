"""QwenImageEditPlusTextEmbedStage — edit chat-template text (+ optional image) → TextEmbedCondition."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from unirl.models.qwen_image.text_embed import extract_masked_hidden
from unirl.models.types.embedding import ImageConditionedEmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import ImagePrimitive, Texts, as_image_sets

from .bundle import QwenImageEditPlusBundle

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
_SIZE_ALIGN = 32


def _condition_size_for_aspect(width: int, height: int) -> Tuple[int, int]:
    """Aspect-preserving ``(w, h)`` at ≈``CONDITION_IMAGE_AREA``, 32-aligned."""
    ratio = float(width) / float(height)
    cond_w = math.sqrt(CONDITION_IMAGE_AREA * ratio)
    cond_h = cond_w / ratio
    cond_w = round(cond_w / _SIZE_ALIGN) * _SIZE_ALIGN
    cond_h = round(cond_h / _SIZE_ALIGN) * _SIZE_ALIGN
    return int(cond_w), int(cond_h)


class QwenImageEditPlusTextEmbedStage(ImageConditionedEmbedStage[Texts, ImagePrimitive, TextEmbedCondition]):
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
        self.bundle = bundle
        self.max_sequence_length = int(max_sequence_length)
        self.processor = self._load_processor(processor_path or bundle.pretrained_path)

    @staticmethod
    def _load_processor(path: str):
        """Load Qwen2VLProcessor from the checkpoint ``processor/`` subfolder."""
        from transformers import Qwen2VLProcessor

        return Qwen2VLProcessor.from_pretrained(path, subfolder="processor")

    def embed(self, p: Texts, images: Optional[ImagePrimitive] = None) -> TextEmbedCondition:
        """Encode prompts; optionally condition on source images."""
        prompt_embeds, prompt_embeds_mask = self._encode(list(p.texts), images)
        return TextEmbedCondition(
            embeds=prompt_embeds,
            attn_mask=prompt_embeds_mask,
            pooled=None,
        )

    def _condition_pil_rows(self, images: ImagePrimitive):
        """Convert ordered source rows to resized PIL images."""
        import PIL.Image

        rows = []
        for source_row in as_image_sets(images).to_pil_rows():
            resized = []
            for pil in source_row:
                cond_w, cond_h = _condition_size_for_aspect(pil.width, pil.height)
                if pil.width != cond_w or pil.height != cond_h:
                    pil = pil.resize((cond_w, cond_h), PIL.Image.LANCZOS)
                resized.append(pil)
            rows.append(resized)
        return rows

    def _encode(self, prompts: List[str], images: Optional[ImagePrimitive]) -> Tuple[torch.Tensor, torch.Tensor]:
        bundle = self.bundle
        device = bundle.device
        dtype = next(bundle.text_encoder.parameters()).dtype

        condition_pils = None
        if images is not None:
            image_rows = self._condition_pil_rows(images)
            if len(image_rows) != len(prompts):
                raise ValueError(
                    f"QwenImageEditPlusTextEmbedStage._encode: image row count {len(image_rows)} "
                    f"!= prompt count {len(prompts)}"
                )
            counts = [len(row) for row in image_rows]
            if any(count < 1 for count in counts):
                raise ValueError(
                    f"QwenImageEditPlusTextEmbedStage._encode requires at least one image per row; counts={counts}."
                )
            base_img_prompts = [
                "".join(IMG_PROMPT_TEMPLATE.format(index + 1) for index in range(len(row))) for row in image_rows
            ]
            condition_pils = [image for row in image_rows for image in row]
        else:
            base_img_prompts = [""] * len(prompts)

        txt = [PROMPT_TEMPLATE.format(prefix + prompt) for prefix, prompt in zip(base_img_prompts, prompts)]

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
        pixel_values = getattr(model_inputs, "pixel_values", None)
        image_grid_thw = getattr(model_inputs, "image_grid_thw", None)
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values.to(dtype=dtype)
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw

        with torch.no_grad():
            encoder_out = bundle.text_encoder(**forward_kwargs)
        hidden_states = encoder_out.hidden_states[-1]

        split_hidden_states = extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [item[PROMPT_TEMPLATE_START_IDX:] for item in split_hidden_states]
        if images is not None and any(count > 1 for count in counts):
            overlong = [
                index for index, item in enumerate(split_hidden_states) if item.size(0) > self.max_sequence_length
            ]
            if overlong:
                raise ValueError(
                    "QwenImageEditPlusTextEmbedStage: multi-reference prompt exceeds max_sequence_length="
                    f"{self.max_sequence_length}; first overlong row={overlong[0]}, "
                    f"length={split_hidden_states[overlong[0]].size(0)}. Raise max_sequence_length or use fewer refs."
                )
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
        return prompt_embeds.to(device=device, dtype=dtype), prompt_embeds_mask


__all__ = ["QwenImageEditPlusTextEmbedStage"]
