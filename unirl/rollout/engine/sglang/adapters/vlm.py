"""``VLMAdapter`` — the narrowest VLM overrides on the text base."""

from __future__ import annotations

from typing import Any, Dict, List

from unirl.config.require import require
from unirl.rollout.engine.sglang.adapters.base import (
    MMEncoding,
    PreparedInputs,
    register_adapter,
)
from unirl.rollout.engine.sglang.adapters.text import TextLMAdapter
from unirl.rollout.engine.sglang.backends import RawResult
from unirl.rollout.engine.sglang.utils import (
    ResolvedSampling,
    build_vision_conversations,
    pil_to_base64,
)
from unirl.types.sample import Sample


@register_adapter("vlm")
class VLMAdapter(TextLMAdapter):
    """VLM conversion (e.g. Qwen2.5-VL): processor-encoded multimodal prompts."""

    def validate(self) -> None:
        super().validate()
        require(
            self.cfg.image_token is not None,
            f"{type(self).__name__} requires config.image_token (the VLM switch)",
        )
        require(
            self._processor is not None,
            f"{type(self).__name__} requires an AutoProcessor (the engine loads one when config.image_token is set)",
        )

    def build_inputs(self, sample: Sample, *, sampling: ResolvedSampling) -> PreparedInputs:
        conversations, images_list, k = build_vision_conversations(sample, sampling.system_instruction)
        require(
            k == sampling.n,
            f"{type(self).__name__}.build_inputs: de-expanded fan-out k={k} != "
            f"resolved n={sampling.n}; conversation grouping and the sampling block "
            "disagree on the gen branch.",
        )

        wire: List[Dict[str, Any]] = []
        prompt_token_ids: List[List[int]] = []
        mm_encs: List[MMEncoding] = []
        for messages, images in zip(conversations, images_list):
            mm = self.encode_mm(messages, images)
            mm_encs.append(mm)
            payload = self.base_payload(sampling)
            payload["text"] = mm.text
            payload["image_data"] = pil_to_base64(mm.image)
            wire.append(payload)
            prompt_token_ids.append(list(mm.input_ids))

        return PreparedInputs(
            wire=wire,
            prompt_token_ids=prompt_token_ids,
            resolved_n=sampling.n,
            mm=mm_encs,
        )

    def encode_mm(self, messages: List[Dict[str, Any]], images: List[Any]) -> MMEncoding:
        """Processor-encode one conversation + its image(s) into the native layout."""
        require(
            len(images) == 1,
            f"{type(self).__name__}.encode_mm: expected exactly one image per request, "
            f"got {len(images)} (multi-image conversations are unsupported).",
        )
        template_kwargs: Dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        template_kwargs.update(self.cfg.chat_template_kwargs or {})
        processor_messages = [
            {
                **message,
                "content": (
                    [{"type": "text", "text": message["content"]}]
                    if isinstance(message.get("content"), str)
                    else message.get("content")
                ),
            }
            for message in messages
        ]
        text = self._processor.apply_chat_template(processor_messages, **template_kwargs)
        enc = self._processor(text=[text], images=images, return_tensors="pt")
        return MMEncoding(
            image=images[0],
            text=text,
            input_ids=enc["input_ids"][0].tolist(),
            pixel_values=enc["pixel_values"],
            image_grid_thw=enc["image_grid_thw"],
        )

    def build_conditions(self, sample: Sample, prepared: PreparedInputs, raw: List[RawResult]) -> Dict[str, Any]:
        """Add per-sample ``pixel_values`` / ``image_grid_thw`` to the base."""
        conditions = super().build_conditions(sample, prepared, raw)
        if prepared.mm:
            _, prompt_index = self.replicate_per_sample(prepared)
            per_sample_pixel_values = [prepared.mm[i].pixel_values for i in prompt_index]
            per_sample_image_grid_thw = [prepared.mm[i].image_grid_thw for i in prompt_index]
            if any(p is not None for p in per_sample_pixel_values):
                conditions["pixel_values"] = per_sample_pixel_values
                conditions["image_grid_thw"] = per_sample_image_grid_thw
        return conditions


__all__ = ["VLMAdapter"]
