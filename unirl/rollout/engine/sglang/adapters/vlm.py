"""``VLMAdapter`` — the narrowest VLM overrides on the text base.

Differs from :class:`TextLMAdapter` in exactly the steps the modality forces:
``build_inputs`` processor-encodes each ``(prompt, image)`` pair (the
chat-templated TEXT with a single placeholder + base64 ``image_data`` go to SRT,
which re-expands it server-side; the processor's EXPANDED ids become the replay
prompt), and ``build_conditions`` adds the per-sample ``pixel_values`` /
``image_grid_thw`` so the replay teacher-forces over the IDENTICAL multimodal
input — the importance ratio stays consistent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.config.require import require
from unirl.rollout.engine.sglang.adapters.base import (
    MMEncoding,
    PreparedInputs,
    register_adapter,
)
from unirl.rollout.engine.sglang.adapters.text import TextLMAdapter
from unirl.rollout.engine.sglang.backends import RawResult
from unirl.rollout.engine.sglang.utils import ResolvedSampling, pil_to_base64
from unirl.types.primitives import Images
from unirl.types.rollout_req import RolloutReq


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

    # ------------------------------------------------------------------ #
    # build_inputs — processor path (overrides the chat-template path)
    # ------------------------------------------------------------------ #

    def build_inputs(self, req: RolloutReq, *, sampling: ResolvedSampling) -> PreparedInputs:
        prompts = self.extract_prompts(req)
        pil_images = self.extract_images(req, n_prompts=len(prompts))

        wire: List[Dict[str, Any]] = []
        prompt_token_ids: List[List[int]] = []
        mm_encs: List[MMEncoding] = []
        for prompt, image in zip(prompts, pil_images):
            mm = self.encode_mm(prompt, image, sampling.system_instruction)
            mm_encs.append(mm)
            payload = self.base_payload(sampling)
            # Send the chat-templated TEXT (single placeholder) + image_data so
            # SRT's processor expands the placeholder and the model actually
            # attends the image. (Sending the pre-expanded input_ids +
            # image_data makes SRT return HTTP 500.)
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

    def extract_images(self, req: RolloutReq, *, n_prompts: int) -> List[Any]:
        image_prim = req.primitives.get("image")
        require(
            image_prim is not None and isinstance(image_prim, Images),
            f"{type(self).__name__} requires req.primitives['image']: Images",
        )
        require(
            len(image_prim) == n_prompts,
            f"{type(self).__name__}: image batch {len(image_prim)} != prompt count {n_prompts}",
        )
        return image_prim.to_pils()

    def encode_mm(
        self,
        user_prompt: str,
        image: Any,
        system_instruction: Optional[str] = None,
    ) -> MMEncoding:
        """Processor-encode one (prompt, image) into the model's native layout.

        Returns a fully-populated :class:`MMEncoding`: ``input_ids`` already has
        the image placeholder expanded to the per-image vision-token count. This
        is the SAME encoding the trainside replay (``chat_template.embed``) uses,
        so the rollout (sent to SRT as ``text`` + ``image_data``) and the replay
        (teacher-forced over ``input_ids`` + ``pixel_values``) are token-for-token
        identical.
        """
        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]})
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self._processor(text=[text], images=[image], return_tensors="pt")
        return MMEncoding(
            image=image,
            text=text,
            input_ids=enc["input_ids"][0].tolist(),
            pixel_values=enc["pixel_values"],
            image_grid_thw=enc["image_grid_thw"],
        )

    # ------------------------------------------------------------------ #
    # build_conditions — prompt condition + the multimodal replay conditions
    # ------------------------------------------------------------------ #

    def build_conditions(self, req: RolloutReq, prepared: PreparedInputs, raw: List[RawResult]) -> Dict[str, Any]:
        """Add per-sample ``pixel_values`` / ``image_grid_thw`` to the base.

        Replicated from the prompt-level processor encoding so each sibling
        sample carries the image condition its rollout was generated under
        (per-sample lists with FieldKind.CONCAT semantics — they survive the
        DP split/merge and reach the replay aligned with ``prompt``).
        """
        conditions = super().build_conditions(req, prepared, raw)
        if prepared.mm:
            _, prompt_index = self.replicate_per_sample(prepared)
            per_sample_pixel_values = [prepared.mm[i].pixel_values for i in prompt_index]
            per_sample_image_grid_thw = [prepared.mm[i].image_grid_thw for i in prompt_index]
            if any(p is not None for p in per_sample_pixel_values):
                conditions["pixel_values"] = per_sample_pixel_values
                conditions["image_grid_thw"] = per_sample_image_grid_thw
        return conditions


__all__ = ["VLMAdapter"]
