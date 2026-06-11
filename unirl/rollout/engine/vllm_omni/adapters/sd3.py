"""SD3.5 family: output sub-adapter + the ``sd35_t2i`` modality class.

Single diffusion stage, TP=1. The request side is the shared
:class:`~.dit.DitInputAdapter` skeleton used directly (prompt dicts are the
``{"prompt", "negative_prompt"}`` shape ``StableDiffusion3Pipeline.forward``
accepts); the response side derives from :class:`~.dit.DitOutputAdapter`
with conditions from the ``encode_prompt`` text capture.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult
from unirl.rollout.engine.vllm_omni.utils import collect_dit_outputs
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


class Sd3OutputAdapter(DitOutputAdapter):
    """Single-"image"-track response with the SD3 text-capture conditions."""

    def build_conditions(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Concat the per-request SD3 ``text_capture`` dicts into one condition.

        Written by ``RLStableDiffusion3Pipeline`` after intercepting
        ``encode_prompt``. All per-request encodes share the same ``L`` (T5
        padding to ``max_sequence_length`` is fixed), so a plain dim-0 concat
        suffices.
        """
        del req
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )

        captures = [(getattr(d, "custom_output", None) or {}).get("text_capture") for d in diff_outputs]
        if any(c is None for c in captures):
            raise RuntimeError(
                "build_response: SD3 rollout returned no 'text_capture' on "
                "DiffusionOutput.custom_output. Check that "
                "RLStableDiffusion3Pipeline._install_encode_prompt_hook ran "
                "in every DiT worker — the subclass swap may not have taken "
                "effect (verify custom_pipeline_args.pipeline_class in the "
                "stage YAML)."
            )

        text_cond = TextEmbedCondition(
            embeds=torch.cat([c["prompt_embeds"] for c in captures], dim=0),
            pooled=torch.cat([c["pooled_prompt_embeds"] for c in captures], dim=0),
            attn_mask=None,  # SD3 uses fixed-length T5 padding; no attn mask needed
        )
        return {"text": text_cond}


@register_adapter("sd3_t2i")
class Sd3T2iAdapter(ModelAdapter):
    """SD3.5-medium text → image (single diffusion stage, TP=1)."""

    stage_yaml = "sd35_t2i_rl.yaml"
    omni_mode = "text-to-image"
    # SD3.5 has no top-level tokenizer (only subfolder CLIP/T5 ones) and the
    # single-stage path never calls build_prompt_tokens.
    needs_driver_tokenizer = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = DitInputAdapter(self.modality)
        self.output_adapter = Sd3OutputAdapter(self.modality)

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


__all__ = ["Sd3OutputAdapter", "Sd3T2iAdapter"]
