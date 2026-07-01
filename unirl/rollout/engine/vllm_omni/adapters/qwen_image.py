"""Qwen-Image family: input/output sub-adapters + the ``qwen_image_t2i`` modality class.

Single diffusion stage, TP=1, no AR prelude (the Qwen2.5-VL text encoder is
co-resident in the diffusion worker). Two family quirks force overrides on
both conversion sides — everything else is the shared DiT skeleton:

- **CFG semantics.** Qwen-Image's CFG knob is ``true_cfg_scale`` (two-pass,
  norm-corrected), not the embedded ``guidance_scale``, and upstream's
  ``forward`` defaults it via ``sp.true_cfg_scale or 4.0`` while
  ``_extract_prompts`` treats an EMPTY-STRING negative prompt as present
  (only an all-``None`` list disarms CFG). The shared skeleton's
  ``negative_prompt: ""`` dicts would therefore silently arm CFG@4.0. The
  input adapter maps the typed ``guidance_scale`` onto ``true_cfg_scale``
  explicitly and emits the ``negative_prompt`` key only when CFG is armed
  (> 1.0) — the trainside oracle recipes run guidance 1.0 = CFG off.
- **Variable-length text conditioning.** Qwen2.5-VL embeds are
  variable-length after the 34-token chat-template prefix strip and each
  request is encoded alone (``runtime.max_inflight: 1``), so per-request
  capture lengths differ — the output adapter ragged-pads to the batch max
  before the dim-0 concat (the attention mask keeps the padding numerically
  inert; the trainer's ``predict_noise`` consumes it as
  ``encoder_hidden_states_mask``). No pooled vector exists for Qwen-Image.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult, StageSampling
from unirl.rollout.engine.vllm_omni.utils import collect_dit_outputs, grouped_texts_from_req, texts_from_req
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp


def _ragged_pad_cat(pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> TextEmbedCondition:
    """Per-request ``(embeds [b, L_i, D], mask [b, L_i])`` pairs → one
    ``TextEmbedCondition`` right-padded to the batch-max ``L``."""
    max_len = max(int(e.shape[1]) for e, _ in pairs)
    embeds: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for e, m in pairs:
        pad = max_len - int(e.shape[1])
        if pad:
            e = torch.cat([e, e.new_zeros(e.shape[0], pad, e.shape[2])], dim=1)
            m = torch.cat([m, m.new_zeros(m.shape[0], pad)], dim=1)
        embeds.append(e)
        masks.append(m)
    return TextEmbedCondition(embeds=torch.cat(embeds, dim=0), pooled=None, attn_mask=torch.cat(masks, dim=0))


class QwenImageInputAdapter(DitInputAdapter):
    """SD3-style request side with the Qwen CFG mapping.

    Carries the model config so ``max_sequence_length`` can be pinned to the
    trainer's text-embed budget (512) when the request doesn't set one —
    upstream would otherwise default to 1024 and the conditioning would
    diverge from the trainside oracle.
    """

    def __init__(self, modality: str, *, model_config: Any = None) -> None:
        super().__init__(modality)
        self.model_config = model_config

    def build_prompts(self, req: RolloutReq) -> List[Any]:
        """``{"prompt"}`` dicts; ``negative_prompt`` ONLY when CFG is armed.

        Upstream ``_extract_prompts`` disarms CFG only when EVERY dict lacks
        the key (``""`` counts as present), so the shared skeleton's
        unconditional ``negative_prompt: ""`` cannot be reused here.
        """
        if req.primitives.get("image") is not None:
            raise ValueError(f"modality={self.modality!r} does not accept req.primitives['image']")
        texts = texts_from_req(req)
        diff_params = req.sampling_params.get("diffusion")
        if float(diff_params.guidance_scale) > 1.0:
            negative_prompt = str(getattr(diff_params, "negative_prompt", "") or "")
            return [{"prompt": text, "negative_prompt": negative_prompt} for text in texts.texts]
        return [{"prompt": text} for text in texts.texts]

    def build_sampling(self, req: RolloutReq) -> List[StageSampling]:
        sampling = super().build_sampling(req)
        diff_params = req.sampling_params.get("diffusion")
        kwargs = sampling[0].kwargs
        # Qwen's CFG knob: set it ALWAYS so upstream's ``or 4.0`` default
        # never fires (at <= 1.0 ``do_true_cfg`` stays False regardless of
        # the prompt-side gate — belt and suspenders).
        kwargs["true_cfg_scale"] = float(diff_params.guidance_scale)
        if "max_sequence_length" not in kwargs:
            max_seq_len = getattr(self.model_config, "max_sequence_length", None)
            if max_seq_len is not None:
                kwargs["max_sequence_length"] = int(max_seq_len)
        return sampling


class QwenImageGroupedInputAdapter(QwenImageInputAdapter):
    """Qwen-Image request builder using vLLM-Omni's native multi-output prompt shape.

    Unlike SD3 where the conditioning tap fires BEFORE upstream's internal
    embed repeat, Qwen-Image's ``encode_prompt`` accepts ``num_images_per_prompt``
    and repeats embeddings internally before returning them. The tap therefore
    captures ALREADY-repeated embeddings — the output adapter needs no
    ``repeat_interleave``.
    """

    def _spp(self, req: RolloutReq) -> int:
        diff_params = req.sampling_params.get("diffusion")
        return int(getattr(diff_params, "samples_per_prompt", 1) or 1)

    def build_prompts(self, req: RolloutReq) -> List[Any]:
        spp = self._spp(req)
        grouped_texts, _ = grouped_texts_from_req(
            req,
            samples_per_prompt=spp,
            caller=f"{self.modality}.build_prompts",
        )
        diff_params = req.sampling_params.get("diffusion")
        if float(diff_params.guidance_scale) > 1.0:
            negative_prompt = str(getattr(diff_params, "negative_prompt", "") or "")
            return [{"prompt": text, "negative_prompt": negative_prompt} for text in grouped_texts]
        return [{"prompt": text} for text in grouped_texts]

    def build_sampling(self, req: RolloutReq) -> List[StageSampling]:
        spp = self._spp(req)
        grouped_texts_from_req(
            req,
            samples_per_prompt=spp,
            caller=f"{self.modality}.build_sampling",
        )
        sampling = super().build_sampling(req)
        sampling[0].kwargs["num_outputs_per_prompt"] = spp
        return sampling


class QwenImageOutputAdapter(DitOutputAdapter):
    """Single-"image"-track response with the Qwen text-capture conditions."""

    _MISSING_CAPTURE_MSG = (
        "build_response: Qwen-Image rollout returned no 'text_capture' on "
        "DiffusionOutput.custom_output. Check that RLQwenImagePipeline's "
        "encode_prompt tap ran in every DiT worker — the subclass swap may "
        "not have taken effect (verify custom_pipeline_args.pipeline_class "
        "in the stage YAML)."
    )

    def build_conditions(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Ragged-pad-concat the per-request Qwen ``text_capture`` dicts.

        Written by ``RLQwenImagePipeline`` after intercepting
        ``encode_prompt``. Keys align with ``QwenImageConditions``:
        ``text`` always; ``negative_text`` only when the negative encode
        fired (CFG armed) — and then it must have fired for every request
        of the call (sampling params are uniform across a generate call).
        """
        del req
        diff_outputs, _, _ = collect_dit_outputs(
            per_request, final_output_type=self.final_output_type, stage_id=self.stage_id, modality=self.modality
        )

        captures = [(getattr(d, "custom_output", None) or {}).get("text_capture") for d in diff_outputs]
        if any(c is None for c in captures):
            raise RuntimeError(self._MISSING_CAPTURE_MSG)

        cond_dict: Dict[str, Any] = {
            "text": _ragged_pad_cat([(c["prompt_embeds"], c["prompt_embeds_mask"]) for c in captures])
        }
        neg_present = [c.get("negative_prompt_embeds") is not None for c in captures]
        if any(neg_present):
            if not all(neg_present):
                raise RuntimeError(
                    "build_response: Qwen-Image negative text captured on some "
                    "requests but not others — CFG arming must be uniform "
                    "across a generate call."
                )
            cond_dict["negative_text"] = _ragged_pad_cat(
                [(c["negative_prompt_embeds"], c["negative_prompt_embeds_mask"]) for c in captures]
            )
        return cond_dict


@register_adapter("qwen_image_t2i")
class QwenImageT2iAdapter(ModelAdapter):
    """Qwen-Image text → image (single diffusion stage, TP=1)."""

    stage_yaml = "qwen_image_t2i_rl.yaml"
    omni_mode = "text-to-image"
    # The Qwen2.5-VL tokenizer lives in the tokenizer/ subfolder; the worker
    # loads it and the single-stage path never calls build_prompt_tokens.
    needs_driver_tokenizer = False

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None, tokenize_fn: Any = None) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)
        self.input_adapter = QwenImageGroupedInputAdapter(self.modality, model_config=model_config)
        self.output_adapter = QwenImageOutputAdapter(self.modality)

    def validate_request(self, req: RolloutReq) -> None:
        if req.primitives.get("image") is not None:
            raise ValueError(
                f"modality={self.modality!r} rejects image-bearing requests; use an image-conditioned modality instead."
            )

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        return self.input_adapter.build(req)

    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        return self.output_adapter.build(req, per_request)


__all__ = ["QwenImageGroupedInputAdapter", "QwenImageInputAdapter", "QwenImageOutputAdapter", "QwenImageT2iAdapter"]
