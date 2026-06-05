"""HI3 standalone DiT — the two-engine trainer's image half.

Eats the recaption that the ``ar_recaption`` engine produced (injected by the
trainer as ``req.primitives['cot_text']``) and renders it on a DiT-only HI3
stage. Inherits the single-"image"-track response side from
:class:`DiTImageAdapter`; conditions come from the fused multimodal capture.
"""

from __future__ import annotations

from typing import Any, Dict, List

from unirl.rollout.engine.vllm_omni_v2.adapters.base import register_adapter
from unirl.rollout.engine.vllm_omni_v2.adapters.dit_image import DiTImageAdapter
from unirl.rollout.engine.vllm_omni_v2.backends import (
    STAGE_KIND_DIFFUSION,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.rollout.engine.vllm_omni_v2.utils import (
    build_fused_mm_condition,
    seed_from_sample_id,
    texts_from_req,
)
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import get_diffusion_params


@register_adapter("dit_recaption")
class DitRecaptionAdapter(DiTImageAdapter):
    """Standalone HI3 DiT — eats an externally-injected recaption.

    The two-engine trainer puts the AR-generated recaption per sample on
    ``req.primitives['cot_text']`` (aligned 1:1 with ``primitives['text']``).
    Each per-prompt dict carries ``extra['ar_generated_text']`` — exactly the
    key the upstream DiT ``forward`` reads as ``cot_text`` — plus
    ``use_system_prompt`` so the DiT rebuilds the same system prefix the AR
    used.

    **One call per prompt, seeded here.** Per-image distinct seeds cannot
    travel through the sampling params: vllm-omni requires one params object
    per STAGE (not per prompt) and shares it across all prompts of a
    ``generate()`` call — ``OmniDiffusionRequest.__post_init__`` assigns a
    random seed only on the FIRST request and the mutated object poisons the
    rest (byte-identical images → diffusion advantage 0). So ``build_inputs``
    emits one single-prompt :class:`GenerateCall` per sample with its own
    ``seed_from_sample_id`` seed and its own x_T recipe gid slice (a shared
    full-batch gid list would make the worker's ``NoiseRecipe.for_batch(1)``
    hand gids[0] to EVERY image).
    """

    stage_yaml = "hunyuan_image3_dit_recaption_rl.yaml"
    # v1 loads a driver tokenizer for dit_recaption even though this builder
    # never tokenizes — kept for parity (health semantics, warm cache).
    clear_cuda_visible = True
    #: HI3 two-engine stages are TP>1 — wake-time LoRA re-push must use the
    #: byte-copy transport.
    lora_copy_transport = True

    #: System-prompt preset for ``use_system_prompt`` — the only piece of the
    #: HI3 chat-template row this DiT-only stage consumes (no task: the
    #: recaption text is injected directly via ``extra['ar_generated_text']``).
    sys_type = "en_unified"

    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        if req.primitives.get("image") is not None:
            raise ValueError("modality='dit_recaption' does not accept req.primitives['image']")

        texts = texts_from_req(req)
        cot = req.primitives.get("cot_text")
        if not isinstance(cot, Texts):
            raise TypeError(
                "modality='dit_recaption' requires req.primitives['cot_text'] (Texts of recaptions); "
                f"got {type(cot).__name__ if cot is not None else 'None'}."
            )
        if len(cot.texts) != len(texts.texts):
            raise ValueError(f"dit_recaption: cot_text count {len(cot.texts)} != prompt count {len(texts.texts)}.")

        sys_type = (req.stage_config or {}).get("sys_type") or self.sys_type
        diff_params = get_diffusion_params(req.sampling_params)

        base_kwargs = self.core_diff_kwargs(req, diff_params)
        height = int(base_kwargs["height"])
        width = int(base_kwargs["width"])

        # Base extra_args mirror the v1 builder: sparse SDE indices + the
        # WHOLE batch's x_T recipe gids (+ the regen base seed — distinct from
        # the per-image SAMPLING seed below; per-image x_T variety comes from
        # the gid, not this seed). NO init_noise_latent_shape — HI3's DiT
        # latent shape is AR-dynamic and resolved in the worker.
        base_extra = self.sde_extra_args(diff_params)
        recipe_gids = list(req.init_noise_group_ids or [])
        if recipe_gids:
            base_extra["init_noise_group_ids"] = [str(g) for g in recipe_gids]
            base_extra["init_noise_seed"] = (
                int(diff_params.seed) if getattr(diff_params, "seed", None) is not None else 0
            )

        calls: List[GenerateCall] = []
        for idx, (sample_id, text, recap) in enumerate(zip(req.sample_ids, texts.texts, cot.texts)):
            prompt = {
                "prompt": text,
                "height": height,
                "width": width,
                "use_system_prompt": sys_type,
                "extra": {"ar_generated_text": recap},
            }
            kwargs = dict(base_kwargs)
            kwargs["seed"] = seed_from_sample_id(sample_id)
            extra_args = dict(base_extra)
            # Each single-prompt generate runs with batch_size=1 in the
            # worker, so ship ONLY this sample's x_T recipe gid.
            gid = recipe_gids[idx] if idx < len(recipe_gids) else None
            if gid is not None and extra_args.get("init_noise_group_ids"):
                extra_args["init_noise_group_ids"] = [str(gid)]
            if extra_args:
                kwargs["extra_args"] = extra_args
            calls.append(
                GenerateCall(
                    prompts=[prompt],
                    sampling=[StageSampling(kind=STAGE_KIND_DIFFUSION, kwargs=kwargs)],
                    # Single-prompt call: its flat output list IS the group.
                    group_by_request_id=False,
                )
            )
        return calls

    def build_dit_condition(self, diff_outputs: List[OmniRawResult]) -> Dict[str, Any]:
        fused = build_fused_mm_condition(diff_outputs)
        if fused is None:
            raise RuntimeError(
                "build_response: HI3 rollout (modality='dit_recaption') "
                "returned no 'fused_mm_capture' on DiffusionOutput.custom_output. "
                "Check that RLHunyuanImage3Pipeline.prepare_inputs_for_generation "
                "hook ran in every DiT worker — the subclass swap may not have "
                "taken effect (verify custom_pipeline_args.pipeline_class in "
                "the stage YAML)."
            )
        return {"fused": fused}


__all__ = ["DitRecaptionAdapter"]
