"""i2t — image-to-text autoregressive generation.

Reads ``primitives["text"]: Texts`` (the prompt) and
``primitives["image"]: Images`` (the image to caption / answer about),
plus ``stage_params["ar"]: dict`` (optional). Builds chat-templated
``input_ids`` with embedded ``<img>`` markers via the chat-template
wrapper, then runs ``HunyuanImage3ARStage.autoregress`` against the
backbone in ``mode="gen_text"`` -- the unified MM forward scatters
ViT patch embeddings into the prompt's ``<img>`` slots via
``instantiate_vit_image_tokens``.

Conditions on the response carry the chat-templated ``input_ids`` plus
the ``cond_vit_*`` / ``vit_kwargs`` tensors that drove the ViT-tokens
scatter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import torch

from unirl.models.types.ar import ARSamplingParams
from unirl.types.conditions import ImageEmbedCondition, ImageLatentCondition
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import get_ar_params

from ..ar import HunyuanImage3ARParams
from ..conditions import HunyuanImage3ARConditions
from .t2t import _resolve_system_prompt, _stop_tokens_for_bot_task, _tokenizer_bot_task

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def generate(pipeline: "HunyuanImage3Pipeline", req: RolloutReq) -> RolloutResp:
    """i2t — AR-stage rollout with image comprehension."""
    texts = req.primitives.get("text")
    if not isinstance(texts, Texts):
        raise TypeError(
            f"HunyuanImage3Pipeline.generate (i2t): "
            f"req.primitives['text'] must be Texts, "
            f"got {type(texts).__name__ if texts is not None else 'None'}"
        )
    images = req.primitives.get("image")
    if not isinstance(images, Images):
        raise TypeError(
            f"HunyuanImage3Pipeline.generate (i2t): "
            f"req.primitives['image'] must be Images, "
            f"got {type(images).__name__ if images is not None else 'None'}"
        )

    # Build HunyuanImage3ARParams from typed sampling params + model-specific stage_config.
    ar = get_ar_params(req.sampling_params)
    model_cfg: Dict[str, Any] = dict(req.stage_config.get("ar") or {})
    ar_params = HunyuanImage3ARParams(
        max_tokens=ar.max_new_tokens if ar is not None else model_cfg.get("max_tokens", 2048),
        temperature=ar.temperature if ar is not None else model_cfg.get("temperature", 0.6),
        top_p=ar.top_p if ar is not None else model_cfg.get("top_p", 0.95),
        top_k=ar.top_k if ar is not None else model_cfg.get("top_k", 1024),
        bot_task=model_cfg.get("bot_task", "auto"),
        cot_text=model_cfg.get("cot_text"),
        system_prompt=model_cfg.get("system_prompt"),
        use_system_prompt=model_cfg.get("use_system_prompt"),
        stop_token_ids=model_cfg.get("stop_token_ids", []),
        taylor_cache_interval=model_cfg.get("taylor_cache_interval"),
        taylor_cache_order=model_cfg.get("taylor_cache_order"),
    )
    bot_task = str(ar_params.bot_task)
    tok_bot_task = _tokenizer_bot_task(bot_task)

    system_prompt = _resolve_system_prompt(
        pipeline.bundle, tok_bot_task, ar_params.use_system_prompt, ar_params.system_prompt
    )
    system_prompt_list = [system_prompt] * len(texts.texts) if system_prompt is not None else None

    # vit: {"joint_image_info": [[JointImageInfo]]*B, "cond_vit_images":
    #       list[Tensor [S_b, D]]*B, "vit_kwargs": {"spatial_shapes",
    #       "attention_mask"}}
    vit = pipeline.vit_encode.encode_for_cond_vit(images)

    # HI3-Instruct represents a cond image as a DUAL VAE + ViT block (the
    # chat template splices VAE <img> slots + a cond <timestep> alongside the
    # ViT <img> slots). ``_encode_cond_image`` VAE-encodes the cond image and
    # returns the per-sample VAE latents + timestep + the (re-shaped) ViT
    # features the unified-MM forward scatters — same call it2i uses.
    # ``cfg_factor=1``: the AR/comprehension forward has no CFG batching.
    # Without the VAE half, the 4096 VAE <img> slots stay bare <img>
    # embeddings → the model sees garbage and can't comprehend the image.
    cond_vae_images, cond_timestep, cond_vit_images = pipeline.bundle.transformer._encode_cond_image(
        vit["joint_image_info"], cfg_factor=1
    )

    # ``cond_vae_images`` is the raw VAE-input image (float). The AR forward runs
    # the bf16 VAE encoder WITHOUT autocast (the diffusion path wraps its forward
    # in torch.autocast; the autoregress loop does not), so a float input hits
    # bf16 conv weights → dtype mismatch. Cast float tensors to the model dtype.
    def _cast_floats(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=pipeline.bundle.dtype) if x.is_floating_point() else x
        if isinstance(x, (list, tuple)):
            return type(x)(_cast_floats(e) for e in x)
        return x

    cond_vae_images = _cast_floats(cond_vae_images)

    # Chat template path: pass batch_cond_image_info so the wrapper splices in
    # the cond-image markers; the resulting cond_vae_image_mask /
    # cond_vit_image_mask / cond_timestep_scatter_index (now on ``fused``) pin
    # which ``input_ids`` positions hold the VAE / ViT / timestep scatter targets.
    mm = pipeline.text_embed.embed_for_ar(
        texts,
        bot_task=tok_bot_task,
        system_prompt=system_prompt_list,
        cot_text=([ar_params.cot_text] * len(texts.texts) if ar_params.cot_text else None),
        batch_cond_image_info=vit["joint_image_info"],
    )

    cond_vae = ImageLatentCondition(latents=cond_vae_images)
    cond_vit = ImageEmbedCondition(
        embeds=cond_vit_images,
        attn_mask=vit["vit_kwargs"]["attention_mask"],
        spatial_shapes=vit["vit_kwargs"]["spatial_shapes"],
    )
    ar_conds = HunyuanImage3ARConditions(
        fused=mm["fused"],
        cond_vae=cond_vae,
        cond_vit=cond_vit,
        cond_timestep=cond_timestep,
        tokenizer_output=mm["tokenizer_output"],
    )

    stop_ids: List[int] = list(ar_params.stop_token_ids or [])
    if not stop_ids:
        stop_ids = _stop_tokens_for_bot_task(pipeline.bundle, bot_task)
    sampling_params = ARSamplingParams(
        max_new_tokens=int(ar_params.max_tokens),
        temperature=float(ar_params.temperature),
        top_p=float(ar_params.top_p),
        top_k=int(ar_params.top_k),
        stop_token_id=stop_ids[0] if stop_ids else None,
    )
    ar_params_with_stops = HunyuanImage3ARParams(
        bot_task=ar_params.bot_task,
        max_tokens=ar_params.max_tokens,
        temperature=ar_params.temperature,
        top_p=ar_params.top_p,
        top_k=ar_params.top_k,
        stop_token_ids=stop_ids,
        cot_text=ar_params.cot_text,
        system_prompt=ar_params.system_prompt,
        use_system_prompt=ar_params.use_system_prompt,
        taylor_cache_interval=ar_params.taylor_cache_interval,
        taylor_cache_order=ar_params.taylor_cache_order,
    )

    text_seg = pipeline.ar.autoregress(ar_conds, sampling_params=sampling_params, params=ar_params_with_stops)

    decoded_texts = pipeline._detokenize_text_segment(text_seg)

    return RolloutResp(
        tracks={
            "ar": RolloutTrack(
                sample_ids=list(req.sample_ids),
                parent_ids=list(req.group_ids),
                conditions=ar_conds.to_dict(),
                segment=text_seg,
                decoded=decoded_texts,
            ),
        }
    )
