"""it2i — image-edit (text + cond image conditioning, image output).

Reads ``primitives["text"]: Texts`` + ``primitives["image"]: Images``
(the source image to edit) and ``stage_params["diffusion"]: dict``.
Encodes the source image via the upstream
``HunyuanImage3VitEncodeStage.encode_for_cond_vit`` (image_processor)
and the model's own ``_encode_cond_image`` for VAE latents, builds the
chat-templated unified-MM tensors with cond-image markers, then runs
the diffusion stage and VAE-decodes the output.

The unified-MM forward consumes the cond_* tensors on the first
diffusion step to scatter VAE latents and ViT features at their pinned
slots in ``inputs_embeds`` (mirroring upstream
``HunyuanImage3ForCausalMM.forward(mode="gen_image")`` at
``hunyuan.py:1991-2017``); subsequent steps reuse the cached K/V at
those slots via the ``HunyuanStaticCache``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unirl.config.require import require
from unirl.types.conditions import ImageEmbedCondition, ImageLatentCondition
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams, get_diffusion_params

from ..conditions import HunyuanImage3DiffusionConditions

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def generate(pipeline: "HunyuanImage3Pipeline", req: RolloutReq) -> RolloutResp:
    """it2i — image edit. Single diffusion stage with cond-image scatter."""
    texts = req.primitives.get("text")
    require(
        isinstance(texts, Texts),
        f"HunyuanImage3Pipeline.generate (it2i): req.primitives['text'] "
        f"must be Texts, "
        f"got {type(texts).__name__ if texts is not None else 'None'}",
    )
    images = req.primitives.get("image")
    require(
        isinstance(images, Images),
        f"HunyuanImage3Pipeline.generate (it2i): req.primitives['image'] "
        f"must be Images, "
        f"got {type(images).__name__ if images is not None else 'None'}",
    )
    require(
        req.primitives.get("negative_text") is None,
        "HunyuanImage3Pipeline.generate (it2i): negative_text is not supported — "
        "the HI3 tokenizer never consumes negative-prompt text; CFG is derived from "
        "guidance_scale > 1.0 (the unconditional branch is built internally from <cfg> tokens).",
    )

    params: DiffusionSamplingParams = get_diffusion_params(req.sampling_params)
    if req.sigmas is None:
        raise ValueError(
            "HunyuanImage3 it2i: req.sigmas is None. Engine adapter must call "
            "unirl.sde.runtime.ensure_req_sigmas before pipeline.generate."
        )
    schedule = req.sigmas.to(pipeline.bundle.device)
    # Single CFG derivation feeding the chat template, ``_encode_cond_image``,
    # and the vit_kwargs duplication below — they must agree on the batch axis.
    cfg = float(params.guidance_scale) > 1.0
    cfg_factor = 2 if cfg else 1

    # 1. ViT cond features. Returns joint_image_info (forwarded to chat
    #    template), cond_vit_images, vit_kwargs.
    vit = pipeline.vit_encode.encode_for_cond_vit(images)

    # 2. VAE encode + ViT-cond duplication for CFG, all via the upstream
    #    ``_encode_cond_image`` so per-sample list shapes match what the
    #    unified-MM forward iterates with at hunyuan.py:1903.
    cond_vae_images, cond_timestep, cond_vit_images = pipeline.bundle.transformer._encode_cond_image(
        vit["joint_image_info"], cfg_factor=cfg_factor
    )

    # 3. vit_kwargs duplicated for CFG -- mirror upstream pipeline
    #    (hunyuan.py:2298-2299).
    vit_kwargs = vit["vit_kwargs"]
    if cfg_factor > 1:
        vit_kwargs = {
            "spatial_shapes": vit_kwargs["spatial_shapes"] * cfg_factor,
            "attention_mask": vit_kwargs["attention_mask"] * cfg_factor,
        }

    # 4. Build the unified-MM tensors with cond-image markers spliced in.
    bot_task = str(req.stage_config.get("bot_task", "image"))
    mm = pipeline.text_embed.embed_for_gen_image(
        texts,
        cfg=cfg,
        height=int(params.height),
        width=int(params.width),
        bot_task=bot_task,
        batch_cond_image_info=vit["joint_image_info"],
    )

    # 5. Pack into the typed conditions container. The chat-template
    #    path drives the fused sequence via input_ids; cond-image data
    #    flows through the typed ImageLatentCondition / ImageEmbedCondition
    #    primitives.
    cond_vae = ImageLatentCondition(latents=cond_vae_images)
    cond_vit = ImageEmbedCondition(
        embeds=cond_vit_images,
        attn_mask=vit_kwargs["attention_mask"],
        spatial_shapes=vit_kwargs["spatial_shapes"],
    )
    diff_conds = HunyuanImage3DiffusionConditions(
        fused=mm["fused"],
        cond_vae=cond_vae,
        cond_vit=cond_vit,
        cond_timestep=cond_timestep,
        tokenizer_output=mm["tokenizer_output"],
    )

    latent_seg = pipeline.diffusion.diffuse(diff_conds, schedule=schedule, params=params)
    edited = pipeline.vae_decode.decode(latent_seg)

    return RolloutResp(
        tracks={
            "image": RolloutTrack(
                sample_ids=list(req.sample_ids),
                parent_ids=list(req.group_ids),
                conditions=diff_conds.to_dict(),
                segment=latent_seg,
                decoded=edited,
            ),
        }
    )
