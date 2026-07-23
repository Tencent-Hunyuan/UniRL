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

from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from unirl.config.require import require
from unirl.types.conditions import ImageEmbedCondition, ImageLatentCondition
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams

from ..conditions import HunyuanImage3DiffusionConditions
from ..seed import make_cpu_generators

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def _prepare_seeded_sampling(
    req: RolloutReq,
) -> Tuple[DiffusionSamplingParams, Optional[List[torch.Generator]], Optional[List[str]]]:
    """Prepare request-local RNG streams without touching process-global RNG.

    UniRL's shared trainside contract makes only the initial ``x_T``
    driver-authoritative. HI3 additionally needs deterministic source-image VAE
    posterior samples and per-step SDE noise to satisfy strict same-seed
    end-to-end image reproducibility.
    """
    params: DiffusionSamplingParams = req.sampling_params.get("diffusion")
    recipe_ids = [str(noise_id) for noise_id in (req.init_noise_group_ids or [])]
    if params.seed is None or not recipe_ids:
        return params, None, None

    sample_ids = [str(sample_id) for sample_id in (req.sample_ids or [])]
    if len(sample_ids) != len(recipe_ids):
        raise ValueError(
            "HunyuanImage3 it2i seeded sampling requires sample_ids aligned with "
            f"init_noise_group_ids; got {len(sample_ids)} and {len(recipe_ids)}."
        )

    seeded_params = replace(params, noise_group_ids=recipe_ids)
    condition_vae_generators = make_cpu_generators(
        int(params.seed),
        [f"cond-vae:{recipe_id}" for recipe_id in recipe_ids],
    )
    sde_sample_keys = [f"{recipe_id}:sample:{sample_id}" for recipe_id, sample_id in zip(recipe_ids, sample_ids)]
    return seeded_params, condition_vae_generators, sde_sample_keys


def _encode_cond_images_per_sample(
    transformer,
    batch_cond_images,
    generators: Optional[List[torch.Generator]],
):
    """Encode each outer sample with its matching VAE RNG stream."""
    if generators is None:
        return transformer._encode_cond_image(batch_cond_images, cfg_factor=1, generator=None)
    if len(generators) != len(batch_cond_images):
        raise ValueError(
            "HunyuanImage3 it2i condition-VAE generator count "
            f"{len(generators)} != condition batch {len(batch_cond_images)}."
        )

    vae_items, timestep_items, vit_items = [], [], []
    for cond_images, generator in zip(batch_cond_images, generators):
        cond_vae, cond_timestep, cond_vit = transformer._encode_cond_image(
            [cond_images],
            cfg_factor=1,
            generator=[generator],
        )
        vae_items.append(cond_vae)
        timestep_items.append(cond_timestep)
        if cond_vit is not None:
            vit_items.extend(cond_vit)

    if all(isinstance(item, torch.Tensor) for item in vae_items) and all(
        item.shape[1:] == vae_items[0].shape[1:] for item in vae_items
    ):
        cond_vae_images = torch.cat(vae_items, dim=0)
        cond_timestep = torch.cat(timestep_items, dim=0)
    else:
        cond_vae_images = vae_items
        cond_timestep = timestep_items
    return cond_vae_images, cond_timestep, vit_items or None


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

    params, condition_vae_generators, sde_sample_keys = _prepare_seeded_sampling(req)
    require(
        int(params.samples_per_prompt) <= 2,
        f"HunyuanImage3 it2i: samples_per_prompt={params.samples_per_prompt} is not supported yet — "
        "the per-sample cond_vit lists (spatial_shapes / attn_mask) trip the dp>1 track merge above 2 "
        "(see the pin in examples/unified_model/hi3_it2i.yaml; per-sample tensors are the tracked follow-up).",
    )
    if req.sigmas is None:
        raise ValueError(
            "HunyuanImage3 it2i: req.sigmas is None. Engine adapter must call "
            "unirl.sde.runtime.ensure_req_sigmas before pipeline.generate."
        )
    schedule = req.sigmas.to(pipeline.bundle.device)
    cfg = float(params.guidance_scale) > 1.0

    # 1. ViT cond features. Returns joint_image_info (forwarded to chat
    #    template), cond_vit_images, vit_kwargs.
    vit = pipeline.vit_encode.encode_for_cond_vit(images)

    # 2. VAE encode + ViT cond, built at cfg_factor=1 (ONE copy per sample). The
    #    cfg uncond branch keeps the SAME source image, so cfg doubling of these
    #    payloads is a pure block duplication (_encode_cond_image / vit_kwargs do
    #    ``.repeat`` / ``list*cfg``); the diffusion stage re-applies it when it
    #    expands CFG. Keeping them B-batched (not doubled) means they survive the
    #    B-sample track transport that a 2B batch would not.
    cond_vae_images, cond_timestep, cond_vit_images = _encode_cond_images_per_sample(
        pipeline.bundle.transformer,
        vit["joint_image_info"],
        condition_vae_generators,
    )
    vit_kwargs = vit["vit_kwargs"]  # B-batched (cfg doubling deferred to the stage)

    # 3. Build the unified-MM tensors with cond-image markers spliced in. With
    #    cfg=True the fused is the cfg-doubled [cond; uncond] N=2B batch.
    bot_task = str(req.stage_config.get("bot_task", "image"))
    mm = pipeline.text_embed.embed_for_gen_image(
        texts,
        cfg=cfg,
        height=int(params.height),
        width=int(params.width),
        bot_task=bot_task,
        batch_cond_image_info=vit["joint_image_info"],
    )

    # 4. Split the cfg-doubled fused into cond (-> fused) + uncond (-> fused_uncond),
    #    both B-batched (one row per sample). The uncond branch genuinely differs from
    #    cond (its prompt tokens are replaced by <cfg> tokens), so it must be carried
    #    explicitly; storing it as its own B-aligned field lets it survive the
    #    B-sample track transport, and the stage re-stacks [cond; uncond] for a GUIDED
    #    replay (ratio=1 at cfg>1). cfg=False -> single branch, fused_uncond=None.
    fused_full = mm["fused"]
    if cfg:
        n = int(fused_full.input_ids.shape[0]) // 2
        fused_cond = fused_full.slice(0, n)
        fused_uncond = fused_full.slice(n, 2 * n)
    else:
        fused_cond = fused_full
        fused_uncond = None

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
        fused=fused_cond,
        fused_uncond=fused_uncond,
        cond_vae=cond_vae,
        cond_vit=cond_vit,
        cond_timestep=cond_timestep,
        tokenizer_output=mm["tokenizer_output"],
    )

    latent_seg = pipeline.diffusion.diffuse(
        diff_conds,
        schedule=schedule,
        params=params,
        sde_sample_keys=sde_sample_keys,
    )
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
