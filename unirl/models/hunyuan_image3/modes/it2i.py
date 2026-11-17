"""it2i — image-edit (text + cond image conditioning, image output)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from unirl.config.require import require
from unirl.types.conditions import ImageEmbedCondition
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from ..conditions import HunyuanImage3DiffusionConditions, HunyuanImage3VAECondition, split_cfg_fused
from ..seed import make_cpu_generators, prepare_seeded_sampling
from .t2t import _as_sample_batches

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def _encode_cond_images(
    transformer,
    batch_cond_images,
    generators: Optional[List[torch.Generator]],
):
    """Encode every source image with the matching per-sample VAE RNG."""
    if generators is None:
        cond_vae, cond_timestep, cond_vit = transformer._encode_cond_image(
            batch_cond_images, cfg_factor=1, generator=None
        )
        return (
            _as_sample_batches(cond_vae),
            _as_sample_batches(cond_timestep),
            None if cond_vit is None else _as_sample_batches(cond_vit),
        )
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
        vae_items.extend(_as_sample_batches(cond_vae))
        timestep_items.extend(_as_sample_batches(cond_timestep))
        if cond_vit is not None:
            vit_items.extend(_as_sample_batches(cond_vit))

    return vae_items, timestep_items, vit_items or None


def generate(pipeline: "HunyuanImage3Pipeline", sample: Sample) -> Sample:
    """it2i — image edit. Single diffusion stage with cond-image scatter."""
    frontier = sample.frontier_gen_part(DiffusionSamplingParams)
    params = frontier.sampling_params
    params, sde_sample_keys = prepare_seeded_sampling(sample, frontier, params)
    condition_vae_generators = (
        make_cpu_generators(params.seed, [f"cond-vae:{recipe_id}" for recipe_id in params.noise_group_ids])
        if sde_sample_keys is not None
        else None
    )
    if params.sigmas is None:
        raise ValueError(
            "HunyuanImage3 it2i: gen part sampling_params.sigmas is None. The hosting engine must "
            "pin σ before pipeline.generate."
        )

    conditioning = sample.conditioning()
    texts = conditioning[0] if conditioning else None
    require(
        isinstance(texts, Texts),
        f"HunyuanImage3Pipeline.generate (it2i): prompt from sample.conditioning()[0] "
        f"must be Texts, "
        f"got {type(texts).__name__ if texts is not None else 'None'}",
    )
    images = next((c for c in conditioning[1:] if isinstance(c, Images)), None)
    require(
        isinstance(images, Images),
        "HunyuanImage3Pipeline.generate (it2i): expected a chained Images input in sample.conditioning(), found none",
    )

    require(
        params.samples_per_prompt <= 2,
        f"HunyuanImage3 it2i: samples_per_prompt={params.samples_per_prompt} is not supported yet; "
        "per-sample cond_vit lists are not transport-safe above 2.",
    )
    schedule = params.sigmas.to(pipeline.bundle.device)
    cfg = params.guidance_scale > 1.0

    vit = pipeline.vit_encode.encode_for_cond_vit(images)

    cond_vae_images, cond_timestep, cond_vit_images = _encode_cond_images(
        pipeline.bundle.transformer,
        vit["joint_image_info"],
        condition_vae_generators,
    )
    vit_kwargs = vit["vit_kwargs"]  # CFG duplication is deferred to the stage.

    mm = pipeline.text_embed.embed_for_gen_image(
        texts,
        cfg=cfg,
        height=params.height,
        width=params.width,
        bot_task=sample.parts[0].control.get("bot_task", "image"),
        batch_cond_image_info=vit["joint_image_info"],
    )

    fused_cond, fused_uncond = split_cfg_fused(mm["fused"], frontier.batch_size, cfg=cfg)

    cond_vae = HunyuanImage3VAECondition(latents=cond_vae_images)
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

    filled = frontier.fill(segment=latent_seg, primitives={"image": edited}, conditions=diff_conds.to_dict())
    return sample.with_parts([*sample.parts[:-1], filled])
