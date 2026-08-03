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
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams

from ..conditions import HunyuanImage3DiffusionConditions
from ..seed import make_cpu_generators

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def _prepare_seeded_sampling(
    sample: Sample,
    frontier: Part,
    params: DiffusionSamplingParams,
) -> Tuple[DiffusionSamplingParams, Optional[List[torch.Generator]], Optional[List[str]]]:
    """Prepare request-local RNG streams without touching global RNG state."""
    recipe_ids = [str(key) for key in NoiseRecipe.from_sample(sample).noise_group_ids]
    if params.seed is None or not recipe_ids:
        return params, None, None
    if len(recipe_ids) != frontier.batch_size:
        raise ValueError(
            "HunyuanImage3 it2i seeded sampling requires noise keys aligned with "
            f"the frontier batch; got {len(recipe_ids)} for {frontier.batch_size}."
        )

    sample_ids = [str(sample_id) for sample_id in frontier.sample_ids]
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
    """Encode every source image with the matching per-sample VAE RNG."""
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


def generate(pipeline: "HunyuanImage3Pipeline", sample: Sample) -> Sample:
    """it2i — image edit. Single diffusion stage with cond-image scatter."""
    frontier = sample.frontier_gen_part(DiffusionSamplingParams)
    params = frontier.sampling_params
    params, condition_vae_generators, sde_sample_keys = _prepare_seeded_sampling(sample, frontier, params)
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
        int(params.samples_per_prompt) <= 2,
        f"HunyuanImage3 it2i: samples_per_prompt={params.samples_per_prompt} is not supported yet; "
        "per-sample cond_vit lists are not transport-safe above 2.",
    )
    schedule = params.sigmas.to(pipeline.bundle.device)
    cfg = float(params.guidance_scale) > 1.0

    vit = pipeline.vit_encode.encode_for_cond_vit(images)

    cond_vae_images, cond_timestep, cond_vit_images = _encode_cond_images_per_sample(
        pipeline.bundle.transformer,
        vit["joint_image_info"],
        condition_vae_generators,
    )
    vit_kwargs = vit["vit_kwargs"]  # CFG duplication is deferred to the stage.

    bot_task = str((sample.parts[0].control or {}).get("bot_task", "image"))
    mm = pipeline.text_embed.embed_for_gen_image(
        texts,
        cfg=cfg,
        height=int(params.height),
        width=int(params.width),
        bot_task=bot_task,
        batch_cond_image_info=vit["joint_image_info"],
    )

    # Store cond and uncond branches separately so transport preserves both.
    fused_full = mm["fused"]
    expected_fused_rows = frontier.batch_size * (2 if cfg else 1)
    actual_fused_rows = int(fused_full.input_ids.shape[0])
    if actual_fused_rows != expected_fused_rows:
        raise ValueError(
            "HunyuanImage3 it2i tokenizer batch mismatch: "
            f"expected {expected_fused_rows} fused rows for B={frontier.batch_size}, "
            f"got {actual_fused_rows}."
        )
    if cfg:
        n = frontier.batch_size
        fused_cond = fused_full.slice(0, n)
        fused_uncond = fused_full.slice(n, 2 * n)
    else:
        fused_cond = fused_full
        fused_uncond = None

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

    filled = frontier.fill(segment=latent_seg, primitives={"image": edited}, conditions=diff_conds.to_dict())
    return sample.with_parts([*sample.parts[:-1], filled])
