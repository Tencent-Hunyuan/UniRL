"""t2i — text-to-image diffusion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from unirl.config.require import require
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from ..conditions import HunyuanImage3DiffusionConditions, split_cfg_fused
from ..seed import prepare_seeded_sampling

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def generate(pipeline: "HunyuanImage3Pipeline", sample: Sample) -> Sample:
    """t2i — single-stage text-to-image, filling the frontier (pre-forked) gen Part."""
    frontier = sample.frontier_gen_part(DiffusionSamplingParams)
    params = frontier.sampling_params
    params, sde_sample_keys = prepare_seeded_sampling(sample, frontier, params)
    if params.sigmas is None:
        raise ValueError(
            "HunyuanImage3 t2i: gen part sampling_params.sigmas is None. The hosting engine must "
            "pin σ before pipeline.generate."
        )

    conditioning = sample.conditioning()
    texts = conditioning[0] if conditioning else None
    require(
        isinstance(texts, Texts),
        f"HunyuanImage3Pipeline.generate (t2i): prompt from sample.conditioning()[0] must be Texts, "
        f"got {type(texts).__name__ if texts is not None else 'None'}",
    )

    control = sample.parts[0].control
    cfg = params.guidance_scale > 1.0
    mm = pipeline.text_embed.embed_for_gen_image(
        texts,
        cfg=cfg,
        height=params.height,
        width=params.width,
        bot_task=control.get("bot_task", "image"),
        sys_type=control.get("sys_type"),
        sequence_template=control.get("sequence_template"),
    )

    fused, fused_uncond = split_cfg_fused(mm["fused"], frontier.batch_size, cfg=cfg)
    diff_conds = HunyuanImage3DiffusionConditions(
        fused=fused,
        fused_uncond=fused_uncond,
        tokenizer_output=mm["tokenizer_output"],
    )
    schedule = params.sigmas.to(pipeline.bundle.device)

    latent_seg = pipeline.diffusion.diffuse(
        diff_conds,
        schedule=schedule,
        params=params,
        sde_sample_keys=sde_sample_keys,
    )
    images = pipeline.vae_decode.decode(latent_seg)

    filled = frontier.fill(segment=latent_seg, primitives={"image": images}, conditions=diff_conds.to_dict())
    return sample.with_parts([*sample.parts[:-1], filled])
