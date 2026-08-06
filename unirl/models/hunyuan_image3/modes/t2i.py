"""t2i — text-to-image diffusion.

Reads ``primitives["text"]: Texts`` plus ``stage_params["diffusion"]:
dict``. Builds the unified-MM input tensors via
``HunyuanImage3TextEmbedStage.embed_for_gen_image``, runs the diffusion
stage in ``mode="gen_image"``, and decodes the final latent to pixels.

``negative_text`` is rejected: the HI3 tokenizer never consumes
negative-prompt text — CFG is derived from ``guidance_scale > 1.0`` and
the unconditional branch is built internally from ``<cfg>`` tokens.

The ``bot_task`` knob (``stage_params["bot_task"]``) is a chat-template
flag: ``"think"`` / ``"recaption"`` / ``"think_recaption"`` insert static
reasoning-mode markers, ``"image"`` inserts none. ``"image"`` alone is not
vllm-omni's ``t2i_vanilla`` preset, which also pins ``sys_type`` and
``sequence_template``; all three come from the root Part's ``control``.
None of them starts an AR pass -- t2i is a single diffusion stage and the
prefix lives in ``input_ids`` only (see vllm-omni ``prompt_utils.py:23-31``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, List, Optional, Tuple

from unirl.config.require import require
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams

from ..conditions import HunyuanImage3DiffusionConditions

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def _prepare_seeded_sampling(
    sample: Sample,
    frontier: Part,
    params: DiffusionSamplingParams,
) -> Tuple[DiffusionSamplingParams, Optional[List[str]]]:
    """Prepare request-local RNG streams without touching global RNG state."""
    recipe_ids = [str(key) for key in NoiseRecipe.from_sample(sample).noise_group_ids]
    if params.seed is None or not recipe_ids:
        return params, None
    if len(recipe_ids) != frontier.batch_size:
        raise ValueError(
            "HunyuanImage3 t2i seeded sampling requires noise keys aligned with "
            f"the frontier batch; got {len(recipe_ids)} for {frontier.batch_size}."
        )

    sample_ids = [str(sample_id) for sample_id in frontier.sample_ids]
    seeded_params = replace(params, noise_group_ids=recipe_ids)
    sde_sample_keys = [f"{recipe_id}:sample:{sample_id}" for recipe_id, sample_id in zip(recipe_ids, sample_ids)]
    return seeded_params, sde_sample_keys


def generate(pipeline: "HunyuanImage3Pipeline", sample: Sample) -> Sample:
    """t2i — single-stage text-to-image, filling the frontier (pre-forked) gen Part."""
    frontier = sample.frontier_gen_part(DiffusionSamplingParams)
    params = frontier.sampling_params
    params, sde_sample_keys = _prepare_seeded_sampling(sample, frontier, params)
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

    control = sample.parts[0].control or {}
    bot_task: str = str(control.get("bot_task", "image"))
    sys_type = control.get("sys_type")
    sequence_template = control.get("sequence_template")

    mm = pipeline.text_embed.embed_for_gen_image(
        texts,
        cfg=float(params.guidance_scale) > 1.0,
        height=int(params.height),
        width=int(params.width),
        bot_task=bot_task,
        sys_type=None if sys_type is None else str(sys_type),
        sequence_template=None if sequence_template is None else str(sequence_template),
    )

    diff_conds = HunyuanImage3DiffusionConditions(
        fused=mm["fused"],
        # Unread with the cache off, and Part.concat cannot merge it across prompts.
        tokenizer_output=mm["tokenizer_output"] if pipeline.diffusion.diffuse_kv_cache else None,
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
