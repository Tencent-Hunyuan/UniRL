"""t2i — text-to-image diffusion.

Reads ``primitives["text"]: Texts`` plus ``stage_params["diffusion"]:
dict``. Builds the unified-MM input tensors via
``HunyuanImage3TextEmbedStage.embed_for_gen_image``, runs the diffusion
stage in ``mode="gen_image"``, and decodes the final latent to pixels.

``negative_text`` is rejected: the HI3 tokenizer never consumes
negative-prompt text — CFG is derived from ``guidance_scale > 1.0`` and
the unconditional branch is built internally from ``<cfg>`` tokens.

The ``bot_task`` knob (``stage_params["bot_task"]``) is a chat-template
flag: ``"image"`` is vllm-omni's t2i_vanilla preset; ``"think"`` /
``"recaption"`` / ``"think_recaption"`` insert static markers that the
model treats as reasoning-mode hints. This is NOT a separate AR-then-
diffuse pass -- vllm-omni's t2i is a single diffusion stage and the
prefix lives in ``input_ids`` only (see vllm-omni
``prompt_utils.py:23-31``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unirl.config.require import require
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from ..conditions import HunyuanImage3DiffusionConditions

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def generate(pipeline: "HunyuanImage3Pipeline", sample: Sample) -> Sample:
    """t2i — single-stage text-to-image, filling the frontier (pre-forked) gen Part."""
    frontier = sample.frontier_gen_part(DiffusionSamplingParams)
    params = frontier.sampling_params
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

    bot_task: str = str((sample.parts[0].control or {}).get("bot_task", "image"))

    # Build the upstream multimodal input tensors. CFG-batched [cond, uncond]
    # when guidance > 1; else single batch axis. ``mm`` is
    # ``{"fused": HunyuanImage3FusedMultimodalCondition, "tokenizer_output": Any}``.
    mm = pipeline.text_embed.embed_for_gen_image(
        texts,
        cfg=float(params.guidance_scale) > 1.0,
        height=int(params.height),
        width=int(params.width),
        bot_task=bot_task,
    )

    diff_conds = HunyuanImage3DiffusionConditions(
        fused=mm["fused"],
        tokenizer_output=mm["tokenizer_output"],
    )
    schedule = params.sigmas.to(pipeline.bundle.device)

    latent_seg = pipeline.diffusion.diffuse(diff_conds, schedule=schedule, params=params)
    images = pipeline.vae_decode.decode(latent_seg)

    filled = frontier.fill(segment=latent_seg, primitives={"image": images}, conditions=diff_conds.to_dict())
    return sample.with_parts([*sample.parts[:-1], filled])
