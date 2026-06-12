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
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams, get_diffusion_params

from ..conditions import HunyuanImage3DiffusionConditions

if TYPE_CHECKING:
    from ..pipeline import HunyuanImage3Pipeline


def generate(pipeline: "HunyuanImage3Pipeline", req: RolloutReq) -> RolloutResp:
    """t2i — single-stage text-to-image."""
    texts = req.primitives.get("text")
    require(
        isinstance(texts, Texts),
        f"HunyuanImage3Pipeline.generate (t2i): input must be Texts, got {type(texts).__name__ if texts is not None else 'None'}",
    )
    require(
        req.primitives.get("negative_text") is None,
        "HunyuanImage3Pipeline.generate (t2i): negative_text is not supported — "
        "the HI3 tokenizer never consumes negative-prompt text; CFG is derived from "
        "guidance_scale > 1.0 (the unconditional branch is built internally from <cfg> tokens).",
    )

    params: DiffusionSamplingParams = get_diffusion_params(req.sampling_params)
    bot_task: str = str(req.stage_config.get("bot_task", "image"))

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
    if req.sigmas is None:
        raise ValueError(
            "HunyuanImage3 t2i: req.sigmas is None. Engine adapter must call "
            "unirl.sde.runtime.ensure_req_sigmas before pipeline.generate."
        )
    schedule = req.sigmas.to(pipeline.bundle.device)

    latent_seg = pipeline.diffusion.diffuse(diff_conds, schedule=schedule, params=params)
    images = pipeline.vae_decode.decode(latent_seg)

    return RolloutResp(
        tracks={
            "image": RolloutTrack(
                sample_ids=list(req.sample_ids),
                parent_ids=list(req.group_ids),
                conditions=diff_conds.to_dict(),
                segment=latent_seg,
                decoded=images,
            ),
        }
    )
