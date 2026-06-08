"""t2i — text-to-image diffusion.

Reads ``primitives["text"]: Texts`` (and optionally ``negative_text``)
plus ``stage_params["diffusion"]: dict``. Builds the unified-MM input
tensors via ``Bundle.build_t2i_inputs``, runs the diffusion stage in
``mode="gen_image"``, and decodes the final latent to pixels.

The ``bot_task`` knob (``stage_params["bot_task"]``) is a chat-template
flag: ``"image"`` is vllm-omni's t2i_vanilla preset; ``"think"`` /
``"recaption"`` / ``"think_recaption"`` insert static markers that the
model treats as reasoning-mode hints. This is NOT a separate AR-then-
diffuse pass -- vllm-omni's t2i is a single diffusion stage and the
prefix lives in ``input_ids`` only (see vllm-omni
``prompt_utils.py:23-31``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

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
    if not isinstance(texts, Texts):
        raise TypeError(
            f"HunyuanImage3Pipeline.generate (t2i): req.primitives['text'] "
            f"must be Texts, "
            f"got {type(texts).__name__ if texts is not None else 'None'}"
        )
    negatives_raw = req.primitives.get("negative_text")
    negatives = negatives_raw if isinstance(negatives_raw, Texts) else None
    if negatives is not None and len(negatives.texts) != len(texts.texts):
        raise ValueError(
            f"HunyuanImage3Pipeline.generate (t2i): negative_text length "
            f"{len(negatives.texts)} != text length {len(texts.texts)}"
        )

    params: DiffusionSamplingParams = get_diffusion_params(req.sampling_params)
    bot_task: str = str(req.stage_config.get("bot_task", "image"))

    # Build the upstream multimodal input tensors. CFG-batched [cond, uncond]
    # when guidance > 1; else single batch axis. ``mm`` is
    # ``{"fused": HunyuanImage3FusedMultimodalCondition, "tokenizer_output": Any}``.
    cfg_on = float(params.guidance_scale) > 1.0
    if negatives is not None:
        neg_strs: Optional[List[str]] = list(negatives.texts)
    elif cfg_on:
        neg_strs = ["" for _ in texts.texts]
    else:
        neg_strs = None
    mm = pipeline.bundle.build_t2i_inputs(
        list(texts.texts),
        neg_strs,
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
