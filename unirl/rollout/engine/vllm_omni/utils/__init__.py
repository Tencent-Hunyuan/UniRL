"""Pure helpers the adapters' conversion steps call (role 3).

No engine state, no runtime imports, no I/O — everything here unit-tests
with canned wire data (``SimpleNamespace`` fakes of the seam's
``OmniRawResult`` protocol). Conversion *logic* lives on the adapters'
input/output sub-adapters; these are the mechanics they lean on.
"""

from unirl.rollout.engine.vllm_omni.utils.diff_kwargs import core_diff_kwargs, sde_extra_args
from unirl.rollout.engine.vllm_omni.utils.noise import pack_initial_noise_extra_args
from unirl.rollout.engine.vllm_omni.utils.prompts import (
    grouped_texts_from_req,
    pil_images_from_req,
    texts_from_req,
)
from unirl.rollout.engine.vllm_omni.utils.sigmas import sigmas_list_from_req
from unirl.rollout.engine.vllm_omni.utils.tracks import (
    assemble_tracks,
    build_ar_segment,
    build_image_segment,
    collect_dit_outputs,
    decoded_text_from_ar,
    grouped_pils_to_videos,
    pick_stage_output,
    pils_to_images,
    seed_from_sample_id,
)

__all__ = [
    "assemble_tracks",
    "build_ar_segment",
    "build_image_segment",
    "collect_dit_outputs",
    "core_diff_kwargs",
    "decoded_text_from_ar",
    "grouped_texts_from_req",
    "grouped_pils_to_videos",
    "pack_initial_noise_extra_args",
    "pick_stage_output",
    "pil_images_from_req",
    "pils_to_images",
    "sde_extra_args",
    "seed_from_sample_id",
    "sigmas_list_from_req",
    "texts_from_req",
]
