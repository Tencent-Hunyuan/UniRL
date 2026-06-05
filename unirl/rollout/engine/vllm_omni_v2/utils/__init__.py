"""Pure helpers the adapters' conversion steps call (role 3).

No engine state, no runtime imports, no I/O — everything here unit-tests
with canned wire data (``SimpleNamespace`` fakes of the seam's
``OmniRawResult`` protocol). Conversion *logic* lives on the per-shape base
adapters; these are the mechanics they lean on.
"""

from unirl.rollout.engine.vllm_omni_v2.utils.conditions import (
    build_ar_fused_condition,
    build_fused_mm_condition,
    build_hv15_conditions,
    build_sd3_text_condition,
)
from unirl.rollout.engine.vllm_omni_v2.utils.noise import pack_initial_noise_extra_args
from unirl.rollout.engine.vllm_omni_v2.utils.prompts import (
    build_prompt_entries,
    pil_images_from_req,
    resolve_task,
    texts_from_req,
)
from unirl.rollout.engine.vllm_omni_v2.utils.sigmas import sigmas_list_from_req
from unirl.rollout.engine.vllm_omni_v2.utils.tracks import (
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
    "build_ar_fused_condition",
    "build_ar_segment",
    "build_fused_mm_condition",
    "build_hv15_conditions",
    "build_image_segment",
    "build_prompt_entries",
    "build_sd3_text_condition",
    "collect_dit_outputs",
    "decoded_text_from_ar",
    "grouped_pils_to_videos",
    "pack_initial_noise_extra_args",
    "pick_stage_output",
    "pil_images_from_req",
    "pils_to_images",
    "resolve_task",
    "seed_from_sample_id",
    "sigmas_list_from_req",
    "texts_from_req",
]
