"""Pure, model-agnostic helpers the ``sglang_diffusion`` adapters call.

No SGLang import, no engine state — fully unit-testable with canned data. The
*customizable* conversion steps live as overridable methods on the adapters; this
package is the generic mechanics those methods lean on.
"""

from unirl.rollout.engine.sglang_diffusion.utils.prompts import (
    deexpand_prompts_from_groups,
)
from unirl.rollout.engine.sglang_diffusion.utils.tensors import (
    decode_sample,
    fuse_encoder_outputs,
    normalize_media,
    tensorize,
)
from unirl.rollout.engine.sglang_diffusion.utils.tracks import (
    build_latent_segment,
    collect_trajectory_latents,
    derive_timestep_alignment,
    fuse_text_conditions,
    stack_decoded_images,
    stack_decoded_videos,
    validate_packed_trajectory,
)

__all__ = [
    "deexpand_prompts_from_groups",
    "decode_sample",
    "fuse_encoder_outputs",
    "normalize_media",
    "tensorize",
    "build_latent_segment",
    "collect_trajectory_latents",
    "derive_timestep_alignment",
    "fuse_text_conditions",
    "stack_decoded_images",
    "stack_decoded_videos",
    "validate_packed_trajectory",
]
