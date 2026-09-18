"""Vendored MiniMax-H3 model code from the unmerged diffusers PR #14355.

See ``VENDOR_COMMIT.txt`` for the pinned commit, the file map, and the
mechanical import rewrite that is the only edit applied. Nothing here is
UniRL code -- keep it pristine so a re-vendor is a recopy plus
``vendorize.py``, not a merge.
"""

from .autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3
from .autoencoder_kl_minimax_h3_audio import AutoencoderKLMiniMaxH3Audio
from .packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_SHORT_EDGE,
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    MINIMAX_H3_TEXT_TAG,
    MiniMaxH3PackedSequence,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .scheduling_minimax_h3 import MiniMaxH3Scheduler
from .transformer_minimax_h3 import MiniMaxH3Transformer3DModel

__all__ = [
    "MINIMAX_H3_AUDIO_CHANNELS",
    "MINIMAX_H3_AUDIO_LATENTS_PER_SECOND",
    "MINIMAX_H3_CANVAS_MULTIPLE",
    "MINIMAX_H3_FPS",
    "MINIMAX_H3_MAX_ASPECT_RATIO",
    "MINIMAX_H3_MAX_DURATION",
    "MINIMAX_H3_MAX_PIXELS",
    "MINIMAX_H3_MIN_ASPECT_RATIO",
    "MINIMAX_H3_MIN_DURATION",
    "MINIMAX_H3_SHORT_EDGE",
    "MINIMAX_H3_TEXT_ENCODER_LAYER",
    "MINIMAX_H3_TEXT_TAG",
    "AutoencoderKLMiniMaxH3",
    "AutoencoderKLMiniMaxH3Audio",
    "MiniMaxH3PackedSequence",
    "MiniMaxH3Scheduler",
    "MiniMaxH3Transformer3DModel",
    "align_num_frames",
    "audio_latent_num_frames",
    "build_packed_sequence",
    "build_row_timesteps",
    "patchify_video_latents",
    "resolve_canvas_size",
    "unpack_audio_tokens",
    "unpatchify_video_tokens",
    "video_latent_num_frames",
]
