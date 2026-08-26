"""Construction config for the MiniMax-H3 t2va (text -> video+audio) pipeline.

MiniMax-H3 is a 33B dense omni-modal transformer that denoises video and stereo
audio jointly in ONE packed sequence. Three properties drive most of this file
and are checkpoint contracts, not knobs:

* **Guidance is distilled into the weights** -- no CFG, no negative prompt, one
  transformer forward per step. ``guidance_scale`` is unused.
* **Two schedules, one per modality**: video ``shift=12.0``, audio ``shift=3.0``.
  Both are plain rectified-flow grids, so UniRL's *static* schedule branch
  reproduces them exactly (see ``pipeline.build_schedule_policy``).
* **The checkpoint is mixed-dtype**: patch projections, timestep MLP and output
  heads are fp32 while the block stack is bf16. See ``meta_init_transformer``
  below and the recipe's ``root_wrap: false``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.config.validation import validate_precision_type

# MiniMax-H3 geometry (fixed for the family): the video VAE compresses 16x
# spatially and 4x temporally into 24 latent channels; the transformer patchifies
# (t, h, w) = (1, 2, 2). Module-level so pipeline (driver ``latent_shape`` +
# unpack) and the diffusion stage (row geometry) share one source without a cycle.
MINIMAX_H3_SPATIAL_COMPRESSION = 16
MINIMAX_H3_TEMPORAL_COMPRESSION = 4
MINIMAX_H3_LATENT_CHANNELS = 24
MINIMAX_H3_PATCH_SIZE = (1, 2, 2)
# Audio VAE latent width -- the feature dim of ONE packed audio row, and the
# transformer's ``audio_in_channels``. Distinct from
# ``vendor.MINIMAX_H3_AUDIO_CHANNELS`` (= 2), which counts the STEREO channels
# that make audio occupy two channel-major row blocks. Conflating the two makes
# the audio x_T 2 wide instead of 32 and mis-shapes ``audio_proj_in``.
MINIMAX_H3_AUDIO_LATENT_CHANNELS = 32


@dataclass
class MiniMaxH3PipelineConfig:
    """Construction args for ``MiniMaxH3Pipeline.from_config``."""

    pretrained_model_ckpt_path: str
    vae_ckpt_path: Optional[str] = None
    text_encoder_ckpt_path: Optional[str] = None

    model_precision: Any = "bf16"
    # Both VAEs are pinned fp32 by the checkpoint. The audio VAE in particular
    # is NOT safe in bf16 -- the reference reports output roughly 20 dB too
    # quiet. These default to fp32 rather than to ``model_precision`` for that
    # reason; overriding them is a deliberate act.
    vae_dtype: Any = "fp32"
    audio_vae_dtype: Any = "fp32"
    text_encoder_dtype: Any = None
    device: Any = None

    # Stage-level precision / numerical policy.
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    # Per-modality rectified-flow shifts. These are the released inference
    # values; they are NOT free parameters -- the model was distilled against
    # them, and video/audio alignment depends on both grids.
    video_shift: float = 12.0
    audio_shift: float = 3.0

    # MiniMax-H3 conditions on the UNNORMALIZED hidden state its Qwen3-VL
    # conditioner produces after its 50th decoder layer, i.e.
    # ``hidden_states[50]`` -- not ``last_hidden_state``.
    text_encoder_hidden_layer: int = 50
    max_sequence_length: int = 512

    # When True (default) video AND audio form a single joint SDE policy: audio
    # is SDE-stepped on its OWN schedule with the same ``eta`` as video, emits
    # its own per-step log-prob, and the two merge by an element-weighted mean
    # (the mean a single SDE over the concatenated ``[video|audio]`` latent
    # would produce). This keeps the RL importance ratio consistent with the
    # audio<->video coupling the packed sequence creates. False denoises audio
    # with ODE (``eta=0``, no log-prob) so only video carries policy signal.
    audio_joint_sde: bool = True

    # Used only when a recipe pins no height/width: ``resolve_canvas_size`` turns
    # an aspect ratio into a canvas by starting the short edge at 768. That is
    # this helper's default rather than a limit, so a recipe may pin a smaller
    # canvas -- see ``MiniMaxH3Geometry.resolve`` for the real constraints.
    # Duration is clamped to [5, 15]s at a fixed 24 fps.
    default_aspect_width: float = 1.0
    default_aspect_height: float = 1.0
    default_duration_seconds: float = 5.0

    # Build the trainable transformer on meta and materialize after sharding,
    # avoiding the ~66 GB/rank load spike. The bundle forwards the model's own
    # ``_keep_in_fp32_modules`` so the mixed-dtype layout survives; the eager
    # path (False) gets that from ``from_pretrained`` natively.
    meta_init_transformer: bool = False

    # Keep the frozen 32B Qwen3-VL conditioner on CPU instead of the train
    # device. It is ~64 GB in bf16 -- larger than the trainable DiT's per-rank
    # shard -- and is only needed to embed prompts, once per rollout, so the
    # transfer cost is negligible against a 22k-row denoising loop. This is what
    # makes an 8-GPU trainside recipe fit at all.
    aux_components_on_cpu: bool = False

    # The two VAEs are a SEPARATE decision from the conditioner, and default to
    # the train device even when the conditioner is parked. Together they are
    # only ~10 GB fp32, and decoding 124 frames of 768x768 through the video VAE
    # on CPU takes minutes per sample -- enough to dominate a rollout that
    # spends ~2 minutes denoising. Set True only if GPU memory genuinely forces
    # it, and expect decode to become the bottleneck.
    vae_components_on_cpu: bool = False

    weight_sync_param_name_prefix: str = "transformer."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="MiniMaxH3PipelineConfig.model_precision")


__all__ = [
    "MINIMAX_H3_AUDIO_LATENT_CHANNELS",
    "MINIMAX_H3_LATENT_CHANNELS",
    "MINIMAX_H3_PATCH_SIZE",
    "MINIMAX_H3_SPATIAL_COMPRESSION",
    "MINIMAX_H3_TEMPORAL_COMPRESSION",
    "MiniMaxH3PipelineConfig",
]
