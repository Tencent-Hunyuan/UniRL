"""Cosmos3 SFT configuration.

One plain dataclass consumed by :class:`~unirl.models.cosmos3.bundle.Cosmos3Bundle`
and the packed Cosmos3 pipeline. Recipes reference it by
``_target_: unirl.models.cosmos3.config.Cosmos3SFTConfig`` — no registration.

Cosmos3-Nano is a 16B Mixture-of-Transformers: a causal "understanding" (und)
text stream and a bidirectional "generation" (gen) stream share each
``Cosmos3VLTextMoTDecoderLayer`` with disjoint parameter sets. SFT here trains
the gen stream (velocity prediction) and freezes the und stream by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Cosmos3SFTConfig:
    """Weights + flow-matching-SFT knobs for the Cosmos3 family."""

    pretrained_model_ckpt_path: str

    # -- precision / placement -------------------------------------------------
    # FSDP compute dtype.
    model_precision: str = "bf16"
    # Uniform transformer storage / optimizer-master dtype. fp32 storage with
    # bf16 mixed-precision compute matches the official Cosmos recipes.
    master_precision: str = "fp32"
    # WanVAE was trained with amp off; encode/decode run in the VAE's own dtype.
    vae_precision: str = "fp32"
    device: str = "cuda"

    # -- trainability ----------------------------------------------------------
    # Freeze the und (AR/text) parameter set: embed_tokens, lm_head, und
    # attention/MLP/norms. The gen stream (add_*_proj / mlp_moe_gen /
    # *_moe_gen norms / proj_in / proj_out / time_embedder / modality heads)
    # stays trainable.
    freeze_understanding: bool = True

    # -- flow-matching training schedule ----------------------------------------
    # Explicit fixed override. None selects the per-resolution table below,
    # falling back to the checkpoint scheduler for an unknown tier.
    flow_shift: Optional[float] = None
    # Official short-edge tier mapping: tier-256/480/720 -> shift 3/5/10.
    flow_shift_by_resolution: Dict[str, float] = field(default_factory=lambda: {"256": 3.0, "480": 5.0, "720": 10.0})
    # Training-time sigma distribution before the shift warp:
    # "logitnormal" (sigmoid of N(mean, std), the upstream action-SFT choice)
    # or "uniform".
    time_dist: str = "logitnormal"
    logitnormal_mean: float = 0.0
    logitnormal_std: float = 1.0

    # -- prompting (must mirror inference so tokenize_prompt output matches) ----
    use_system_prompt: bool = True
    add_resolution_template: bool = True
    add_duration_template: bool = True

    # -- vision task -------------------------------------------------------------
    fps: float = 15.0
    # Video prediction / i2v-style SFT: latent frame 0 is clean conditioning
    # (the observation), later frames are noised. False -> pure t2v/t2i SFT.
    condition_on_first_frame: bool = True
    vision_loss_weight: float = 1.0

    # -- action BC (policy-mode) -------------------------------------------------
    # Embodiment domain for the DomainAwareLinear action heads. DROID/Franka
    # single-arm = "droid_lerobot" (domain id 8). The canonical Cosmos3 width
    # for that domain is 10 (9D EE pose + 1D gripper); debug datasets that only
    # carry another action layout (e.g. droid_100's 7-D flattened action) may
    # override ``raw_action_dim`` — fine for finetuning, but then the head no
    # longer matches the base checkpoint's pretrained action semantics.
    action_domain_name: str = "droid_lerobot"
    action_chunk_size: int = 16
    raw_action_dim: int = 7
    action_loss_weight: float = 10.0
    action_view_point: str = "concat_view"


__all__ = ["Cosmos3SFTConfig"]
