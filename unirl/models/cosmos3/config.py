"""Cosmos3 SFT configuration dataclass (MoT streams + knob semantics: see README.md)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Cosmos3SFTConfig:
    """Weights + flow-matching-SFT knobs for the Cosmos3 family."""

    pretrained_model_ckpt_path: str

    # -- precision / placement -------------------------------------------------
    model_precision: str = "bf16"  # FSDP compute dtype
    master_precision: str = "fp32"  # uniform transformer storage / optimizer-master dtype
    vae_precision: str = "fp32"  # WanVAE runs in its own dtype (trained with amp off)
    device: str = "cuda"
    meta_init_transformer: bool = False  # build on meta; backend loads sharded weights post-wrap

    # -- trainability ----------------------------------------------------------
    freeze_understanding: bool = True  # freeze the und (AR/text) stream; gen stream stays trainable

    # -- flow-matching training schedule ----------------------------------------
    flow_shift: Optional[float] = None  # fixed override; None -> per-resolution table below
    # Official short-edge tier mapping: tier-256/480/720 -> shift 3/5/10.
    flow_shift_by_resolution: Dict[str, float] = field(default_factory=lambda: {"256": 3.0, "480": 5.0, "720": 10.0})
    time_dist: str = "logitnormal"  # sigma distribution before the shift warp: "logitnormal" | "uniform"
    logitnormal_mean: float = 0.0
    logitnormal_std: float = 1.0

    # -- prompting (must mirror inference so tokenize_prompt output matches) ----
    use_system_prompt: bool = True
    add_resolution_template: bool = True
    add_duration_template: bool = True

    # -- vision task -------------------------------------------------------------
    fps: float = 15.0
    condition_on_first_frame: bool = True  # latent frame 0 stays clean (i2v-style); False -> pure t2v
    vision_loss_weight: float = 1.0

    # -- action BC (policy-mode; domain/width caveats: README # Gotchas) ---------
    action_domain_name: str = "droid_lerobot"
    action_chunk_size: int = 16
    raw_action_dim: int = 7
    action_loss_weight: float = 10.0
    action_view_point: str = "concat_view"


__all__ = ["Cosmos3SFTConfig"]
