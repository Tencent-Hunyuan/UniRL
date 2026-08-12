"""Construction config for the Qwen3-Omni thinker AR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unirl.config.validation import validate_precision_type


@dataclass
class Qwen3OmniPipelineConfig:
    """Arguments for constructing a bundle and pipeline."""

    pretrained_model_ckpt_path: str
    tokenizer_ckpt_path: Optional[str] = None
    trust_remote_code: bool = True

    model_precision: Any = "bf16"
    # Global HF attention backend used by both replay and autoregression.
    attn_implementation: Optional[str] = None
    device: Any = None

    autocast_precision: str = "bf16"
    logprob_precision: str = "fp32"

    use_gradient_checkpointing: bool = False

    # Weight-sync prefix for the thinker's decoder under ``model``.
    weight_sync_param_name_prefix: str = "model."

    use_lora: bool = False
    lora_target_modules: Optional[List[str]] = None

    # Freeze embedded encoders while training the decoder.
    freeze_vision_tower: bool = True
    freeze_audio_tower: bool = True

    max_prompt_length: int = 4096

    # Video sampling rate used to derive TMRoPE timing.
    video_fps: float = 1.0
    video_max_frames: Optional[int] = None
    # Per-frame pixel cap passed to the processor as ``size.longest_edge``.
    video_max_pixels: Optional[int] = None
    # Whether to include the video's audio track in TMRoPE inputs.
    use_audio_in_video: bool = False

    # Talker direct-TTS supports meta-init and prefix-selective ``talker.*``
    # loading. The legacy Thinker-only path remains eager-only.
    meta_init_transformer: bool = False

    system_instruction: Optional[str] = None
    # Extra non-structural kwargs forwarded to the processor chat template
    # (for example tool schemas). Required tensor/tokenization return-shape
    # kwargs are enforced by the chat-template stage.
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)

    # --- Talker TTS RL / SFT ---
    # Compatibility switch: Qwen3OmniBundle.from_config delegates to the
    # standalone Qwen3OmniTalkerBundle. It loads only the frozen Thinker input
    # embedding table, Talker+MTP, and frozen Code2Wav.
    enable_talker: bool = False
    # Legacy option retained for recipe compatibility; direct TTS has no full
    # Thinker to freeze and always freezes its embedding provider.
    freeze_thinker: bool = True
    # Always freeze Code2Wav (decode-for-reward / SFT observation only).
    freeze_code2wav: bool = True
    # Phase-1 RL freezes the residual decoder and both embedding providers.
    freeze_mtp: bool = False
    freeze_talker_embeddings: bool = True
    phase1_layer0_lora_only: bool = False
    # Default TTS system prompt when the dataset does not supply one.
    tts_system_instruction: Optional[str] = (
        "You are a high-quality TTS model. Convert the user's text into natural speech audio."
    )
    # Default speaker when Part metadata omits ``speaker``.
    default_speaker: str = "Ethan"

    def __post_init__(self) -> None:
        validate_precision_type(self.model_precision, field="Qwen3OmniPipelineConfig.model_precision")
        if float(self.video_fps) <= 0.0:
            raise ValueError(f"Qwen3OmniPipelineConfig.video_fps must be > 0, got {self.video_fps!r}")
        if self.video_max_frames is not None and int(self.video_max_frames) < 1:
            raise ValueError(f"Qwen3OmniPipelineConfig.video_max_frames must be >= 1, got {self.video_max_frames!r}")
        if self.video_max_pixels is not None and int(self.video_max_pixels) < 1:
            raise ValueError(f"Qwen3OmniPipelineConfig.video_max_pixels must be >= 1, got {self.video_max_pixels!r}")
        if self.phase1_layer0_lora_only and not self.use_lora:
            raise ValueError(
                "Qwen3OmniPipelineConfig.phase1_layer0_lora_only requires use_lora=True"
            )
        if self.phase1_layer0_lora_only and not (
            self.freeze_mtp and self.freeze_code2wav and self.freeze_talker_embeddings
        ):
            raise ValueError(
                "Phase-1 layer0-LoRA mode requires freeze_mtp, freeze_code2wav, "
                "and freeze_talker_embeddings"
            )


__all__ = ["Qwen3OmniPipelineConfig"]
