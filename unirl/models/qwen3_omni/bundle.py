"""Weights, processor, and tokenizer for the Qwen3-Omni Thinker path."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from unirl.models.types.bundle import Bundle
from unirl.utils.dtypes import parse_torch_dtype

from .config import Qwen3OmniPipelineConfig
logger = logging.getLogger(__name__)


class Qwen3OmniBundle(Bundle):
    """Qwen3-Omni Thinker bundle.

    Thinker-only mode (default)
        ``transformer`` is the standalone Thinker CausalLM (existing RL path).

    Compatibility Talker mode (``config.enable_talker=True``)
        :meth:`from_config` delegates to the independent
        :class:`Qwen3OmniTalkerBundle`; it does not load a full Omni model.
    """

    def __init__(
        self,
        *,
        transformer: nn.Module,
        processor: Any,
        tokenizer: Any,
        dtype: torch.dtype,
        device: torch.device,
        pretrained_path: str,
        omni: Optional[nn.Module] = None,
        enable_talker: bool = False,
        default_speaker: str = "Ethan",
        tts_system_instruction: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.processor = processor
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.device = device
        self.pretrained_path = pretrained_path
        self.omni = omni
        self.enable_talker = bool(enable_talker)
        self.default_speaker = str(default_speaker)
        self.tts_system_instruction = tts_system_instruction

    @property
    def thinker(self) -> nn.Module:
        if self.omni is not None and hasattr(self.omni, "thinker"):
            return self.omni.thinker
        return self.transformer

    @property
    def talker(self) -> nn.Module:
        if self.omni is None or not hasattr(self.omni, "talker"):
            raise RuntimeError("Qwen3OmniBundle.talker requires enable_talker=True (full Omni load).")
        return self.omni.talker

    @property
    def code2wav(self) -> nn.Module:
        if self.omni is None or not hasattr(self.omni, "code2wav"):
            raise RuntimeError("Qwen3OmniBundle.code2wav requires enable_talker=True (full Omni load).")
        return self.omni.code2wav

    @classmethod
    def from_config(cls, config: Qwen3OmniPipelineConfig) -> "Qwen3OmniBundle":
        from transformers import AutoConfig, AutoProcessor

        if config.enable_talker:
            # Compatibility entry for existing recipes. Direct TTS now uses a
            # standalone Talker bundle and never constructs/full-forwards the
            # 30B Thinker. Thinker-only construction below remains unchanged.
            from .talker_bundle import Qwen3OmniTalkerBundle

            return Qwen3OmniTalkerBundle.from_config(config)

        path = config.pretrained_model_ckpt_path

        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        dtype = parse_torch_dtype(config.model_precision, field_name="model_precision")

        if config.meta_init_transformer:
            raise NotImplementedError(
                "Qwen3OmniBundle Thinker path: meta_init_transformer=True is not "
                "yet supported (needs a strip-'thinker.' checkpoint mapping). "
                "The standalone Qwen3OmniTalkerBundle supports Talker meta-init."
            )

        load_kwargs = {}
        if getattr(config, "attn_implementation", None):
            load_kwargs["attn_implementation"] = str(config.attn_implementation)

        omni: Optional[nn.Module] = None
        from transformers.models.qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration

        full_cfg = AutoConfig.from_pretrained(path, trust_remote_code=bool(config.trust_remote_code))
        thinker_cfg = full_cfg.thinker_config
        transformer = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            path,
            config=thinker_cfg,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **load_kwargs,
        ).to(device)

        if config.freeze_vision_tower and hasattr(transformer, "visual"):
            transformer.visual.requires_grad_(False)
            logger.info("Froze thinker vision tower (%d params).", sum(1 for _ in transformer.visual.parameters()))
        if config.freeze_audio_tower and hasattr(transformer, "audio_tower"):
            transformer.audio_tower.requires_grad_(False)
            logger.info("Froze thinker audio tower (%d params).", sum(1 for _ in transformer.audio_tower.parameters()))

        if config.use_gradient_checkpointing:
            if hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            else:
                logger.warning(
                    "Qwen3-Omni thinker %s does not expose gradient_checkpointing_enable; skipping.",
                    type(transformer).__name__,
                )

        processor = AutoProcessor.from_pretrained(
            path,
            trust_remote_code=bool(config.trust_remote_code),
        )
        tokenizer = getattr(processor, "tokenizer", None) or processor
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        return cls(
            transformer=transformer,
            processor=processor,
            tokenizer=tokenizer,
            dtype=dtype,
            device=device,
            pretrained_path=path,
            omni=omni,
            enable_talker=bool(config.enable_talker),
            default_speaker=str(config.default_speaker),
            tts_system_instruction=config.tts_system_instruction,
        )


__all__ = ["Qwen3OmniBundle"]
